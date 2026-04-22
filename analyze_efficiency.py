import os
import sys
import torch
import numpy as np
import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import metrics from root
from metrics import evaluate_all
from ultils import np_rounding

def load_data(args, current_dir):
    data_path = os.path.abspath(os.path.join(current_dir, 'Data', args.dataset))
    with open(os.path.join(data_path, 'vital_sign_24hrs.pkl'), 'rb') as f:
        continuous_x = pickle.load(f)
    with open(os.path.join(data_path, 'med_interv_24hrs.pkl'), 'rb') as f:
        discrete_x = pickle.load(f)

    discrete_x = np.nan_to_num(discrete_x, nan=0.0)
    discrete_x = np.clip(discrete_x, 0.0, 1.0)
    continuous_x = np.nan_to_num(continuous_x, nan=0.0)
    
    return continuous_x, discrete_x

def get_model_versions():
    """
    Returns the network classes for both versions.
    We import them dynamically to avoid namespace collisions if possible,
    or just use the ones from the respective directories.
    """
    # 1. Root Version
    import networks as root_net
    
    # 2. Neo Version (Adding to path temporarily)
    sys.path.insert(0, os.path.abspath("neo_m3gan"))
    import networks as neo_net
    sys.path.pop(0)
    
    return root_net, neo_net

def run_test_iteration(model_type, checkpoint_path, real_c, real_d, args, device):
    root_net, neo_net = get_model_versions()
    
    c_dim, d_dim = real_c.shape[2], real_d.shape[2]
    time_steps = real_c.shape[1]
    latent_dim = 25
    noise_dim = min(int(c_dim / 2), int(d_dim / 2))
    
    net = neo_net if model_type == "neo" else root_net
    
    c_vae = net.AutoregressiveVAE(c_dim, args.gen_num_units, latent_dim, args.enc_layers, args.dec_layers, time_steps).to(device)
    d_vae = net.AutoregressiveVAE(d_dim, args.gen_num_units, latent_dim, args.enc_layers, args.dec_layers, time_steps).to(device)
    joint_gen = net.JointGenerator(noise_dim, noise_dim, args.gen_num_units, latent_dim, latent_dim, args.gen_num_layers).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    c_vae.load_state_dict(checkpoint['c_vae'])
    d_vae.load_state_dict(checkpoint['d_vae'])
    joint_gen.c_gen.load_state_dict(checkpoint['c_gen'])
    joint_gen.d_gen.load_state_dict(checkpoint['d_gen'])
    
    c_vae.eval(); d_vae.eval(); joint_gen.eval()
    
    c_gen_data = []
    d_gen_data = []
    num_batches = int(np.ceil(args.num_samples / args.batch_size))

    with torch.no_grad():
        for _ in range(num_batches):
            noise_c = torch.randn(args.batch_size, time_steps, noise_dim, device=device)
            noise_d = torch.randn(args.batch_size, time_steps, noise_dim, device=device)
            fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)
            fake_c_seq, _ = c_vae.reconstruct_decoder(fake_z_c)
            fake_d_seq, _ = d_vae.reconstruct_decoder(fake_z_d)
            c_gen_data.append(fake_c_seq.cpu().numpy())
            d_gen_data.append(fake_d_seq.cpu().numpy())

    c_gen_data = np.concatenate(c_gen_data, axis=0)[:args.num_samples]
    d_gen_data = np_rounding(np.concatenate(d_gen_data, axis=0))[:args.num_samples]

    indices = np.random.choice(real_c.shape[0], args.num_samples, replace=False)
    scores = evaluate_all(real_c[indices], c_gen_data, real_d[indices], d_gen_data)
    return scores

def main():
    parser = argparse.ArgumentParser(description="Efficiency Analysis: M3GAN vs Neo-M3GAN")
    parser.add_argument('--dataset', type=str, default="Mimic3")
    parser.add_argument('--checkpoint_root', type=str, required=True, help="Path to standard m3gan checkpoint")
    parser.add_argument('--checkpoint_neo', type=str, required=True, help="Path to neo_m3gan checkpoint")
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_iterations', type=int, default=50, help="Number of times to repeat the test for stats")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    real_c, real_d = load_data(args, current_dir)

    results = {"root": [], "neo": []}
    
    for i in range(args.num_iterations):
        print(f"\n--- Iteration {i+1}/{args.num_iterations} ---")
        print("Testing Standard M3GAN...")
        results["root"].append(run_test_iteration("root", args.checkpoint_root, real_c, real_d, args, device))
        
        print("Testing Neo M3GAN...")
        results["neo"].append(run_test_iteration("neo", args.checkpoint_neo, real_c, real_d, args, device))

    # Calculate statistics
    stats = []
    for model_name, scores_list in results.items():
        df = pd.DataFrame(scores_list)
        summary = df.describe().loc[['mean', 'max', 'min', 'std']]
        summary.index = [f"{model_name}_{idx}" for idx in summary.index]
        # Rename 'std' to 'var' and square it for variance if requested, 
        # or just keep std. User asked for variance.
        summary.loc[f"{model_name}_var"] = df.var()
        stats.append(summary)

    final_stats = pd.concat(stats)
    print("\nFinal Comparison Summary:")
    print(final_stats)

    # Save to PDF
    os.makedirs("Output/analysis/", exist_ok=True)
    pdf_path = "Output/analysis/efficiency_comparison.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Table Page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        ax.set_title(f"Efficiency Comparison (n={args.num_iterations}, samples={args.num_samples})", fontsize=16)
        table = ax.table(cellText=final_stats.values.round(5),
                         colLabels=final_stats.columns,
                         rowLabels=final_stats.index,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        pdf.savefig(fig)
        plt.close()

        # Visualization Page: Boxplots for metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Metric Distribution Comparison", fontsize=18)
        metrics = ['mmd', 'rmse', 'corr_c', 'corr_d']
        titles = ["Continuous MMD", "Discrete RMSE", "Continuous Corr Error", "Discrete Corr Error"]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            plot_data = [pd.DataFrame(results["root"])[metric], pd.DataFrame(results["neo"])[metric]]
            ax.boxplot(plot_data, labels=["Standard", "Neo"])
            ax.set_title(titles[idx])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

    print(f"\nAnalysis complete. Results saved to: {pdf_path}")

if __name__ == "__main__":
    main()
