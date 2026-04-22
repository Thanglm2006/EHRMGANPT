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

import importlib.util

def load_networks(model_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if model_type == "neo":
        module_path = os.path.join(current_dir, "neo_m3gan", "networks.py")
        module_name = "networks_neo"
    else:
        module_path = os.path.join(current_dir, "networks.py")
        module_name = "networks_root"
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    return net

def run_test_iteration(model_type, checkpoint_path, real_c, real_d, args, device):
    # Dynamically load the correct network module
    net = load_networks(model_type)
    
    c_dim, d_dim = real_c.shape[2], real_d.shape[2]
    time_steps = real_c.shape[1]
    latent_dim = 25
    noise_dim = min(int(c_dim / 2), int(d_dim / 2))
    
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

import re

def find_checkpoints(directory):
    checkpoints = {}
    if not os.path.exists(directory):
        return checkpoints
    
    for file in os.listdir(directory):
        if not file.endswith(".pth"):
            continue
            
        # Ignore pretrain or best files to avoid confusion
        if "pretrain" in file.lower() or "best" in file.lower():
            continue
            
        # Extract epoch number using regex
        # Matches any sequence of digits at the end of the filename (before .pth)
        match = re.search(r'_(\d+)\.pth$', file)
        if match:
            epoch = int(match.group(1))
            checkpoints[epoch] = os.path.join(directory, file)
            
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Efficiency Analysis: M3GAN vs Neo-M3GAN (All Epochs)")
    parser.add_argument('--dataset', type=str, default="Mimic3")
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_iterations', type=int, default=50, help="Iterations per epoch for statistical stability")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    real_c, real_d = load_data(args, current_dir)

    # 1. Discover Checkpoints
    root_ckpt_dir = os.path.join(current_dir, "Output", "checkpoint")
    neo_ckpt_dir = os.path.join(current_dir, "neo_m3gan", "Output", "checkpoint")
    
    root_ckpts = find_checkpoints(root_ckpt_dir)
    neo_ckpts = find_checkpoints(neo_ckpt_dir)
    
    common_epochs = sorted(list(set(root_ckpts.keys()) & set(neo_ckpts.keys())))
    
    if not common_epochs:
        print("❌ No matching epochs found between root and neo checkpoints!")
        print(f"Root epochs: {sorted(root_ckpts.keys())}")
        print(f"Neo epochs: {sorted(neo_ckpts.keys())}")
        return

    print(f"Found {len(common_epochs)} matching epochs: {common_epochs}")

    all_results = []
    
    # 2. Evaluate All Epochs
    for epoch in common_epochs:
        print(f"\n" + "="*50)
        print(f" EVALUATING EPOCH {epoch}")
        print("="*50)
        
        for i in range(args.num_iterations):
            # Test Root
            res_root = run_test_iteration("root", root_ckpts[epoch], real_c, real_d, args, device)
            res_root.update({'epoch': epoch, 'model': 'standard', 'run': i})
            all_results.append(res_root)
            
            # Test Neo
            res_neo = run_test_iteration("neo", neo_ckpts[epoch], real_c, real_d, args, device)
            res_neo.update({'epoch': epoch, 'model': 'neo', 'run': i})
            all_results.append(res_neo)

    df_results = pd.DataFrame(all_results)
    
    # 3. Save and Plot
    os.makedirs("Result/analysis/", exist_ok=True)
    pdf_path = "Result/analysis/training_efficiency_comparison.pdf"
    csv_path = "Result/analysis/full_metrics_long.csv"
    comp_path = "Result/analysis/metrics_comparison_summary.csv"
    
    # Save long format
    df_results.to_csv(csv_path, index=False)
    
    # Create comprehensive comparison summary (Mean, Std, Min, Max)
    # We aggregate first to get raw numbers for plotting
    stats_raw = df_results.groupby(['epoch', 'model']).agg(['mean', 'std', 'min', 'max']).reset_index()
    # Flatten columns: ('mmd', 'mean') -> 'mmd_mean'
    stats_raw.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in stats_raw.columns]
    
    # Create formatted strings 'mean ± std' for the CSV
    summary_formatted = pd.DataFrame()
    summary_formatted['epoch'] = stats_raw['epoch']
    summary_formatted['model'] = stats_raw['model']
    
    for metric in ['mmd', 'rmse', 'corr_c', 'corr_d']:
        summary_formatted[f'{metric}_mean_std'] = [
            f"{m:.5f} ± {s:.5f}" for m, s in zip(stats_raw[f'{metric}_mean'], stats_raw[f'{metric}_std'])
        ]
        summary_formatted[f'{metric}_min'] = stats_raw[f'{metric}_min']
        summary_formatted[f'{metric}_max'] = stats_raw[f'{metric}_max']
    
    summary_formatted.to_csv(comp_path, index=False)
    
    # Create a pivoted mean-only summary for plotting and PDF table
    summary_mean = df_results.groupby(['epoch', 'model']).mean().reset_index()
    comparison_pivot = summary_mean.pivot(index='epoch', columns='model')
    comparison_pivot.columns = [f"{col[0]}_{col[1]}" for col in comparison_pivot.columns]
    
    with PdfPages(pdf_path) as pdf:
        # Trend Plots
        metrics = ['mmd', 'rmse', 'corr_c', 'corr_d']
        titles = ["Continuous MMD Trend", "Discrete RMSE Trend", "Continuous Corr Error Trend", "Discrete Corr Error Trend"]
        
        for idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(10, 6))
            for model_type in ['standard', 'neo']:
                m_data = stats_raw[stats_raw['model'] == model_type]
                ax.errorbar(m_data['epoch'], m_data[f'{metric}_mean'], yerr=m_data[f'{metric}_std'], 
                            label=model_type.capitalize(), marker='o', capsize=5)
            
            ax.set_title(titles[idx], fontsize=14)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score (Lower is Better)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close()

        # Summary Table Page
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        ax.set_title("Training Metrics Side-by-Side Comparison", fontsize=16)
        
        # Format the pivoted data for the table
        table_df = comparison_pivot.round(5).reset_index()
        table = ax.table(cellText=table_df.values,
                         colLabels=table_df.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        pdf.savefig(fig)
        plt.close()

    print(f"\nAnalysis complete!")
    print(f"PDF Report: {pdf_path}")
    print(f"CSV Data:   {csv_path}")

if __name__ == "__main__":
    main()
