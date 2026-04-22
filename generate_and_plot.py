import os
import sys
import torch
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Import components (Standard M3GAN by default, but can be switched)
from ultils import np_rounding, renormlizer

import importlib.util

def load_networks(is_neo):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if is_neo:
        module_path = os.path.join(current_dir, "neo_m3gan", "networks.py")
        module_name = "networks_neo"
    else:
        module_path = os.path.join(current_dir, "networks.py")
        module_name = "networks_root"
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    return net

def generate_single_sample(models, args, device, time_steps, noise_dim):
    c_vae, d_vae, joint_gen = models
    
    noise_c = torch.randn(1, time_steps, noise_dim, device=device)
    noise_d = torch.randn(1, time_steps, noise_dim, device=device)
    
    with torch.no_grad():
        fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)
        fake_c, _ = c_vae.reconstruct_decoder(fake_z_c)
        fake_d, _ = d_vae.reconstruct_decoder(fake_z_d)
        
    fake_c = fake_c.cpu().numpy()
    fake_d = np_rounding(fake_d.cpu().numpy())
    return fake_c[0], fake_d[0]

def plot_interactive(models, real_c, real_d, c_names, d_names, scaler, args, device, time_steps, noise_dim):
    sns.set_style("whitegrid")
    num_c_feats = len(c_names)
    
    print("\nFeature Selection:")
    print(f"Available Continuous Features: {', '.join([f'[{i}] {name}' for i, name in enumerate(c_names)])}")
    selection = input("Enter feature indices to plot (e.g. 0,1,5), or 'all', or 'random': ").strip().lower()
    
    if selection == 'all':
        plot_indices = list(range(num_c_feats))
    elif selection == 'random':
        plot_indices = np.random.choice(num_c_feats, min(5, num_c_feats), replace=False).tolist()
    else:
        try:
            plot_indices = [int(i.strip()) for i in selection.split(',') if i.strip().isdigit()]
            if not plot_indices: plot_indices = [0]
        except:
            plot_indices = [0]

    while True:
        print(f"\nGenerating NEW synthetic patient data...")
        fake_c, fake_d = generate_single_sample(models, args, device, time_steps, noise_dim)
        
        # Pick a random REAL patient for comparison
        real_idx = np.random.randint(0, len(real_c))
        r_c_sample = real_c[real_idx]
        r_d_sample = real_d[real_idx]

        # Un-normalize both if scaler provided
        if scaler:
            fake_c = renormlizer(fake_c.reshape(1, time_steps, -1), 
                                 scaler['maxes'] - scaler['mins'], scaler['mins'])[0]
            r_c_sample = renormlizer(r_c_sample.reshape(1, time_steps, -1), 
                                     scaler['maxes'] - scaler['mins'], scaler['mins'])[0]

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        
        # Column 1: Synthetic
        # Plot 1.1: Selected Continuous Features (Synthetic)
        ax0 = axes[0, 0]
        for i in plot_indices:
            if i < num_c_feats:
                ax0.plot(fake_c[:, i], label=c_names[i])
        ax0.set_title(f"Synthetic Patient: Continuous Features")
        ax0.legend(loc='upper right', fontsize='xx-small', ncol=1)
        
        # Column 2: Real
        # Plot 1.2: Selected Continuous Features (Real)
        ax2 = axes[0, 1]
        for i in plot_indices:
            if i < num_c_feats:
                ax2.plot(r_c_sample[:, i], label=c_names[i])
        ax2.set_title(f"Real Patient (ID {real_idx}): Continuous Features")

        # Synchronize Y-axis for continuous features
        y_min = min(ax0.get_ylim()[0], ax2.get_ylim()[0])
        y_max = max(ax0.get_ylim()[1], ax2.get_ylim()[1])
        ax0.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        # Plot 2.1: Discrete Features (Synthetic)
        ax1 = axes[1, 0]
        sns.heatmap(fake_d.T[:15, :], ax=ax1, cmap="Reds", cbar=False, yticklabels=d_names[:15])
        ax1.set_title(f"Synthetic Patient: Discrete Features (Top 15)")
        ax1.set_xlabel("Time Steps")

        # Plot 2.2: Discrete Features (Real)
        ax3 = axes[1, 1]
        sns.heatmap(r_d_sample.T[:15, :], ax=ax3, cmap="Greens", cbar=False, yticklabels=d_names[:15])
        ax3.set_title(f"Real Patient (ID {real_idx}): Discrete Features (Top 15)")
        ax3.set_xlabel("Time Steps")
        
        plt.tight_layout()
        plt.show()
        
        cont = input("Generate another NEW patient? (y/n): ")
        if cont.lower() != 'y':
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Mimic3")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Checkpoint and Detect Version
    checkpoint = torch.load(args.checkpoint, map_location=device)
    is_neo = False
    if 'c_gen' in checkpoint:
        first_key = list(checkpoint['c_gen'].keys())[0]
        if "cl." in first_key:
            is_neo = True
            print("[Info] Detected Neo-M3GAN architecture.")

    net = load_networks(is_neo)

    # 2. Load Metadata
    data_path = os.path.join('Data/', args.dataset)
    with open(os.path.join(data_path, 'vital_sign_24hrs.pkl'), 'rb') as f:
        real_c = pickle.load(f)
    time_steps, c_dim = real_c.shape[1], real_c.shape[2]
    
    with open(os.path.join(data_path, 'med_interv_24hrs.pkl'), 'rb') as f:
        real_d = pickle.load(f)
    d_dim = real_d.shape[2]
    
    # Pre-sanitize real data
    real_d = np.nan_to_num(real_d, nan=0.0)
    real_c = np.nan_to_num(real_c, nan=0.0)
    
    noise_dim = min(int(c_dim / 2), int(d_dim / 2))

    # 3. Initialize Models
    c_vae = net.AutoregressiveVAE(c_dim, args.gen_num_units, 25, args.enc_layers, args.dec_layers, time_steps).to(device)
    d_vae = net.AutoregressiveVAE(d_dim, args.gen_num_units, 25, args.enc_layers, args.dec_layers, time_steps).to(device)
    joint_gen = net.JointGenerator(noise_dim, noise_dim, args.gen_num_units, 25, 25, args.gen_num_layers).to(device)

    c_vae.load_state_dict(checkpoint['c_vae'])
    d_vae.load_state_dict(checkpoint['d_vae'])
    joint_gen.c_gen.load_state_dict(checkpoint['c_gen'])
    joint_gen.d_gen.load_state_dict(checkpoint['d_gen'])
    c_vae.eval(); d_vae.eval(); joint_gen.eval()

    # 4. Handle Scaler
    scaler = None
    scaler_path = os.path.join(data_path, 'clinical_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        c_names = scaler.get('feature_names')
        d_names = scaler.get('discrete_names')
    else:
        c_names = [f"C-Feat {i}" for i in range(c_dim)]
        d_names = [f"D-Feat {i}" for i in range(d_dim)]

    # 5. Start Plotting
    plot_interactive((c_vae, d_vae, joint_gen), real_c, real_d, c_names, d_names, scaler, args, device, time_steps, noise_dim)


if __name__ == "__main__":
    main()
