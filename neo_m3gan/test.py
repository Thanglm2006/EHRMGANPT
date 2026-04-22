import torch
import numpy as np
import pickle
import argparse
import os
from networks import AutoregressiveVAE, JointGenerator
from metrics import evaluate_all
from ultils import np_rounding
from visualise import visualise_gan


def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get the directory where test.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Real Data (Data is always in the root folder, one level up from neo_m3gan/)
    data_path = os.path.abspath(os.path.join(current_dir, '..', 'Data', args.dataset))
    print(f"Using Data directory: {data_path}")

    with open(os.path.join(data_path, 'vital_sign_24hrs.pkl'), 'rb') as f:
        continuous_x = pickle.load(f)
    with open(os.path.join(data_path, 'med_interv_24hrs.pkl'), 'rb') as f:
        discrete_x = pickle.load(f)

    # Sanitize the real data exactly as we did in training
    discrete_x = np.nan_to_num(discrete_x, nan=0.0)
    discrete_x = np.clip(discrete_x, 0.0, 1.0)
    continuous_x = np.nan_to_num(continuous_x, nan=0.0)

    # Load scaling metadata and feature names
    c_feature_names = None
    d_feature_names = None
    
    # Auto-detect scaler if not provided
    scaler_to_use = args.clinical_scaler
    if not scaler_to_use:
        potential_scaler = os.path.join(data_path, 'clinical_scaler.pkl')
        if os.path.exists(potential_scaler):
            scaler_to_use = potential_scaler
            print(f"Auto-detected clinical scaler at: {scaler_to_use}")

    if scaler_to_use and os.path.exists(scaler_to_use):
        print(f"Applying clinical scaler from: {scaler_to_use}")
        with open(scaler_to_use, 'rb') as f:
            scaler = pickle.load(f)
        min_val_con = scaler.get('mins')
        range_val_con = scaler.get('maxes') - min_val_con
        c_feature_names = scaler.get('feature_names')
        d_feature_names = scaler.get('discrete_names')
    else:
        # If no scaler provided, calculate local stats from the loaded data 
        # (which is likely already 0-1 normalized).
        min_val_con = np.min(continuous_x, axis=(0, 1))
        max_val_con = np.max(continuous_x, axis=(0, 1))
        range_val_con = max_val_con - min_val_con
        range_val_con[range_val_con == 0] = 1e-6

    # Data is already 0-1 from the .pkl, so we do not re-normalize.
    # The min_val_con and range_val_con will be used by the visualizer to un-normalize.
    
    time_steps = continuous_x.shape[1]
    c_dim = continuous_x.shape[2]
    d_dim = discrete_x.shape[2]

    latent_dim = 25
    noise_dim = min(int(c_dim / 2), int(d_dim / 2))

    # --- SMART CHECKPOINT RESOLUTION ---
    # We prioritize the provided path, but fallback to root/parent folders and 
    # handle common naming discrepancies (like the 'neo_' prefix) to be helpful.
    
    checkpoint_path = args.checkpoint
    search_paths = [
        checkpoint_path,                                      # 1. Exact path provided
        os.path.join(current_dir, checkpoint_path),           # 2. Local to script
        os.path.join(current_dir, '..', checkpoint_path),     # 3. One level up (project root)
    ]
    
    # Also search for 'm3gan_...' if 'neo_m3gan_...' was requested (and vice-versa)
    base = os.path.basename(checkpoint_path)
    alt_base = base.replace("neo_", "", 1) if base.startswith("neo_") else f"neo_{base}"
    if alt_base != base:
        dir_name = os.path.dirname(checkpoint_path)
        search_paths.extend([
            os.path.join(dir_name, alt_base),
            os.path.join(current_dir, dir_name, alt_base),
            os.path.join(current_dir, '..', dir_name, alt_base),
        ])

    found_path = None
    for p in search_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and os.path.isfile(abs_p):
            found_path = abs_p
            break
            
    if not found_path:
        print("\n❌ ERROR: Checkpoint not found!")
        print(f"Tried searching for: '{args.checkpoint}' and variants.")
        print("\nAvailable checkpoints in this project:")
        # List a few available ones to help the user
        root_dir = os.path.abspath(os.path.join(current_dir, '..'))
        for r, d, f in os.walk(root_dir):
            if "checkpoint" in r.lower():
                for file in f:
                    if file.endswith(".pth"):
                        print(f" - {os.path.relpath(os.path.join(r, file), current_dir)}")
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    print(f"[OK] Found checkpoint: {found_path}")
    checkpoint = torch.load(found_path, map_location=device)

    # 2. Initialize Networks
    c_vae = AutoregressiveVAE(c_dim, args.gen_num_units, latent_dim, args.enc_layers, args.dec_layers, time_steps).to(device)
    d_vae = AutoregressiveVAE(d_dim, args.gen_num_units, latent_dim, args.enc_layers, args.dec_layers, time_steps).to(device)

    joint_gen = JointGenerator(noise_dim, noise_dim, args.gen_num_units, latent_dim, latent_dim, args.gen_num_layers).to(device)
    c_gen = joint_gen.c_gen
    d_gen = joint_gen.d_gen

    # 3. Load Saved Weights
    c_vae.load_state_dict(checkpoint['c_vae'])
    d_vae.load_state_dict(checkpoint['d_vae'])
    c_gen.load_state_dict(checkpoint['c_gen'])
    d_gen.load_state_dict(checkpoint['d_gen'])

    c_vae.eval()
    d_vae.eval()
    c_gen.eval()
    d_gen.eval()

    # 4. Generate Synthetic Data
    print(f"Generating {args.num_samples} synthetic patients...")
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

    # Combine batches and trim to exact requested sample size
    c_gen_data = np.concatenate(c_gen_data, axis=0)[:args.num_samples]
    d_gen_data = np_rounding(np.concatenate(d_gen_data, axis=0))[:args.num_samples]

    # Subsample real data to match the synthetic data size for a fair metric calculation
    indices = np.random.choice(continuous_x.shape[0], args.num_samples, replace=False)
    real_c_eval = continuous_x[indices]
    real_d_eval = discrete_x[indices]

    # 5. Run Evaluation Metrics
    evaluate_all(real_c_eval, c_gen_data, real_d_eval, d_gen_data)

    # 6. Save Visualization PDF
    print("Generating visualization plots...")
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    num_plot = 10  # Number of patient samples to overlay in each plot

    # Extract epoch or filename to use as an identifier in the visualization saved images
    inx_id = checkpoint.get('epoch', os.path.basename(found_path).split('.')[0])

    # Passes the data to visualise_gan to output the PDF
    visualise_gan(real_c_eval, c_gen_data, real_d_eval, d_gen_data, inx_id, range_val_con, min_val_con,
                  num_dim=12, num_plot=num_plot, SAVE_PATH=save_dir,
                  c_feature_names=c_feature_names, d_feature_names=d_feature_names)

    print(f"Success! PDF Visualizations saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Mimic3", help="Dataset folder name")
    parser.add_argument('--checkpoint', type=str, default="Output/checkpoint/neo_m3gan_100.pth", help="Path to the saved .pth file")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help="Number of synthetic patients to generate for testing")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for generation")
    parser.add_argument('--clinical_scaler', type=str, default=None, help="Path to clinical_scaler.pkl to restore raw units")

    # Model architecture parameters (must match the training config)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default="Output/test_plots/")

    args = parser.parse_args()
    test_model(args)