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
    
    # 1. Load Real Data
    data_path = os.path.abspath(os.path.join(current_dir, '..', 'Data', args.dataset))
    print(f"Using Data directory: {data_path}")

    vital_path = os.path.join(data_path, 'vital_sign_24hrs.pkl')
    med_path = os.path.join(data_path, 'med_interv_24hrs.pkl')

    if not os.path.exists(vital_path) or not os.path.exists(med_path):
        raise FileNotFoundError(f"Data files not found in {data_path}. Please check your path.")

    with open(vital_path, 'rb') as f:
        continuous_x = pickle.load(f)
    with open(med_path, 'rb') as f:
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
        min_val_con = np.min(continuous_x, axis=(0, 1))
        max_val_con = np.max(continuous_x, axis=(0, 1))
        range_val_con = max_val_con - min_val_con
        range_val_con[range_val_con == 0] = 1e-6

    # Normalise continuous data to [0, 1] range to evaluate metrics properly
    continuous_x = (continuous_x - min_val_con) / range_val_con
    
    time_steps = continuous_x.shape[1]
    c_dim = continuous_x.shape[2]
    d_dim = discrete_x.shape[2]

    latent_dim = 25
    noise_dim = min(int(c_dim / 2), int(d_dim / 2))

    # Search checkpoint
    checkpoint_path = args.checkpoint
    search_paths = [
        checkpoint_path,
        os.path.join(current_dir, checkpoint_path),
        os.path.join(current_dir, "..", checkpoint_path)
    ]
    
    found_path = None
    for p in search_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and os.path.isfile(abs_p):
            found_path = abs_p
            break
            
    if not found_path:
        print(f"\n❌ ERROR: Checkpoint not found! Checked search paths: {search_paths}")
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    print(f"[OK] Found V2 checkpoint: {found_path}")
    checkpoint = torch.load(found_path, map_location=device)

    # 2. Initialize Networks
    c_vae = AutoregressiveVAE(c_dim, args.gen_num_units, latent_dim, args.enc_layers, args.dec_layers, time_steps).to(device)
    d_vae = AutoregressiveVAE(d_dim, args.gen_num_units, latent_dim, args.enc_layers, args.dec_layers, time_steps).to(device)
    joint_gen = JointGenerator(noise_dim, noise_dim, args.gen_num_units, latent_dim, latent_dim, args.gen_num_layers).to(device)

    # 3. Load Saved Weights
    c_vae.load_state_dict(checkpoint['c_vae'])
    d_vae.load_state_dict(checkpoint['d_vae'])
    
    # Check if they are stored in joint_gen or sub-generators
    if 'c_gen' in checkpoint:
        joint_gen.c_gen.load_state_dict(checkpoint['c_gen'])
    if 'd_gen' in checkpoint:
        joint_gen.d_gen.load_state_dict(checkpoint['d_gen'])

    c_vae.eval()
    d_vae.eval()
    joint_gen.eval()

    # 4. Generate Synthetic Data
    print(f"Generating {args.num_samples} synthetic patients using V2 Coupled Generator...")
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
    print("Generating validation plots...")
    save_dir = os.path.abspath(os.path.join(current_dir, "..", args.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    num_plot = 10  # Number of patient samples to overlay in each plot

    # Extract epoch or filename to use as an identifier in the visualization saved images
    inx_id = checkpoint.get('epoch', os.path.basename(found_path).split('.')[0])

    # Un-normalize data before visualizing so graphs plot clinical range units
    real_c_eval = real_c_eval * range_val_con + min_val_con
    c_gen_data = c_gen_data * range_val_con + min_val_con

    # Passes the data to visualise_gan to output the PDF
    visualise_gan(real_c_eval, c_gen_data, real_d_eval, d_gen_data, f"v2_{inx_id}", range_val_con, min_val_con,
                  num_dim=12, num_plot=num_plot, SAVE_PATH=save_dir,
                  c_feature_names=c_feature_names, d_feature_names=d_feature_names)

    print(f"Success! PDF Visualizations saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mimic", help="Dataset folder name")
    parser.add_argument('--checkpoint', type=str, default="Output/checkpoint/best_m3gan_v2.pth", help="Path to the saved .pth file")
    parser.add_argument('--num_samples', type=int, default=5000, help="Number of synthetic patients to generate for testing")
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
