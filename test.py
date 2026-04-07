import torch
import numpy as np
import pickle
import argparse
import os
from networks import VAE_Decoder, BilateralGenerator
from metrics import evaluate_all
from ultils import np_rounding
from visualise import visualise_gan


def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Real Data (for comparison)
    data_path = os.path.join('Data/', args.dataset)
    with open(os.path.join(data_path, 'vital_sign_24hrs.pkl'), 'rb') as f:
        continuous_x = pickle.load(f)
    with open(os.path.join(data_path, 'med_interv_24hrs.pkl'), 'rb') as f:
        discrete_x = pickle.load(f)

    # Sanitize the real data exactly as we did in training
    discrete_x = np.nan_to_num(discrete_x, nan=0.0)
    discrete_x = np.clip(discrete_x, 0.0, 1.0)
    continuous_x = np.nan_to_num(continuous_x, nan=0.0)

    # Calculate Normalization Stats (Needed for plotting real-world values)
    min_val_con = np.min(continuous_x, axis=(0, 1))
    max_val_con = np.max(continuous_x, axis=(0, 1))
    range_val_con = max_val_con - min_val_con
    range_val_con[range_val_con == 0] = 1e-6

    # Normalize continuous data to [0, 1] for the network
    continuous_x = (continuous_x - min_val_con) / range_val_con
    max_val_con = range_val_con  # Adjust max for the renormlizer function

    time_steps = continuous_x.shape[1]
    c_dim = continuous_x.shape[2]
    d_dim = discrete_x.shape[2]

    latent_dim = 25
    noise_dim = min(int(c_dim / 2), int(d_dim / 2))

    print(f"Loading checkpoint: {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 2. Initialize Networks (Generators and Decoders only)
    c_dec = VAE_Decoder(latent_dim, args.gen_num_units, c_dim, args.dec_layers).to(device)
    d_dec = VAE_Decoder(latent_dim, args.gen_num_units, d_dim, args.dec_layers).to(device)

    c_gen = BilateralGenerator(noise_dim, args.gen_num_units, latent_dim, args.gen_num_layers).to(device)
    d_gen = BilateralGenerator(noise_dim, args.gen_num_units, latent_dim, args.gen_num_layers).to(device)

    # 3. Load Saved Weights
    c_dec.load_state_dict(checkpoint['c_dec'])
    d_dec.load_state_dict(checkpoint['d_dec'])
    c_gen.load_state_dict(checkpoint['c_gen'])
    d_gen.load_state_dict(checkpoint['d_gen'])

    c_dec.eval();
    d_dec.eval();
    c_gen.eval();
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

            h_cpl_c = [torch.zeros(args.batch_size, args.gen_num_units, device=device) for _ in
                       range(args.gen_num_layers)]
            h_cpl_d = [torch.zeros(args.batch_size, args.gen_num_units, device=device) for _ in
                       range(args.gen_num_layers)]

            fake_z_c, h_cpl_d = c_gen(noise_c, h_cpl_d)
            fake_z_d, h_cpl_c = d_gen(noise_d, h_cpl_c)

            fake_c_seq, _ = c_dec(fake_z_c)
            fake_d_seq, _ = d_dec(fake_z_d)

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
    save_dir = "Output/test_plots/"
    os.makedirs(save_dir, exist_ok=True)

    # Passes the data to visualise_gan to output the PDF
    visualise_gan(real_c_eval, c_gen_data, real_d_eval, d_gen_data,
                  inx="Test_Run", max_val_con=max_val_con, min_val_con=min_val_con, SAVE_PATH=save_dir)

    print(f"✅ Success! PDF Visualizations saved to: {save_dir}visualise_gan_epoch_Test_Run.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Mimic3", help="Dataset folder name")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the saved .pth file")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help="Number of synthetic patients to generate for testing")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for generation")

    # Model architecture parameters (must match the training config)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)

    args = parser.parse_args()
    test_model(args)