# main.py
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import argparse
from trainer import train_m3gan
from metrics import evaluate_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main(args):
    # 1. Load Real Data
    data_path = os.path.join('Data/', args.dataset)

    with open(os.path.join(data_path, 'vital_sign_24hrs.pkl'), 'rb') as f:
        continuous_x = pickle.load(f)

    with open(os.path.join(data_path, 'med_interv_24hrs.pkl'), 'rb') as f:
        discrete_x = pickle.load(f)

    discrete_x = np.nan_to_num(discrete_x, nan=0.0)
    discrete_x = np.clip(discrete_x, 0.0, 1.0)
    continuous_x = np.nan_to_num(continuous_x, nan=0.0)

    min_val_con = np.min(continuous_x, axis=(0, 1))
    max_val_con = np.max(continuous_x, axis=(0, 1))

    # Prevent division by zero for constant features
    range_val_con = max_val_con - min_val_con
    range_val_con[range_val_con == 0] = 1e-6

    # Normalize the data to [0, 1] for the VAE Sigmoid output
    continuous_x = (continuous_x - min_val_con) / range_val_con
    max_val_con = range_val_con

    # 2. Extract Dimensions
    time_steps = continuous_x.shape[1]
    c_dim = continuous_x.shape[2]
    d_dim = discrete_x.shape[2]

    # Paper defaults
    shared_latent_dim = 25
    noise_dim = min(int(c_dim / 2), int(d_dim / 2))

    # 3. Create PyTorch DataLoader
    dataset = TensorDataset(
        torch.tensor(continuous_x, dtype=torch.float32),
        torch.tensor(discrete_x, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 4. Setup Configuration Dictionary
    config = {
        'batch_size': args.batch_size,
        'time_steps': time_steps,
        'c_dim': c_dim,
        'd_dim': d_dim,
        'latent_dim': shared_latent_dim,
        'hidden_dim': args.gen_num_units,
        'noise_dim': noise_dim,

        'enc_layers': args.enc_layers,
        'dec_layers': args.dec_layers,
        'gen_layers': args.gen_num_layers,
        'dis_layers': args.dis_num_layers,

        'num_pre_epochs': args.num_pre_epochs,
        'num_epochs': args.num_epochs,
        'epoch_ckpt_freq': args.epoch_ckpt_freq,

        'v_lr_pre': args.v_lr_pre,
        'v_lr': args.v_lr,
        'g_lr': args.g_lr,
        'd_lr': args.d_lr,

        'd_rounds': args.d_rounds,
        'g_rounds': args.g_rounds,
        'v_rounds': args.v_rounds,

        'alpha_re': args.alpha_re,
        'alpha_kl': args.alpha_kl,
        'alpha_mt': args.alpha_mt,
        'alpha_ct': args.alpha_ct,

        'c_beta_adv': args.c_beta_adv,
        'c_beta_fm': args.c_beta_fm,
        'd_beta_adv': args.d_beta_adv,
        'd_beta_fm': args.d_beta_fm,

        'resume_checkpoint': args.resume_checkpoint,
        'patience': args.patience,
        'skip_pretrain': args.skip_pretrain
    }

    # 5. Start Training
    print("Starting EHR-M-GAN PyTorch Training...")
    train_m3gan(dataloader, config, max_val_con, min_val_con)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mimic", choices=['Mimic3', 'eicu', 'hirid'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_pre_epochs', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--epoch_ckpt_freq', type=int, default=100)
    parser.add_argument('--d_rounds', type=int, default=2)
    parser.add_argument('--g_rounds', type=int, default=1)
    parser.add_argument('--v_rounds', type=int, default=1)
    parser.add_argument('--v_lr_pre', type=float, default=0.0005)
    parser.add_argument('--v_lr', type=float, default=0.0001)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--alpha_re', type=float, default=1)
    parser.add_argument('--alpha_kl', type=float, default=0.5)
    parser.add_argument('--alpha_mt', type=float, default=0.1)
    parser.add_argument('--alpha_ct', type=float, default=0.1)
    parser.add_argument('--c_beta_adv', type=float, default=1)
    parser.add_argument('--c_beta_fm', type=float, default=10.0)
    parser.add_argument('--d_beta_adv', type=float, default=1.0)
    parser.add_argument('--d_beta_fm', type=float, default=10.0)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--dis_num_layers', type=int, default=3)
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to .pth file to load weights from")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience for VAE pretraining")
    parser.add_argument('--skip_pretrain', action='store_true', help="Skip Phase 1: VAE Pretraining and go straight to Joint GAN")


    args = parser.parse_args()
    main(args)