import torch
import os
import numpy as np
import pickle
import argparse
from torch.utils.data import TensorDataset

from trainer import train_m3gan


def main(args):
    # Setup dataset paths
    data_path = os.path.join('Data/', args.dataset)
    
    vital_path = os.path.join(data_path, 'vital_sign_24hrs.pkl')
    med_path = os.path.join(data_path, 'med_interv_24hrs.pkl')
    
    if not os.path.exists(vital_path) or not os.path.exists(med_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please make sure 'vital_sign_24hrs.pkl' and 'med_interv_24hrs.pkl' exist.")

    # Load vital signs (continuous) and medication interventions (discrete)
    with open(vital_path, 'rb') as f:
        continuous_x = pickle.load(f)
    with open(med_path, 'rb') as f:
        discrete_x = pickle.load(f)

    # Preprocess discrete and continuous features
    discrete_x = np.nan_to_num(np.clip(discrete_x, 0.0, 1.0), nan=0.0)
    continuous_x = np.nan_to_num(continuous_x, nan=0.0)

    # Normalize continuous vital signs to [0, 1] range
    min_val_con = np.min(continuous_x, axis=(0, 1))
    max_val_con = np.max(continuous_x, axis=(0, 1))
    range_con = max_val_con - min_val_con
    range_con[range_con == 0.0] = 1e-6 # Avoid division by zero
    
    continuous_x = (continuous_x - min_val_con) / range_con

    # Prepare configuration dictionary
    config = {
        'batch_size': args.batch_size,
        'pre_batch_size': args.pre_batch_size,
        'time_steps': continuous_x.shape[1],
        'c_dim': continuous_x.shape[2],
        'd_dim': discrete_x.shape[2],
        'latent_dim': 25,
        'hidden_dim': args.gen_num_units,
        'noise_dim': min(continuous_x.shape[2] // 2, discrete_x.shape[2] // 2),
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
        'skip_pretrain': args.skip_pretrain,
        'use_amp': args.use_amp,
        'resume_checkpoint': args.resume_checkpoint,
        'resume_vae_only': args.resume_vae_only
    }

    # Load custom scale names if available
    c_feature_names = None
    d_feature_names = None
    
    # Feature columns names (for MIMIC-III standard)
    if args.dataset.lower() == 'mimic':
        c_feature_names = ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Mean BP', 'Resp Rate', 'Temp C', 'SpO2', 'Glucose']
        d_feature_names = ['Adenosine', 'Dobutamine', 'Dopamine', 'Epinephrine', 'Isuprel', 'Milrinone', 'Norepinephrine', 'Phenylephrine', 'Vasopressin']

    # Convert to Tensor Dataset and trigger training
    dataset = TensorDataset(
        torch.tensor(continuous_x, dtype=torch.float32),
        torch.tensor(discrete_x, dtype=torch.float32)
    )

    print("\n" + "=" * 50)
    print(" EHR-M-GAN V2 TRAINING INITIATED")
    print(f" Dataset:    {args.dataset}")
    print(f" Devices:    {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f" Pre-Epochs: {args.num_pre_epochs} (Skip: {args.skip_pretrain})")
    print(f" GAN Epochs: {args.num_epochs}")
    print(f" TTUR:       d_lr={args.d_lr}, g_lr={args.g_lr}")
    print(f" WGAN-GP:    d_rounds={args.d_rounds}")
    print("=" * 50 + "\n")

    train_m3gan(dataset, config, range_con, min_val_con, c_feature_names, d_feature_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EHR-M-GAN V2 Command Line Interface")
    parser.add_argument('--dataset', type=str, default="mimic", help="Dataset directory under 'Data/' (e.g., mimic)")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for Phase 2 joint training")
    parser.add_argument('--pre_batch_size', type=int, default=1024, help="Batch size for Phase 1 VAE pretraining")
    parser.add_argument('--num_pre_epochs', type=int, default=500, help="Number of pretraining epochs")
    parser.add_argument('--num_epochs', type=int, default=800, help="Number of joint VAE-GAN training epochs")
    parser.add_argument('--epoch_ckpt_freq', type=int, default=50, help="Epoch frequency for evaluation and checkpoints")
    
    # Optimizers learning rates (TTUR settings optimized for WGAN-GP by default)
    parser.add_argument('--v_lr_pre', type=float, default=0.0005, help="Learning rate for Phase 1 VAE")
    parser.add_argument('--v_lr', type=float, default=0.0001, help="Learning rate for Phase 2 VAE")
    parser.add_argument('--g_lr', type=float, default=0.0001, help="Learning rate for Phase 2 Generator")
    parser.add_argument('--d_lr', type=float, default=0.0003, help="Learning rate for Phase 2 Discriminator (TTUR)")
    
    # Critic updates per generator update (standard WGAN-GP default = 5)
    parser.add_argument('--d_rounds', type=int, default=5, help="Number of discriminator updates per G step")
    parser.add_argument('--g_rounds', type=int, default=1)
    parser.add_argument('--v_rounds', type=int, default=1)
    
    # VAE Loss Scaling weights
    parser.add_argument('--alpha_re', type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument('--alpha_kl', type=float, default=0.5, help="KL divergence loss weight")
    parser.add_argument('--alpha_mt', type=float, default=0.1, help="Bilateral latent matching loss weight")
    parser.add_argument('--alpha_ct', type=float, default=0.1, help="Latent contrastive loss weight")
    
    # GAN Loss Scaling weights
    parser.add_argument('--c_beta_adv', type=float, default=1.0, help="Continuous GAN loss scaling")
    parser.add_argument('--c_beta_fm', type=float, default=10.0, help="Continuous Feature Matching loss scaling")
    parser.add_argument('--d_beta_adv', type=float, default=1.0, help="Discrete GAN loss scaling")
    parser.add_argument('--d_beta_fm', type=float, default=10.0, help="Discrete Feature Matching loss scaling")
    
    # Network layers
    parser.add_argument('--enc_layers', type=int, default=3, help="Number of layers in VAE Encoder LSTM")
    parser.add_argument('--dec_layers', type=int, default=3, help="Number of layers in VAE Decoder LSTM")
    parser.add_argument('--gen_num_units', type=int, default=512, help="Hidden units in Bilateral cells")
    parser.add_argument('--gen_num_layers', type=int, default=3, help="Number of layers in Generator LSTMs")
    parser.add_argument('--dis_num_layers', type=int, default=3, help="Number of layers in Discriminator LSTMs")
    
    parser.add_argument('--skip_pretrain', action='store_true', default=False, help="Skip Phase 1 and go straight to joint VAE-GAN training")
    parser.add_argument('--use_amp', action='store_true', default=True, help="Enable PyTorch mixed precision (AMP) for speed")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint .pth file to resume training")
    parser.add_argument('--resume_vae_only', action='store_true', default=False, help="If true, only load VAE weights from checkpoint and skip GAN weights.")

    main(parser.parse_args())
