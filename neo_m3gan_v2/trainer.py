import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks import AutoregressiveVAE, SequenceDiscriminator, JointGenerator
from ultils import nt_xent_loss, kl_divergence, feature_matching_loss, np_rounding
from visualise import visualise_gan, plot_metrics_trend
from metrics import evaluate_all


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty for WGAN-GP (Enforces Float32 for double-backward stability)."""
    with torch.amp.autocast(device.type, enabled=False):
        alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
        interpolates = (alpha * real_samples.float() + ((1 - alpha) * fake_samples.float())).requires_grad_(True)
        
        # Disable cuDNN benchmarking during gradient penalty calculation to prevent memory leaks
        with torch.backends.cudnn.flags(enabled=False):
            d_interpolates, _ = discriminator(interpolates)
            
        fake = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    # Add epsilon to L2 norm calculation to prevent division-by-zero NaN in autograd
    grad_norms = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((grad_norms - 1) ** 2).mean()
    return gradient_penalty


def train_m3gan(dataset, config, max_val_con, min_val_con, c_feature_names=None, d_feature_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c_dim, d_dim = config['c_dim'], config['d_dim']
    latent_dim = config['latent_dim']
    hidden_dim = config['hidden_dim']
    noise_dim = config['noise_dim']
    time_steps = config['time_steps']

    # Using VAE Class to wrap the new Autoregressive architecture
    c_vae = AutoregressiveVAE(c_dim, hidden_dim, latent_dim, config['enc_layers'], config['dec_layers'], time_steps).to(device)
    d_vae = AutoregressiveVAE(d_dim, hidden_dim, latent_dim, config['enc_layers'], config['dec_layers'], time_steps).to(device)

    # Initialize Joint Generator (handles bilateral coupling internally, with mapping networks in V2)
    joint_gen = JointGenerator(noise_dim, noise_dim, hidden_dim, latent_dim, latent_dim, config['gen_layers']).to(device)
    c_gen = joint_gen.c_gen
    d_gen = joint_gen.d_gen

    # Spectral Normalization Critic to enforce Lipschitz continuity
    c_dis = SequenceDiscriminator(c_dim, hidden_dim, time_steps, config.get('dis_layers', 3)).to(device)
    d_dis = SequenceDiscriminator(d_dim, hidden_dim, time_steps, config.get('dis_layers', 3)).to(device)

    # Separate parameters for optimizing
    vae_params = list(c_vae.parameters()) + list(d_vae.parameters())
    dis_params = list(c_dis.parameters()) + list(d_dis.parameters())

    optimizer_VAE_pre = optim.Adam(vae_params, lr=config['v_lr_pre'])
    optimizer_VAE = optim.Adam(vae_params, lr=config['v_lr'])
    optimizer_G = optim.Adam(joint_gen.parameters(), lr=config['g_lr'])
    optimizer_D = optim.Adam(dis_params, lr=config['d_lr'])

    scaler = torch.amp.GradScaler('cuda', enabled=config.get('use_amp', False))

    # Resume from checkpoint if specified
    if config.get('resume_checkpoint'):
        print(f"\nLoading checkpoint from {config['resume_checkpoint']}...")
        checkpoint = torch.load(config['resume_checkpoint'], map_location=device)
        
        c_vae.load_state_dict(checkpoint['c_vae'])
        d_vae.load_state_dict(checkpoint['d_vae'])
        print("Successfully loaded VAE weights from checkpoint.")
        
        if not config.get('resume_vae_only', False):
            if 'c_gen' in checkpoint:
                print("Detected pretrained GAN weights in checkpoint. Loading...")
                c_gen.load_state_dict(checkpoint['c_gen'])
                d_gen.load_state_dict(checkpoint['d_gen'])
                c_dis.load_state_dict(checkpoint['c_dis'])
                d_dis.load_state_dict(checkpoint['d_dis'])
        else:
            print("Flag --resume_vae_only is active. Skipping GAN generator/discriminator weight initialization.")

    # ----------------------------------------------------
    # Phase 1: VAE Pretraining (Contrastive VAE Learning)
    # ----------------------------------------------------
    if not config.get('skip_pretrain', False):
        print("\n=== Starting Phase 1: VAE Pretraining ===")
        dataloader = DataLoader(dataset, batch_size=config['pre_batch_size'], shuffle=True, drop_last=True)
        print(f"Phase 1 Batch Size: {config['pre_batch_size']}, Iterations: {len(dataloader)}")

        for epoch in range(config['num_pre_epochs']):
            # Linear KL weight annealing over the first 100 epochs to prevent posterior collapse
            kl_anneal_factor = min(1.0, epoch / 100.0)
            c_real_lst, c_rec_lst, d_real_lst, d_rec_lst = [], [], [], []
            epoch_total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Pretrain Epoch [{epoch + 1}/{config['num_pre_epochs']}]", leave=True)

            for continuous_x, discrete_x in pbar:
                continuous_x = torch.clamp(torch.nan_to_num(continuous_x.to(device), nan=0.0), 0.0, 1.0)
                discrete_x = torch.clamp(torch.nan_to_num(discrete_x.to(device), nan=0.0), 0.0, 1.0)

                optimizer_VAE_pre.zero_grad()

                with torch.amp.autocast(device.type, enabled=config.get('use_amp', False)):
                    c_rec, _, c_mu, c_logvar, c_z = c_vae(continuous_x)
                    d_rec, d_logits, d_mu, d_logvar, d_z = d_vae(discrete_x)

                    loss_c_rec = F.mse_loss(c_rec, continuous_x)
                    loss_d_rec = F.binary_cross_entropy_with_logits(d_logits, discrete_x)

                    loss_c_kl = kl_divergence(c_mu, c_logvar)
                    loss_d_kl = kl_divergence(d_mu, d_logvar)

                    c_z_flat = c_z.view(c_z.size(0), -1)
                    d_z_flat = d_z.view(d_z.size(0), -1)
                    loss_contrastive = nt_xent_loss(c_z_flat, d_z_flat)
                    loss_matching = F.mse_loss(c_z, d_z)

                    total_vae_loss = (config['alpha_re'] * (loss_c_rec + loss_d_rec) +
                                      (config['alpha_kl'] * kl_anneal_factor) * (loss_c_kl + loss_d_kl) +
                                      config['alpha_ct'] * loss_contrastive +
                                      config['alpha_mt'] * loss_matching)

                scaler.scale(total_vae_loss).backward()
                # Unscale gradients before clipping when using AMP
                scaler.unscale_(optimizer_VAE_pre)
                torch.nn.utils.clip_grad_norm_(vae_params, max_norm=5.0)
                scaler.step(optimizer_VAE_pre)
                scaler.update()

                epoch_total_loss += total_vae_loss.item()
                pbar.set_postfix(
                    c_rec=f"{loss_c_rec.item():.4f}",
                    d_rec=f"{loss_d_rec.item():.4f}",
                    tot_vae=f"{total_vae_loss.item():.4f}"
                )

            print(f"Pretrain Epoch {epoch + 1}: Loss = {epoch_total_loss / len(dataloader):.4f}")
            
            # Save periodic pretrain check
            if (epoch + 1) % 50 == 0:
                os.makedirs("Output/checkpoint", exist_ok=True)
                torch.save({
                    'c_vae': c_vae.state_dict(),
                    'd_vae': d_vae.state_dict()
                }, f"Output/checkpoint/vae_pretrain_{epoch + 1}.pth")

    # ----------------------------------------------------
    # Phase 2: Joint VAE-GAN Training
    # ----------------------------------------------------
    print("\n=== Starting Phase 2: Joint VAE-GAN Training (V2) ===")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    # Trackers for quantitative validation
    history_dict = {'epochs': [], 'mmd': [], 'rmse': [], 'corr_c': [], 'corr_d': []}
    best_gan_score = float('inf')

    for epoch in range(config['num_epochs']):
        c_dis_loss_lst, d_dis_loss_lst, g_loss_lst, vae_loss_lst = [], [], [], []
        pbar = tqdm(dataloader, desc=f"GAN Epoch [{epoch + 1}/{config['num_epochs']}]", leave=True)

        for continuous_x, discrete_x in pbar:
            continuous_x = torch.clamp(torch.nan_to_num(continuous_x.to(device), nan=0.0), 0.0, 1.0)
            discrete_x = torch.clamp(torch.nan_to_num(discrete_x.to(device), nan=0.0), 0.0, 1.0)
            batch_size = continuous_x.size(0)

            # ----------------------------------------------------
            # 1. TRAIN DISCRIMINATORS (CRITICS)
            # ----------------------------------------------------
            for _ in range(config['d_rounds']):
                optimizer_D.zero_grad()

                # Generate fake latents
                noise_c = torch.randn(batch_size, time_steps, noise_dim, device=device)
                noise_d = torch.randn(batch_size, time_steps, noise_dim, device=device)

                with torch.no_grad():
                    fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)
                    fake_c_seq, _ = c_vae.reconstruct_decoder(fake_z_c)
                    fake_d_seq, _ = d_vae.reconstruct_decoder(fake_z_d)

                with torch.amp.autocast(device.type, enabled=config.get('use_amp', False)):
                    # Compute Wasserstein Loss with spectral norm and gradient penalty
                    real_c_score, _ = c_dis(continuous_x)
                    fake_c_score, _ = c_dis(fake_c_seq.detach())

                    real_d_score, _ = d_dis(discrete_x)
                    fake_d_score, _ = d_dis(fake_d_seq.detach())

                    loss_c_gp = compute_gradient_penalty(c_dis, continuous_x, fake_c_seq.detach(), device)
                    loss_d_gp = compute_gradient_penalty(d_dis, discrete_x, fake_d_seq.detach(), device)

                    # Wasserstein Critics minimize Wasserstein distance negative
                    loss_c_dis = fake_c_score.mean() - real_c_score.mean() + 10 * loss_c_gp
                    loss_d_dis = fake_d_score.mean() - real_d_score.mean() + 10 * loss_d_gp
                    total_dis_loss = config.get('c_beta_adv', 1.0) * loss_c_dis + config.get('d_beta_adv', 1.0) * loss_d_dis

                scaler.scale(total_dis_loss).backward()
                # Unscale gradients before clipping when using AMP
                scaler.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(dis_params, max_norm=5.0)
                scaler.step(optimizer_D)
                scaler.update()

                c_dis_loss_lst.append(loss_c_dis.item())
                d_dis_loss_lst.append(loss_d_dis.item())

            # ----------------------------------------------------
            # 2. TRAIN GENERATOR
            # ----------------------------------------------------
            optimizer_G.zero_grad()

            with torch.amp.autocast(device.type, enabled=config.get('use_amp', False)):
                noise_c = torch.randn(batch_size, time_steps, noise_dim, device=device)
                noise_d = torch.randn(batch_size, time_steps, noise_dim, device=device)

                # Bilateral Coupled Forward pass
                fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)

                fake_c_seq, _ = c_vae.reconstruct_decoder(fake_z_c)
                fake_d_seq, _ = d_vae.reconstruct_decoder(fake_z_d)

                # Extract features from critics to apply Feature Matching loss
                fake_c_score, fake_c_feat = c_dis(fake_c_seq)
                fake_d_score, fake_d_feat = d_dis(fake_d_seq)

                with torch.no_grad():
                    _, real_c_feat = c_dis(continuous_x)
                    _, real_d_feat = d_dis(discrete_x)

                loss_c_gen = -fake_c_score.mean()
                loss_d_gen = -fake_d_score.mean()

                loss_c_fm = feature_matching_loss(fake_c_feat, real_c_feat)
                loss_d_fm = feature_matching_loss(fake_d_feat, real_d_feat)

                # Generator loss consists of adversarial and feature matching metrics
                total_gen_loss = (config.get('c_beta_adv', 1.0) * loss_c_gen +
                                  config.get('d_beta_adv', 1.0) * loss_d_gen +
                                  config.get('c_beta_fm', 10.0) * loss_c_fm +
                                  config.get('d_beta_fm', 10.0) * loss_d_fm)

            scaler.scale(total_gen_loss).backward()
            # Unscale gradients before clipping when using AMP
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(joint_gen.parameters(), max_norm=5.0)
            scaler.step(optimizer_G)
            scaler.update()
            g_loss_lst.append(total_gen_loss.item())

            # ----------------------------------------------------
            # 3. TRAIN VAE (Joint VAE Loss)
            # ----------------------------------------------------
            optimizer_VAE.zero_grad()

            with torch.amp.autocast(device.type, enabled=config.get('use_amp', False)):
                c_rec, _, c_mu, c_logvar, c_z = c_vae(continuous_x)
                d_rec, d_logits, d_mu, d_logvar, d_z = d_vae(discrete_x)

                loss_c_rec = F.mse_loss(c_rec, continuous_x)
                loss_d_rec = F.binary_cross_entropy_with_logits(d_logits, discrete_x)

                loss_c_kl = kl_divergence(c_mu, c_logvar)
                loss_d_kl = kl_divergence(d_mu, d_logvar)

                c_z_flat = c_z.view(c_z.size(0), -1)
                d_z_flat = d_z.view(d_z.size(0), -1)
                loss_contrastive = nt_xent_loss(c_z_flat, d_z_flat)
                loss_matching = F.mse_loss(c_z, d_z)

                total_vae_loss = (config['alpha_re'] * (loss_c_rec + loss_d_rec) +
                                  config['alpha_kl'] * (loss_c_kl + loss_d_kl) +
                                  config['alpha_ct'] * loss_contrastive +
                                  config['alpha_mt'] * loss_matching)

            scaler.scale(total_vae_loss).backward()
            # Unscale gradients before clipping when using AMP
            scaler.unscale_(optimizer_VAE)
            torch.nn.utils.clip_grad_norm_(vae_params, max_norm=5.0)
            scaler.step(optimizer_VAE)
            scaler.update()
            vae_loss_lst.append(total_vae_loss.item())

            pbar.set_postfix(
                D_c=f"{np.mean(c_dis_loss_lst):.3f}",
                D_d=f"{np.mean(d_dis_loss_lst):.3f}",
                G=f"{np.mean(g_loss_lst):.3f}",
                VAE=f"{np.mean(vae_loss_lst):.3f}"
            )

        # ----------------------------------------------------
        # Epoch-based Validation & Fast Metric Calculation
        # ----------------------------------------------------
        if (epoch + 1) % 5 == 0:
            print(f"\n✅ Epoch {epoch + 1} completed.")

        if (epoch + 1) % config.get('epoch_ckpt_freq', 100) == 0 or (epoch + 1) == config['num_epochs']:
            # Run Fast Metric Evaluation on 1000 generated sequences
            eval_batches = 1000 // config['batch_size']
            if eval_batches == 0: eval_batches = 1
            
            c_real_eval, d_real_eval = [], []
            c_gen_eval, d_gen_eval = [], []

            c_vae.eval()
            d_vae.eval()
            joint_gen.eval()

            # Grab some real comparison batches
            for idx, (real_c, real_d) in enumerate(dataloader):
                if idx >= eval_batches:
                    break
                c_real_eval.append(real_c.numpy())
                d_real_eval.append(real_d.numpy())

            with torch.no_grad():
                for _ in range(eval_batches):
                    noise_c = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)
                    noise_d = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)

                    # ROOT FIX: Huấn luyện dùng joint_gen thì đánh giá cũng PHẢI dùng joint_gen!
                    # Không dùng c_gen/d_gen riêng lẻ vì chúng bị thiếu Attention và bị ép qua Sigmoid.
                    fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)

                    fake_c_seq, _ = c_vae.reconstruct_decoder(fake_z_c)
                    fake_d_seq, _ = d_vae.reconstruct_decoder(fake_z_d)

                    c_gen_eval.append(fake_c_seq.cpu().numpy())
                    d_gen_eval.append(fake_d_seq.cpu().numpy())

            c_real_eval = np.concatenate(c_real_eval, axis=0)
            d_real_eval = np.concatenate(d_real_eval, axis=0)
            c_gen_eval = np.concatenate(c_gen_eval, axis=0)
            d_gen_eval = np.concatenate(d_gen_eval, axis=0)

            # Round discrete probabilities
            d_gen_rounded = np_rounding(d_gen_eval)

            # Evaluate Metrics
            scores = evaluate_all(c_real_eval, c_gen_eval, d_real_eval, d_gen_rounded)
            
            # Save history
            history_dict['epochs'].append(epoch + 1)
            history_dict['mmd'].append(scores['mmd'])
            history_dict['rmse'].append(scores['rmse'])
            history_dict['corr_c'].append(scores['corr_c'])
            history_dict['corr_d'].append(scores['corr_d'])

            # Plot training metrics trend and patient curves
            plot_metrics_trend(history_dict, save_path="Output/test_plots/")
            visualise_gan(
                c_real_eval, c_gen_eval, d_real_eval, d_gen_eval,
                epoch + 1, max_val_con, min_val_con,
                SAVE_PATH="Output/test_plots/",
                c_feature_names=c_feature_names,
                d_feature_names=d_feature_names
            )

            # Restore models back to training mode
            c_vae.train()
            d_vae.train()
            joint_gen.train()

            # Save checkpoint
            os.makedirs("Output/checkpoint", exist_ok=True)
            checkpoint_path = f"Output/checkpoint/m3gan_v2_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'c_gen': c_gen.state_dict(),
                'd_gen': d_gen.state_dict(),
                'c_vae': c_vae.state_dict(),
                'd_vae': d_vae.state_dict(),
                'c_dis': c_dis.state_dict(),
                'd_dis': d_dis.state_dict(),
                'history': history_dict
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            # Keep track of the best model (MMD + RMSE score combined)
            combined_score = scores['mmd'] + scores['rmse']
            if combined_score < best_gan_score:
                best_gan_score = combined_score
                best_checkpoint_path = "Output/checkpoint/best_m3gan_v2.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'c_gen': c_gen.state_dict(),
                    'd_gen': d_gen.state_dict(),
                    'c_vae': c_vae.state_dict(),
                    'd_vae': d_vae.state_dict(),
                    'c_dis': c_dis.state_dict(),
                    'd_dis': d_dis.state_dict(),
                    'scores': scores
                }, best_checkpoint_path)
                print(f"🥇 New best model found! Saved to {best_checkpoint_path}")
