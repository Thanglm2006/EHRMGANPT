import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from networks import VAE_Encoder, VAE_Decoder, SequenceDiscriminator, BilateralGenerator
from ultils import nt_xent_loss, kl_divergence, feature_matching_loss, np_rounding
from visualise import visualise_gan, visualise_vae


def train_m3gan(dataloader, config, max_val_con, min_val_con):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c_dim, d_dim = config['c_dim'], config['d_dim']
    latent_dim = config['latent_dim']
    hidden_dim = config['hidden_dim']
    noise_dim = config['noise_dim']
    time_steps = config['time_steps']

    c_enc = VAE_Encoder(c_dim, hidden_dim, latent_dim, config['enc_layers']).to(device)
    c_dec = VAE_Decoder(latent_dim, hidden_dim, c_dim, config['dec_layers']).to(device)
    d_enc = VAE_Encoder(d_dim, hidden_dim, latent_dim, config['enc_layers']).to(device)
    d_dec = VAE_Decoder(latent_dim, hidden_dim, d_dim, config['dec_layers']).to(device)

    c_gen = BilateralGenerator(noise_dim, hidden_dim, latent_dim, config['gen_layers']).to(device)
    d_gen = BilateralGenerator(noise_dim, hidden_dim, latent_dim, config['gen_layers']).to(device)

    c_dis = SequenceDiscriminator(c_dim, hidden_dim, config['dis_layers']).to(device)
    d_dis = SequenceDiscriminator(d_dim, hidden_dim, config['dis_layers']).to(device)

    vae_params = list(c_enc.parameters()) + list(c_dec.parameters()) + \
                 list(d_enc.parameters()) + list(d_dec.parameters())
    optimizer_VAE_pre = optim.Adam(vae_params, lr=config['v_lr_pre'])
    optimizer_VAE = optim.Adam(vae_params, lr=config['v_lr'])

    optimizer_G = optim.Adam(list(c_gen.parameters()) + list(d_gen.parameters()), lr=config['g_lr'])
    optimizer_D = optim.Adam(list(c_dis.parameters()) + list(d_dis.parameters()), lr=config['d_lr'])

    bce_with_logits = nn.BCEWithLogitsLoss()

    if config.get('resume_checkpoint') is not None:
        print(f"\nLoading pretrained weights from {config['resume_checkpoint']}...")
        checkpoint = torch.load(config['resume_checkpoint'], map_location=device)

        # Use strictly to allow loading partial checkpoints (like just the VAE)
        if 'c_enc' in checkpoint: c_enc.load_state_dict(checkpoint['c_enc'])
        if 'c_dec' in checkpoint: c_dec.load_state_dict(checkpoint['c_dec'])
        if 'd_enc' in checkpoint: d_enc.load_state_dict(checkpoint['d_enc'])
        if 'd_dec' in checkpoint: d_dec.load_state_dict(checkpoint['d_dec'])

        # If resuming Phase 2, load the GAN components too
        if 'c_gen' in checkpoint: c_gen.load_state_dict(checkpoint['c_gen'])
        if 'd_gen' in checkpoint: d_gen.load_state_dict(checkpoint['d_gen'])
        if 'c_dis' in checkpoint: c_dis.load_state_dict(checkpoint['c_dis'])
        if 'd_dis' in checkpoint: d_dis.load_state_dict(checkpoint['d_dis'])
        print("Weights successfully loaded!\n")

    # ---------------------------------------------------------
    # Phase 1: Pretrain VAE
    # ---------------------------------------------------------
    c_enc.train()
    c_dec.train()
    d_enc.train()
    d_dec.train()

    best_vae_loss = float('inf')
    epochs_no_improve = 0
    patience = config.get('patience', 200)
    ckpt_dir = "Output/checkpoint/"

    print("\n=== Starting Phase 1: VAE Pretraining ===")
    for epoch in range(config['num_pre_epochs']):
        c_real_lst, c_rec_lst, d_real_lst, d_rec_lst = [], [], [], []
        epoch_total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch [{epoch + 1}/{config['num_pre_epochs']}]", leave=True)

        for continuous_x, discrete_x in pbar:
            # Absolute input sanitization
            continuous_x = torch.clamp(torch.nan_to_num(continuous_x.to(device), nan=0.0), 0.0, 1.0)
            discrete_x = torch.clamp(torch.nan_to_num(discrete_x.to(device), nan=0.0), 0.0, 1.0)

            optimizer_VAE_pre.zero_grad()

            c_z, c_mu, c_logvar = c_enc(continuous_x)
            c_rec, _ = c_dec(c_z)

            d_z, d_mu, d_logvar = d_enc(discrete_x)
            d_rec, d_logits = d_dec(d_z)

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

            total_vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_params, max_norm=5.0)
            optimizer_VAE_pre.step()

            epoch_total_loss += total_vae_loss.item()
            pbar.set_postfix(
                c_rec=f"{loss_c_rec.item():.4f}",
                d_rec=f"{loss_d_rec.item():.4f}",
                tot_vae=f"{total_vae_loss.item():.4f}"
            )

            if (epoch + 1) % config['epoch_ckpt_freq'] == 0 or epoch == config['num_pre_epochs'] - 1:
                c_real_lst.append(continuous_x.detach().cpu().numpy())
                c_rec_lst.append(c_rec.detach().cpu().numpy())
                d_real_lst.append(discrete_x.detach().cpu().numpy())
                d_rec_lst.append(np_rounding(d_rec.detach().cpu().numpy()))

        # --- Phase 1 Early Stopping Logic ---
        avg_epoch_loss = epoch_total_loss / len(dataloader)
        if avg_epoch_loss < best_vae_loss:
            best_vae_loss = avg_epoch_loss
            epochs_no_improve = 0
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'c_enc': c_enc.state_dict(),
                'c_dec': c_dec.state_dict(),
                'd_enc': d_enc.state_dict(),
                'd_dec': d_dec.state_dict(),
            }, os.path.join(ckpt_dir, "best_pretrain_vae.pth"))
        else:
            epochs_no_improve += 1
            print(f"--- No improvement in VAE loss for {epochs_no_improve} epochs ---")

            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}! VAE has converged.")
                print(f"Loading the best VAE weights (Loss: {best_vae_loss:.4f}) before proceeding to Phase 2...\n")

                best_ckpt = torch.load(os.path.join(ckpt_dir, "best_pretrain_vae.pth"))
                c_enc.load_state_dict(best_ckpt['c_enc'])
                c_dec.load_state_dict(best_ckpt['c_dec'])
                d_enc.load_state_dict(best_ckpt['d_enc'])
                d_dec.load_state_dict(best_ckpt['d_dec'])
                break

        # Save checkpoints and plots based on frequency
        if (epoch + 1) % config['epoch_ckpt_freq'] == 0 or epoch == config['num_pre_epochs'] - 1:
            visualise_vae(np.vstack(c_real_lst), np.vstack(c_rec_lst),
                          np.vstack(d_real_lst), np.vstack(d_rec_lst),
                          inx=(epoch + 1), max_val_con=max_val_con, min_val_con=min_val_con)

            os.makedirs('Output/fake', exist_ok=True)
            np.savez('Output/fake/vae.npz', c_real=np.vstack(c_real_lst), c_rec=np.vstack(c_rec_lst),
                     d_real=np.vstack(d_real_lst), d_rec=np.vstack(d_rec_lst))

            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'c_enc': c_enc.state_dict(),
                'c_dec': c_dec.state_dict(),
                'd_enc': d_enc.state_dict(),
                'd_dec': d_dec.state_dict(),
            }, os.path.join(ckpt_dir, f"pretrain_vae_epoch_{epoch + 1}.pth"))

    # ---------------------------------------------------------
    # Phase 2: Joint GAN Training
    # ---------------------------------------------------------
    print("\n=== Starting Phase 2: Joint GAN Training ===")

    best_gan_loss = float('inf')
    epochs_no_improve_gan = 0

    for epoch in range(config['num_epochs']):
        c_real_lst, d_real_lst = [], []
        epoch_g_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Joint GAN Epoch [{epoch + 1}/{config['num_epochs']}]", leave=True)

        for continuous_x, discrete_x in pbar:
            batch_size = continuous_x.size(0)

            continuous_x = torch.clamp(torch.nan_to_num(continuous_x.to(device), nan=0.0), 0.0, 1.0)
            discrete_x = torch.clamp(torch.nan_to_num(discrete_x.to(device), nan=0.0), 0.0, 1.0)

            if (epoch + 1) % config['epoch_ckpt_freq'] == 0 or epoch == config['num_epochs'] - 1:
                c_real_lst.append(continuous_x.detach().cpu().numpy())
                d_real_lst.append(discrete_x.detach().cpu().numpy())

            real_labels = torch.clamp(torch.empty((batch_size, time_steps), device=device).uniform_(0.8, 1.0), 0.0, 1.0)
            fake_labels = torch.empty((batch_size, time_steps), device=device).uniform_(0.0, 0.3)

            # --- Update Discriminator ---
            d_loss = torch.tensor(0.0)
            for _ in range(config['d_rounds']):
                optimizer_D.zero_grad()
                c_dec.eval()
                d_dec.eval()

                noise_c = torch.randn(batch_size, time_steps, noise_dim, device=device)
                noise_d = torch.randn(batch_size, time_steps, noise_dim, device=device)

                h_cpl_c = [torch.zeros(batch_size, hidden_dim, device=device) for _ in range(config['gen_layers'])]
                h_cpl_d = [torch.zeros(batch_size, hidden_dim, device=device) for _ in range(config['gen_layers'])]

                fake_z_c, h_cpl_d = c_gen(noise_c, h_cpl_d)
                fake_z_d, h_cpl_c = d_gen(noise_d, h_cpl_c)

                with torch.no_grad():
                    fake_c_seq, _ = c_dec(fake_z_c)
                    fake_d_seq, _ = d_dec(fake_z_d)

                c_real_logits, c_real_fm = c_dis(continuous_x)
                d_real_logits, _ = d_dis(discrete_x)
                c_fake_logits, c_fake_fm = c_dis(fake_c_seq.detach())
                d_fake_logits, _ = d_dis(fake_d_seq.detach())

                d_loss_c = bce_with_logits(c_real_logits, real_labels) + bce_with_logits(c_fake_logits, fake_labels)
                d_loss_d = bce_with_logits(d_real_logits, real_labels) + bce_with_logits(d_fake_logits, fake_labels)
                d_loss = d_loss_c + d_loss_d

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(c_dis.parameters()) + list(d_dis.parameters()), max_norm=5.0)
                optimizer_D.step()

            # --- Update Generator ---
            g_loss = torch.tensor(0.0)
            for _ in range(config['g_rounds']):
                optimizer_G.zero_grad()
                noise_c = torch.randn(batch_size, time_steps, noise_dim, device=device)
                noise_d = torch.randn(batch_size, time_steps, noise_dim, device=device)

                h_cpl_c = [torch.zeros(batch_size, hidden_dim, device=device) for _ in range(config['gen_layers'])]
                h_cpl_d = [torch.zeros(batch_size, hidden_dim, device=device) for _ in range(config['gen_layers'])]

                fake_z_c, h_cpl_d = c_gen(noise_c, h_cpl_d)
                fake_z_d, h_cpl_c = d_gen(noise_d, h_cpl_c)

                c_dec.train()
                d_dec.train()
                fake_c_seq, _ = c_dec(fake_z_c)
                fake_d_seq, _ = d_dec(fake_z_d)

                c_fake_logits, c_fake_fm = c_dis(fake_c_seq)
                d_fake_logits, _ = d_dis(fake_d_seq)

                with torch.no_grad():
                    _, c_real_fm = c_dis(continuous_x)

                g_adv_loss_c = bce_with_logits(c_fake_logits, torch.ones_like(c_fake_logits))
                g_adv_loss_d = bce_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))
                g_fm_loss_c = feature_matching_loss(c_fake_fm, c_real_fm)
                g_fm_loss_d = feature_matching_loss(fake_d_seq, discrete_x)

                g_loss_c = config['c_beta_adv'] * g_adv_loss_c + config['c_beta_fm'] * g_fm_loss_c
                g_loss_d = config['d_beta_adv'] * g_adv_loss_d + config['d_beta_fm'] * g_fm_loss_d
                g_loss = g_loss_c + g_loss_d

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(c_gen.parameters()) + list(d_gen.parameters()), max_norm=5.0)
                optimizer_G.step()

            epoch_g_loss += g_loss.item()

            # --- Update VAE ---
            total_vae_loss = torch.tensor(0.0)
            for _ in range(config['v_rounds']):
                c_enc.train()
                c_dec.train()
                d_enc.train()
                d_dec.train()
                optimizer_VAE.zero_grad()

                c_z, c_mu, c_logvar = c_enc(continuous_x)
                c_rec, _ = c_dec(c_z)

                d_z, d_mu, d_logvar = d_enc(discrete_x)
                d_rec, d_logits = d_dec(d_z)

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

                total_vae_loss.backward()
                torch.nn.utils.clip_grad_norm_(vae_params, max_norm=5.0)
                optimizer_VAE.step()

            pbar.set_postfix(
                d_loss=f"{d_loss.item():.4f}",
                g_loss=f"{g_loss.item():.4f}",
                v_loss=f"{total_vae_loss.item():.4f}"
            )

        # --- Phase 2 Early Stopping Logic ---
        avg_epoch_g_loss = epoch_g_loss / len(dataloader)
        if avg_epoch_g_loss < best_gan_loss:
            best_gan_loss = avg_epoch_g_loss
            epochs_no_improve_gan = 0

            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'c_gen': c_gen.state_dict(),
                'd_gen': d_gen.state_dict(),
                'c_dis': c_dis.state_dict(),
                'd_dis': d_dis.state_dict(),
                'c_enc': c_enc.state_dict(),
                'c_dec': c_dec.state_dict(),
                'd_enc': d_enc.state_dict(),
                'd_dec': d_dec.state_dict(),
            }, os.path.join(ckpt_dir, "best_m3gan.pth"))
        else:
            epochs_no_improve_gan += 1
            print(f"--- No improvement in GAN Generator loss for {epochs_no_improve_gan} epochs ---")

            if epochs_no_improve_gan >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}! Joint GAN has converged.")
                print(f"Loading the best GAN weights (G Loss: {best_gan_loss:.4f}) before finishing...\n")

                best_ckpt = torch.load(os.path.join(ckpt_dir, "best_m3gan.pth"))
                c_gen.load_state_dict(best_ckpt['c_gen'])
                d_gen.load_state_dict(best_ckpt['d_gen'])
                c_dis.load_state_dict(best_ckpt['c_dis'])
                d_dis.load_state_dict(best_ckpt['d_dis'])
                c_enc.load_state_dict(best_ckpt['c_enc'])
                c_dec.load_state_dict(best_ckpt['c_dec'])
                d_enc.load_state_dict(best_ckpt['d_enc'])
                d_dec.load_state_dict(best_ckpt['d_dec'])
                break

        # Save checkpoints and plots based on frequency
        if (epoch + 1) % config['epoch_ckpt_freq'] == 0 or epoch == config['num_epochs'] - 1:
            c_dec.eval()
            d_dec.eval()
            c_gen.eval()
            d_gen.eval()
            d_gen_data, c_gen_data = [], []

            with torch.no_grad():
                for _ in range(len(dataloader)):
                    noise_c = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)
                    noise_d = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)

                    h_cpl_c = [torch.zeros(config['batch_size'], hidden_dim, device=device) for _ in
                               range(config['gen_layers'])]
                    h_cpl_d = [torch.zeros(config['batch_size'], hidden_dim, device=device) for _ in
                               range(config['gen_layers'])]

                    fake_z_c, h_cpl_d = c_gen(noise_c, h_cpl_d)
                    fake_z_d, h_cpl_c = d_gen(noise_d, h_cpl_c)

                    fake_c_seq, _ = c_dec(fake_z_c)
                    fake_d_seq, _ = d_dec(fake_z_d)

                    c_gen_data.append(fake_c_seq.cpu().numpy())
                    d_gen_data.append(fake_d_seq.cpu().numpy())

            c_gen_data = np.concatenate(c_gen_data, axis=0)
            d_gen_data = np_rounding(np.concatenate(d_gen_data, axis=0))

            c_real_full = np.vstack(c_real_lst)
            d_real_full = np.vstack(d_real_lst)

            visualise_gan(c_real_full, c_gen_data,
                          d_real_full, d_gen_data,
                          inx=(epoch + 1), max_val_con=max_val_con, min_val_con=min_val_con)

            data_gen_path = os.path.join("Output/fake/", f"epoch{epoch + 1}")
            os.makedirs(data_gen_path, exist_ok=True)
            np.savez(os.path.join(data_gen_path, "gen_data.npz"), c_gen_data=c_gen_data, d_gen_data=d_gen_data)

            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'c_gen': c_gen.state_dict(),
                'd_gen': d_gen.state_dict(),
                'c_dis': c_dis.state_dict(),
                'd_dis': d_dis.state_dict(),
                'c_enc': c_enc.state_dict(),
                'c_dec': c_dec.state_dict(),
                'd_enc': d_enc.state_dict(),
                'd_dec': d_dec.state_dict(),
            }, os.path.join(ckpt_dir, f"m3gan_epoch_{epoch + 1}.pth"))

            c_dec.train()
            d_dec.train()
            c_gen.train()
            d_gen.train()