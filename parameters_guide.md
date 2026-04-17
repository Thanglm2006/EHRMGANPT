# EHR-M-GAN Parameter, Object, and Variable Guide

This guide provides a deep dive into the configuration, objects, and internal variables that drive the EHR-M-GAN training process in `trainer.py`.

---

## 1. Global Hyperparameters (Config Dictionary)

These parameters are passed from `main.py` and control the overall behavior of the experiment.

| Key | Description |
| :--- | :--- |
| `batch_size` | Number of samples per training step in Phase 2. |
| `pre_batch_size` | Large batch size for stable VAE pretraining (Phase 1). |
| `time_steps` | Length of the EHR sequence (e.g., 24 hours). |
| `c_dim` / `d_dim` | Number of features in the continuous and discrete data streams. |
| `latent_dim` | size of the bottleneck representation $z$ in the VAE. |
| `hidden_dim` | Number of units in the LSTM hidden layers. |
| `noise_dim` | Size of the random seed vector for the Generators. |
| `enc_layers` / `dec_layers` | Depth of the VAE architectures. |
| `gen_layers` / `dis_layers` | Depth of the Generator and Discriminator architectures. |
| `alpha_...` | Weights for VAE loss components (Rec, KL, Contrastive, Matching). |
| `..._beta_...` | Weights for GAN loss components (Adversarial, Feature Matching). |

---

## 2. Core Training Objects

These are the primary PyTorch objects initialized at the start of `train_m3gan`.

| Object | Type | Role |
| :--- | :--- | :--- |
| `c_vae` / `d_vae` | `AutoregressiveVAE` | Autoencoders that learn the underlying distribution of EHR timestamps. |
| `c_gen` / `d_gen` | `BilateralGenerator` | Generators that produce synthetic latent codes $z$ using across-stream coupling. |
| `c_dis` / `d_dis` | `SequenceDiscriminator` | Classifiers that distinguish real EHR sequences from synthetic ones. |
| `optimizer_VAE` | `Adam` | Updates the VAE weights to improve reconstruction and latent alignment. |
| `optimizer_G` | `Adam` | Updates Generator weights based on Discriminator feedback. |
| `optimizer_D` | `Adam` | Updates Discriminator weights to improve classification accuracy. |
| `scaler` | `GradScaler` | Manages Mixed Precision (AMP) to prevent gradient underflow in FP16. |
| `dataloader` | `DataLoader` | Iterates over the dataset, handling shuffling and batching. |

---

## 3. Key Tensors and Data Flow

These tensors represent the flow of data through the network during a single training step.

| Tensor | Shape | Role |
| :--- | :--- | :--- |
| `continuous_x` | `[B, T, C]` | Real continuous vital sign data for the current batch. |
| `discrete_x` | `[B, T, D]` | Real discrete medication/intervention data for the current batch. |
| `noise_c` / `noise_d`| `[B, T, N]` | Gaussian noise used as the "seed" for the generators. |
| `h_cpl_c` / `h_cpl_d`| List of `[B, H]` | The "Coupled" hidden states passed between generators (Bilateral interaction). |
| `fake_z_c` / `fake_z_d`| `[B, T, L]` | Synthetic latent codes produced by the generators. |
| `fake_c_seq` / `fake_d_seq`| `[B, T, C/D]` | Final synthetic EHR data after passing `fake_z` through the VAE Decoders. |
| `real_labels` | `[B]` | Target values (0.8–1.0) for real data classification. |
| `fake_labels` | `[B]` | Target values (0.0–0.3) for fake data classification. |
| `c_mu` / `c_logvar` | `[B, T, L]` | Mean and Variance parameters for the VAE latent distribution. |

---

## 4. Phase 2: Joint Training Internal Logic

### A. Data Scopes
For every batch, we apply:
- **`nan_to_num`**: Safeguard against missing data points.
- **`clamp(0.0, 1.0)`**: Ensures stability for `BCE` and `Sigmoid` layers.

### B. The GAN Update Cycle
- **Discriminator Step (`d_rounds`)**:
    1. Generate noise and hidden coupled states.
    2. Pass noise through Generators to get `fake_z`.
    3. Decode `fake_z` to get `fake_seq`.
    4. Score both `real_seq` and `fake_seq`.
    5. Backpropagate the combined error to the Discriminator **only**.
- **Generator Step (`g_rounds`)**:
    1. Re-run the generation process.
    2. Pass `fake_seq` through the Discriminator.
    3. Calculate **Adversarial Loss** (can I fool D?) and **Feature Matching Loss** (do my internal stats match real data?).
    4. Backpropagate to the Generator **only**.
- **VAE Step (`v_rounds`)**:
    1. Calculate standard VAE Reconstruction and KL losses.
    2. Apply **Contrastive Alignment** (NT-Xent) to ensure continuous and discrete latents for the same patient stay synchronized.

---

## 5. Fixed Constants and Hardcoded Parameters

| Value | Scope | Description |
| :--- | :--- | :--- |
| `5.0` | `clip_grad_norm_` | Gradient clipping threshold to prevent RNN exploding gradients. |
| `0.8 - 1.0` | `uniform_` | Smoothing range for Real Labels. Prevents D from becoming too "strict". |
| `0.0 - 0.3` | `uniform_` | Smoothing range for Fake Labels. Prevents G from losing gradient signals. |
| `1e-8` | `nt_xent_loss` | Epsilon added to logs and divisions to prevent `NaN` values. |
| `inf` | `best_..._loss` | Initial value for the best loss trackers used for checkpointing. |
