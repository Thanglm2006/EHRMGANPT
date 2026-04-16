# EHR-M-GAN: Project Overview

EHR-M-GAN is a deep learning framework designed to generate realistic, synthetic Electronic Health Record (EHR) data. It specifically focuses on mixed-type time-series data: **Continuous vitals** and **Discrete medical interventions**.

## 1. How Training Works

The training process is divided into two distinct phases to ensure the model captures both the individual characteristics of the data streams and their complex correlations.

### Phase 1: VAE Pretraining
The goal of this phase is to learn a structured latent representation for both continuous and discrete data.
- **Continuous VAE**: Learns to encode and reconstruct vitals (e.g., heart rate, blood pressure).
- **Discrete VAE**: Learns to encode and reconstruct medical interventions (e.g., medication doses, procedures).
- **Joint Optimization**: Both VAEs are trained together with **Contrastive** and **Matching** losses. This forces the latent spaces of the two streams to be aligned.

### Phase 2: Joint GAN Training
Once the VAEs provide a stable latent space, the GAN takes over to generate sequences from random noise.
- **Generator**: Takes Gaussian noise and generates a sequence in the **latent space** ($z$).
- **Bilateral Mechanism**: The generators for continuous and discrete data are "coupled" using a bilateral LSTM cell, ensuring that the generated interventions correlate with the generated vitals.
- **Decoder Reconstruction**: The generated latent sequences are passed through the pretrained VAE Decoders to produce the final EHR data.
- **Discriminator**: Tries to distinguish between real temporal sequences and synthetic ones.

---

## 2. Explanation of Losses

### VAE Losses (State Representation)
- **Reconstruction Loss (`loss_rec`)**:
    - **Continuous**: Mean Squared Error (MSE) between original and reconstructed vitals.
    - **Discrete**: Binary Cross Entropy (BCE) for intervention flags.
- **KL Divergence (`loss_kl`)**: Regularizes the latent space manually to follow a normal distribution.
- **Contrastive Loss (`loss_ct`)**: Uses **NT-Xent** to align latent vectors of the same patient.
- **Matching Loss (`loss_mt`)**: MSE between the latent vectors of the two streams.

### GAN Losses (Generation Quality)
- **Adversarial Loss (`adv_loss`)**: Standard GAN loss (Real vs. Fake).
- **Feature Matching Loss (`fm_loss`)**: Generator tries to match the statistical distributions of features in the Discriminator.

---

## 3. How the Model Learns

The model learns through a multi-stage objective:
1.  **Compression**: VAEs compress 24-hour sequences into a "clinical state" ($z$).
2.  **Correlation**: Contrastive loss teaches the model that certain vitals happen alongside certain treatments.
3.  **Temporal Dynamics**: LSTMs in the GAN learn patient state evolution.
4.  **Refinement**: The Discriminator provides feedback on overall sequence realism.

---

## 4. Data Structure

The project uses data from the **MIMIC-III** dataset.

### Input Format
The data is structured as 3D Tensors: `(Batch Size, Time Steps, Features)`

| Data Stream | Shape | Examples of Features |
| :--- | :--- | :--- |
| **Continuous (Vitals)** | `(N, 24, 104)` | Heart Rate, SpO2, Systolic BP, Temp, etc. |
| **Discrete (Interventions)** | `(N, 24, 13)` | Vasopressors, Ventilation, Sedation, etc. |

-   **Normalization**: Min-max scaled to `[0, 1]`.
-   **Timesteps**: 24 hours at 1-hour intervals.
