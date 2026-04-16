# EHR-M-GAN: Project Overview & Architecture

EHR-M-GAN is a generative model specifically designed for **Electronic Health Records (EHR)**. It specializes in generating dual-mode longitudinal data: **Continuous** (vitals signs) and **Discrete** (medical interventions).

## 1. Core Model Architecture

The project is built on a "Bilateral" architecture that couples two separate pipelines (one for continuous data, one for discrete data) to ensure they stay synchronized.

### A. Autoregressive VAE (Auto-VAE)
The foundation of the model is a Temporal VAE that handles the time-series nature of EHR data.
*   **Dual-Input Encoder**: At each timestep $t$, the encoder receives the current data point $x_t$ AND the residual error from the previous reconstruction (to enforce autoregressive learning).
*   **Latent Bottleneck**: It compresses the input into a latent vector $z_t$ (Patient State).
*   **LSTM Backbone**: Uses multi-layer LSTMs to maintain long-term memory of a patient's medical history.

### B. Bilateral Generator (Coupled GAN)
The Generator is the "Brain" of the project. It doesn't generate raw data; instead, it generates the **latent traces** ($z$) that the VAE decoders then turn into real data.
*   **Coupled LSTM Cells**: A custom LSTM cell that takes its own previous state PLUS the hidden state of the *other* data stream. 
    *   *Example*: The "Vitals" generator knows what the "Intervention" generator is doing at every second.
*   **Synchronized Sampling**: This ensures that if the model generates a "High Fever" in the vitals stream, it simultaneously generates "Antipyretics" in the intervention stream.

### C. Sequence Discriminator
The Discriminator acts as the "Judge."
*   **Full Temporal Analysis**: Unlike simple discriminators that look at one point in time, this model flattens the *entire* sequence of LSTM hidden states. It judges if the **entire progression** of the patient's stay looks realistic.

---

## 2. Loss Functions (The Learning Signal)

The model learns through a multi-objective loss function:

### Phase 1: VAE Pretraining
1.  **Reconstruction Loss**: Ensures the VAE can accurately rebuild a patient's record (MSE for vitals, BCE for interventions).
2.  **KL Divergence**: Regularizes the latent space.
3.  **Contrastive Loss (NT-Xent)**: Forces the Continuous and Discrete latent vectors to be similar if they belong to the same patient.
4.  **Latent Matching**: Minimizes the distance between the two streams to enforce coupling.

### Phase 2: Joint GAN Training
1.  **Adversarial Loss**: The classic GAN "cat-and-mouse" game between G and D.
2.  **Feature Matching (FM)**: A statistical stabilizer. It forces the generated data to have the same **mean and standard deviation** as the real medical data across the whole batch.

---

## 3. Data Structure
The model expects data in 3D tensors: `[Batch Size, Time Steps, Features]`.
*   **Continuous Features (104)**: Vitals like heart rate, oxygen saturation, blood pressure.
*   **Discrete Features (13)**: Interventions like mechanical ventilation, drug administration, or dialysis.
*   **Time Steps (24)**: Represents a 24-hour window of clinical monitoring.

---

## 4. Hardware Optimization (Tesla M40 & RTX 3080)
The code is optimized for two specific environments:
*   **RTX 3080**: Uses **Mixed Precision (AMP)** to fit large batches into 10GB VRAM and leverages Tensor Cores for 2-3x faster training.
*   **Tesla M40**: Leverages the large **24GB VRAM** to handle higher complexity (`hidden_dim=512`) and full precision training.
