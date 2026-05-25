import json
import os

# Paths to python files
root_dir = '/home/thanglm2006/EHRMGANPT'
v1_dir = '/home/thanglm2006/EHRMGANPT/neo_m3gan'
v2_dir = '/home/thanglm2006/EHRMGANPT/neo_m3gan_v2'
notebook_path = '/home/thanglm2006/EHRMGANPT/neo_m3gan_efficiency_colab.ipynb'

# Helper to read file and return list of lines
def read_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

# Initialize notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# 1. Add Markdown Header
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# EHR-M-GAN: 3-Model Training Efficiency & Quality Analysis on Google Colab\n",
        "\n",
        "This notebook allows you to load checkpoints from **all 3 versions** of EHR-M-GAN and compare their performance, convergence rate, and continuous/discrete generation quality side-by-side:\n",
        "\n",
        "1. **Model 1: Standard M3GAN (Baseline)**: The basic Coupled VAE-GAN implementation.\n",
        "2. **Model 2: Neo M3GAN V1**: Introduced Temporal Self-Attention and synchronized generators.\n",
        "3. **Model 3: Neo M3GAN V2 (Ours)**: Added **3-layer MLP Mapping Networks** in G, **Minibatch Standard Deviation** in D, **KL weight annealing** in Phase 1, and **numerical autograd stability** to fully prevent mode collapse.\n",
        "\n",
        "---"
    ]
})

# 2. Add Setup & Drive Mount cell
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# 1. Mount Google Drive to load datasets/checkpoints\n",
        "from google.colab import drive\n",
        "import os\n",
        "import numpy as np\n",
        "import pickle\n",
        "import pandas as pd\n",
        "import torch\n",
        "\n",
        "# drive.mount('/content/drive')\n",
        "\n",
        "# 2. Create local directory structure\n",
        "os.makedirs('Data/mimic', exist_ok=True)\n",
        "os.makedirs('Output/analysis', exist_ok=True)"
    ]
})

# 3. Add Instruction to Upload/Link Data cell
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### ⚠️ Data Upload Instructions:\n",
        "Please upload your patient data (`vital_sign_24hrs.pkl` and `med_interv_24hrs.pkl`) to your Google Drive or upload them directly to the Colab files section under `Data/mimic/`.\n",
        "\n",
        "Once uploaded, the cell below will verify and load the patient data."
    ]
})

# 4. Add Verify & Load Data cell
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# Load and normalize real dataset for validation comparison\n",
        "vital_path = 'Data/mimic/vital_sign_24hrs.pkl'\n",
        "med_path = 'Data/mimic/med_interv_24hrs.pkl'\n",
        "\n",
        "if not os.path.exists(vital_path) or not os.path.exists(med_path):\n",
        "    print(\"❌ ERROR: Please upload 'vital_sign_24hrs.pkl' and 'med_interv_24hrs.pkl' to 'Data/mimic/' folder first!\")\n",
        "else:\n",
        "    with open(vital_path, 'rb') as f:\n",
        "        real_c = pickle.load(f)\n",
        "    with open(med_path, 'rb') as f:\n",
        "        real_d = pickle.load(f)\n",
        "        \n",
        "    # Sanitize and normalize\n",
        "    real_d = np.nan_to_num(np.clip(real_d, 0.0, 1.0), nan=0.0)\n",
        "    real_c = np.nan_to_num(real_c, nan=0.0)\n",
        "    \n",
        "    min_val = np.min(real_c, axis=(0, 1))\n",
        "    max_val = np.max(real_c, axis=(0, 1))\n",
        "    range_val = max_val - min_val\n",
        "    range_val[range_val == 0.0] = 1e-6\n",
        "    real_c_norm = (real_c - min_val) / range_val\n",
        "    \n",
        "    print(f\"✅ Successfully loaded dataset!\")\n",
        "    print(f\"Continuous shape: {real_c_norm.shape} | Discrete shape: {real_d.shape}\")"
    ]
})

# 5. Add %%writefile cells for networks, metrics, utils
# Standard Networks
std_net_lines = read_file_lines(os.path.join(root_dir, 'networks.py'))
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": ["%%writefile networks_standard.py\n"] + std_net_lines
})

# V1 Networks
v1_net_lines = read_file_lines(os.path.join(v1_dir, 'networks.py'))
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": ["%%writefile networks_v1.py\n"] + v1_net_lines
})

# V2 Networks
v2_net_lines = read_file_lines(os.path.join(v2_dir, 'networks.py'))
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": ["%%writefile networks_v2.py\n"] + v2_net_lines
})

# Shared Utils
utils_lines = read_file_lines(os.path.join(root_dir, 'ultils.py'))
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": ["%%writefile ultils.py\n"] + utils_lines
})

# Shared Metrics
metrics_lines = read_file_lines(os.path.join(root_dir, 'metrics.py'))
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": ["%%writefile metrics.py\n"] + metrics_lines
})

# 6. Add 3-Model Side-by-side Evaluation Code
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# Dynamic testing module loader\n",
        "import importlib.util\n",
        "from ultils import np_rounding\n",
        "from metrics import evaluate_all\n",
        "\n",
        "def load_network_module(name, filepath):\n",
        "    spec = importlib.util.spec_from_file_location(name, filepath)\n",
        "    module = importlib.util.module_from_spec(spec)\n",
        "    spec.loader.exec_module(module)\n",
        "    return module\n",
        "\n",
        "def evaluate_checkpoint(model_version, checkpoint_path, real_c, real_d, num_samples=1000, batch_size=256):\n",
        "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "    \n",
        "    # Load correct architecture\n",
        "    if model_version == 'standard':\n",
        "        net = load_network_module('net_std', 'networks_standard.py')\n",
        "    elif model_version == 'v1':\n",
        "        net = load_network_module('net_v1', 'networks_v1.py')\n",
        "    elif model_version == 'v2':\n",
        "        net = load_network_module('net_v2', 'networks_v2.py')\n",
        "    else:\n",
        "        raise ValueError(\"Invalid model version\")\n",
        "        \n",
        "    c_dim, d_dim = real_c.shape[2], real_d.shape[2]\n",
        "    time_steps = real_c.shape[1]\n",
        "    latent_dim = 25\n",
        "    noise_dim = min(int(c_dim / 2), int(d_dim / 2))\n",
        "    \n",
        "    c_vae = net.AutoregressiveVAE(c_dim, 512, latent_dim, 3, 3, time_steps).to(device)\n",
        "    d_vae = net.AutoregressiveVAE(d_dim, 512, latent_dim, 3, 3, time_steps).to(device)\n",
        "    joint_gen = net.JointGenerator(noise_dim, noise_dim, 512, latent_dim, latent_dim, 3).to(device)\n",
        "    \n",
        "    print(f\"Loading {model_version.upper()} checkpoint from {checkpoint_path}...\")\n",
        "    checkpoint = torch.load(checkpoint_path, map_location=device)\n",
        "    \n",
        "    c_vae.load_state_dict(checkpoint['c_vae'])\n",
        "    d_vae.load_state_dict(checkpoint['d_vae'])\n",
        "    \n",
        "    # Support loading joint_gen directly (v2) or single generator states (standard/v1)\n",
        "    if 'c_gen' in checkpoint:\n",
        "        if hasattr(joint_gen, 'c_gen'):\n",
        "            joint_gen.c_gen.load_state_dict(checkpoint['c_gen'])\n",
        "            joint_gen.d_gen.load_state_dict(checkpoint['d_gen'])\n",
        "    \n",
        "    c_vae.eval(); d_vae.eval(); joint_gen.eval()\n",
        "    \n",
        "    c_gen_data, d_gen_data = [], []\n",
        "    num_batches = int(np.ceil(num_samples / batch_size))\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        for _ in range(num_batches):\n",
        "            noise_c = torch.randn(batch_size, time_steps, noise_dim, device=device)\n",
        "            noise_d = torch.randn(batch_size, time_steps, noise_dim, device=device)\n",
        "            \n",
        "            fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)\n",
        "            fake_c_seq, _ = c_vae.reconstruct_decoder(fake_z_c)\n",
        "            fake_d_seq, _ = d_vae.reconstruct_decoder(fake_z_d)\n",
        "            \n",
        "            c_gen_data.append(fake_c_seq.cpu().numpy())\n",
        "            d_gen_data.append(fake_d_seq.cpu().numpy())\n",
        "            \n",
        "    c_gen_data = np.concatenate(c_gen_data, axis=0)[:num_samples]\n",
        "    d_gen_data = np_rounding(np.concatenate(d_gen_data, axis=0))[:num_samples]\n",
        "    \n",
        "    indices = np.random.choice(real_c.shape[0], num_samples, replace=False)\n",
        "    scores = evaluate_all(real_c[indices], c_gen_data, real_d[indices], d_gen_data)\n",
        "    return scores"
    ]
})

# 7. Add Checkpoint Paths configuration
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 🏃‍♂️ Run 3-Model Side-by-Side Comparison:\n",
        "Update the paths below to point to the checkpoints you downloaded/trained for **Standard (Baseline)**, **V1 (Self-Attention)**, and **V2 (Ours)** versions."
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# UPDATE THESE PATHS TO YOUR SAVED MODEL CHECKPOINTS\n",
        "ckpt_standard = '/content/m3gan_standard_epoch_100.pth'\n",
        "ckpt_v1 = '/content/m3gan_v1_epoch_100.pth'\n",
        "ckpt_v2 = '/content/m3gan_v2_epoch_100.pth'\n",
        "\n",
        "results = []\n",
        "\n",
        "# 1. Evaluate Standard Model (Baseline)\n",
        "if os.path.exists(ckpt_standard):\n",
        "    scores_std = evaluate_checkpoint('standard', ckpt_standard, real_c_norm, real_d, num_samples=1000)\n",
        "    scores_std.update({'model': 'Standard (Baseline)'})\n",
        "    results.append(scores_std)\n",
        "else:\n",
        "    print(\"⚠️ Skip Standard (Baseline) - Checkpoint file not found.\")\n",
        "    \n",
        "# 2. Evaluate V1 Model (Self-Attention)\n",
        "if os.path.exists(ckpt_v1):\n",
        "    scores_v1 = evaluate_checkpoint('v1', ckpt_v1, real_c_norm, real_d, num_samples=1000)\n",
        "    scores_v1.update({'model': 'Neo M3GAN V1'})\n",
        "    results.append(scores_v1)\n",
        "else:\n",
        "    print(\"⚠️ Skip Neo M3GAN V1 - Checkpoint file not found.\")\n",
        "\n",
        "# 3. Evaluate V2 Model (Ours: Mapping Network + Minibatch StdDev)\n",
        "if os.path.exists(ckpt_v2):\n",
        "    scores_v2 = evaluate_checkpoint('v2', ckpt_v2, real_c_norm, real_d, num_samples=1000)\n",
        "    scores_v2.update({'model': 'Neo M3GAN V2 (Ours)'})\n",
        "    results.append(scores_v2)\n",
        "else:\n",
        "    print(\"⚠️ Skip Neo M3GAN V2 (Ours) - Checkpoint file not found.\")"
    ]
})

# 8. Add Plotting & Results Tabulation
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Visualizing Results & Statistical Report:\n",
        "The cell below compiles a visual side-by-side comparison chart and displays a cleanly formatted markdown table comparing the models across all key metrics."
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "if len(results) > 0:\n",
        "    df = pd.DataFrame(results)\n",
        "    # Reorder columns\n",
        "    cols = ['model', 'mmd', 'rmse', 'corr_c', 'corr_d']\n",
        "    df = df[cols]\n",
        "    \n",
        "    print(\"\\n\" + \"=\"*60)\n",
        "    print(\"              EHR-M-GAN 3-MODEL EFFICIENCY SUMMARY\")\n",
        "    print(\"=\"*60)\n",
        "    # Avoid using tabulate library by hand-formatting a simple markdown table\n",
        "    print(f\"| {'Model':<25} | {'MMD':<10} | {'RMSE':<10} | {'Corr C':<10} | {'Corr D':<10} |\")\n",
        "    print(f\"| {'-'*25} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} |\")\n",
        "    for _, row in df.iterrows():\n",
        "        print(f\"| {row['model']:<25} | {row['mmd']:<10.5f} | {row['rmse']:<10.5f} | {row['corr_c']:<10.5f} | {row['corr_d']:<10.5f} |\")\n",
        "    print(\"=\"*60 + \"\\n\")\n",
        "    \n",
        "    # Plotting\n",
        "    import matplotlib.pyplot as plt\n",
        "    fig, axes = plt.subplots(1, 3, figsize=(20, 5))\n",
        "    fig.suptitle('Side-by-Side Model Efficiency & Quality Comparison (Lower is Better)', fontsize=16, y=1.05)\n",
        "    \n",
        "    # 1. Continuous MMD\n",
        "    axes[0].bar(df['model'], df['mmd'], color=['gray', 'orange', 'green'][:len(df)])\n",
        "    axes[0].set_title('Continuous MMD')\n",
        "    axes[0].set_ylabel('MMD Score')\n",
        "    \n",
        "    # 2. Discrete RMSE\n",
        "    axes[1].bar(df['model'], df['rmse'], color=['gray', 'orange', 'green'][:len(df)])\n",
        "    axes[1].set_title('Discrete Prob RMSE')\n",
        "    axes[1].set_ylabel('RMSE Score')\n",
        "    \n",
        "    # 3. Continuous Correlation Error\n",
        "    axes[2].bar(df['model'], df['corr_c'], color=['gray', 'orange', 'green'][:len(df)])\n",
        "    axes[2].set_title('Continuous Correlation Error')\n",
        "    axes[2].set_ylabel('Absolute Error')\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.savefig('Output/analysis/3_model_efficiency_comparison.png', dpi=300)\n",
        "    plt.show()\n",
        "else:\n",
        "    print(\"❌ No results to show! Please check your checkpoint paths.\")"
    ]
})

# Write the final notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print("Successfully generated neo_m3gan_efficiency_colab.ipynb!")
