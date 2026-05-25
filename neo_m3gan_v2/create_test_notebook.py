import json
import os

# Paths to the V2 python files
v2_dir = '/home/thanglm2006/EHRMGANPT/neo_m3gan_v2'
notebook_path = '/home/thanglm2006/EHRMGANPT/neo_m3gan_v2_test_kaggle.ipynb'

# Helper to read file and return list of lines
def read_file_lines(filename):
    path = os.path.join(v2_dir, filename)
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
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "version": "3.12.12"
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [
                {
                    "sourceType": "datasetVersion",
                    "sourceId": 15835044,
                    "datasetId": 10150501,
                    "databundleVersionId": 16784966
                }
            ],
            "dockerImageVersionId": 31329,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 1. Add Intro cell
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# EHR-M-GAN (Neo M3GAN V2) Testing & Inference on Kaggle\n",
        "\n",
        "This notebook is prepared for testing the improved **neo_m3gan_v2** model. It loads a trained checkpoint, generates synthetic EHR sequences, evaluates quantitative metrics (MMD, RMSE, Correlation Error), and saves high-fidelity validation plots.\n",
        "\n",
        "### Before you start:\n",
        "1. **Add Data**: Click **+ Add Data** -> Add your dataset **`mimic3`**.\n",
        "2. **Add Checkpoint**: Click **+ Add Data** -> Add your trained model checkpoint dataset.\n",
        "3. **Update Paths**: Update `--checkpoint` in the last cell to point to your loaded checkpoint file path.\n",
        "\n",
        "---"
    ]
})

# 2. Add Directory Setup cell
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# 1. Create directory structure\n",
        "import os\n",
        "os.makedirs('Data/mimic', exist_ok=True)\n",
        "os.makedirs('Output/test_plots', exist_ok=True)\n",
        "\n",
        "# 2. Link Kaggle Data to local Data directory\n",
        "dataset_root = '/kaggle/input/datasets/lmnhthng/mimic3'\n",
        "\n",
        "for filename in ['vital_sign_24hrs.pkl', 'med_interv_24hrs.pkl', 'clinical_scaler.pkl']:\n",
        "    src = os.path.join(dataset_root, filename)\n",
        "    dst = os.path.join('Data/mimic', filename)\n",
        "    if os.path.exists(src):\n",
        "        if os.path.exists(dst): os.remove(dst)\n",
        "        os.symlink(src, dst)\n",
        "        print(f\"Linked {filename}\")\n",
        "    else:\n",
        "        print(f\"WARNING: {filename} not found in {dataset_root}\")"
    ]
})

# 3. Add %%writefile cells for each code file
files_to_write = ['ultils.py', 'networks.py', 'metrics.py', 'visualise.py', 'test.py']
for filename in files_to_write:
    lines = read_file_lines(filename)
    source = [f"%%writefile {filename}\n"] + lines
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source
    })

# 4. Add Inference cell
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# Run Evaluation and Inference\n",
        "# Please verify the path to your checkpoint.pth below\n",
        "!python test.py --dataset mimic --checkpoint /kaggle/input/models/lmnhthng/neo-mgan3-epoch100/pytorch/default/1/neo_m3gan_100.pth --num_samples 5000"
    ]
})

# Write the final notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print("Successfully generated neo_m3gan_v2_test_kaggle.ipynb!")
