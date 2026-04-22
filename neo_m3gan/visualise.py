import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
from ultils import renormlizer


def visualise_gan(data_continuous_real, data_continuous_syn, data_discrete_real, data_discrete_syn, inx, max_val_con,
                  min_val_con, num_dim=12, num_plot=10, SAVE_PATH="logs/", c_feature_names=None, d_feature_names=None):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    data_continuous_real = renormlizer(data_continuous_real, max_val_con, min_val_con)
    data_continuous_syn = renormlizer(data_continuous_syn, max_val_con, min_val_con)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, num_dim, figsize=(100, 40))
    fig.suptitle(f"EHR-M-GAN Synthesized Data Validation (Epoch {inx})", fontsize=60, y=0.98)
    plt.setp(axes, xticks=[0, 3, 6, 9, 12, 15, 18, 21, 24])

    c_dim_list = random.sample(list(range(data_continuous_real.shape[2])), num_dim)
    c_pid_index = random.sample(list(range(len(data_continuous_real))), num_plot)
    c_pid_index_syn = random.sample(list(range(len(data_continuous_syn))), num_plot)

    for i in range(num_dim):
        c_title = c_feature_names[c_dim_list[i]] if c_feature_names and c_dim_list[i] < len(c_feature_names) else f"Continuous Feature {c_dim_list[i]}"
        
        # Continuous Real
        ax = axes[0, i]
        df = pd.DataFrame(data_continuous_real[c_pid_index, :, c_dim_list[i]])
        sns.lineplot(ax=ax, data=df.T, palette=sns.color_palette('Greens', n_colors=num_plot), legend=False, alpha=0.3)
        sns.lineplot(ax=ax, data=df.T.mean(axis=1), color='black', linewidth=3, label='Mean')
        ax.set_title(f"Real - {c_title}", fontsize=24)
        ax.set_ylabel("Value", fontsize=20)
        ax.set_xlabel("Time Step (hrs)", fontsize=20)

        # Continuous Synthetic
        ax_syn = axes[1, i]
        df_syn = pd.DataFrame(data_continuous_syn[c_pid_index_syn, :, c_dim_list[i]])
        sns.lineplot(ax=ax_syn, data=df_syn.T, palette=sns.color_palette('Reds', n_colors=num_plot), legend=False, alpha=0.3)
        sns.lineplot(ax=ax_syn, data=df_syn.T.mean(axis=1), color='black', linewidth=3, label='Mean')
        ax_syn.set_title(f"Synthetic - {c_title}", fontsize=24)
        ax_syn.set_ylabel("Value", fontsize=20)
        ax_syn.set_xlabel("Time Step (hrs)", fontsize=20)
        
        # Exact vertical range match with Real plot
        ax_syn.set_ylim(axes[0, i].get_ylim())

    d_dim_list = random.sample(list(range(data_discrete_real.shape[2])), num_dim)
    d_pid_index = random.sample(list(range(len(data_discrete_real))), num_plot)
    d_pid_index_syn = random.sample(list(range(len(data_discrete_syn))), num_plot)

    for i in range(num_dim):
        d_title = d_feature_names[d_dim_list[i]] if d_feature_names and d_dim_list[i] < len(d_feature_names) else f"Discrete Feature {d_dim_list[i]}"
        
        # Discrete Real (Heatmap is much better for binary on/off data)
        ax = axes[2, i]
        d_data_real = data_discrete_real[d_pid_index, :, d_dim_list[i]]
        if hasattr(d_data_real, "detach"): d_data_real = d_data_real.detach().cpu().numpy()
        
        sns.heatmap(d_data_real, ax=ax, cmap="Greens", cbar=True, vmin=0, vmax=1, linewidths=0.05, linecolor='lightgray')
        ax.set_title(f"Real - {d_title}", fontsize=24)
        ax.set_ylabel("Patient Sample", fontsize=20)
        ax.set_xlabel("Time Step (hrs)", fontsize=20)

        # Discrete Synthetic
        ax = axes[3, i]
        d_data_syn = data_discrete_syn[d_pid_index_syn, :, d_dim_list[i]]
        if hasattr(d_data_syn, "detach"): d_data_syn = d_data_syn.detach().cpu().numpy()

        sns.heatmap(d_data_syn, ax=ax, cmap="Reds", cbar=True, vmin=0, vmax=1, linewidths=0.05, linecolor='lightgray')
        ax.set_title(f"Synthetic - {d_title}", fontsize=24)
        ax.set_ylabel("Patient Sample", fontsize=20)
        ax.set_xlabel("Time Step (hrs)", fontsize=20)

    # Adjust layout to prevent overlap with huge suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Sanitize inx for filename (removes invalid filesystem characters)
    safe_inx = str(inx)
    if len(safe_inx) > 100: safe_inx = safe_inx[:100]
    for char in ['{', '}', ':', '"', "'", '[', ']', ' ', '\n', '\t', '\\', '/']:
        safe_inx = safe_inx.replace(char, '_')

    fig.savefig(os.path.join(SAVE_PATH, f'visualise_gan_epoch_{safe_inx}.pdf'), format='pdf')
    plt.close(fig)


def visualise_vae(data_continuous_real, data_continuous_syn, data_discrete_real, data_discrete_syn, inx, max_val_con,
                  min_val_con, num_dim=8, num_plot=10, SAVE_PATH="logs/"):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    data_continuous_real = renormlizer(data_continuous_real, max_val_con, min_val_con)
    data_continuous_syn = renormlizer(data_continuous_syn, max_val_con, min_val_con)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, num_dim, figsize=(100, 30))
    fig.suptitle(f"EHR-M-GAN VAE Reconstruction Validation (Epoch {inx})", fontsize=60, y=0.98)
    plt.setp(axes, xticks=[0, 3, 6, 9, 12, 15, 18, 21, 24])

    c_dim_list = random.sample(list(range(data_continuous_real.shape[2])), num_dim)
    c_pid_index = random.sample(list(range(len(data_continuous_syn))), num_plot)

    for i in range(len(c_dim_list)):
        ax = axes[0, i]
        df = pd.DataFrame(data_continuous_real[c_pid_index, :, c_dim_list[i]])
        sns.lineplot(ax=ax, data=df.T, palette=sns.color_palette('Greens', n_colors=num_plot), legend=False)
        df_syn = pd.DataFrame(data_continuous_syn[c_pid_index, :, c_dim_list[i]])
        sns.lineplot(ax=ax, data=df_syn.T, marker='o', palette=sns.color_palette('Reds', n_colors=num_plot), legend=False)
        ax.set_title(f"Continuous Feature {c_dim_list[i]} (Green=Real, Red=Rec)", fontsize=24)
        ax.set_ylabel("Value", fontsize=20)
        ax.set_xlabel("Time Step (hrs)", fontsize=20)

    d_dim_list = random.sample(list(range(data_discrete_real.shape[2])), num_dim)
    d_pid_index = random.sample(list(range(len(data_discrete_syn))), num_plot)

    for i in range(len(d_dim_list)):
        ax = axes[1, i]
        df = pd.DataFrame(data_discrete_real[d_pid_index, :, d_dim_list[i]])
        sns.lineplot(ax=ax, data=df.T, palette=sns.color_palette('Greens', n_colors=num_plot), legend=False)
        df_syn = pd.DataFrame(data_discrete_syn[d_pid_index, :, d_dim_list[i]])
        sns.lineplot(ax=ax, data=df_syn.T, marker='o', palette=sns.color_palette('Reds', n_colors=num_plot), legend=False)
        ax.set_title(f"Discrete Feature {d_dim_list[i]} (Green=Real, Red=Rec)", fontsize=24)
        ax.set_ylabel("Probability", fontsize=20)
        ax.set_xlabel("Time Step (hrs)", fontsize=20)

    # Adjust layout to prevent overlap with huge suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Sanitize inx for filename
    safe_inx = str(inx)
    if len(safe_inx) > 100: safe_inx = safe_inx[:100]
    for char in ['{', '}', ':', '"', "'", '[', ']', ' ', '\n', '\t', '\\', '/']:
        safe_inx = safe_inx.replace(char, '_')

    fig.savefig(os.path.join(SAVE_PATH, f'visualise_vae_epoch_{safe_inx}.pdf'), format='pdf')
    plt.close(fig)


def plot_metrics_trend(history_dict, save_path="logs/"):
    """
    Plots the quantitative metric trajectory over time and saves as a single multi-panel PDF.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    epochs = history_dict.get('epochs', [])
    if len(epochs) == 0:
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("EHR-M-GAN Quantitative Training Metrics Trend", fontsize=20, y=1.05)

    # Plot 1: MMD
    if 'mmd' in history_dict and len(history_dict['mmd']) > 0:
        axes[0].plot(epochs, history_dict['mmd'], marker='o', color='blue', linewidth=2)
        axes[0].set_title("Continuous MMD (Lower is Better)", fontsize=16)
        axes[0].set_xlabel("Epochs", fontsize=14)
        axes[0].set_ylabel("MMD Score", fontsize=14)
        axes[0].tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: RMSE
    if 'rmse' in history_dict and len(history_dict['rmse']) > 0:
        axes[1].plot(epochs, history_dict['rmse'], marker='o', color='green', linewidth=2)
        axes[1].set_title("Discrete Probability RMSE (Lower is Better)", fontsize=16)
        axes[1].set_xlabel("Epochs", fontsize=14)
        axes[1].set_ylabel("RMSE Score", fontsize=14)
        axes[1].tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 3: Correlation Errors
    if 'corr_c' in history_dict and len(history_dict['corr_c']) > 0:
        axes[2].plot(epochs, history_dict['corr_c'], marker='o', label="Continuous Corr Error", color='red', linewidth=2)
        if 'corr_d' in history_dict and len(history_dict['corr_d']) > 0:
            axes[2].plot(epochs, history_dict['corr_d'], marker='s', label="Discrete Corr Error", color='orange', linewidth=2)
        axes[2].set_title("Feature Correlation Error (Lower is Better)", fontsize=16)
        axes[2].set_xlabel("Epochs", fontsize=14)
        axes[2].set_ylabel("Absolute Error", fontsize=14)
        axes[2].legend(fontsize=12)
        axes[2].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'metrics_history.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)