import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load CSV file with semicolon as separator
file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\experimente\Overfitting\augmentation\on_val_data\epoch_metrics_base_model.csv"
df = pd.read_csv(file_path, sep=";")

# Ensure correct sorting by epoch
df = df.sort_values(by="epoch")

# Set output directory to same folder as CSV file
output_dir = os.path.dirname(file_path)

# Define x-axis tick positions (starting from 5 in steps of 5)
x_ticks = np.arange(5, df["epoch"].max() + 1, step=5)

# Plot 1: Triplet Loss per Epoch
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_triplet_loss"], marker='o', linestyle='-', color='blue', label="Triplet Loss")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Triplet Loss", fontsize=12)
plt.title("Triplet Loss per Epoch", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Adjust axes
plt.xticks(x_ticks)  # X-axis starts at 5 and then increments in steps of 5
min_loss = np.floor(min(df["train_triplet_loss"]) * 10) / 10  # Round down to nearest 0.1
max_loss = np.ceil(max(df["train_triplet_loss"]) * 10) / 10   # Round up to nearest 0.1
plt.yticks(np.arange(0, max_loss + 0.1, step=0.1))  # Y-axis in 0.1 steps, always starting at 0
plt.ylim(0, max_loss)  # Fix y-axis limits, starting at 0

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "triplet_loss_per_epoch.png"), dpi=300)
plt.close()  # Close the plot to avoid interactive issues

# Plot 2: Information Retrieval Metrics per Epoch
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["val_precisionK"], marker='o', linestyle='-', label="Precision@K", color='red')
plt.plot(df["epoch"], df["val_recallK"], marker='s', linestyle='--', label="Recall@K", color='purple')
plt.plot(df["epoch"], df["val_mAP"], marker='d', linestyle=':', label="mAP", color='green')

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Information Retrieval Metrics per Epoch", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Adjust axes
plt.xticks(x_ticks)  # X-axis starts at 5 and then increments in steps of 5
plt.yticks(np.arange(0, 1.1, step=0.1))  # Y-axis fixed from 0 to 1 in 0.1 steps
plt.ylim(0, 1)  # Y-axis always from 0 to 1

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ir_metrics_per_epoch.png"), dpi=300)
plt.close()  # Close the plot to avoid interactive issues
