import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load CSV file with semicolon as separator
file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\experimente\normal_vs_abnormal\epoch_metrics_base_model.csv"
df = pd.read_csv(file_path, sep=";")

# Clean column names (remove spaces, keep original casing)
df.columns = df.columns.str.strip()

# Debug: Print columns to check their names
print("Columns in CSV:", df.columns)

# Ensure correct sorting by Epoch (use correct capitalization!)
df = df.sort_values(by="Epoch")

# Set output directory to same folder as CSV file
output_dir = os.path.dirname(file_path)

# Define x-axis tick positions (starting from 5 in steps of 5)
x_ticks = np.arange(5, df["Epoch"].max() + 1, step=5)

# Compute total loss (TripLoss + BCELoss)
df["TotalLoss"] = df["TripLoss"] + df["BCELoss"]

# Plot 1: Loss per Epoch (TripLoss, BCELoss, Total Loss)
plt.figure(figsize=(8, 5))
plt.plot(df["Epoch"], df["TripLoss"], marker='o', linestyle='-', color='blue', label="Triplet Loss")
plt.plot(df["Epoch"], df["BCELoss"], marker='s', linestyle='--', color='orange', label="BCE Loss")
plt.plot(df["Epoch"], df["TotalLoss"], marker='d', linestyle=':', color='red', label="Total Loss (Triplet + BCE)")

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Loss per Epoch", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Adjust axes
plt.xticks(x_ticks)
min_loss = np.floor(min(df["TotalLoss"]) * 10) / 10
max_loss = np.ceil(max(df["TotalLoss"]) * 10) / 10
plt.yticks(np.arange(0, max_loss + 0.1, step=0.1))
plt.ylim(0, max_loss)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_per_epoch.png"), dpi=300)
plt.close()

# Plot 2: Information Retrieval Metrics per Epoch
plt.figure(figsize=(8, 5))
plt.plot(df["Epoch"], df["P@K"], marker='o', linestyle='-', label="Precision@K", color='red')
plt.plot(df["Epoch"], df["R@K"], marker='s', linestyle='--', label="Recall@K", color='purple')
plt.plot(df["Epoch"], df["mAP"], marker='d', linestyle=':', label="mAP", color='green')

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Information Retrieval Metrics per Epoch", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Adjust axes
plt.xticks(x_ticks)
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ir_metrics_per_epoch.png"), dpi=300)
plt.close()

# Plot 3: ACC and AUC per Epoch
plt.figure(figsize=(8, 5))
plt.plot(df["Epoch"], df["ACC"], marker='o', linestyle='-', label="Accuracy", color='blue')
plt.plot(df["Epoch"], df["AUC"], marker='s', linestyle='--', label="AUC", color='red')

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Accuracy and AUC per Epoch", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Adjust axes
plt.xticks(x_ticks)
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acc_auc_per_epoch.png"), dpi=300)
plt.close()
