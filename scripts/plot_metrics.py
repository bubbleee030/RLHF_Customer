import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import glob

# Dynamically find the latest run directory
run_dirs = glob.glob("cost_output/run_deberta_official*")
latest_run_dir = max(run_dirs, key=os.path.getctime)
log_path = os.path.join(latest_run_dir, "training_log.json")
print(f"Loading data from: {log_path}")

# Load data
with open(log_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# The new train_cost_model_v2.py already computes EMA internally!
# Let's map them to the names expected by the plotting code below.
if 'loss_ema' in df.columns:
    df['loss_smooth'] = df['loss_ema']
    df['accuracy_smooth'] = df['accuracy_ema']
    df['accuracy_sign_smooth'] = df['accuracy_sign_ema']
else:
    # Fallback for old logs
    df['loss_smooth'] = df['loss'].ewm(alpha=0.1).mean()
    df['accuracy_smooth'] = df['accuracy'].ewm(alpha=0.1).mean()
    df['accuracy_sign_smooth'] = df['accuracy_sign'].ewm(alpha=0.1).mean()

# Set style
sns.set_theme(style="whitegrid")

# 1. Plot Loss
plt.figure(figsize=(10, 6))
# Plot original faintly in background
sns.lineplot(data=df, x="step", y="loss", color="red", alpha=0.2, linewidth=1)
# Plot smoothed line
sns.lineplot(data=df, x="step", y="loss_smooth", color="red", linewidth=2.5)
plt.title("Training Loss Over Steps (DeBERTa-v3-large)", fontsize=16)
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Loss (3-Term PKU Loss)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/plots/cost_loss_curve.png", dpi=300)
plt.close()

# 2. Plot Accuracy (Pairwise and Sign)
plt.figure(figsize=(10, 6))
# Plot original faintly in background
sns.lineplot(data=df, x="step", y="accuracy", color="blue", alpha=0.1, linewidth=1)
sns.lineplot(data=df, x="step", y="accuracy_sign", color="green", alpha=0.1, linewidth=1)
# Plot smoothed line
sns.lineplot(data=df, x="step", y="accuracy_smooth", label="Pairwise Accuracy (Smoothed)", color="blue", linewidth=2.5)
sns.lineplot(data=df, x="step", y="accuracy_sign_smooth", label="Sign Accuracy (Smoothed)", color="green", linewidth=2.5)
plt.title("Training Accuracy Over Steps (DeBERTa-v3-large)", fontsize=16)
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("outputs/plots/cost_acc_curve.png", dpi=300)
plt.close()

print("Plots saved as outputs/plots/cost_loss_curve.png and outputs/plots/cost_acc_curve.png")
