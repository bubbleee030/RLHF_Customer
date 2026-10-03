import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

log_path = "/tmp/ministral_pipeline_full.log"

metrics = []

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Split by \r and \n
lines = re.split(r'[\r\n]+', content)

# Regex to extract metrics from tqdm output
# Example: 1315/1315 [46:05<00:00,  2.10s/it, acc=0.6562, acc_s=0.8125, loss=1.4883, lr=8.58e-06]
pattern = re.compile(r'(\d+)/1315 \[.*acc=([\d.]+),\s*acc_s=([\d.]+),\s*loss=([\d.]+),\s*lr=([\d.e-]+)\]')

current_epoch = 1
last_step = 0
for line in lines:
    if "Epoch" in line and "/" in line:
        try:
            ep_match = re.search(r'Epoch (\d+)/3', line)
            if ep_match:
                current_epoch = int(ep_match.group(1))
        except:
            pass
    
    match = pattern.search(line)
    if match:
        step_in_epoch = int(match.group(1))
        # Total steps calculation (1315 steps per epoch)
        global_step = (current_epoch - 1) * 1315 + step_in_epoch
        
        # Avoid duplicate global steps if \r overwrites
        if global_step > last_step:
            acc = float(match.group(2))
            acc_s = float(match.group(3))
            loss = float(match.group(4))
            lr = float(match.group(5))
            
            metrics.append({
                "step": global_step,
                "epoch": current_epoch,
                "accuracy": acc,
                "accuracy_sign": acc_s,
                "loss": loss,
                "lr": lr
            })
            last_step = global_step

if not metrics:
    print("No metrics found!")
    exit(1)

df = pd.DataFrame(metrics)

# Exponential Moving Average for smoothing
df['loss_smooth'] = df['loss'].ewm(alpha=0.1).mean()
df['accuracy_smooth'] = df['accuracy'].ewm(alpha=0.1).mean()
df['accuracy_sign_smooth'] = df['accuracy_sign'].ewm(alpha=0.1).mean()

os.makedirs("outputs/plots", exist_ok=True)

sns.set_theme(style="whitegrid")

# Plot Loss
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="step", y="loss", color="red", alpha=0.2, linewidth=1)
sns.lineplot(data=df, x="step", y="loss_smooth", color="red", linewidth=2.5)
plt.title("Training Loss Over Steps (Ministral Instruct Cutoff)", fontsize=16)
plt.xlabel("Global Steps", fontsize=14)
plt.ylabel("Loss (Sequence-wise)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/plots/cutoff_loss_curve.png", dpi=300)
plt.close()

# Plot Accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="step", y="accuracy", color="blue", alpha=0.1, linewidth=1)
sns.lineplot(data=df, x="step", y="accuracy_sign", color="green", alpha=0.1, linewidth=1)
sns.lineplot(data=df, x="step", y="accuracy_smooth", label="Pairwise Accuracy (Smoothed)", color="blue", linewidth=2.5)
sns.lineplot(data=df, x="step", y="accuracy_sign_smooth", label="Sign Accuracy (Smoothed)", color="green", linewidth=2.5)
plt.title("Training Accuracy Over Steps (Ministral Instruct Cutoff)", fontsize=16)
plt.xlabel("Global Steps", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("outputs/plots/cutoff_acc_curve.png", dpi=300)
plt.close()

print(f"Parsed {len(df)} steps. Plots saved to outputs/plots/")
print(df.tail(1))
