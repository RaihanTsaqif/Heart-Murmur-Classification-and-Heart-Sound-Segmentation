"""Plot training progress for the folds completed before early stop."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CLASS_NAMES = ["S1", "Systole", "S2", "Diastole"]
COLORS = ["#4C9BE8", "#E8804C", "#5BC85B", "#C85BC8"]
FOLDS = [("version_0", "Fold 0"), ("version_1", "Fold 1")]

def epoch_rows(path):
    df = pd.read_csv(path)
    # keep one consolidated row per epoch: epoch-level metrics live where val_accuracy is present
    df = df[df["val_accuracy"].notna()].copy()
    df = df.groupby("epoch", as_index=False).last()
    return df

data = {name: epoch_rows(f"lightning_logs/{v}/metrics.csv") for v, name in FOLDS}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Heart-Sound Segmenter (LSTM+CRF) — Training Progress (2 folds before early stop)",
             fontsize=14, fontweight="bold")

# 1) Loss
ax = axes[0, 0]
for (name, df), ls in zip(data.items(), ["-", "--"]):
    ax.plot(df["epoch"], df["train_loss_epoch"], ls, color="#E8804C", label=f"{name} train")
    ax.plot(df["epoch"], df["val_loss_epoch"], ls, color="#4C9BE8", label=f"{name} val")
ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2) Accuracy
ax = axes[0, 1]
for (name, df), ls in zip(data.items(), ["-", "--"]):
    ax.plot(df["epoch"], df["train_accuracy_epoch"], ls, color="#E8804C", label=f"{name} train")
    ax.plot(df["epoch"], df["val_accuracy"], ls, color="#4C9BE8", label=f"{name} val")
ax.set_title("Overall Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 3) Validation F1
ax = axes[1, 0]
for (name, df), ls in zip(data.items(), ["-", "--"]):
    ax.plot(df["epoch"], df["val_f1"], ls, color="#5BC85B", label=f"{name} val F1")
ax.set_title("Validation F1 (macro)"); ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 4) Per-class val accuracy at best (last) epoch, averaged across folds
ax = axes[1, 1]
finals = {name: df.iloc[-1] for name, df in data.items()}
import numpy as np
x = np.arange(4); w = 0.35
for i, (name, row) in enumerate(finals.items()):
    vals = [row[f"val_accuracy_{c}"] for c in range(4)]
    ax.bar(x + (i - 0.5) * w, vals, w, label=name,
           color=[COLORS[c] for c in range(4)], alpha=0.7 if i == 0 else 1.0,
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_title("Per-Class Validation Accuracy (final epoch)")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("training_progress.png", dpi=130)
print("Saved training_progress.png\n")

# Text summary
for name, df in data.items():
    last = df.iloc[-1]
    print(f"=== {name}: {len(df)} epochs ===")
    print(f"  val_accuracy : {last['val_accuracy']*100:.2f}%")
    print(f"  val_f1       : {last['val_f1']*100:.2f}%")
    print(f"  per-class acc: " + ", ".join(
        f"{CLASS_NAMES[c]} {last[f'val_accuracy_{c}']*100:.1f}%" for c in range(4)))
