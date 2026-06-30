"""
ResNet-LSTM Results Visualization
use:
    python ResNet_lstm_visualize.py

Generated files (saved in the plots/ directory):
    - training_curves.png      → Loss and accuracy curves
    - confusion_matrix.png     → Confusion matrix heatmap for the validation set
    - per_class_acc.png        → Accuracy bar chart per emotion class
    - speech_vs_song.png       → Accuracy comparison between speech and song
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # No gui
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix

# PATHS 
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_JSON = os.path.join(ROOT_DIR, "models", "train_history.json")
VAL_CSV      = os.path.join(ROOT_DIR, "resnet_lstm_val_results.csv")
PLOTS_DIR    = os.path.join(ROOT_DIR, "plots")

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]

os.makedirs(PLOTS_DIR, exist_ok=True)


# TRAINING CURVES  (loss + accuracy)
def plot_training_curves():
    if not os.path.exists(HISTORY_JSON):
        print(f"[SKIP] train_history.json not found: {HISTORY_JSON}")
        return

    with open(HISTORY_JSON) as f:
        history = json.load(f)

    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]
    val_acc     = [h["val_acc"]    for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ResNet-LSTM Training History", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, "b-o", markersize=4, label="Train Loss")
    ax.plot(epochs, val_loss,   "r-o", markersize=4, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in val_acc], "g-o", markersize=4, label="Val Accuracy")
    best_epoch = epochs[val_acc.index(max(val_acc))]
    best_acc   = max(val_acc) * 100
    ax.axvline(best_epoch, color="orange", linestyle="--", linewidth=1.2,
               label=f"Best: epoch {best_epoch} (%{best_acc:.1f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-class accuracy son epoch
    if "per_class_acc" in history[-1]:
        last = history[-1]["per_class_acc"]
        note = "Last epoch per-class: " + ", ".join(
            f"{e}={v*100:.0f}%" for e, v in last.items()
        )
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved succesfully: {out}")



# CONFUSION MATRIX

def plot_confusion_matrix():
    if not os.path.exists(VAL_CSV):
        print(f"[SKIP] val CSV not found: {VAL_CSV}")
        return

    df = pd.read_csv(VAL_CSV)
    y_true = df["True_Label"].str.lower().tolist()
    y_pred = df["Predicted_Label"].str.lower().tolist()

    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    # Normalize row based
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — ResNet-LSTM", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Ham sayılar",   "d"),
        (axes[1], cm_norm, "Normalize (%)", ".1%"),
    ]:
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if fmt == "d" else "RdYlGn")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_xticklabels([e.capitalize() for e in EMOTIONS], rotation=35, ha="right")
        ax.set_yticklabels([e.capitalize() for e in EMOTIONS])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        thresh = data.max() / 2.0
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                val = data[i, j]
                label = f"{val:{fmt}}" if fmt == "d" else f"{val*100:.1f}%"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=9,
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out}")



#  PER-CLASS ACCURACY 
def plot_per_class_accuracy():
    if not os.path.exists(VAL_CSV):
        print(f"[SKIP] val CSV not found: {VAL_CSV}")
        return

    df = pd.read_csv(VAL_CSV)
    accs = []
    counts = []
    for emo in EMOTIONS:
        sub = df[df["True_Label"] == emo.upper()]
        if sub.empty:
            accs.append(0)
            counts.append(0)
        else:
            acc = (sub["Correct"] == "YES").sum() / len(sub) * 100
            accs.append(acc)
            counts.append(len(sub))

    colors = ["#4CAF50" if a >= 70 else "#FFC107" if a >= 50 else "#F44336"
              for a in accs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([e.capitalize() for e in EMOTIONS], accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy — ResNet-LSTM (Val Set)", fontweight="bold")
    ax.axhline(sum(accs) / len(accs), color="navy", linestyle="--",
               linewidth=1.2, label=f"average: %{sum(accs)/len(accs):.1f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, acc, n in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"%{acc:.1f}\n(n={n})", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "per_class_acc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved succesfully: {out}")


#  SPEECH vs SONG ACCURACY
def plot_speech_vs_song():
    if not os.path.exists(VAL_CSV):
        print(f"[SKIP] val CSV not found: {VAL_CSV}")
        return

    df = pd.read_csv(VAL_CSV)
    if "Modality" not in df.columns:
        print("[SKIP] CSV'de Modality column not found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Speech vs Song — Per-Class Accuracy", fontweight="bold")

    x = np.arange(len(EMOTIONS))
    width = 0.35

    for ax, mod, color in [(axes[0], "speech", "steelblue"), (axes[1], "song", "coral")]:
        sub = df[df["Modality"] == mod]
        if sub.empty:
            ax.set_title(f"{mod.capitalize()} — no data")
            continue

        mod_accs = []
        mod_ns   = []
        for emo in EMOTIONS:
            s = sub[sub["True_Label"] == emo.upper()]
            if s.empty:
                mod_accs.append(0)
                mod_ns.append(0)
            else:
                mod_accs.append((s["Correct"] == "YES").sum() / len(s) * 100)
                mod_ns.append(len(s))

        overall = (sub["Correct"] == "YES").sum() / len(sub) * 100
        bars = ax.bar([e.capitalize() for e in EMOTIONS], mod_accs,
                      color=color, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{mod.capitalize()} (overall: %{overall:.1f}, n={len(sub)})")
        ax.axhline(overall, color="black", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.3)

        for bar, acc, n in zip(bars, mod_accs, mod_ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"%{acc:.0f}\n(n={n})", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "speech_vs_song.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved succesfully: {out}")



#  PER-CLASS ACC OVER EPOCHS (train history)
def plot_per_class_over_epochs():
    if not os.path.exists(HISTORY_JSON):
        print(f"[SKIP] train_history.json not found.")
        return

    with open(HISTORY_JSON) as f:
        history = json.load(f)

    if "per_class_acc" not in history[0]:
        print("[SKIP] History'de per_class_acc not found.")
        return

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_map = ["#2196F3", "#4CAF50", "#9C27B0", "#F44336", "#FF9800"]

    for i, emo in enumerate(EMOTIONS):
        vals = [h["per_class_acc"].get(emo, 0) * 100 for h in history]
        ax.plot(epochs, vals, "-o", markersize=3, linewidth=1.5,
                color=colors_map[i], label=emo.capitalize())

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy Over Epochs", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "per_class_over_epochs.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out}")



# MAIN
if __name__ == "__main__":
    print(f"graphs will be saved: {PLOTS_DIR}\n")
    plot_training_curves()
    plot_per_class_over_epochs()
    plot_confusion_matrix()
    plot_per_class_accuracy()
    plot_speech_vs_song()
    print("\n all graphics has generated.")
