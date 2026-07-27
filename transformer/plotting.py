from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_curves(history: dict, best_epoch: int, path: Path, title_suffix: str = ""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs_range = list(range(1, len(history.get("train_loss_curve", [])) + 1))
    ax.plot(epochs_range, history.get("train_loss_curve", []), label="Train Loss (Huber)", color="#1f77b4", lw=2)
    ax.plot(epochs_range, history.get("val_loss_curve", []), label="Val Loss (Huber)", color="#ff7f0e", lw=2)
    if best_epoch and best_epoch > 0:
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label="Best epoch")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Huber Loss", fontsize=12, fontweight="bold")
    base_title = "Training vs Validation Huber Loss"
    ax.set_title(base_title + (f" {title_suffix}" if title_suffix else ""), fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_val_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, s=36, color="#5DADE2", edgecolor="black")
    min_val, max_val = float(np.min(y_true)), float(np.max(y_true))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual pCMC (val)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted pCMC (val)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
