"""
evaluate.py
===========
Evaluation metrics and visualization for MalScan-ML.

Functions:
    - plot_roc_curves(): Overlay ROC curves for multiple models
    - plot_confusion_matrix(): Heatmap confusion matrix
    - plot_feature_importance(): Horizontal bar chart of top features
    - print_classification_report(): Formatted console report
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report
)


# ─── Plot style ──────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = {
    "random_forest": "#2196F3",
    "xgboost": "#FF5722",
    "malware": "#F44336",
    "benign": "#4CAF50",
}


def plot_roc_curves(
    y_true: np.ndarray,
    roc_data: dict,          # {model_name: (y_proba, auc_score)}
    save_path: Path = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        y_true: True binary labels
        roc_data: dict mapping model name → (predicted probabilities, AUC score)
        save_path: If provided, save figure here
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, (y_proba, auc_score) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        color = PALETTE.get(model_name, "#9C27B0")
        label = f"{model_name.replace('_', ' ').title()} (AUC = {auc_score:.3f})"
        ax.plot(fpr, tpr, lw=2, color=color, label=label)

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Malware Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate operating point hint
    ax.annotate(
        "Operating point\n(FPR=0.05)",
        xy=(0.05, 0.0),
        xytext=(0.15, 0.05),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8, color="gray",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] ROC plot saved → {save_path}")
    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    labels: list = ["Benign", "Malware"],
    save_path: Path = None,
    figsize: tuple = (6, 5),
) -> None:
    """
    Plot a styled confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        labels: Class names [negative, positive]
        save_path: If provided, save figure here
    """
    cm = confusion_matrix(y_true, y_pred)

    # Compute percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: green for correct, red for wrong
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    # Custom annotations with count + percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            is_correct = (i == j)
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=color,
            )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Add TP/TN/FP/FN labels
    tn, fp, fn, tp = cm.ravel()
    stats_text = f"TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}"
    fig.text(0.5, -0.02, stats_text, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] Confusion matrix saved → {save_path}")
    plt.show()
    plt.close()


def plot_feature_importance(
    importances,        # pd.Series: feature_name → importance score
    title: str = "Feature Importances",
    save_path: Path = None,
    figsize: tuple = (10, 8),
    color: str = "#2196F3",
) -> None:
    """
    Horizontal bar chart of feature importances.

    Args:
        importances: pd.Series sorted descending
        title: Chart title
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = len(importances)
    colors = [plt.cm.Blues(0.4 + 0.6 * (1 - i / n)) for i in range(n)]

    bars = ax.barh(
        range(n),
        importances.values[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(importances.index[::-1], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importances.values[::-1])):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8, color="gray",
        )

    # Highlight top 5
    for bar in bars[-5:]:
        bar.set_edgecolor("#FF5722")
        bar.set_linewidth(1.5)

    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, importances.values.max() * 1.12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] Feature importance plot saved → {save_path}")
    plt.show()
    plt.close()


def plot_entropy_distribution(
    df,                  # DataFrame with 'file_entropy' and 'label' columns
    save_path: Path = None,
    figsize: tuple = (10, 5),
) -> None:
    """
    Distribution of file entropy for malware vs benign samples.
    High entropy in malware is a key signal.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    feature_pairs = [
        ("file_entropy", "Whole-file Entropy"),
        ("mean_entropy", "Mean Section Entropy"),
    ]

    for ax, (feature, label) in zip(axes, feature_pairs):
        if feature not in df.columns:
            continue

        malware = df[df["label"] == 1][feature].dropna()
        benign = df[df["label"] == 0][feature].dropna()

        ax.hist(benign, bins=40, alpha=0.6, color=PALETTE["benign"],
                label=f"Benign (n={len(benign):,})", density=True)
        ax.hist(malware, bins=40, alpha=0.6, color=PALETTE["malware"],
                label=f"Malware (n={len(malware):,})", density=True)

        ax.axvline(x=7.0, color="black", linestyle="--", lw=1.5,
                   alpha=0.7, label="Threshold (7.0)")

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{label} Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Entropy: Malware vs Benign", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] Entropy plot saved → {save_path}")
    plt.show()
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Print a formatted classification report to console."""
    print(f"\n{'─'*50}")
    print(f"  Classification Report: {model_name.replace('_', ' ').title()}")
    print(f"{'─'*50}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Benign", "Malware"],
        digits=4,
    ))
