"""
Evaluation Module
====================

Provides comprehensive model evaluation including metrics computation
and publication-quality visualization for both pipelines.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    COLOR_PALETTE,
    FIGURE_DPI,
    FIGURE_SIZE_LARGE,
    FIGURE_SIZE_MEDIUM,
    FIGURE_SIZE_SMALL,
    FIGURES_DIR,
    HEATMAP_CMAP,
)
from src.utils import format_percentage, setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for ROC-AUC). Optional.
        average: Averaging method ('binary', 'macro', 'weighted').

    Returns:
        Dictionary with metric names and values.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # ROC-AUC (requires probabilities)
    if y_prob is not None:
        try:
            if average == "binary":
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
                )
        except ValueError as e:
            logger.warning(f"Cannot compute ROC-AUC: {e}")
            metrics["roc_auc"] = float("nan")

    # Multiclass-specific metrics
    if average != "binary":
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    logger.info(
        f"Metrics: " +
        ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not np.isnan(v))
    )

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Print and return sklearn's detailed classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        target_names: Optional list of class names.

    Returns:
        Formatted classification report string.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )
    print(report)
    return report


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_name: Optional[str] = None,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class label names for axes.
        title: Plot title.
        save_name: Filename to save (without path). Saved to FIGURES_DIR.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_name: Optional[str] = None,
) -> None:
    """
    Plot the ROC curve with AUC annotation.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        title: Plot title.
        save_name: Filename to save.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9E9E9E", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    plt.show()
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_name: Optional[str] = None,
) -> None:
    """
    Plot the Precision-Recall curve.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        title: Plot title.
        save_name: Filename to save.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)
    ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#4CAF50")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Precision-Recall curve saved to {save_path}")

    plt.show()
    plt.close(fig)


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_name: Optional[str] = None,
) -> None:
    """
    Plot horizontal bar chart of feature importances.

    Args:
        importance: Array of importance values.
        feature_names: List of feature names.
        top_n: Number of top features to display.
        title: Plot title.
        save_name: Filename to save.
    """
    # Sort by importance
    indices = np.argsort(importance)[-top_n:]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))
    ax.barh(range(len(sorted_names)), sorted_importance, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()
    plt.close(fig)


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_name: Optional[str] = None,
) -> None:
    """
    Plot a correlation matrix heatmap for numeric features.

    Args:
        df: DataFrame with numeric columns.
        title: Plot title.
        save_name: Filename to save.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap=HEATMAP_CMAP,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Correlation matrix saved to {save_path}")

    plt.show()
    plt.close(fig)


def plot_target_distribution(
    y: pd.Series,
    title: str = "Target Distribution",
    save_name: Optional[str] = None,
) -> None:
    """
    Plot the distribution of the target variable.

    Args:
        y: Target Series.
        title: Plot title.
        save_name: Filename to save.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)
    counts = y.value_counts()
    colors = ["#4CAF50", "#F44336"] if len(counts) == 2 else plt.cm.Set2(range(len(counts)))
    counts.plot(kind="bar", color=colors, ax=ax, edgecolor="black", alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    # Add count labels on bars
    for i, (idx, val) in enumerate(counts.items()):
        ax.text(i, val + max(counts) * 0.01, str(val),
                ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Target distribution plot saved to {save_path}")

    plt.show()
    plt.close(fig)
