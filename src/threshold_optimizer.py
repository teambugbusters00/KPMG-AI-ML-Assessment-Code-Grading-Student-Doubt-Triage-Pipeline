"""
Threshold Optimization Module
================================

Evaluates and selects optimal confidence thresholds for the doubt triage
pipeline's routing logic:
  - Confidence >= threshold → Auto Approval
  - Confidence <  threshold → Teacher Review

Evaluates thresholds: [0.60, 0.70, 0.80, 0.85, 0.90]
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CONFIDENCE_THRESHOLDS,
    FIGURE_DPI,
    FIGURE_SIZE_MEDIUM,
    FIGURE_SIZE_SMALL,
    FIGURES_DIR,
)
from src.utils import format_percentage, setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Threshold Evaluation
# =============================================================================


def evaluate_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_confidence: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Evaluate multiple confidence thresholds on validation data.

    For each threshold, computes:
      - Accuracy of auto-approved predictions
      - Auto-approval rate (% of predictions above threshold)
      - Number of auto-approved vs teacher-reviewed

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_confidence: Confidence scores (max probability).
        thresholds: List of thresholds to evaluate. Defaults to config.

    Returns:
        DataFrame with evaluation results per threshold.
    """
    thresholds = thresholds or CONFIDENCE_THRESHOLDS
    results = []

    for threshold in thresholds:
        auto_mask = y_confidence >= threshold
        n_auto = auto_mask.sum()
        n_review = (~auto_mask).sum()
        n_total = len(y_true)

        auto_approval_rate = n_auto / n_total if n_total > 0 else 0

        # Accuracy of auto-approved predictions
        if n_auto > 0:
            auto_accuracy = float(
                np.mean(y_pred[auto_mask] == y_true[auto_mask])
            )
        else:
            auto_accuracy = 0.0

        # Overall accuracy when using this threshold
        # (auto-approved use model prediction, teacher-reviewed are assumed correct)
        effective_correct = (
            np.sum(y_pred[auto_mask] == y_true[auto_mask]) +
            np.sum(~auto_mask)  # teacher-reviewed = always correct
        )
        effective_accuracy = float(effective_correct / n_total) if n_total > 0 else 0

        results.append({
            "threshold": threshold,
            "auto_approved": int(n_auto),
            "teacher_reviewed": int(n_review),
            "auto_approval_rate": float(auto_approval_rate),
            "auto_accuracy": auto_accuracy,
            "effective_accuracy": effective_accuracy,
        })

        logger.info(
            f"Threshold {threshold:.2f}: "
            f"Auto={n_auto} ({auto_approval_rate:.1%}), "
            f"Review={n_review}, "
            f"Auto Acc={auto_accuracy:.4f}, "
            f"Effective Acc={effective_accuracy:.4f}"
        )

    return pd.DataFrame(results)


def select_best_threshold(
    threshold_results: pd.DataFrame,
    metric: str = "effective_accuracy",
    min_auto_rate: float = 0.3,
) -> float:
    """
    Select the best threshold based on evaluation results.

    Prioritizes the specified metric while ensuring a minimum auto-approval rate.

    Args:
        threshold_results: DataFrame from evaluate_thresholds().
        metric: Column to optimize ('effective_accuracy' or 'auto_accuracy').
        min_auto_rate: Minimum acceptable auto-approval rate.

    Returns:
        Best threshold value.
    """
    # Filter by minimum auto-approval rate
    eligible = threshold_results[
        threshold_results["auto_approval_rate"] >= min_auto_rate
    ]

    if eligible.empty:
        logger.warning(
            f"No threshold meets min auto-approval rate {min_auto_rate}. "
            f"Using lowest threshold."
        )
        best_threshold = float(threshold_results["threshold"].min())
    else:
        best_idx = eligible[metric].idxmax()
        best_threshold = float(eligible.loc[best_idx, "threshold"])

    best_row = threshold_results[
        threshold_results["threshold"] == best_threshold
    ].iloc[0]

    logger.info(
        f"Best threshold: {best_threshold:.2f} "
        f"(Auto Rate: {best_row['auto_approval_rate']:.1%}, "
        f"{metric}: {best_row[metric]:.4f})"
    )

    return best_threshold


# =============================================================================
# Routing Logic
# =============================================================================


def route_prediction(
    confidence: float,
    threshold: float,
) -> str:
    """
    Apply routing logic based on prediction confidence.

    Args:
        confidence: Model's prediction confidence (max probability).
        threshold: Confidence threshold for auto-approval.

    Returns:
        'Auto Approval' if confidence >= threshold, else 'Teacher Review'.
    """
    return "Auto Approval" if confidence >= threshold else "Teacher Review"


def route_predictions_batch(
    confidences: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Apply routing logic to a batch of predictions.

    Args:
        confidences: Array of confidence scores.
        threshold: Confidence threshold.

    Returns:
        Array of route labels ('Auto Approval' or 'Teacher Review').
    """
    routes = np.where(
        confidences >= threshold,
        "Auto Approval",
        "Teacher Review",
    )

    n_auto = (routes == "Auto Approval").sum()
    n_review = (routes == "Teacher Review").sum()
    logger.info(
        f"Routing: {n_auto} auto-approved ({n_auto / len(routes):.1%}), "
        f"{n_review} teacher review ({n_review / len(routes):.1%})"
    )

    return routes


# =============================================================================
# Visualization
# =============================================================================


def plot_threshold_vs_accuracy(
    threshold_results: pd.DataFrame,
    best_threshold: Optional[float] = None,
    save_name: Optional[str] = None,
) -> None:
    """
    Plot threshold vs accuracy curves.

    Args:
        threshold_results: DataFrame from evaluate_thresholds().
        best_threshold: Highlight the selected threshold.
        save_name: Filename to save the plot.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)

    ax.plot(
        threshold_results["threshold"],
        threshold_results["auto_accuracy"],
        marker="o", linewidth=2, color="#2196F3",
        label="Auto-Approved Accuracy",
    )
    ax.plot(
        threshold_results["threshold"],
        threshold_results["effective_accuracy"],
        marker="s", linewidth=2, color="#4CAF50",
        label="Effective Accuracy",
    )

    if best_threshold is not None:
        ax.axvline(x=best_threshold, color="#F44336", linestyle="--",
                    linewidth=1.5, label=f"Best Threshold ({best_threshold:.2f})")

    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Threshold vs Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Threshold vs accuracy plot saved to {save_path}")

    plt.show()
    plt.close(fig)


def plot_threshold_vs_approval_rate(
    threshold_results: pd.DataFrame,
    best_threshold: Optional[float] = None,
    save_name: Optional[str] = None,
) -> None:
    """
    Plot threshold vs auto-approval rate.

    Args:
        threshold_results: DataFrame from evaluate_thresholds().
        best_threshold: Highlight the selected threshold.
        save_name: Filename to save the plot.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)

    ax.plot(
        threshold_results["threshold"],
        threshold_results["auto_approval_rate"],
        marker="D", linewidth=2, color="#FF9800",
        label="Auto-Approval Rate",
    )
    ax.fill_between(
        threshold_results["threshold"],
        threshold_results["auto_approval_rate"],
        alpha=0.15, color="#FF9800",
    )

    if best_threshold is not None:
        ax.axvline(x=best_threshold, color="#F44336", linestyle="--",
                    linewidth=1.5, label=f"Best Threshold ({best_threshold:.2f})")

    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Auto-Approval Rate", fontsize=12)
    ax.set_title("Threshold vs Auto-Approval Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Threshold vs approval rate plot saved to {save_path}")

    plt.show()
    plt.close(fig)


def plot_confidence_distribution(
    confidences: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Prediction Confidence Distribution",
    save_name: Optional[str] = None,
) -> None:
    """
    Plot histogram of prediction confidence scores.

    Args:
        confidences: Array of confidence scores.
        threshold: Optional threshold line to display.
        title: Plot title.
        save_name: Filename to save the plot.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)

    ax.hist(confidences, bins=30, color="#7E57C2", alpha=0.7,
            edgecolor="black", linewidth=0.5)

    if threshold is not None:
        ax.axvline(x=threshold, color="#F44336", linestyle="--",
                    linewidth=2, label=f"Threshold ({threshold:.2f})")
        ax.legend(fontsize=11)

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Confidence distribution plot saved to {save_path}")

    plt.show()
    plt.close(fig)
