"""
Explainability Module
========================

Provides model interpretability using SHAP (SHapley Additive exPlanations)
for both pipelines. Uses TreeExplainer for LightGBM and KernelExplainer
as fallback.
"""

import logging
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_DPI, FIGURE_SIZE_LARGE, FIGURES_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 500,
) -> Any:
    """
    Compute SHAP values for the given model and data.

    Automatically selects the appropriate SHAP explainer:
      - TreeExplainer for tree-based models (LightGBM, RF)
      - LinearExplainer for linear models
      - KernelExplainer as fallback

    Args:
        model: Trained model (can be a Pipeline).
        X: Feature matrix to explain.
        feature_names: Optional feature names for display.
        max_samples: Maximum samples for KernelExplainer background.

    Returns:
        SHAP Explanation object with computed values.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    # Extract the actual classifier if model is a Pipeline
    classifier = model
    X_transformed = X
    if hasattr(model, "named_steps"):
        # It's a Pipeline — transform X through preprocessing steps
        classifier = model.named_steps.get("classifier", model[-1])
        preprocessors = [
            step for name, step in model.named_steps.items()
            if name != "classifier"
        ]
        for preprocessor in preprocessors:
            if hasattr(preprocessor, "transform"):
                X_transformed = preprocessor.transform(X_transformed)

    # Choose explainer
    model_type = type(classifier).__name__
    logger.info(f"Computing SHAP values using {model_type}...")

    try:
        if hasattr(classifier, "booster_") or "LGBM" in model_type or "GBM" in model_type:
            explainer = shap.TreeExplainer(classifier)
        elif "Forest" in model_type or "Tree" in model_type:
            explainer = shap.TreeExplainer(classifier)
        elif "Linear" in model_type or "Logistic" in model_type:
            explainer = shap.LinearExplainer(
                classifier,
                X_transformed[:max_samples],
            )
        else:
            # Fallback to KernelExplainer (slower but universal)
            background = shap.sample(X_transformed, min(max_samples, len(X_transformed)))
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
    except Exception as e:
        logger.warning(f"Primary explainer failed ({e}), falling back to KernelExplainer")
        background = shap.sample(X_transformed, min(100, len(X_transformed)))
        explainer = shap.KernelExplainer(
            classifier.predict if not hasattr(classifier, "predict_proba")
            else classifier.predict_proba,
            background,
        )

    # Compute SHAP values
    shap_values = explainer(X_transformed)

    logger.info(f"SHAP values computed for {X_transformed.shape[0]} samples")
    return shap_values


def plot_shap_summary(
    shap_values: Any,
    feature_names: Optional[List[str]] = None,
    title: str = "SHAP Feature Importance",
    save_name: Optional[str] = None,
    max_display: int = 20,
) -> None:
    """
    Create a SHAP summary (beeswarm) plot.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: Feature names for the y-axis.
        title: Plot title.
        save_name: Filename to save.
        max_display: Maximum number of features to show.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required for explainability plots")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)

    # Handle binary classification (SHAP returns 2D for positive class)
    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        # Multi-output: take the positive class (index 1)
        sv = sv[:, :, 1]

    shap.summary_plot(
        sv,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"SHAP summary plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_shap_bar(
    shap_values: Any,
    feature_names: Optional[List[str]] = None,
    title: str = "SHAP Mean Absolute Impact",
    save_name: Optional[str] = None,
    max_display: int = 20,
) -> None:
    """
    Create a SHAP bar plot showing mean absolute SHAP values.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: Feature names.
        title: Plot title.
        save_name: Filename to save.
        max_display: Maximum features to display.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required for explainability plots")

    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[:, :, 1]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"SHAP bar plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_shap_waterfall(
    shap_values: Any,
    sample_index: int = 0,
    title: str = "SHAP Waterfall — Single Prediction",
    save_name: Optional[str] = None,
) -> None:
    """
    Create a SHAP waterfall plot for a single prediction explanation.

    Args:
        shap_values: SHAP Explanation object.
        sample_index: Index of the sample to explain.
        title: Plot title.
        save_name: Filename to save.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required for explainability plots")

    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[:, :, 1]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    shap.plots.waterfall(sv[sample_index], show=False)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        save_path = FIGURES_DIR / save_name
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"SHAP waterfall plot saved to {save_path}")

    plt.show()
    plt.close()
