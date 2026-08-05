"""
Data Preprocessing Module
============================

Handles data quality checks, cleaning, and preparation for both pipelines:
  - Duplicate detection
  - Missing value analysis and imputation
  - Outlier detection (IQR + Z-score)
  - Data leakage detection
  - Class imbalance handling
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FIGURE_DPI,
    FIGURE_SIZE_LARGE,
    FIGURE_SIZE_MEDIUM,
    FIGURES_DIR,
    HEATMAP_CMAP,
    LEAKAGE_CORRELATION_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_ZSCORE_THRESHOLD,
    RANDOM_STATE,
    STRATIFY,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Duplicate Detection
# =============================================================================


def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Identify and report duplicate rows in the dataset.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for duplicate detection. None = all columns.

    Returns:
        Tuple of (duplicate rows DataFrame, count of duplicates).
    """
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    n_dupes = df.duplicated(subset=subset).sum()

    logger.info(
        f"Duplicate analysis: {n_dupes} duplicate rows found "
        f"({n_dupes / len(df) * 100:.2f}% of dataset)"
    )

    if n_dupes > 0:
        logger.info(f"First 5 duplicate rows:\n{duplicates.head()}")

    return duplicates, n_dupes


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove duplicate rows, keeping the first occurrence.

    Args:
        df: Input DataFrame.
        subset: Columns to consider. None = all columns.

    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    logger.info(f"Removed {n_removed} duplicate rows ({n_before} → {len(df_clean)})")
    return df_clean


# =============================================================================
# Missing Value Analysis
# =============================================================================


def analyze_missing_values(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Analyze and visualize missing values in the dataset.

    Creates a missing value heatmap and returns summary statistics.

    Args:
        df: Input DataFrame.
        save_path: Path to save the heatmap figure. Defaults to figures dir.

    Returns:
        DataFrame with columns: column, missing_count, missing_pct.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
    }).sort_values("missing_count", ascending=False)

    logger.info(
        f"Missing value summary: "
        f"{(missing > 0).sum()} columns with missing values, "
        f"{missing.sum()} total missing cells"
    )

    # Plot missing value heatmap
    if missing.sum() > 0:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
        sns.heatmap(
            df.isnull().astype(int),
            cbar=True,
            cmap="YlOrRd",
            yticklabels=False,
            ax=ax,
        )
        ax.set_title("Missing Value Heatmap", fontsize=14, fontweight="bold")
        ax.set_xlabel("Features")
        ax.set_ylabel("Samples")
        plt.tight_layout()
        save_path = save_path or str(FIGURES_DIR / "missing_values_heatmap.png")
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Missing value heatmap saved to {save_path}")
    else:
        logger.info("No missing values found — skipping heatmap")

    return missing_df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values using the specified strategy.

    Args:
        df: Input DataFrame.
        strategy: Imputation strategy ('median', 'mean', 'mode', 'drop').
        columns: Specific columns to impute. None = all numeric columns.

    Returns:
        DataFrame with missing values handled.
    """
    df_clean = df.copy()

    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if strategy == "drop":
        n_before = len(df_clean)
        df_clean = df_clean.dropna(subset=columns).reset_index(drop=True)
        logger.info(f"Dropped {n_before - len(df_clean)} rows with missing values")
    elif strategy in ("mean", "median", "mode"):
        for col in columns:
            if df_clean[col].isnull().sum() > 0:
                if strategy == "mean":
                    fill_val = df_clean[col].mean()
                elif strategy == "median":
                    fill_val = df_clean[col].median()
                else:
                    fill_val = df_clean[col].mode()[0]
                df_clean[col].fillna(fill_val, inplace=True)
                logger.info(f"Imputed '{col}' with {strategy}: {fill_val:.4f}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'median', 'mean', 'mode', or 'drop'.")

    return df_clean


# =============================================================================
# Outlier Detection
# =============================================================================


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    save_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Detect outliers using IQR or Z-score method.

    Args:
        df: Input DataFrame.
        columns: Columns to check. None = all numeric columns.
        method: Detection method ('iqr' or 'zscore').
        save_path: Path to save outlier visualization.

    Returns:
        Dictionary mapping column names to outlier statistics.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_info: Dict[str, Dict] = {}

    for col in columns:
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            mask = (df[col] < lower) | (df[col] > upper)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores > OUTLIER_ZSCORE_THRESHOLD
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

        n_outliers = mask.sum()
        outlier_info[col] = {
            "n_outliers": int(n_outliers),
            "pct_outliers": float(n_outliers / len(df) * 100),
            "method": method,
        }

    # Log summary
    total_cols_with_outliers = sum(
        1 for v in outlier_info.values() if v["n_outliers"] > 0
    )
    logger.info(
        f"Outlier detection ({method}): "
        f"{total_cols_with_outliers}/{len(columns)} columns have outliers"
    )

    return outlier_info


# =============================================================================
# Data Leakage Detection
# =============================================================================


def check_data_leakage(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = LEAKAGE_CORRELATION_THRESHOLD,
) -> List[Tuple[str, str, float]]:
    """
    Detect potential data leakage through high correlations with the target.

    Checks for:
      1. Features with very high correlation to the target (potential leakage).
      2. Feature pairs with near-perfect correlation (redundancy).

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        threshold: Correlation threshold above which leakage is suspected.

    Returns:
        List of tuples (feature1, feature2, correlation) for suspicious pairs.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col not in numeric_df.columns:
        logger.warning(f"Target '{target_col}' is not numeric — skipping leakage check")
        return []

    corr_matrix = numeric_df.corr()
    suspicious_pairs: List[Tuple[str, str, float]] = []

    # Check target correlations
    target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    high_target_corr = target_corrs[target_corrs > threshold]

    if len(high_target_corr) > 0:
        logger.warning(
            f"⚠ Potential target leakage detected! "
            f"Features with correlation > {threshold} to '{target_col}':"
        )
        for feat, corr_val in high_target_corr.items():
            logger.warning(f"  - {feat}: {corr_val:.4f}")
            suspicious_pairs.append((str(feat), target_col, float(corr_val)))

    # Check inter-feature correlations
    features = [c for c in numeric_df.columns if c != target_col]
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            corr_val = abs(corr_matrix.loc[f1, f2])
            if corr_val > threshold:
                suspicious_pairs.append((f1, f2, float(corr_val)))

    n_redundant = len(suspicious_pairs) - len(high_target_corr)
    logger.info(
        f"Leakage check complete: {len(high_target_corr)} target-leakage suspects, "
        f"{n_redundant} highly redundant feature pairs"
    )

    return suspicious_pairs


# =============================================================================
# Train / Validation / Test Split
# =============================================================================


def split_data(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification.

    Uses two-stage splitting: first train+val vs test, then train vs val.

    Args:
        df: Input DataFrame.
        target_col: Target column name for stratification.
        train_size: Proportion for training (default 0.60).
        val_size: Proportion for validation (default 0.20).
        test_size: Proportion for testing (default 0.20).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split proportions must sum to 1.0"

    stratify_col = df[target_col] if STRATIFY else None

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_col,
    )

    # Second split: separate validation from training
    val_relative = val_size / (train_size + val_size)
    stratify_col_tv = train_val_df[target_col] if STRATIFY else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=RANDOM_STATE,
        stratify=stratify_col_tv,
    )

    logger.info(
        f"Data split: Train={len(train_df)} ({train_size:.0%}), "
        f"Val={len(val_df)} ({val_size:.0%}), "
        f"Test={len(test_df)} ({test_size:.0%})"
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =============================================================================
# Class Imbalance
# =============================================================================


def check_class_imbalance(
    df: pd.DataFrame,
    target_col: str,
    imbalance_ratio_threshold: float = 3.0,
) -> Dict[str, float]:
    """
    Analyze class distribution and report imbalance ratio.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        imbalance_ratio_threshold: Ratio above which the dataset is considered
            imbalanced (majority_count / minority_count).

    Returns:
        Dictionary with class counts and imbalance ratio.
    """
    counts = df[target_col].value_counts()
    imbalance_ratio = counts.max() / counts.min()

    result = {
        "counts": counts.to_dict(),
        "imbalance_ratio": float(imbalance_ratio),
        "is_imbalanced": imbalance_ratio > imbalance_ratio_threshold,
    }

    if result["is_imbalanced"]:
        logger.warning(
            f"⚠ Class imbalance detected! Ratio: {imbalance_ratio:.2f}x "
            f"(threshold: {imbalance_ratio_threshold}x). "
            f"Distribution: {counts.to_dict()}"
        )
    else:
        logger.info(
            f"Class distribution acceptable. Ratio: {imbalance_ratio:.2f}x. "
            f"Distribution: {counts.to_dict()}"
        )

    return result
