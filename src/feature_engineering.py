"""
Feature Engineering Module
============================

Engineers meaningful software metrics from the NASA KC1 dataset columns.
Derives composite features (Complexity Ratio, Maintainability Index, etc.)
and documents which features are unavailable.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import safe_divide, setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Feature Mapping: KC1 columns → Requested features
# =============================================================================

# Direct mappings from KC1 column names to requested feature names
KC1_DIRECT_FEATURES: Dict[str, str] = {
    "loc": "LOC",
    "v(g)": "Cyclomatic_Complexity",
    "ev(g)": "Essential_Complexity",
    "iv(g)": "Design_Complexity",
    "n": "Halstead_Total_Operators_Operands",
    "v": "Halstead_Volume",
    "l": "Halstead_Program_Length",
    "d": "Halstead_Difficulty",
    "i": "Halstead_Intelligence",
    "e": "Halstead_Effort",
    "b": "Halstead_Bug_Estimate",
    "t": "Halstead_Time_Estimate",
    "branchCount": "Branch_Count",
    "lOCode": "Lines_of_Code",
    "lOComment": "Lines_of_Comment",
    "lOBlank": "Lines_of_Blank",
    "uniq_Op": "Unique_Operators",
    "uniq_Opnd": "Unique_Operands",
    "total_Op": "Total_Operators",
    "total_Opnd": "Total_Operands",
}

# Features that CANNOT be computed from KC1 (documented limitations)
UNAVAILABLE_FEATURES: List[str] = [
    "Fan_In",
    "Fan_Out",
    "Runtime_Efficiency",
    "Memory_Efficiency",
    "Function_Count",
]


def get_feature_availability_report() -> str:
    """
    Generate a report documenting which requested features are available
    and which are unavailable in the KC1 dataset.

    Returns:
        Formatted string report of feature availability.
    """
    report_lines = [
        "=" * 60,
        "FEATURE AVAILABILITY REPORT — NASA KC1 Dataset",
        "=" * 60,
        "",
        "AVAILABLE (Direct from KC1):",
    ]
    for kc1_col, feat_name in KC1_DIRECT_FEATURES.items():
        report_lines.append(f"  ✓ {feat_name} (from '{kc1_col}' column)")

    report_lines.extend([
        "",
        "AVAILABLE (Derived / Engineered):",
        "  ✓ Code_Size (LOC + lOComment + lOBlank)",
        "  ✓ Complexity_Ratio (Cyclomatic Complexity / LOC)",
        "  ✓ Maintainability_Index (SEI formula)",
        "  ✓ Comment_Density (lOComment / (LOC + lOComment))",
        "",
        "UNAVAILABLE (Documented Limitations):",
    ])
    for feat in UNAVAILABLE_FEATURES:
        report_lines.append(f"  ✗ {feat} — Requires data not present in KC1")

    report_lines.extend([
        "",
        "NOTE: Fan In/Fan Out require call-graph analysis data.",
        "Runtime/Memory Efficiency require profiling data.",
        "Function Count requires AST-level parsing not available in KC1.",
        "=" * 60,
    ])

    report = "\n".join(report_lines)
    logger.info("Feature availability report generated")
    return report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all available software metrics from the NASA KC1 dataset.

    Adds derived features to the DataFrame:
      - Code_Size: Total lines (code + comment + blank)
      - Complexity_Ratio: Cyclomatic complexity per LOC
      - Maintainability_Index: SEI maintainability formula
      - Comment_Density: Proportion of comment lines

    Args:
        df: DataFrame with raw KC1 columns.

    Returns:
        DataFrame with additional engineered feature columns.

    Notes:
        Features that cannot be computed from KC1 are NOT fabricated.
        See UNAVAILABLE_FEATURES and the final report for details.
    """
    df_eng = df.copy()
    engineered_count = 0

    # -----------------------------------------------------------------
    # Code Size = LOC + Lines of Comment + Lines of Blank
    # -----------------------------------------------------------------
    loc_col = _find_column(df_eng, ["loc", "LOC", "lOCode"])
    comment_col = _find_column(df_eng, ["lOComment", "LOComment", "loc_comment"])
    blank_col = _find_column(df_eng, ["lOBlank", "LOBlank", "loc_blank"])

    if loc_col and comment_col and blank_col:
        df_eng["Code_Size"] = (
            df_eng[loc_col] + df_eng[comment_col] + df_eng[blank_col]
        )
        engineered_count += 1
        logger.info(f"Engineered Code_Size from {loc_col}, {comment_col}, {blank_col}")
    else:
        logger.warning("Cannot compute Code_Size: missing LOC/comment/blank columns")

    # -----------------------------------------------------------------
    # Complexity Ratio = Cyclomatic Complexity / LOC
    # -----------------------------------------------------------------
    vg_col = _find_column(df_eng, ["v(g)", "vg", "cyclomatic_complexity"])
    if vg_col and loc_col:
        df_eng["Complexity_Ratio"] = df_eng.apply(
            lambda row: safe_divide(row[vg_col], row[loc_col]),
            axis=1,
        )
        engineered_count += 1
        logger.info(f"Engineered Complexity_Ratio from {vg_col} / {loc_col}")
    else:
        logger.warning("Cannot compute Complexity_Ratio: missing v(g) or loc")

    # -----------------------------------------------------------------
    # Maintainability Index = 171 - 5.2*ln(V) - 0.23*v(g) - 16.2*ln(LOC)
    # (SEI / Microsoft formula, clamped to [0, 100])
    # -----------------------------------------------------------------
    vol_col = _find_column(df_eng, ["v", "volume", "halstead_volume"])
    if vol_col and vg_col and loc_col:
        def _calc_mi(row: pd.Series) -> float:
            v_val = max(row[vol_col], 1)  # Avoid log(0)
            loc_val = max(row[loc_col], 1)
            vg_val = row[vg_col]
            mi = 171 - 5.2 * np.log(v_val) - 0.23 * vg_val - 16.2 * np.log(loc_val)
            return max(0, min(100, mi))  # Clamp to [0, 100]

        df_eng["Maintainability_Index"] = df_eng.apply(_calc_mi, axis=1)
        engineered_count += 1
        logger.info("Engineered Maintainability_Index (SEI formula)")
    else:
        logger.warning(
            "Cannot compute Maintainability_Index: "
            "missing volume, v(g), or loc columns"
        )

    # -----------------------------------------------------------------
    # Comment Density = lOComment / (LOC + lOComment)
    # -----------------------------------------------------------------
    if comment_col and loc_col:
        df_eng["Comment_Density"] = df_eng.apply(
            lambda row: safe_divide(
                row[comment_col],
                row[loc_col] + row[comment_col],
            ),
            axis=1,
        )
        engineered_count += 1
        logger.info(f"Engineered Comment_Density from {comment_col}")
    else:
        logger.warning("Cannot compute Comment_Density: missing comment or LOC columns")

    # -----------------------------------------------------------------
    # Halstead Derived: Bug Density = Halstead Bugs / LOC
    # -----------------------------------------------------------------
    bug_col = _find_column(df_eng, ["b", "halstead_bugs", "bug_estimate"])
    if bug_col and loc_col:
        df_eng["Bug_Density"] = df_eng.apply(
            lambda row: safe_divide(row[bug_col], row[loc_col]),
            axis=1,
        )
        engineered_count += 1
        logger.info(f"Engineered Bug_Density from {bug_col} / {loc_col}")

    # -----------------------------------------------------------------
    # Effort Density = Halstead Effort / LOC
    # -----------------------------------------------------------------
    effort_col = _find_column(df_eng, ["e", "halstead_effort", "effort"])
    if effort_col and loc_col:
        df_eng["Effort_Density"] = df_eng.apply(
            lambda row: safe_divide(row[effort_col], row[loc_col]),
            axis=1,
        )
        engineered_count += 1
        logger.info(f"Engineered Effort_Density from {effort_col} / {loc_col}")

    logger.info(
        f"Feature engineering complete: {engineered_count} new features derived. "
        f"Total columns: {df_eng.shape[1]}"
    )

    return df_eng


def get_feature_names(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Get the list of feature column names (excluding the target).

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.

    Returns:
        List of feature column names.
    """
    return [col for col in df.select_dtypes(include=[np.number]).columns
            if col != target_col]


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Find the first matching column name from a list of candidates.

    Args:
        df: DataFrame to search.
        candidates: List of possible column names.

    Returns:
        The first matching column name, or empty string if none found.
    """
    for name in candidates:
        if name in df.columns:
            return name
    return ""
