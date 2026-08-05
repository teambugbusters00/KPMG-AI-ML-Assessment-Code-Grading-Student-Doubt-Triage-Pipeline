"""
Central Configuration Module
==============================

All constants, paths, hyperparameter grids, and thresholds are defined here.
No hardcoded values should exist outside this module.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_STATE: int = 42
N_JOBS: int = -1

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Dataset Configuration
# =============================================================================
# NASA KC1 (OpenML ID: 1067)
NASA_DATASET_ID: int = 1067
NASA_DATASET_NAME: str = "kc1"
NASA_TARGET_COLUMN: str = "defects"

# CS1QA
CS1QA_REPO_URL: str = "https://github.com/cyoon47/CS1QA.git"
CS1QA_DATA_SUBDIR: str = "data"
CS1QA_TARGET_COLUMN: str = "question_type"

# =============================================================================
# Data Splitting
# =============================================================================
TRAIN_SIZE: float = 0.60
VAL_SIZE: float = 0.20
TEST_SIZE: float = 0.20
STRATIFY: bool = True

# =============================================================================
# Cross-Validation
# =============================================================================
CV_FOLDS: int = 5
CV_SCORING: str = "roc_auc"
CV_SCORING_MULTICLASS: str = "f1_macro"

# =============================================================================
# Pipeline 1 — Code Grading Configuration
# =============================================================================
# Outlier detection
OUTLIER_IQR_MULTIPLIER: float = 1.5
OUTLIER_ZSCORE_THRESHOLD: float = 3.0

# Leakage detection correlation threshold
LEAKAGE_CORRELATION_THRESHOLD: float = 0.95

# Baseline model: Random Forest
RF_BASELINE_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
}

# Final model: LightGBM
LGBM_DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
}

# Optuna hyperparameter search space
OPTUNA_N_TRIALS: int = 15
OPTUNA_TIMEOUT: int = 60  # seconds
LGBM_PARAM_SPACE: Dict[str, Any] = {
    "n_estimators": {"low": 100, "high": 1000, "step": 50},
    "max_depth": {"low": 3, "high": 12},
    "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
    "num_leaves": {"low": 15, "high": 127},
    "min_child_samples": {"low": 5, "high": 100},
    "subsample": {"low": 0.5, "high": 1.0},
    "colsample_bytree": {"low": 0.5, "high": 1.0},
    "reg_alpha": {"low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-8, "high": 10.0, "log": True},
}

# =============================================================================
# Pipeline 2 — Doubt Triage Configuration
# =============================================================================
# TF-IDF
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: tuple = (1, 2)

# Logistic Regression baseline
LR_BASELINE_PARAMS: Dict[str, Any] = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "multi_class": "ovr",
    "solver": "lbfgs",
}

# Linear SVC final model
SVC_PARAMS: Dict[str, Any] = {
    "max_iter": 5000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}

# Confidence thresholds to evaluate
CONFIDENCE_THRESHOLDS: List[float] = [0.60, 0.70, 0.80, 0.85, 0.90]

# Urgency derivation keywords
URGENCY_HIGH_KEYWORDS: List[str] = [
    "error", "bug", "crash", "wrong", "fail", "broken", "urgent",
    "deadline", "help", "stuck", "cannot", "impossible", "emergency",
]
URGENCY_MEDIUM_KEYWORDS: List[str] = [
    "confused", "unclear", "doubt", "question", "understand",
    "explain", "why", "how", "what",
]

# =============================================================================
# Visualization
# =============================================================================
FIGURE_DPI: int = 150
FIGURE_SIZE_LARGE: tuple = (12, 8)
FIGURE_SIZE_MEDIUM: tuple = (10, 6)
FIGURE_SIZE_SMALL: tuple = (8, 5)
COLOR_PALETTE: str = "viridis"
HEATMAP_CMAP: str = "RdYlBu_r"

# =============================================================================
# Model Artifacts
# =============================================================================
CODE_GRADING_MODEL_PATH: Path = MODELS_DIR / "code_grading_model.pkl"
CODE_GRADING_SCALER_PATH: Path = MODELS_DIR / "code_grading_scaler.pkl"
DOUBT_TRIAGE_MODEL_PATH: Path = MODELS_DIR / "doubt_triage_model.pkl"
DOUBT_TRIAGE_VECTORIZER_PATH: Path = MODELS_DIR / "doubt_triage_vectorizer.pkl"

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
