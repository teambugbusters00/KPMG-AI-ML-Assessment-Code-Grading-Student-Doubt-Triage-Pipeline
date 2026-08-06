"""
Model Training Module
========================

Handles training, cross-validation, and hyperparameter optimization for
both ML pipelines using sklearn Pipelines.

Pipeline 1: Random Forest (baseline) → LightGBM (final) with Optuna
Pipeline 2: Logistic Regression (baseline) → LinearSVC + Calibration (final)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import (
    CODE_GRADING_MODEL_PATH,
    CODE_GRADING_SCALER_PATH,
    CV_FOLDS,
    CV_SCORING,
    CV_SCORING_MULTICLASS,
    DOUBT_TRIAGE_MODEL_PATH,
    DOUBT_TRIAGE_VECTORIZER_PATH,
    LGBM_DEFAULT_PARAMS,
    LGBM_PARAM_SPACE,
    LR_BASELINE_PARAMS,
    N_JOBS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    RANDOM_STATE,
    RF_BASELINE_PARAMS,
    SVC_PARAMS,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Cross-Validation
# =============================================================================


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    scoring: Optional[str] = None,
    cv_folds: int = CV_FOLDS,
    is_multiclass: bool = False,
) -> Dict[str, float]:
    """
    Perform stratified k-fold cross-validation and report results.

    Args:
        model: Fitted or unfitted sklearn-compatible estimator.
        X: Feature matrix.
        y: Target array.
        scoring: Scoring metric name. Auto-detected if None.
        cv_folds: Number of cross-validation folds.
        is_multiclass: If True, uses multiclass scoring.

    Returns:
        Dictionary with 'mean', 'std', 'scores' keys.
    """
    if scoring is None:
        scoring = CV_SCORING_MULTICLASS if is_multiclass else CV_SCORING

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Use multiple scoring metrics
    scoring_dict = _get_scoring_dict(is_multiclass)

    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring_dict,
        n_jobs=N_JOBS,
        return_train_score=True,
    )

    # Log primary metric
    primary_key = f"test_{scoring}"
    if primary_key not in results:
        # Fallback to first available
        primary_key = list(k for k in results.keys() if k.startswith("test_"))[0]

    scores = results[primary_key]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    logger.info(
        f"Cross-validation ({cv_folds}-fold): "
        f"{primary_key.replace('test_', '')} = {mean_score:.4f} ± {std_score:.4f}"
    )

    return {
        "mean": mean_score,
        "std": std_score,
        "scores": scores.tolist(),
        "all_results": {k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in results.items()},
    }


def _get_scoring_dict(is_multiclass: bool) -> Dict[str, str]:
    """
    Get the scoring dictionary for cross-validation.

    Args:
        is_multiclass: Whether this is a multiclass problem.

    Returns:
        Dictionary mapping metric names to sklearn scoring strings.
    """
    if is_multiclass:
        return {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        }
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }


# =============================================================================
# Pipeline 1: Code Grading Models
# =============================================================================


def train_baseline_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a Random Forest baseline model with StandardScaler in a pipeline.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.

    Returns:
        Tuple of (fitted Pipeline, cross-validation results dict).
    """
    logger.info("Training baseline Random Forest model...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(**RF_BASELINE_PARAMS)),
    ])

    # Cross-validate first
    cv_results = cross_validate_model(pipeline, X_train, y_train)

    # Fit on full training data
    pipeline.fit(X_train, y_train)

    logger.info(
        f"Baseline RF trained. CV ROC-AUC: "
        f"{cv_results['mean']:.4f} ± {cv_results['std']:.4f}"
    )

    return pipeline, cv_results


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a LightGBM model with StandardScaler in a pipeline.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        params: LightGBM hyperparameters. Defaults to LGBM_DEFAULT_PARAMS.

    Returns:
        Tuple of (fitted Pipeline, cross-validation results dict).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is required. Install with: pip install lightgbm")

    model_params = {**LGBM_DEFAULT_PARAMS, **(params or {})}
    logger.info(f"Training LightGBM with params: {model_params}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", lgb.LGBMClassifier(**model_params)),
    ])

    # Cross-validate
    cv_results = cross_validate_model(pipeline, X_train, y_train)

    # Fit on full training data
    pipeline.fit(X_train, y_train)

    logger.info(
        f"LightGBM trained. CV ROC-AUC: "
        f"{cv_results['mean']:.4f} ± {cv_results['std']:.4f}"
    )

    return pipeline, cv_results


def tune_lightgbm_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
) -> Tuple[Dict[str, Any], float]:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        n_trials: Number of Optuna trials.
        timeout: Maximum optimization time in seconds.

    Returns:
        Tuple of (best_params dict, best_score float).
    """
    try:
        import lightgbm as lgb
        import optuna
    except ImportError:
        raise ImportError("Optuna and LightGBM required for hyperparameter tuning")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info(f"Starting Optuna hyperparameter search ({n_trials} trials, {timeout}s timeout)...")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for LightGBM tuning."""
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": N_JOBS,
            "n_estimators": trial.suggest_int(
                "n_estimators",
                LGBM_PARAM_SPACE["n_estimators"]["low"],
                LGBM_PARAM_SPACE["n_estimators"]["high"],
                step=LGBM_PARAM_SPACE["n_estimators"]["step"],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                LGBM_PARAM_SPACE["max_depth"]["low"],
                LGBM_PARAM_SPACE["max_depth"]["high"],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                LGBM_PARAM_SPACE["learning_rate"]["low"],
                LGBM_PARAM_SPACE["learning_rate"]["high"],
                log=LGBM_PARAM_SPACE["learning_rate"].get("log", False),
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                LGBM_PARAM_SPACE["num_leaves"]["low"],
                LGBM_PARAM_SPACE["num_leaves"]["high"],
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                LGBM_PARAM_SPACE["min_child_samples"]["low"],
                LGBM_PARAM_SPACE["min_child_samples"]["high"],
            ),
            "subsample": trial.suggest_float(
                "subsample",
                LGBM_PARAM_SPACE["subsample"]["low"],
                LGBM_PARAM_SPACE["subsample"]["high"],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                LGBM_PARAM_SPACE["colsample_bytree"]["low"],
                LGBM_PARAM_SPACE["colsample_bytree"]["high"],
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                LGBM_PARAM_SPACE["reg_alpha"]["low"],
                LGBM_PARAM_SPACE["reg_alpha"]["high"],
                log=LGBM_PARAM_SPACE["reg_alpha"].get("log", False),
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                LGBM_PARAM_SPACE["reg_lambda"]["low"],
                LGBM_PARAM_SPACE["reg_lambda"]["high"],
                log=LGBM_PARAM_SPACE["reg_lambda"].get("log", False),
            ),
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", lgb.LGBMClassifier(**params)),
        ])

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,  # Optuna handles parallelism
        )
        return float(np.mean(results["test_score"]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_score = study.best_value

    logger.info(
        f"Optuna optimization complete. "
        f"Best ROC-AUC: {best_score:.4f}. "
        f"Best params: {best_params}"
    )

    return best_params, best_score


def _apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling using NearestNeighbors for robust, dependency-free class balancing.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    maj_class = classes[np.argmax(counts)]
    min_class = classes[np.argmin(counts)]
    n_target = np.max(counts)
    
    X_min = X[y == min_class]
    n_min = len(X_min)
    n_needed = n_target - n_min
    if n_needed <= 0:
        return X, y
        
    from sklearn.neighbors import NearestNeighbors
    k_neighbors = min(5, n_min - 1)
    if k_neighbors >= 1:
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X_min)
        indices = nn.kneighbors(X_min, return_distance=False)
        
        rng = np.random.RandomState(random_state)
        synthetic_samples = []
        for _ in range(n_needed):
            idx = rng.randint(0, n_min)
            neighbor_idx = rng.choice(indices[idx, 1:])
            diff = X_min[neighbor_idx] - X_min[idx]
            gap = rng.uniform(0, 1)
            synthetic_samples.append(X_min[idx] + gap * diff)
            
        X_res = np.vstack([X, np.array(synthetic_samples)])
        y_res = np.concatenate([y, np.full(n_needed, min_class)])
        return X_res, y_res
    else:
        from sklearn.utils import resample
        X_min_res, y_min_res = resample(X_min, np.full(n_min, min_class), replace=True, n_samples=n_target, random_state=random_state)
        X_maj = X[y == maj_class]
        y_maj = y[y == maj_class]
        return np.vstack([X_maj, X_min_res]), np.concatenate([y_maj, y_min_res])


def train_lightgbm_with_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train LightGBM with SMOTE oversampling for class imbalance.

    Applies SMOTE to the training data before fitting the model to generate
    synthetic minority class samples and improve recall.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        params: LightGBM hyperparameters. Defaults to LGBM_DEFAULT_PARAMS.

    Returns:
        Tuple of (fitted Pipeline, CV results, SMOTE-resampled X, SMOTE-resampled y).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is required.")

    # Apply SMOTE
    X_resampled, y_resampled = _apply_smote(X_train, y_train, random_state=RANDOM_STATE)
    logger.info(
        f"SMOTE applied: {len(y_train)} -> {len(y_resampled)} samples. "
        f"Class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}"
    )

    model_params = {**LGBM_DEFAULT_PARAMS, **(params or {})}
    # Remove class_weight when using SMOTE (already balanced)
    model_params.pop("class_weight", None)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", lgb.LGBMClassifier(**model_params)),
    ])

    # Cross-validate on resampled data
    cv_results = cross_validate_model(pipeline, X_resampled, y_resampled)

    # Fit on full resampled data
    pipeline.fit(X_resampled, y_resampled)

    logger.info(
        f"LightGBM+SMOTE trained. CV ROC-AUC: "
        f"{cv_results['mean']:.4f} ± {cv_results['std']:.4f}"
    )

    return pipeline, cv_results, X_resampled, y_resampled


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lgbm_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a soft-voting ensemble combining HistGradientBoosting, Random Forest,
    and LightGBM for improved generalization.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        lgbm_params: LightGBM hyperparameters.

    Returns:
        Tuple of (fitted Pipeline, cross-validation results dict).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM required for ensemble.")

    # Apply SMOTE
    X_resampled, y_resampled = _apply_smote(X_train, y_train, random_state=RANDOM_STATE)

    lgbm_model_params = {**LGBM_DEFAULT_PARAMS, **(lgbm_params or {})}
    lgbm_model_params.pop("class_weight", None)
    lgbm_model_params.pop("objective", None)
    lgbm_model_params.pop("metric", None)

    ensemble = VotingClassifier(
        estimators=[
            ("hgb", HistGradientBoostingClassifier(
                max_depth=5,
                learning_rate=0.05,
                max_iter=500,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            )),
            ("lgbm", lgb.LGBMClassifier(**lgbm_model_params)),
        ],
        voting="soft",
        n_jobs=N_JOBS,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", ensemble),
    ])

    cv_results = cross_validate_model(pipeline, X_resampled, y_resampled)
    pipeline.fit(X_resampled, y_resampled)

    logger.info(
        f"Ensemble (HGB+RF+LGBM) trained with SMOTE. CV ROC-AUC: "
        f"{cv_results['mean']:.4f} ± {cv_results['std']:.4f}"
    )

    return pipeline, cv_results


# =============================================================================
# Pipeline 2: Doubt Triage Models
# =============================================================================


def train_baseline_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a Logistic Regression baseline model for multiclass classification.

    Args:
        X_train: Training feature matrix (TF-IDF vectors).
        y_train: Training target array (question types).

    Returns:
        Tuple of (fitted model, cross-validation results dict).
    """
    logger.info("Training baseline Logistic Regression model...")

    model = LogisticRegression(**LR_BASELINE_PARAMS)

    cv_results = cross_validate_model(
        model, X_train, y_train,
        is_multiclass=True,
    )

    model.fit(X_train, y_train)

    logger.info(
        f"Baseline LR trained. CV Macro-F1: "
        f"{cv_results['mean']:.4f} ± {cv_results['std']:.4f}"
    )

    return model, cv_results


def train_calibrated_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """
    Train a LinearSVC with probability calibration for the doubt triage pipeline.

    LinearSVC doesn't natively support predict_proba, so we wrap it with
    CalibratedClassifierCV to get calibrated probability estimates.

    Args:
        X_train: Training feature matrix (TF-IDF vectors).
        y_train: Training target array (question types).

    Returns:
        Tuple of (CalibratedClassifierCV model, cross-validation results dict).
    """
    logger.info("Training LinearSVC with CalibratedClassifierCV...")

    import pandas as pd
    min_class_samples = int(pd.Series(y_train).value_counts().min())
    n_cv = max(2, min(CV_FOLDS, min_class_samples))

    base_svc = LinearSVC(**SVC_PARAMS)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_svc,
        cv=n_cv,
        method="sigmoid",
    )

    try:
        cv_results = cross_validate_model(
            calibrated_model, X_train, y_train,
            cv_folds=n_cv,
            is_multiclass=True,
        )
    except Exception as e:
        logger.warning(f"Cross-validation failed for CalibratedClassifierCV due to small class counts ({e}). Using dummy CV results.")
        cv_results = {"mean": 0.0, "std": 0.0, "scores": np.array([0.0])}

    # Fit on full training data
    calibrated_model.fit(X_train, y_train)

    logger.info(
        f"Calibrated SVC trained. CV Macro-F1: "
        f"{cv_results['mean']:.4f} ± {cv_results['std']:.4f}"
    )

    return calibrated_model, cv_results


# =============================================================================
# Model Persistence
# =============================================================================


def save_model(model: Any, filepath: str, description: str = "") -> None:
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained model or pipeline to save.
        filepath: Output file path.
        description: Optional description for logging.
    """
    joblib.dump(model, filepath)
    logger.info(f"Model saved: {filepath} ({description})")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model file.

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from: {filepath}")
    return model
