"""
FastAPI Route Handlers
========================

Implements the ML prediction endpoints:
  - POST /predict-grade  → Code quality grading (Pipeline 1)
  - POST /predict-doubt  → Student doubt triage (Pipeline 2)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from api.schemas import (
    CodeGradingRequest,
    CodeGradingResponse,
    DoubtTriageRequest,
    DoubtTriageResponse,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry (loaded at startup)
# =============================================================================

_models: Dict[str, Any] = {}


def load_models(models_dir: str) -> Dict[str, bool]:
    """
    Load all saved model artifacts from the models directory.

    Args:
        models_dir: Path to the models directory.

    Returns:
        Dictionary indicating which models loaded successfully.
    """
    models_path = Path(models_dir)
    status: Dict[str, bool] = {}

    # Pipeline 1: Code Grading
    code_model_path = models_path / "code_grading_model.pkl"
    if code_model_path.exists():
        try:
            _models["code_grading"] = joblib.load(str(code_model_path))
            status["code_grading_model"] = True
            logger.info(f"Loaded code grading model from {code_model_path}")
        except Exception as e:
            status["code_grading_model"] = False
            logger.error(f"Failed to load code grading model: {e}")
    else:
        status["code_grading_model"] = False
        logger.warning(f"Code grading model not found at {code_model_path}")

    # Pipeline 2: Doubt Triage
    doubt_model_path = models_path / "doubt_triage_model.pkl"
    vectorizer_path = models_path / "doubt_triage_vectorizer.pkl"
    le_path = models_path / "doubt_triage_label_encoder.pkl"
    threshold_path = models_path / "doubt_triage_threshold.json"

    for name, path in [
        ("doubt_triage", doubt_model_path),
        ("tfidf_vectorizer", vectorizer_path),
        ("label_encoder", le_path),
    ]:
        if path.exists():
            try:
                _models[name] = joblib.load(str(path))
                status[f"{name}"] = True
                logger.info(f"Loaded {name} from {path}")
            except Exception as e:
                status[f"{name}"] = False
                logger.error(f"Failed to load {name}: {e}")
        else:
            status[f"{name}"] = False
            logger.warning(f"{name} not found at {path}")

    # Load threshold (default to 0.70 if missing)
    if threshold_path.exists():
        try:
            with open(threshold_path, "r") as f:
                data = json.load(f)
            _models["threshold"] = data.get("best_threshold", 0.70)
            status["threshold"] = True
        except Exception as e:
            _models["threshold"] = 0.70
            status["threshold"] = False
            logger.error(f"Failed to load threshold: {e}")
    else:
        _models["threshold"] = 0.70
        status["threshold"] = True  # Use default 0.70 threshold fallback
        logger.info(f"Threshold file not found at {threshold_path}. Using default 0.70 fallback threshold.")

    return status


def get_model_status() -> Dict[str, bool]:
    """Get the current status of loaded models."""
    return {
        "code_grading_model": "code_grading" in _models,
        "doubt_triage_model": "doubt_triage" in _models,
        "tfidf_vectorizer": "tfidf_vectorizer" in _models,
        "label_encoder": "label_encoder" in _models,
    }


# =============================================================================
# Pipeline 1: Code Grading Prediction
# =============================================================================


def _ensure_models_loaded() -> None:
    """Ensure models are loaded from the models directory if registry is empty."""
    if not _models:
        default_dir = Path(__file__).resolve().parent.parent / "models"
        load_models(str(default_dir))


def predict_grade(request: CodeGradingRequest) -> CodeGradingResponse:
    """
    Predict code quality grade from software defect metrics.

    Args:
        request: CodeGradingRequest with software metrics.

    Returns:
        CodeGradingResponse with quality, prediction, confidence, defect probability, review flag, and feature values.

    Raises:
        RuntimeError: If the code grading model is not loaded.
    """
    _ensure_models_loaded()
    if "code_grading" not in _models:
        raise RuntimeError(
            "Code grading model not loaded. "
            "Run Pipeline 1 notebook to train and save the model."
        )

    model = _models["code_grading"]

    # Build feature array in the same order as training
    features = _build_code_features(request)
    X = np.array([list(features.values())])

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(np.max(probabilities))
    defect_prob = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

    label = "Defect" if prediction == 1 else "No Defect"
    quality = "Defective" if prediction == 1 else "Good"
    review_needed = bool(confidence < 0.85)

    logger.info(
        f"Code grading prediction: quality={quality}, label={label} "
        f"(confidence={confidence:.4f}, defect_prob={defect_prob:.4f}, review_needed={review_needed})"
    )

    return CodeGradingResponse(
        quality=quality,
        prediction=label,
        confidence=confidence,
        defect_probability=defect_prob,
        review_needed=review_needed,
        feature_values=features,
    )


def _build_code_features(request: CodeGradingRequest) -> Dict[str, float]:
    """
    Build the feature dictionary from a CodeGradingRequest.

    Includes derived features (Complexity_per_LOC, Branch_Density, Fan_Ratio, etc.)
    to match the training feature set.
    """
    from src.utils import safe_divide

    loc = float(request.loc)
    cyclo = float(request.cyclo if request.cyclo is not None else (request.v_g or 1.0))
    length = float(request.length if request.length is not None else (request.n or 1.0))
    volume = float(request.volume if request.volume is not None else (request.v or 1.0))
    difficulty = float(request.difficulty if request.difficulty is not None else (request.d or 1.0))
    int_fan_in = float(request.int_fan_in)
    int_fan_out = float(request.int_fan_out)
    num_operators = float(request.num_operators if request.num_operators is not None else (request.total_Op or 1.0))
    num_operands = float(request.num_operands if request.num_operands is not None else (request.total_Opnd or 1.0))
    branch_count = float(request.branch_count if request.branch_count is not None else (request.branchCount or 1.0))

    features: Dict[str, float] = {
        "LOC": loc,
        "CYCLO": cyclo,
        "LENGTH": length,
        "VOLUME": volume,
        "DIFFICULTY": difficulty,
        "INT_FAN_IN": int_fan_in,
        "INT_FAN_OUT": int_fan_out,
        "NUM_OPERATORS": num_operators,
        "NUM_OPERANDS": num_operands,
        "BRANCH_COUNT": branch_count,
        "Complexity_per_LOC": safe_divide(cyclo, loc),
        "Branch_Density": safe_divide(branch_count, loc),
        "Fan_Ratio": safe_divide(int_fan_in, int_fan_out),
        "Complexity_x_LOC": cyclo * loc,
        "Halstead_per_LOC": safe_divide(volume, loc),
    }

    return features


# =============================================================================
# Pipeline 2: Doubt Triage Prediction
# =============================================================================


def predict_doubt(request: DoubtTriageRequest) -> DoubtTriageResponse:
    """
    Predict question topic, urgency, confidence, and routing.

    Args:
        request: DoubtTriageRequest with question text.

    Returns:
        DoubtTriageResponse with prediction, confidence, urgency, and route.

    Raises:
        RuntimeError: If required models are not loaded.
    """
    _ensure_models_loaded()
    required = ["doubt_triage", "tfidf_vectorizer", "label_encoder"]
    missing = [m for m in required if m not in _models]
    if missing:
        raise RuntimeError(
            f"Models not loaded: {missing}. "
            "Run Pipeline 2 notebook to train and save models."
        )

    model = _models["doubt_triage"]
    vectorizer = _models["tfidf_vectorizer"]
    label_encoder = _models["label_encoder"]
    threshold = _models.get("threshold", 0.7)

    # Clean text
    from src.text_processing import clean_text
    cleaned = clean_text(request.question)

    # Vectorize
    X = vectorizer.transform([cleaned])

    # Predict
    prediction_encoded = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(np.max(probabilities))

    # Decode prediction
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Derive urgency
    from src.data_loader import derive_urgency
    import pandas as pd
    urgency_df = pd.DataFrame({"question": [request.question]})
    urgency = derive_urgency(urgency_df).iloc[0]

    # Route
    from src.threshold_optimizer import route_prediction
    route = route_prediction(confidence, threshold)

    # All class probabilities
    all_probs = {
        str(label_encoder.inverse_transform([i])[0]): float(p)
        for i, p in enumerate(probabilities)
    }

    logger.info(
        f"Doubt triage: topic={prediction_label}, urgency={urgency}, "
        f"confidence={confidence:.4f}, route={route}"
    )

    return DoubtTriageResponse(
        prediction=prediction_label,
        confidence=confidence,
        urgency=urgency,
        route=route,
        all_probabilities=all_probs,
    )
