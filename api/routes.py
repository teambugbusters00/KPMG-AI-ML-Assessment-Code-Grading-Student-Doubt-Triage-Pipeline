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

    # Load threshold
    if threshold_path.exists():
        try:
            with open(threshold_path, "r") as f:
                data = json.load(f)
            _models["threshold"] = data.get("best_threshold", 0.7)
            status["threshold"] = True
        except Exception as e:
            _models["threshold"] = 0.7
            status["threshold"] = False
            logger.error(f"Failed to load threshold: {e}")
    else:
        _models["threshold"] = 0.7
        status["threshold"] = False

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


def predict_grade(request: CodeGradingRequest) -> CodeGradingResponse:
    """
    Predict code quality grade from software metrics.

    Args:
        request: CodeGradingRequest with software metrics.

    Returns:
        CodeGradingResponse with prediction, confidence, and feature values.

    Raises:
        RuntimeError: If the code grading model is not loaded.
    """
    if "code_grading" not in _models:
        raise RuntimeError(
            "Code grading model not loaded. "
            "Run Pipeline 1 notebook to train and save the model."
        )

    model = _models["code_grading"]

    # Build feature array in the same order as training
    features = _build_code_features(request)
    feature_names = list(features.keys())
    X = np.array([list(features.values())])

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(np.max(probabilities))
    defect_prob = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

    label = "Defect" if prediction == 1 else "No Defect"

    logger.info(
        f"Code grading prediction: {label} "
        f"(confidence={confidence:.4f}, defect_prob={defect_prob:.4f})"
    )

    return CodeGradingResponse(
        prediction=label,
        confidence=confidence,
        defect_probability=defect_prob,
        feature_values=features,
    )


def _build_code_features(request: CodeGradingRequest) -> Dict[str, float]:
    """
    Build the feature dictionary from a CodeGradingRequest.

    Includes derived features (Code_Size, Complexity_Ratio, etc.)
    to match the training feature set.

    Args:
        request: The incoming code grading request.

    Returns:
        Dictionary of feature name → value.
    """
    from src.utils import safe_divide

    # Raw features
    features: Dict[str, float] = {
        "loc": request.loc,
        "v(g)": request.v_g,
        "ev(g)": request.ev_g,
        "iv(g)": request.iv_g,
        "n": request.n,
        "v": request.v,
        "l": request.l,
        "d": request.d,
        "i": request.i,
        "e": request.e,
        "b": request.b,
        "t": request.t,
        "lOCode": request.lOCode,
        "lOComment": request.lOComment,
        "lOBlank": request.lOBlank,
        "uniq_Op": request.uniq_Op,
        "uniq_Opnd": request.uniq_Opnd,
        "total_Op": request.total_Op,
        "total_Opnd": request.total_Opnd,
        "branchCount": request.branchCount,
    }

    # Derived features
    features["Code_Size"] = (
        request.loc + request.lOComment + request.lOBlank
    )
    features["Complexity_Ratio"] = safe_divide(request.v_g, request.loc)

    # Maintainability Index
    v_val = max(request.v, 1)
    loc_val = max(request.loc, 1)
    mi = 171 - 5.2 * np.log(v_val) - 0.23 * request.v_g - 16.2 * np.log(loc_val)
    features["Maintainability_Index"] = max(0, min(100, mi))

    features["Comment_Density"] = safe_divide(
        request.lOComment,
        request.loc + request.lOComment,
    )
    features["Bug_Density"] = safe_divide(request.b, request.loc)
    features["Effort_Density"] = safe_divide(request.e, request.loc)

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
        label_encoder.inverse_transform([i])[0]: float(p)
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
