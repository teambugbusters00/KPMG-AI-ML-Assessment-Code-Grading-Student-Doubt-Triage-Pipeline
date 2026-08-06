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

import threading

_models: Dict[str, Any] = {}
_models_lock = threading.Lock()
_models_loaded_flag = False


def load_models(models_dir: str) -> Dict[str, bool]:
    """
    Load all saved model artifacts from the models directory in a thread-safe manner.

    Args:
        models_dir: Path to the models directory.

    Returns:
        Dictionary indicating which models loaded successfully.
    """
    global _models_loaded_flag
    with _models_lock:
        if _models_loaded_flag and len(_models) >= 4:
            return get_model_status()

        models_path = Path(models_dir)
        status: Dict[str, bool] = {}

        # Pipeline 1: Code Grading
        code_model_path = models_path / "code_grading_model.pkl"
        if code_model_path.exists():
            try:
                _models["code_grading"] = joblib.load(code_model_path)
                status["code_grading_model"] = True
                logger.info(f"Loaded code grading model from {code_model_path}")
            except Exception as e:
                logger.error(f"Error loading code grading model: {e}")
                status["code_grading_model"] = False

        # Pipeline 2: Doubt Triage
        doubt_model_path = models_path / "doubt_triage_model.pkl"
        vectorizer_path = models_path / "doubt_triage_vectorizer.pkl"
        encoder_path = models_path / "doubt_triage_label_encoder.pkl"
        metrics_path = models_path / "doubt_triage_metrics.json"

        if doubt_model_path.exists():
            try:
                _models["doubt_triage"] = joblib.load(doubt_model_path)
                status["doubt_triage"] = True
                logger.info(f"Loaded doubt_triage from {doubt_model_path}")
            except Exception as e:
                logger.error(f"Error loading doubt triage model: {e}")
                status["doubt_triage"] = False

        if vectorizer_path.exists():
            try:
                _models["tfidf_vectorizer"] = joblib.load(vectorizer_path)
                status["tfidf_vectorizer"] = True
                logger.info(f"Loaded tfidf_vectorizer from {vectorizer_path}")
            except Exception as e:
                logger.error(f"Error loading TF-IDF vectorizer: {e}")
                status["tfidf_vectorizer"] = False

        if encoder_path.exists():
            try:
                _models["label_encoder"] = joblib.load(encoder_path)
                status["label_encoder"] = True
                logger.info(f"Loaded label_encoder from {encoder_path}")
            except Exception as e:
                logger.error(f"Error loading label encoder: {e}")
                status["label_encoder"] = False

        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                _models["threshold"] = metrics.get("threshold", 0.7)
                logger.info(f"Loaded decision threshold {metrics.get('threshold')} from {metrics_path}")
            except Exception as e:
                logger.error(f"Error loading metrics JSON: {e}")
                _models["threshold"] = 0.7

        _models_loaded_flag = True
        import gc
        gc.collect()
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
# Helper Functions for Inference Optimization
# =============================================================================


def _derive_urgency_text(text: str) -> str:
    """Derive urgency directly from string without pandas overhead."""
    from src.config import URGENCY_HIGH_KEYWORDS, URGENCY_MEDIUM_KEYWORDS
    text_lower = text.lower()
    if any(kw in text_lower for kw in URGENCY_HIGH_KEYWORDS):
        return "HIGH"
    if any(kw in text_lower for kw in URGENCY_MEDIUM_KEYWORDS):
        return "MEDIUM"
    return "LOW"


def _route_prediction(confidence: float, threshold: float) -> str:
    """Route prediction based on confidence threshold without matplotlib imports."""
    return "Auto Approval" if confidence >= threshold else "Teacher Review"


# =============================================================================
# Pipeline 1: Code Grading Prediction
# =============================================================================


def _ensure_models_loaded() -> None:
    """Ensure models are loaded from the models directory if registry is empty."""
    if not _models_loaded_flag:
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
    feature_list = list(features.values())

    # Safely match feature dimensions if model expects specific number of columns
    scaler = getattr(model, "named_steps", {}).get("scaler", None)
    expected_n = getattr(scaler, "n_features_in_", len(feature_list))
    if len(feature_list) < expected_n:
        feature_list += [0.0] * (expected_n - len(feature_list))
    elif len(feature_list) > expected_n:
        feature_list = feature_list[:expected_n]

    input_feature_matrix = np.array([feature_list])

    # Predict defect probability and model confidence
    prediction = model.predict(input_feature_matrix)[0]
    probabilities = model.predict_proba(input_feature_matrix)[0]
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
        # --- Raw features ---
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
        # --- Ratio features ---
        "Complexity_per_LOC": safe_divide(cyclo, loc),
        "Branch_Density": safe_divide(branch_count, loc),
        "Fan_Ratio": safe_divide(int_fan_in, int_fan_out),
        "Complexity_x_LOC": cyclo * loc,
        "Halstead_per_LOC": safe_divide(volume, loc),
        # --- Interaction features ---
        "LOC_x_DIFFICULTY": loc * difficulty,
        "CYCLO_x_VOLUME": cyclo * volume,
        "OPERATORS_x_OPERANDS": num_operators * num_operands,
        "LOC_x_BRANCH": loc * branch_count,
        "VOLUME_x_DIFFICULTY": volume * difficulty,
        "FAN_IN_x_FAN_OUT": int_fan_in * int_fan_out,
        "LENGTH_x_DIFFICULTY": length * difficulty,
        "CYCLO_x_BRANCH": cyclo * branch_count,
        # --- Difference features ---
        "OPERATOR_OPERAND_DIFF": num_operators - num_operands,
        "FAN_IN_OUT_DIFF": int_fan_in - int_fan_out,
        "LOC_LENGTH_DIFF": loc - length,
        # --- Polynomial features ---
        "LOC_squared": loc ** 2,
        "CYCLO_squared": cyclo ** 2,
        "VOLUME_squared": volume ** 2,
        "BRANCH_squared": branch_count ** 2,
        "DIFFICULTY_squared": difficulty ** 2,
        # --- Log-transformed features ---
        "LOC_log": float(np.log1p(loc)),
        "VOLUME_log": float(np.log1p(volume)),
        "CYCLO_log": float(np.log1p(cyclo)),
        "LENGTH_log": float(np.log1p(length)),
    }

    # --- Statistical aggregation features ---
    import statistics
    complexity_vals = [loc, cyclo, branch_count]
    halstead_vals = [length, volume, difficulty, num_operators, num_operands]
    all_vals = complexity_vals + halstead_vals + [int_fan_in, int_fan_out]

    features["Complexity_Group_Mean"] = statistics.mean(complexity_vals)
    features["Complexity_Group_Std"] = statistics.pstdev(complexity_vals) if len(complexity_vals) > 1 else 0.0
    features["Halstead_Group_Mean"] = statistics.mean(halstead_vals)
    features["Halstead_Group_Std"] = statistics.pstdev(halstead_vals) if len(halstead_vals) > 1 else 0.0
    features["All_Features_Mean"] = statistics.mean(all_vals)
    features["All_Features_Std"] = statistics.pstdev(all_vals) if len(all_vals) > 1 else 0.0
    features["Operator_Ratio"] = safe_divide(num_operators, num_operators + num_operands)

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

    # Derive urgency directly without pandas overhead
    urgency = _derive_urgency_text(request.question)

    # Route directly without matplotlib imports
    route = _route_prediction(confidence, threshold)

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
