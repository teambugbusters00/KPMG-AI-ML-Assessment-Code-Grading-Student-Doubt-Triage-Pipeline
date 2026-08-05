"""
Pydantic Schemas for FastAPI Endpoints
=========================================

Defines request/response models for both ML pipeline endpoints.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Pipeline 1: Code Grading
# =============================================================================


class CodeGradingRequest(BaseModel):
    """Request schema for the /predict-grade endpoint."""

    loc: float = Field(..., description="Lines of code (McCabe)")
    v_g: float = Field(..., alias="v(g)", description="Cyclomatic complexity")
    ev_g: float = Field(..., alias="ev(g)", description="Essential complexity")
    iv_g: float = Field(..., alias="iv(g)", description="Design complexity")
    n: float = Field(..., description="Halstead total operators + operands")
    v: float = Field(..., description="Halstead volume")
    l: float = Field(..., description="Halstead program length")
    d: float = Field(..., description="Halstead difficulty")
    i: float = Field(..., description="Halstead intelligence")
    e: float = Field(..., description="Halstead effort")
    b: float = Field(..., description="Halstead bug estimate")
    t: float = Field(..., description="Halstead time estimate")
    lOCode: float = Field(0, description="Lines of code (Halstead)")
    lOComment: float = Field(0, description="Lines of comment")
    lOBlank: float = Field(0, description="Lines of blank")
    uniq_Op: float = Field(0, description="Unique operators")
    uniq_Opnd: float = Field(0, description="Unique operands")
    total_Op: float = Field(0, description="Total operators")
    total_Opnd: float = Field(0, description="Total operands")
    branchCount: float = Field(0, description="Branch count")

    model_config = {"populate_by_name": True}


class CodeGradingResponse(BaseModel):
    """Response schema for the /predict-grade endpoint."""

    prediction: str = Field(
        ..., description="Defect prediction: 'Defect' or 'No Defect'"
    )
    confidence: float = Field(
        ..., description="Prediction confidence (0.0 to 1.0)"
    )
    defect_probability: float = Field(
        ..., description="Probability of defect (0.0 to 1.0)"
    )
    feature_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Engineered feature values used for prediction",
    )


# =============================================================================
# Pipeline 2: Doubt Triage
# =============================================================================


class DoubtTriageRequest(BaseModel):
    """Request schema for the /predict-doubt endpoint."""

    question: str = Field(
        ...,
        description="Student question text",
        min_length=1,
        max_length=5000,
    )


class DoubtTriageResponse(BaseModel):
    """Response schema for the /predict-doubt endpoint."""

    prediction: str = Field(
        ..., description="Predicted question type/topic"
    )
    confidence: float = Field(
        ..., description="Prediction confidence (0.0 to 1.0)"
    )
    urgency: str = Field(
        ..., description="Derived urgency level: HIGH, MEDIUM, or LOW"
    )
    route: str = Field(
        ..., description="Routing decision: 'Auto Approval' or 'Teacher Review'"
    )
    all_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probabilities for each question type",
    )


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str = Field("ok", description="Service health status")
    models_loaded: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of loaded models",
    )
