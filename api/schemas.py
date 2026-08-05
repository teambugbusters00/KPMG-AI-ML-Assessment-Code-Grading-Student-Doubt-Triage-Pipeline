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
    """Request schema for the /predict-grade endpoint using SoftwareDefectDataset metrics."""

    loc: float = Field(25.0, alias="LOC", description="Lines of Code")
    cyclo: float = Field(4.0, alias="CYCLO", description="Cyclomatic Complexity")
    length: float = Field(75.0, alias="LENGTH", description="Halstead Length")
    volume: float = Field(350.0, alias="VOLUME", description="Halstead Volume")
    difficulty: float = Field(12.0, alias="DIFFICULTY", description="Halstead Difficulty")
    int_fan_in: float = Field(2.0, alias="INT_FAN_IN", description="Internal Fan In")
    int_fan_out: float = Field(1.0, alias="INT_FAN_OUT", description="Internal Fan Out")
    num_operators: float = Field(45.0, alias="NUM_OPERATORS", description="Number of Operators")
    num_operands: float = Field(30.0, alias="NUM_OPERANDS", description="Number of Operands")
    branch_count: float = Field(7.0, alias="BRANCH_COUNT", description="Branch Count")

    # Compatibility aliases
    v_g: Optional[float] = Field(None, alias="v(g)")
    ev_g: Optional[float] = Field(None, alias="ev(g)")
    iv_g: Optional[float] = Field(None, alias="iv(g)")
    n: Optional[float] = Field(None)
    v: Optional[float] = Field(None)
    l: Optional[float] = Field(None)
    d: Optional[float] = Field(None)
    i: Optional[float] = Field(None)
    e: Optional[float] = Field(None)
    b: Optional[float] = Field(None)
    t: Optional[float] = Field(None)
    lOCode: Optional[float] = Field(None)
    lOComment: Optional[float] = Field(None)
    lOBlank: Optional[float] = Field(None)
    uniq_Op: Optional[float] = Field(None)
    uniq_Opnd: Optional[float] = Field(None)
    total_Op: Optional[float] = Field(None)
    total_Opnd: Optional[float] = Field(None)

    model_config = {"populate_by_name": True}


class CodeGradingResponse(BaseModel):
    """Response schema for the /predict-grade endpoint."""

    quality: str = Field(
        ..., description="Code quality assessment: 'Good' or 'Defective'"
    )
    prediction: str = Field(
        ..., description="Defect prediction label: 'Defect' or 'No Defect'"
    )
    confidence: float = Field(
        ..., description="Prediction confidence (0.0 to 1.0)"
    )
    defect_probability: float = Field(
        ..., description="Probability of defect class (0.0 to 1.0)"
    )
    review_needed: bool = Field(
        ..., description="True if confidence < 0.85 (needs manual review)"
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
