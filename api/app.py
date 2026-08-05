"""
FastAPI Application
=====================

Main application entry point for the LMS ML Pipeline API.

Endpoints:
  - GET  /health         → Health check with model status
  - POST /predict-grade  → Code quality grading (Pipeline 1)
  - POST /predict-doubt  → Student doubt triage (Pipeline 2)

Run with:
  uvicorn api.app:app --reload --port 8000
"""

import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress version mismatch warnings & Gradio telemetry
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import get_model_status, load_models, predict_doubt, predict_grade
from api.schemas import (
    CodeGradingRequest,
    CodeGradingResponse,
    DoubtTriageRequest,
    DoubtTriageResponse,
    HealthResponse,
)

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="LMS ML Pipeline API",
    description=(
        "Production-quality ML API for an LMS (Learning Management System). "
        "Provides two endpoints: Code Quality Grading and Student Doubt Triage."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event() -> None:
    """Load trained models in background thread so Uvicorn binds port instantly."""
    import asyncio
    models_dir = str(PROJECT_ROOT / "models")
    logger.info(f"Initiating background model loading from {models_dir}...")
    asyncio.create_task(asyncio.to_thread(load_models, models_dir))


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
async def serve_dashboard() -> HTMLResponse:
    """Serve interactive web dashboard on root path."""
    html_path = PROJECT_ROOT / "api" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>LMS ML Pipeline API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


@app.get("/hybridaction/zybTrackerStatisticsAction", include_in_schema=False)
async def suppress_browser_tracker():
    """Silence browser extension tracker requests."""
    return {}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the service status and loaded model information.
    """
    return HealthResponse(
        status="ok",
        models_loaded=get_model_status(),
    )


@app.post(
    "/predict-grade",
    response_model=CodeGradingResponse,
    tags=["Pipeline 1 — Code Grading"],
    summary="Predict code quality grade",
    description="Accepts software metrics and returns a defect prediction with confidence.",
)
async def api_predict_grade(request: CodeGradingRequest) -> CodeGradingResponse:
    """
    Predict whether a code module is likely defective.

    Accepts McCabe and Halstead software metrics, engineers derived features,
    and returns:
      - Prediction: 'Defect' or 'No Defect'
      - Confidence: Model prediction confidence
      - Defect Probability: Raw probability of defect class
    """
    try:
        return predict_grade(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict-doubt",
    response_model=DoubtTriageResponse,
    tags=["Pipeline 2 — Doubt Triage"],
    summary="Predict student doubt category and route",
    description="Accepts a student question and returns topic, urgency, confidence, and routing.",
)
async def api_predict_doubt(request: DoubtTriageRequest) -> DoubtTriageResponse:
    """
    Classify a student's programming question and determine routing.

    Processes the question text through the NLP pipeline and returns:
      - Prediction: Question topic category
      - Confidence: Model prediction confidence
      - Urgency: Derived urgency level (HIGH/MEDIUM/LOW)
      - Route: 'Auto Approval' or 'Teacher Review'
    """
    try:
        return predict_doubt(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# =============================================================================
# Mount Gradio Interface
# =============================================================================
try:
    import gradio as gr
    from api.gradio_app import demo as gradio_demo
    app = gr.mount_gradio_app(app, gradio_demo, path="/gradio")
    logger.info("Gradio interface mounted successfully at /gradio")
except Exception as e:
    logger.error(f"Could not mount Gradio interface: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
