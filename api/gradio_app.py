"""
Gradio Interface for KPMG LMS ML Pipeline
=========================================

Interactive Web UI for:
  - Pipeline 1: Code Quality Grading (Defect Prediction)
  - Pipeline 2: Student Doubt Triage (Topic, Urgency & Route Prediction)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import gradio as gr
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import load_models, predict_grade, predict_doubt, get_model_status
from api.schemas import CodeGradingRequest, DoubtTriageRequest

# Ensure models are loaded
load_models(str(PROJECT_ROOT / "models"))

# =============================================================================
# Helper Handler Functions
# =============================================================================

def gradio_predict_grade(
    loc: float, cyclo: float, length: float, volume: float, difficulty: float,
    int_fan_in: float, int_fan_out: float, num_operators: float, num_operands: float, branch_count: float
) -> Tuple[str, str, float, float, str, Dict[str, float]]:
    """Handler for Pipeline 1 Code Quality Grading Gradio tab."""
    req = CodeGradingRequest(
        loc=loc, cyclo=cyclo, length=length, volume=volume, difficulty=difficulty,
        int_fan_in=int_fan_in, int_fan_out=int_fan_out,
        num_operators=num_operators, num_operands=num_operands, branch_count=branch_count
    )
    res = predict_grade(req)
    review_str = "Yes (Confidence < 85%)" if res.review_needed else "No (High Quality Prediction)"
    return (
        f"Quality: {res.quality}",
        f"Defect Label: {res.prediction}",
        round(res.confidence * 100, 2),
        round(res.defect_probability * 100, 2),
        review_str,
        res.feature_values
    )


def gradio_predict_doubt(question: str) -> Tuple[str, str, str, float, Dict[str, float]]:
    """Handler for Pipeline 2 Student Doubt Triage Gradio tab."""
    if not question.strip():
        return "N/A", "N/A", "N/A", 0.0, {}
    
    req = DoubtTriageRequest(question=question)
    res = predict_doubt(req)
    
    # Format probabilities for Gradio Label component
    sorted_probs = dict(sorted(res.all_probabilities.items(), key=lambda x: x[1], reverse=True))
    
    return (
        res.prediction,
        res.urgency,
        res.route,
        round(res.confidence * 100, 2),
        sorted_probs
    )


def gradio_health_check() -> Dict[str, Any]:
    """Handler for System Status check tab."""
    return get_model_status()


# Preset values for Code Grading (SoftwareDefectDataset)
CLEAN_CODE_PRESET = [0.08, 0.43, 0.18, 0.16, 0.0, 0.0, 0.22, 0.14, 0.81, 0.64]
DEFECTIVE_CODE_PRESET = [0.89, 0.0, 0.96, 0.75, 0.67, 1.0, 0.0, 0.11, 0.02, 0.0]


# =============================================================================
# Build Gradio Blocks Application
# =============================================================================

custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.header-box {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.header-box h1 {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
"""

def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="LMS ML Pipeline — KPMG Assessment") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        
        with gr.Row():
            gr.HTML("""
                <div class="header-box">
                    <h1>LMS AI/ML Pipeline Dashboard</h1>
                    <p>Interactive Code Quality Grading & Student Doubt Triage System</p>
                </div>
            """)

        with gr.Tabs():
            # TAB 1: Code Quality Grading
            with gr.TabItem("Pipeline 1: Code Quality Grading"):
                gr.Markdown("### Code Software Metrics & Defect Prediction (SoftwareDefectDataset)")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Input Software Metrics")
                        with gr.Row():
                            btn_clean = gr.Button("Load Clean Code Sample", variant="secondary", size="sm")
                            btn_defect = gr.Button("Load Defective Sample", variant="secondary", size="sm")

                        with gr.Row():
                            loc = gr.Number(label="LOC (Lines of Code)", value=0.08)
                            cyclo = gr.Number(label="CYCLO (Cyclomatic)", value=0.43)
                            length = gr.Number(label="LENGTH (Halstead Length)", value=0.18)
                            volume = gr.Number(label="VOLUME (Halstead Volume)", value=0.16)

                        with gr.Row():
                            difficulty = gr.Number(label="DIFFICULTY (Difficulty)", value=0.0)
                            int_fan_in = gr.Number(label="INT_FAN_IN (Fan In)", value=0.0)
                            int_fan_out = gr.Number(label="INT_FAN_OUT (Fan Out)", value=0.22)

                        with gr.Row():
                            num_operators = gr.Number(label="NUM_OPERATORS", value=0.14)
                            num_operands = gr.Number(label="NUM_OPERANDS", value=0.81)
                            branch_count = gr.Number(label="BRANCH_COUNT", value=0.64)

                        btn_predict_grade = gr.Button("Run Code Quality Prediction", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Prediction Output")
                        quality_label = gr.Textbox(label="Submission Quality", interactive=False)
                        grade_label = gr.Textbox(label="Defect Assessment", interactive=False)
                        grade_conf = gr.Number(label="Model Confidence (%)", interactive=False)
                        defect_prob = gr.Number(label="Defect Probability (%)", interactive=False)
                        review_flag = gr.Textbox(label="Needs Manual Review (< 85% conf)", interactive=False)
                        derived_features = gr.JSON(label="Engineered Feature Breakdown")

                all_inputs = [loc, cyclo, length, volume, difficulty, int_fan_in, int_fan_out, num_operators, num_operands, branch_count]
                
                btn_predict_grade.click(
                    fn=gradio_predict_grade,
                    inputs=all_inputs,
                    outputs=[quality_label, grade_label, grade_conf, defect_prob, review_flag, derived_features]
                )

                btn_clean.click(fn=lambda: CLEAN_CODE_PRESET, outputs=all_inputs)
                btn_defect.click(fn=lambda: DEFECTIVE_CODE_PRESET, outputs=all_inputs)

            # TAB 2: Student Doubt Triage
            with gr.TabItem("Pipeline 2: Student Doubt Triage"):
                gr.Markdown("### Question NLP Classification & Urgency Routing")

                with gr.Row():
                    with gr.Column(scale=2):
                        doubt_input = gr.Textbox(
                            label="Student Question Text",
                            placeholder="Enter programming question or query here...",
                            lines=4,
                            value="My code throws NullPointerException when calling array element inside nested loop in Java"
                        )
                        btn_predict_doubt = gr.Button("Classify Question & Route", variant="primary")

                        gr.Examples(
                            examples=[
                                ["My code throws NullPointerException when calling array element inside nested loop in Java"],
                                ["URGENT: Production server database connections timing out during load test deadline in 1 hour!"],
                                ["Where can I find the assignment submission link for module 3?"],
                                ["How do I fix Segmentation Fault in C pointer allocation?"]
                            ],
                            inputs=[doubt_input],
                            label="Preset Example Queries"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### Classification Results")
                        doubt_topic = gr.Textbox(label="Predicted Topic Category", interactive=False)
                        doubt_urgency = gr.Textbox(label="Urgency Level", interactive=False)
                        doubt_route = gr.Textbox(label="Routing Recommendation", interactive=False)
                        doubt_conf = gr.Number(label="Confidence (%)", interactive=False)
                        doubt_probs = gr.Label(label="Topic Category Probabilities", num_top_classes=4)

                btn_predict_doubt.click(
                    fn=gradio_predict_doubt,
                    inputs=[doubt_input],
                    outputs=[doubt_topic, doubt_urgency, doubt_route, doubt_conf, doubt_probs]
                )

            # TAB 3: System Health
            with gr.TabItem("System Status & Models"):
                gr.Markdown("### Status of Saved Model Artifacts")
                btn_status = gr.Button("Refresh Model Status")
                status_json = gr.JSON(label="Loaded Models", value=get_model_status())
                btn_status.click(fn=gradio_health_check, outputs=[status_json])

    return demo


demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
