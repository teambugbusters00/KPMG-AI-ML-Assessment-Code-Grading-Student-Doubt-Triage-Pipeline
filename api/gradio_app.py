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
    loc: float, v_g: float, ev_g: float, iv_g: float,
    n: float, v: float, l: float, d: float, i: float, e: float, b: float, t: float,
    lOCode: float, lOComment: float, lOBlank: float,
    uniq_Op: float, uniq_Opnd: float, total_Op: float, total_Opnd: float, branchCount: float
) -> Tuple[str, float, float, Dict[str, float]]:
    """Handler for Pipeline 1 Code Quality Grading Gradio tab."""
    req = CodeGradingRequest(
        loc=loc, v_g=v_g, ev_g=ev_g, iv_g=iv_g,
        n=n, v=v, l=l, d=d, i=i, e=e, b=b, t=t,
        lOCode=lOCode, lOComment=lOComment, lOBlank=lOBlank,
        uniq_Op=uniq_Op, uniq_Opnd=uniq_Opnd, total_Op=total_Op, total_Opnd=total_Opnd,
        branchCount=branchCount
    )
    res = predict_grade(req)
    return (
        f"Result: {res.prediction}",
        round(res.defect_probability * 100, 2),
        round(res.confidence * 100, 2),
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


# Preset values for Code Grading
CLEAN_CODE_PRESET = [15, 2, 1, 1, 40, 180, 0.15, 6, 30, 1080, 0.06, 60, 12, 4, 2, 8, 8, 24, 16, 3]
DEFECTIVE_CODE_PRESET = [280, 42, 25, 18, 1200, 8500, 0.01, 85, 100, 722500, 2.8, 40138, 240, 5, 35, 45, 80, 700, 500, 83]


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
                    <h1>🎓 LMS AI/ML Pipeline Dashboard</h1>
                    <p>Interactive Code Quality Grading & Student Doubt Triage System</p>
                </div>
            """)

        with gr.Tabs():
            # TAB 1: Code Quality Grading
            with gr.TabItem("📊 Pipeline 1: Code Quality Grading"):
                gr.Markdown("### Code Software Metrics & Defect Prediction")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### McCabe & Halstead Input Metrics")
                        with gr.Row():
                            btn_clean = gr.Button("⚡ Load Clean Code Sample", variant="secondary", size="sm")
                            btn_defect = gr.Button("⚠️ Load Defective Sample", variant="secondary", size="sm")

                        with gr.Row():
                            loc = gr.Number(label="LOC (loc)", value=25)
                            v_g = gr.Number(label="Cyclomatic v(g)", value=4)
                            ev_g = gr.Number(label="Essential ev(g)", value=1)
                            iv_g = gr.Number(label="Design iv(g)", value=3)

                        with gr.Row():
                            n = gr.Number(label="Length (n)", value=75)
                            v = gr.Number(label="Volume (v)", value=350)
                            l = gr.Number(label="Level (l)", value=0.08)
                            d = gr.Number(label="Difficulty (d)", value=12)

                        with gr.Row():
                            i = gr.Number(label="Intelligence (i)", value=29)
                            e = gr.Number(label="Effort (e)", value=4200)
                            b = gr.Number(label="Bugs Est. (b)", value=0.11)
                            t = gr.Number(label="Time Est. (t)", value=233)

                        with gr.Row():
                            lOCode = gr.Number(label="Code Lines", value=20)
                            lOComment = gr.Number(label="Comment Lines", value=3)
                            lOBlank = gr.Number(label="Blank Lines", value=2)

                        with gr.Row():
                            uniq_Op = gr.Number(label="Uniq Operators", value=12)
                            uniq_Opnd = gr.Number(label="Uniq Operands", value=10)
                            total_Op = gr.Number(label="Total Operators", value=45)
                            total_Opnd = gr.Number(label="Total Operands", value=30)
                            branchCount = gr.Number(label="Branch Count", value=7)

                        btn_predict_grade = gr.Button("🔍 Run Code Quality Prediction", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Prediction Output")
                        grade_label = gr.Textbox(label="Assessment Result", interactive=False)
                        defect_prob = gr.Number(label="Defect Probability (%)", interactive=False)
                        grade_conf = gr.Number(label="Model Confidence (%)", interactive=False)
                        derived_features = gr.JSON(label="Engineered Feature Breakdown")

                all_inputs = [loc, v_g, ev_g, iv_g, n, v, l, d, i, e, b, t, lOCode, lOComment, lOBlank, uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount]
                
                btn_predict_grade.click(
                    fn=gradio_predict_grade,
                    inputs=all_inputs,
                    outputs=[grade_label, defect_prob, grade_conf, derived_features]
                )

                def set_preset(preset_list):
                    return preset_list

                btn_clean.click(fn=lambda: CLEAN_CODE_PRESET, outputs=all_inputs)
                btn_defect.click(fn=lambda: DEFECTIVE_CODE_PRESET, outputs=all_inputs)

            # TAB 2: Student Doubt Triage
            with gr.TabItem("🤖 Pipeline 2: Student Doubt Triage"):
                gr.Markdown("### Question NLP Classification & Urgency Routing")

                with gr.Row():
                    with gr.Column(scale=2):
                        doubt_input = gr.Textbox(
                            label="Student Question Text",
                            placeholder="Enter programming question or query here...",
                            lines=4,
                            value="My code throws NullPointerException when calling array element inside nested loop in Java"
                        )
                        btn_predict_doubt = gr.Button("⚡ Classify Question & Route", variant="primary")

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
            with gr.TabItem("⚙️ System Status & Models"):
                gr.Markdown("### Status of Saved Model Artifacts")
                btn_status = gr.Button("↻ Check Model Status")
                status_json = gr.JSON(label="Loaded Models", value=get_model_status())
                btn_status.click(fn=gradio_health_check, outputs=[status_json])

    return demo


demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
