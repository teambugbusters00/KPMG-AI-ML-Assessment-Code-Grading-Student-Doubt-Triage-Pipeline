# 🎓 LMS AI/ML Pipeline — Code Quality Grading & Student Doubt Triage

> Production-quality Machine Learning & NLP Pipeline for Learning Management Systems (LMS).  
> Built for the **KPMG AI/ML Off-Campus Hiring Assessment**.

---

## 🌐 Live Web Interfaces & API Links

- **Interactive Dashboard:** `https://<your-render-url>/` *(Sleek HTML/JS Web App)*
- **Gradio Interactive UI:** `https://<your-render-url>/gradio` *(Full 3-Tab Gradio App)*
- **FastAPI Swagger Docs:** `https://<your-render-url>/docs`
- **API Health Check:** `https://<your-render-url>/health`

---

## 📋 Overview

This project implements **two production-grade ML pipelines**:

| Pipeline | Goal | Dataset | Model |
|----------|------|---------|-------|
| **Pipeline 1: Code Grading** | Predict code quality & defect likelihood from software metrics | `SoftwareDefectDataset.csv` (`Project_CodeNet/assets/`) | LightGBM + Optuna |
| **Pipeline 2: Doubt Triage** | Classify programming questions, estimate urgency & route queries | CS1QA Dataset | LinearSVC + CalibratedClassifierCV |

---

## 🏗️ Project Structure

```
kpmg/
├── data/
│   ├── raw/                    # Original raw datasets
│   └── processed/              # Cleaned & split data
├── Project_CodeNet/
│   └── assets/
│       └── SoftwareDefectDataset.csv # Primary dataset for Pipeline 1
├── notebooks/
│   ├── 01_code_grading.py      # Pipeline 1 notebook (16 sections)
│   └── 02_doubt_triage.py      # Pipeline 2 notebook (16 sections)
├── src/
│   ├── config.py               # Central configuration module
│   ├── utils.py                # Logging, seed management, helpers
│   ├── data_loader.py          # Dataset downloading & loading
│   ├── preprocessing.py        # Duplicates, missing values, outliers, leakage check
│   ├── feature_engineering.py  # Derived software metrics (Complexity/LOC, Fan Ratio, etc.)
│   ├── text_processing.py      # NLP cleaning & TF-IDF vectorization
│   ├── model_training.py       # Baseline RF, LightGBM, Optuna 5-fold CV, LinearSVC
│   ├── evaluation.py           # Metrics, confusion matrix, ROC/PR curves
│   ├── explainability.py       # SHAP TreeExplainer analysis
│   └── threshold_optimizer.py  # Confidence threshold optimization & routing
├── models/                     # Saved model artifacts (.pkl & .json)
├── reports/
│   ├── figures/                # Generated visualization plots
│   └── final_report.md         # Professional technical report
├── api/
│   ├── app.py                  # FastAPI server entry point
│   ├── gradio_app.py           # Full Gradio Blocks interface
│   ├── dashboard.html          # Custom web dashboard landing page
│   ├── schemas.py              # Pydantic request/response models
│   └── routes.py               # Endpoint logic handlers
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start & Local Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks / Pipeline Scripts

```bash
# Run Pipeline 1 (Code Quality Grading)
python notebooks/01_code_grading.py

# Run Pipeline 2 (Student Doubt Triage)
python notebooks/02_doubt_triage.py
```

### 3. Start Combined Server (FastAPI + Gradio)

```bash
uvicorn api.app:app --reload --port 8000
```

- Web Dashboard: `http://localhost:8000/`
- Gradio UI: `http://localhost:8000/gradio`
- Interactive Swagger API: `http://localhost:8000/docs`

---

## 📡 API Endpoints & Request Examples

### 1. Code Quality Grading (`POST /predict-grade`)

**Request Payload:**
```json
{
  "LOC": 0.08,
  "CYCLO": 0.43,
  "LENGTH": 0.18,
  "VOLUME": 0.16,
  "DIFFICULTY": 0.0,
  "INT_FAN_IN": 0.0,
  "INT_FAN_OUT": 0.22,
  "NUM_OPERATORS": 0.14,
  "NUM_OPERANDS": 0.81,
  "BRANCH_COUNT": 0.64
}
```

**Response Payload:**
```json
{
  "quality": "Good",
  "prediction": "No Defect",
  "confidence": 0.94,
  "defect_probability": 0.06,
  "review_needed": false,
  "feature_values": {
    "Complexity_per_LOC": 5.375,
    "Branch_Density": 8.0,
    "Fan_Ratio": 0.0,
    "Complexity_x_LOC": 0.0344,
    "Halstead_per_LOC": 2.0
  }
}
```

### 2. Student Doubt Triage (`POST /predict-doubt`)

**Request Payload:**
```json
{
  "question": "My code throws NullPointerException when calling array element inside loop"
}
```

**Response Payload:**
```json
{
  "prediction": "Java Programming",
  "confidence": 0.91,
  "urgency": "MEDIUM",
  "route": "Teacher Review",
  "all_probabilities": {
    "Java Programming": 0.91,
    "Python Syntax": 0.05,
    "Database Errors": 0.04
  }
}
```

---

## 💡 Tradeoffs & Handling Ambiguity

1. **Defect Prediction as Proxy for Code Quality (Pipeline 1):**
   - **Ambiguity:** Benchmark datasets with human teacher grading rubrics are scarce in open ML literature.
   - **Tradeoff & Resolution:** Software defect prediction metrics (McCabe complexity, Halstead metrics, fan-in/fan-out) serve as an objective proxy for code submission quality. Lower defect probability indicates higher software quality.

2. **Confidence Thresholding for Manual Review:**
   - Predictions with confidence $\ge 0.85$ are flagged as high-confidence automated predictions (`review_needed: false`), while predictions below $0.85$ are routed for manual teacher review (`review_needed: true`).

3. **Urgency Modeling (Pipeline 2):**
   - Derived heuristically from question text keywords (e.g. error/crash/deadline $\rightarrow$ HIGH, explanation/why $\rightarrow$ MEDIUM, general $\rightarrow$ LOW) combined with calibrated prediction probabilities.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy`, `scipy` | Data structures & numerical utilities |
| `scikit-learn` | Preprocessing, Random Forest baseline, LinearSVC, CalibratedClassifierCV |
| `lightgbm` | Gradient boosting for Pipeline 1 |
| `optuna` | Hyperparameter optimization |
| `nltk` | NLP text processing (tokenization, stop words, lemmatization) |
| `shap` | TreeExplainer feature attribution & interpretability |
| `fastapi`, `uvicorn`, `pydantic` | Production REST API server |
| `gradio` | Interactive 3-tab web UI |
| `joblib` | Model artifact serialization |

---

## 📄 License & Attribution

Built for the KPMG AI/ML Assessment.
- Datasets: Project CodeNet, NASA MDP, CS1QA (NAACL 2022).
