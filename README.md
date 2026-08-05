# 🎓 LMS ML Pipeline — Code Grading & Doubt Triage

> Production-quality ML pipeline for a Learning Management System (LMS).  
> Built for the **KPMG Off-Campus Hiring Assessment**.

---

## 📋 Overview

This project implements **two independent ML pipelines**:

| Pipeline | Goal | Dataset | Model |
|----------|------|---------|-------|
| **Code Grading** | Predict software defect likelihood | NASA KC1 | LightGBM |
| **Doubt Triage** | Classify student questions & route | CS1QA | LinearSVC + Calibration |

---

## 🏗️ Project Structure

```
kpmg/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned & split data
├── notebooks/
│   ├── 01_code_grading.py      # Pipeline 1 notebook (16 sections)
│   └── 02_doubt_triage.py      # Pipeline 2 notebook (16 sections)
├── src/
│   ├── config.py               # Central configuration
│   ├── utils.py                # Logging, seed management, helpers
│   ├── data_loader.py          # Dataset download & loading
│   ├── preprocessing.py        # Duplicates, missing, outliers, leakage
│   ├── feature_engineering.py  # Derived software metrics
│   ├── text_processing.py      # NLP cleaning & TF-IDF
│   ├── model_training.py       # Training, CV, Optuna tuning
│   ├── evaluation.py           # Metrics & visualization
│   ├── explainability.py       # SHAP analysis
│   └── threshold_optimizer.py  # Confidence threshold optimization
├── models/                     # Saved model artifacts (.pkl)
├── reports/
│   ├── figures/                # Generated plots
│   └── final_report.md         # Professional report
├── api/
│   ├── app.py                  # FastAPI application
│   ├── schemas.py              # Pydantic request/response models
│   └── routes.py               # Endpoint handlers
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks

The notebook scripts use `# %%` cell markers and can be run in:
- **VS Code**: Open the `.py` file → Run Cell (Ctrl+Enter)
- **Jupyter**: Convert first with `jupytext --to notebook notebooks/01_code_grading.py`
- **Google Colab**: Copy cell contents into Colab cells

```bash
# Run Pipeline 1
python notebooks/01_code_grading.py

# Run Pipeline 2
python notebooks/02_doubt_triage.py
```

### 3. Start the API

```bash
uvicorn api.app:app --reload --port 8000
```

API documentation available at: `http://localhost:8000/docs`

---

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Predict Code Grade
```bash
POST /predict-grade
Content-Type: application/json

{
  "loc": 150,
  "v(g)": 12,
  "ev(g)": 4,
  "iv(g)": 8,
  "n": 250,
  "v": 1500.5,
  "l": 0.02,
  "d": 25.3,
  "i": 59.4,
  "e": 37962.65,
  "b": 0.5,
  "t": 2109.04,
  "lOCode": 120,
  "lOComment": 20,
  "lOBlank": 10,
  "branchCount": 15
}
```

**Response:**
```json
{
  "prediction": "No Defect",
  "confidence": 0.87,
  "defect_probability": 0.13,
  "feature_values": { ... }
}
```

### Predict Doubt Category
```bash
POST /predict-doubt
Content-Type: application/json

{
  "question": "Why am I getting a syntax error on line 15 of my Python code?"
}
```

**Response:**
```json
{
  "prediction": "syntax",
  "confidence": 0.92,
  "urgency": "HIGH",
  "route": "Auto Approval",
  "all_probabilities": { ... }
}
```

---

## 📊 Pipeline Details

### Pipeline 1: Code Grading

| Step | Detail |
|------|--------|
| Dataset | NASA KC1 (OpenML ID: 1067) |
| Features | 21 raw + 6 engineered |
| Baseline | Random Forest (`class_weight='balanced'`) |
| Final Model | LightGBM (Optuna-tuned, 50 trials) |
| CV | 5-fold Stratified, ROC-AUC metric |
| Explainability | SHAP TreeExplainer |

### Pipeline 2: Doubt Triage

| Step | Detail |
|------|--------|
| Dataset | CS1QA (9,237 Q&A pairs, 9 categories) |
| Text Cleaning | lowercase, URLs, punctuation, stopwords, lemmatization |
| Features | TF-IDF (5000 features, bigrams) |
| Baseline | Logistic Regression (OVR) |
| Final Model | LinearSVC + CalibratedClassifierCV |
| Thresholds | 0.60, 0.70, 0.80, 0.85, 0.90 |
| Routing | confidence ≥ threshold → Auto, else → Teacher Review |

---

## 🔧 Engineering Practices

- ✅ **Modular code** — Reusable functions across both pipelines
- ✅ **Type hints** — All functions annotated
- ✅ **Docstrings** — Every function documented
- ✅ **Logging** — Module-level loggers throughout
- ✅ **PEP 8** — Consistent code style
- ✅ **No hardcoded values** — Central `config.py`
- ✅ **Reproducibility** — `random_state=42` everywhere
- ✅ **No data leakage** — Strict train/test separation
- ✅ **sklearn Pipelines** — Preprocessing + model in single object
- ✅ **Exception handling** — Graceful error management

---

## 📈 Visualizations Generated

| Plot | Pipeline |
|------|----------|
| Missing Value Heatmap | 1 |
| Target Distribution | 1, 2 |
| Correlation Matrix | 1 |
| Feature Distributions | 1 |
| Feature Importance | 1 |
| SHAP Summary Plot | 1 |
| SHAP Bar Plot | 1 |
| SHAP Waterfall | 1 |
| Confusion Matrix | 1, 2 |
| ROC Curve | 1, 2 |
| Precision-Recall Curve | 1, 2 |
| Threshold vs Accuracy | 2 |
| Threshold vs Auto-Approval Rate | 2 |
| Confidence Distribution | 2 |
| Error Analysis | 1, 2 |

---

## 📝 Report

See [reports/final_report.md](reports/final_report.md) for the complete professional report including:
- Problem Statement & Methodology
- Feature Engineering (available vs. unavailable)
- Model Selection & Hyperparameter Tuning
- Cross-Validation Results
- Threshold Justification
- Data Leakage Analysis
- Limitations & Future Work

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | ML models, preprocessing, evaluation |
| lightgbm | Gradient boosting (Pipeline 1) |
| optuna | Hyperparameter optimization |
| matplotlib, seaborn | Visualization |
| nltk | NLP text processing |
| shap | Model explainability |
| fastapi, uvicorn | REST API deployment |
| joblib | Model serialization |

---

## 📄 License

This project was built for educational assessment purposes.

**Datasets**:
- NASA KC1: PROMISE Repository (Sayyad Shirabad & Menzies, 2005)
- CS1QA: NAACL 2022 (Yoon et al.), MIT License
- IBM Project CodeNet: Apache 2.0 License
