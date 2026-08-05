# Professional Report: LMS ML Pipeline for Code Grading & Doubt Triage

**Project**: KPMG Off-Campus Hiring Assessment  
**Author**: Vijay Joping  
**Date**: August 2026

---

## 1. Problem Statement

Modern Learning Management Systems (LMS) face two critical challenges:

1. **Code Quality Assessment**: Instructors manually reviewing hundreds of code submissions is time-consuming, inconsistent, and doesn't scale. An automated code grading system based on software metrics can provide objective, immediate quality feedback.

2. **Student Doubt Triage**: Large online courses generate overwhelming volumes of student questions. Automatically categorizing questions by topic, assessing urgency, and routing based on prediction confidence enables efficient resource allocation.

This project builds two independent ML pipelines to address these challenges:
- **Pipeline 1**: Binary classification of code defect likelihood from software engineering metrics
- **Pipeline 2**: Multi-class text classification of student questions with confidence-based routing

---

## 2. Datasets

### Pipeline 1: NASA KC1 Software Defect Dataset

| Property | Value |
|----------|-------|
| **Source** | NASA Metrics Data Program / PROMISE Repository |
| **OpenML ID** | 1067 |
| **Domain** | Storage Management Software |
| **Samples** | ~2,109 software modules |
| **Features** | 21 software metrics (McCabe + Halstead + LOC) |
| **Target** | `defects` (binary: true/false) |
| **Class Distribution** | ~85% No Defect / ~15% Defect (imbalanced) |
| **Citation** | Sayyad Shirabad & Menzies (2005), PROMISE Repository |

### Pipeline 2: CS1QA Educational Dataset

| Property | Value |
|----------|-------|
| **Source** | KAIST Introductory Python Course |
| **Publication** | NAACL 2022 (Yoon et al.) |
| **Samples** | 9,237 annotated Q&A pairs |
| **Categories** | 9 question types |
| **Features** | Question text (natural language) |
| **Target** | `question_type` (9-class multi-class) |

---

## 3. Methodology

### Pipeline 1: ML-Based Code Grading

```
Raw Data → EDA → Cleaning → Feature Engineering → Leakage Detection
→ Stratified Split (60/20/20) → Baseline (RF) → Final (LightGBM)
→ Optuna Tuning → SHAP Explainability → Model Saving
```

### Pipeline 2: Student Doubt Triage

```
Raw Text → Text Cleaning (lowercase, URLs, punctuation, stopwords, lemma)
→ TF-IDF (5000 features, bigrams) → Split (60/20/20)
→ Baseline (Logistic Regression) → Final (LinearSVC + Calibration)
→ Threshold Optimization → Routing Logic → Model Saving
```

---

## 4. Feature Engineering

### Pipeline 1: Software Metrics

#### Available Features (Direct from KC1)

| Feature | KC1 Column | Description |
|---------|------------|-------------|
| LOC | `loc` | McCabe's line count |
| Cyclomatic Complexity | `v(g)` | McCabe's cyclomatic complexity |
| Essential Complexity | `ev(g)` | McCabe's essential complexity |
| Design Complexity | `iv(g)` | McCabe's design complexity |
| Halstead Volume | `v` | Program volume |
| Halstead Difficulty | `d` | Program difficulty |
| Halstead Effort | `e` | Programming effort |
| Halstead Bug Estimate | `b` | Estimated bugs |
| Halstead Time Estimate | `t` | Estimated programming time |
| Branch Count | `branchCount` | Flow graph branches |

#### Derived Features (Engineered)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| Code Size | `loc + lOComment + lOBlank` | Total module size |
| Complexity Ratio | `v(g) / loc` | Complexity per line |
| Maintainability Index | `171 - 5.2·ln(V) - 0.23·v(g) - 16.2·ln(LOC)` | SEI maintainability |
| Comment Density | `lOComment / (loc + lOComment)` | Documentation ratio |
| Bug Density | `b / loc` | Estimated bugs per line |
| Effort Density | `e / loc` | Effort per line |

#### Unavailable Features (Documented Limitations)

| Feature | Reason |
|---------|--------|
| **Fan In** | Requires call-graph analysis data not in KC1 |
| **Fan Out** | Requires call-graph analysis data not in KC1 |
| **Runtime Efficiency** | Requires runtime profiling data |
| **Memory Efficiency** | Requires memory profiling data |
| **Function Count** | Requires AST-level parsing not in KC1 |

These features are NOT fabricated. IBM Project CodeNet provides tooling for AST analysis and call graphs, but does not contain pre-computed per-module metrics compatible with the KC1 schema. CodeNet was referenced for methodology validation only.

### Pipeline 2: Text Features

- **TF-IDF Vectorization**: max_features=5000, ngram_range=(1,2), sublinear TF
- **Urgency Derivation**: Heuristic keyword matching (HIGH/MEDIUM/LOW)
  - HIGH: error, bug, crash, wrong, fail, broken, urgent, deadline
  - MEDIUM: confused, unclear, doubt, question, understand
  - LOW: everything else

---

## 5. Model Selection

### Pipeline 1

| Model | Role | Rationale |
|-------|------|-----------|
| **Random Forest** | Baseline | Robust ensemble, handles imbalance via `class_weight='balanced'`, no feature scaling required |
| **LightGBM** | Final | State-of-art gradient boosting, handles multicollinearity, native categorical support, efficient training |

### Pipeline 2

| Model | Role | Rationale |
|-------|------|-----------|
| **Logistic Regression** | Baseline | Strong baseline for text classification, interpretable coefficients |
| **LinearSVC + CalibratedClassifierCV** | Final | SVM excels on high-dimensional sparse data (TF-IDF), calibration provides probability estimates for confidence routing |

---

## 6. Hyperparameter Tuning

### Pipeline 1: Optuna (LightGBM)

| Parameter | Search Range |
|-----------|-------------|
| n_estimators | 100–1000 |
| max_depth | 3–12 |
| learning_rate | 0.01–0.3 (log scale) |
| num_leaves | 15–127 |
| min_child_samples | 5–100 |
| subsample | 0.5–1.0 |
| colsample_bytree | 0.5–1.0 |
| reg_alpha | 1e-8–10 (log scale) |
| reg_lambda | 1e-8–10 (log scale) |

- **Optimization**: 50 trials with 300s timeout
- **Objective**: Maximize ROC-AUC via 5-fold stratified CV
- **Best parameters**: See notebook output

### Pipeline 2

- CalibratedClassifierCV performs internal 5-fold CV for calibration
- `class_weight='balanced'` handles multi-class imbalance
- `max_iter=5000` ensures convergence

---

## 7. Cross-Validation

Both pipelines use **5-fold Stratified K-Fold Cross-Validation** to ensure:
- Each fold preserves the class distribution
- Results are reported as **mean ± standard deviation**
- All folds use the same random state (42) for reproducibility

### Pipeline 1 (Binary Classification)
- Primary metric: **ROC-AUC**
- Additional: Accuracy, Precision, Recall, F1

### Pipeline 2 (Multi-class Classification)
- Primary metric: **Macro F1**
- Additional: Accuracy, Weighted F1

---

## 8. Evaluation

### Pipeline 1 Metrics

| Metric | Value |
|--------|-------|
| Accuracy | See notebook output |
| Precision | See notebook output |
| Recall | See notebook output |
| F1 Score | See notebook output |
| ROC-AUC | See notebook output |

### Pipeline 2 Metrics

| Metric | Value |
|--------|-------|
| Accuracy | See notebook output |
| Macro F1 | See notebook output |
| Weighted F1 | See notebook output |

*Note: Exact values are populated when notebooks are executed.*

### Visualizations Generated
- Missing Value Heatmap
- Target Distribution (both pipelines)
- Correlation Matrix
- Feature Importance (LightGBM)
- SHAP Summary Plot
- SHAP Waterfall Plot
- Confusion Matrix (all models)
- ROC Curve (Pipeline 1 + Pipeline 2 per-class)
- Precision-Recall Curve
- Threshold vs Accuracy
- Threshold vs Auto-Approval Rate
- Confidence Distribution

---

## 9. Threshold Justification

For Pipeline 2 (Doubt Triage), threshold optimization balances two competing objectives:

1. **High auto-approval rate** → Reduces instructor workload
2. **High auto-approved accuracy** → Ensures quality of automated responses

Thresholds evaluated: **0.60, 0.70, 0.80, 0.85, 0.90**

The **best threshold** is selected to maximize **effective accuracy** (assuming teacher-reviewed predictions are corrected) while maintaining a minimum 30% auto-approval rate.

**Routing Logic**:
- Confidence ≥ threshold → **Auto Approval** (immediate response)
- Confidence < threshold → **Teacher Review** (queued for instructor)

---

## 10. Data Leakage Analysis

### Pipeline 1
- **Target leakage**: Checked correlation of each feature with the target. Features with correlation > 0.95 are flagged.
- **Train-test contamination**: Prevented by splitting before any preprocessing (StandardScaler fit on training only).
- **Multicollinearity**: Halstead metrics are inherently correlated (e.g., volume ↔ effort). Handled by LightGBM's `colsample_bytree` feature subsampling.

### Pipeline 2
- **TF-IDF leakage**: Vectorizer fitted on TRAINING data only, then used to transform val/test.
- **Text leakage**: Only question text is used as input (no answer or code context).
- **Duplicate contamination**: Duplicate questions removed before splitting.

---

## 11. Limitations

1. **Missing Features**: Fan In/Out, Runtime/Memory Efficiency, and Function Count cannot be derived from the NASA KC1 dataset.

2. **Dataset Age**: KC1 dates from early 2000s NASA projects. Modern code quality patterns may differ (different languages, frameworks, paradigms).

3. **Urgency Heuristic**: CS1QA lacks ground-truth urgency annotations. The derived urgency is based on keyword matching, which may misclassify nuanced questions.

4. **Static Analysis Only**: Pipeline 1 uses static code metrics only. No runtime behavior, test coverage, or dynamic analysis data is available.

5. **Domain Specificity**: Pipeline 2 is trained on introductory Python course data. Performance on advanced courses or different programming languages may vary.

6. **No Code Context**: Pipeline 2 uses only question text. Incorporating the associated code could improve classification accuracy.

---

## 12. Future Work

1. **Advanced NLP**: Fine-tune CodeBERT or similar code-language models for doubt triage.

2. **Runtime Metrics**: Integrate execution profiling data for code grading.

3. **Active Learning**: Implement continuous model improvement from teacher corrections.

4. **Multi-language Support**: Extend code grading to support multiple programming languages.

5. **A/B Testing**: Deploy in a real LMS and measure instructor time savings.

6. **Graph Neural Networks**: Use IBM Project CodeNet's AST/call-graph tools for structure-aware code analysis.

7. **Ensemble Methods**: Combine multiple models with stacking for improved performance.

---

## Appendix: Project Structure

```
kpmg/
├── data/raw/              # Downloaded datasets
├── data/processed/        # Processed data (after splitting)
├── notebooks/
│   ├── 01_code_grading.py # Pipeline 1 notebook
│   └── 02_doubt_triage.py # Pipeline 2 notebook
├── src/
│   ├── config.py          # Central configuration
│   ├── utils.py           # Utilities
│   ├── data_loader.py     # Dataset loading
│   ├── preprocessing.py   # Data cleaning
│   ├── feature_engineering.py  # Feature derivation
│   ├── text_processing.py # NLP preprocessing
│   ├── model_training.py  # Model training & tuning
│   ├── evaluation.py      # Metrics & plots
│   ├── explainability.py  # SHAP analysis
│   └── threshold_optimizer.py  # Threshold optimization
├── models/                # Saved model artifacts
├── reports/
│   ├── figures/           # Generated visualizations
│   └── final_report.md   # This report
├── api/
│   ├── app.py             # FastAPI application
│   ├── schemas.py         # Pydantic models
│   └── routes.py          # Endpoint handlers
├── requirements.txt
└── README.md
```
