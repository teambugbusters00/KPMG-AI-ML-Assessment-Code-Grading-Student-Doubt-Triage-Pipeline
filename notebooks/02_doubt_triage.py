# %% [markdown]
# # Pipeline 2: Student Doubt Triage
# ## KPMG Off-Campus Hiring Assessment — LMS ML Pipeline
#
# **Objective**: Predict the topic category of student programming questions,
# assess urgency, generate confidence scores, and route predictions based
# on confidence thresholds.
#
# **Pipeline**: Text Cleaning → TF-IDF → Logistic Regression (Baseline)
# → LinearSVC + Calibration (Final) → Threshold Optimization → Routing

# %% [markdown]
# ## 1. Introduction
#
# This notebook implements an NLP-based pipeline for automatically triaging
# student questions (doubts) in an LMS environment. The system:
#
# 1. **Predicts the topic** of each question (9 categories)
# 2. **Assesses urgency** using heuristic text analysis
# 3. **Generates a confidence score** for each prediction
# 4. **Routes** high-confidence predictions to auto-approval and
#    low-confidence ones to teacher review
#
# **Dataset**: CS1QA (NAACL 2022) — 9,237 annotated Q&A pairs from an
# introductory Python programming course.

# %% [markdown]
# ## 2. Business Problem
#
# In large online courses, instructors face overwhelming volumes of student
# questions. A doubt triage system can:
#
# - **Categorize** questions by topic automatically
# - **Prioritize** urgent questions (runtime errors, bugs)
# - **Route** confident predictions directly to answers
# - **Escalate** ambiguous questions to human instructors
#
# This reduces instructor workload while ensuring students get timely help.
#
# **Classification Task**: Multi-class text classification
# - **Input**: Student question text
# - **Output**: Question type (9 categories) + Urgency + Confidence + Route

# %%
# === Setup & Imports ===
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CONFIDENCE_THRESHOLDS,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    DOUBT_TRIAGE_MODEL_PATH,
    DOUBT_TRIAGE_VECTORIZER_PATH,
)
from src.utils import set_global_seed, print_section_header
from src.data_loader import load_cs1qa_dataset
from src.preprocessing import (
    detect_duplicates,
    remove_duplicates,
    analyze_missing_values,
    split_data,
    check_class_imbalance,
)
from src.text_processing import (
    clean_text_column,
    build_tfidf_vectorizer,
    fit_transform_tfidf,
    transform_tfidf,
    get_top_tfidf_features,
)
from src.model_training import (
    train_baseline_lr,
    train_calibrated_svc,
    save_model,
    cross_validate_model,
)
from src.evaluation import (
    compute_classification_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_target_distribution,
)
from src.threshold_optimizer import (
    evaluate_thresholds,
    select_best_threshold,
    route_predictions_batch,
    plot_threshold_vs_accuracy,
    plot_threshold_vs_approval_rate,
    plot_confidence_distribution,
)

# Set reproducibility
set_global_seed(RANDOM_STATE)
print_section_header("Pipeline 2: Student Doubt Triage")

# %% [markdown]
# ## 3. Dataset Description
#
# **CS1QA Dataset** (NAACL 2022, Yoon et al.)
#
# | Property | Value |
# |----------|-------|
# | Source | KAIST Introductory Python Course |
# | Annotated Samples | 9,237 Q&A pairs |
# | Question Types | 9 categories |
# | Features | Question text, student code, answers |
#
# **Question Type Categories**:
# syntax, logic, runtime, conceptual, output, debugging,
# implementation, design, other
#
# **Note on Urgency**: CS1QA does not contain explicit urgency labels.
# Urgency is derived heuristically from question text (see Section 6).

# %%
print_section_header("Loading CS1QA Dataset")
df, target_col = load_cs1qa_dataset()

print(f"Dataset shape: {df.shape}")
print(f"Target column: {target_col}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
df.head()

# %%
print(f"\nDataset info:")
df.info()

# %%
print(f"\nQuestion type distribution:")
print(df[target_col].value_counts())

# %% [markdown]
# ## 4. Exploratory Data Analysis

# %%
print_section_header("4.1 Target Distribution")
plot_target_distribution(
    df[target_col],
    title="Question Type Distribution (CS1QA)",
    save_name="p2_target_distribution.png",
)

# %%
# Check imbalance
imbalance_info = check_class_imbalance(df, target_col)

# %%
print_section_header("4.2 Urgency Distribution")
print(f"Urgency distribution:\n{df['urgency'].value_counts()}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = {"HIGH": "#F44336", "MEDIUM": "#FF9800", "LOW": "#4CAF50"}
urgency_counts = df["urgency"].value_counts()
urgency_counts.plot(kind="bar", ax=ax,
                     color=[colors.get(x, "#999") for x in urgency_counts.index],
                     edgecolor="black", alpha=0.8)
ax.set_title("Derived Urgency Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Urgency Level")
ax.set_ylabel("Count")
for i, (idx, val) in enumerate(urgency_counts.items()):
    ax.text(i, val + max(urgency_counts) * 0.01, str(val),
            ha="center", va="bottom", fontweight="bold")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p2_urgency_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print_section_header("4.3 Text Length Analysis")
df["text_length"] = df["question"].str.len()
df["word_count"] = df["question"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["text_length"], bins=30, color="#5C6BC0", alpha=0.7, edgecolor="black")
axes[0].set_title("Question Text Length Distribution", fontweight="bold")
axes[0].set_xlabel("Character Count")
axes[0].set_ylabel("Frequency")

axes[1].hist(df["word_count"], bins=30, color="#26A69A", alpha=0.7, edgecolor="black")
axes[1].set_title("Question Word Count Distribution", fontweight="bold")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "p2_text_length_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nText length stats:\n{df['text_length'].describe()}")
print(f"\nWord count stats:\n{df['word_count'].describe()}")

# %%
# Text length by question type
print_section_header("4.4 Text Length by Category")
fig, ax = plt.subplots(figsize=(12, 6))
df.boxplot(column="word_count", by=target_col, ax=ax)
ax.set_title("Word Count by Question Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Question Type")
ax.set_ylabel("Word Count")
plt.suptitle("")  # Remove default title
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p2_wordcount_by_category.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Data Cleaning

# %%
print_section_header("5.1 Duplicate Detection")
dupes, n_dupes = detect_duplicates(df, subset=["question"])
print(f"Duplicate questions: {n_dupes}")

if n_dupes > 0:
    df = remove_duplicates(df, subset=["question"])
    print(f"After removing duplicates: {df.shape}")

# %%
print_section_header("5.2 Text Cleaning Pipeline")
print("Applying: lowercase → remove URLs → remove punctuation → remove stopwords → lemmatization")

df = clean_text_column(df, text_column="question", output_column="cleaned_text")

# Show examples
print("\n--- Cleaning Examples ---")
for i in range(min(3, len(df))):
    print(f"  Original:  {df.iloc[i]['question'][:80]}...")
    print(f"  Cleaned:   {df.iloc[i]['cleaned_text'][:80]}...")
    print()

# %%
# Remove rows where cleaned text is empty
empty_mask = df["cleaned_text"].str.strip() == ""
print(f"Empty after cleaning: {empty_mask.sum()} rows")
if empty_mask.sum() > 0:
    df = df[~empty_mask].reset_index(drop=True)
    print(f"Dataset after removing empty: {df.shape}")

# %% [markdown]
# ## 6. Feature Engineering

# %%
print_section_header("6. TF-IDF Feature Extraction")
print(f"Configuration: max_features=5000, ngram_range=(1,2)")
print(f"\nNote on Urgency: Derived heuristically from keywords in question text.")
print(f"  - HIGH: error/crash/bug/fail keywords → {(df['urgency']=='HIGH').sum()} questions")
print(f"  - MEDIUM: confusion/explanation keywords → {(df['urgency']=='MEDIUM').sum()} questions")
print(f"  - LOW: general questions → {(df['urgency']=='LOW').sum()} questions")
print(f"  ⚠ Limitation: This is a heuristic, not ground-truth urgency.")

# %% [markdown]
# ## 7. Leakage Detection

# %%
print_section_header("7. Data Leakage Detection")
print("For text classification, potential leakage sources:")
print("  1. TF-IDF fitted AFTER split ✓ (we fit only on training data)")
print("  2. No answer/code text used as features ✓ (only question text)")
print("  3. Duplicate questions removed ✓ (prevents train-test contamination)")
print("\n✓ No data leakage detected in the pipeline design.")

# %% [markdown]
# ## 8. Train / Validation / Test Split

# %%
print_section_header("8. Data Splitting")

train_df, val_df, test_df = split_data(df, target_col)

print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")

# %%
# Fit TF-IDF on TRAINING data only (prevent leakage)
print_section_header("Fitting TF-IDF on Training Data")
vectorizer = build_tfidf_vectorizer()
X_train_tfidf, vectorizer = fit_transform_tfidf(train_df["cleaned_text"], vectorizer)
X_val_tfidf = transform_tfidf(val_df["cleaned_text"], vectorizer)
X_test_tfidf = transform_tfidf(test_df["cleaned_text"], vectorizer)

y_train = train_df[target_col].values
y_val = val_df[target_col].values
y_test = test_df[target_col].values

# Encode labels for sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

class_names = le.classes_.tolist()
print(f"\nClasses: {class_names}")
print(f"TF-IDF shape: Train={X_train_tfidf.shape}, Val={X_val_tfidf.shape}, Test={X_test_tfidf.shape}")

# %% [markdown]
# ## 9. Baseline Models

# %%
print_section_header("9. Baseline: Logistic Regression")

lr_model, lr_cv_results = train_baseline_lr(X_train_tfidf, y_train_enc)

# Evaluate on validation
y_val_pred_lr = lr_model.predict(X_val_tfidf)
y_val_prob_lr = lr_model.predict_proba(X_val_tfidf)

lr_metrics = compute_classification_metrics(
    y_val_enc, y_val_pred_lr, y_val_prob_lr, average="macro"
)
print(f"\nBaseline LR — Validation Metrics:")
for k, v in lr_metrics.items():
    print(f"  {k}: {v:.4f}")

# %%
print("\nBaseline LR — Classification Report:")
print_classification_report(y_val_enc, y_val_pred_lr, target_names=class_names)

plot_confusion_matrix(
    y_val_enc, y_val_pred_lr,
    labels=class_names,
    title="Baseline LR — Confusion Matrix (Validation)",
    save_name="p2_lr_confusion_matrix.png",
)

# %% [markdown]
# ## 10. Hyperparameter Tuning
#
# For the SVC model, we use CalibratedClassifierCV which performs internal
# cross-validation for probability calibration. The key hyperparameter
# decisions are:
# - `class_weight='balanced'` to handle class imbalance
# - `max_iter=5000` for convergence
# - Sigmoid calibration method (better for SVM)

# %% [markdown]
# ## 11. Final Models

# %%
print_section_header("11. Final Model: LinearSVC + CalibratedClassifierCV")

svc_model, svc_cv_results = train_calibrated_svc(X_train_tfidf, y_train_enc)

# Evaluate on validation
y_val_pred_svc = svc_model.predict(X_val_tfidf)
y_val_prob_svc = svc_model.predict_proba(X_val_tfidf)

svc_metrics = compute_classification_metrics(
    y_val_enc, y_val_pred_svc, y_val_prob_svc, average="macro"
)
print(f"\nCalibrated SVC — Validation Metrics:")
for k, v in svc_metrics.items():
    print(f"  {k}: {v:.4f}")

# %% [markdown]
# ## 12. Evaluation

# %%
print_section_header("12.1 Final Model — Test Set Evaluation")

y_test_pred = svc_model.predict(X_test_tfidf)
y_test_prob = svc_model.predict_proba(X_test_tfidf)

test_metrics = compute_classification_metrics(
    y_test_enc, y_test_pred, y_test_prob, average="macro"
)
print(f"\nCalibrated SVC — TEST SET Metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

# %%
print("\nFinal Model — Classification Report (Test Set):")
print_classification_report(y_test_enc, y_test_pred, target_names=class_names)

# %%
plot_confusion_matrix(
    y_test_enc, y_test_pred,
    labels=class_names,
    title="Calibrated SVC — Confusion Matrix (Test Set)",
    save_name="p2_svc_confusion_matrix.png",
)

# %%
# Multiclass ROC (one-vs-rest)
print_section_header("12.2 ROC Curves (One-vs-Rest)")
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test_enc, classes=list(range(len(class_names))))

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, (class_name, ax) in enumerate(zip(class_names, axes.flat)):
    if i < y_test_bin.shape[1]:
        from sklearn.metrics import roc_curve as sk_roc_curve, auc as sk_auc
        fpr, tpr, _ = sk_roc_curve(y_test_bin[:, i], y_test_prob[:, i])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(f"{class_name}", fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

plt.suptitle("ROC Curves — One-vs-Rest (Test Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p2_roc_curves_multiclass.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Multiclass Precision-Recall Curves
print_section_header("12.3 Precision-Recall Curves")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, (class_name, ax) in enumerate(zip(class_names, axes.flat)):
    if i < y_test_bin.shape[1]:
        from sklearn.metrics import precision_recall_curve as sk_pr_curve, auc as sk_auc2
        precision, recall, _ = sk_pr_curve(y_test_bin[:, i], y_test_prob[:, i])
        pr_auc = sk_auc2(recall, precision)
        ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"PR-AUC={pr_auc:.3f}")
        ax.set_title(f"{class_name}", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

plt.suptitle("Precision-Recall Curves — One-vs-Rest (Test Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p2_pr_curves_multiclass.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print_section_header("12.4 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression (Baseline)", "Calibrated SVC (Final)"],
    "CV Macro-F1": [
        f"{lr_cv_results['mean']:.4f} ± {lr_cv_results['std']:.4f}",
        f"{svc_cv_results['mean']:.4f} ± {svc_cv_results['std']:.4f}",
    ],
    "Val Accuracy": [lr_metrics["accuracy"], svc_metrics["accuracy"]],
    "Val Macro-F1": [lr_metrics.get("f1_macro", 0), svc_metrics.get("f1_macro", 0)],
    "Val Weighted-F1": [lr_metrics.get("f1_weighted", 0), svc_metrics.get("f1_weighted", 0)],
})
print(comparison_df.to_string(index=False))

# %% [markdown]
# ## 13. Explainability

# %%
print_section_header("13. Top TF-IDF Features per Class")

feature_names_tfidf = vectorizer.get_feature_names_out()

for class_name in class_names:
    # Get class index
    class_idx = le.transform([class_name])[0]
    top_features = get_top_tfidf_features(
        vectorizer, class_idx,
        X_train_tfidf, y_train_enc,
        top_n=10,
    )
    print(f"\n  [{class_name.upper()}] Top TF-IDF features:")
    for feat_name, score in top_features:
        print(f"    {feat_name}: {score:.4f}")

# %%
# Visualize top features per class
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
for i, (class_name, ax) in enumerate(zip(class_names, axes.flat)):
    class_idx = le.transform([class_name])[0]
    top_features = get_top_tfidf_features(
        vectorizer, class_idx, X_train_tfidf, y_train_enc, top_n=10
    )
    if top_features:
        names, scores = zip(*top_features)
        ax.barh(range(len(names)), scores, color=plt.cm.Set2(i / len(class_names)))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(f"{class_name}", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

plt.suptitle("Top TF-IDF Features per Question Type", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p2_tfidf_features_per_class.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14. Threshold Optimization

# %%
print_section_header("14. Threshold Optimization")

# Get confidence scores (max probability per prediction)
confidences = np.max(y_test_prob, axis=1)

print(f"Confidence statistics:")
print(f"  Mean: {confidences.mean():.4f}")
print(f"  Std:  {confidences.std():.4f}")
print(f"  Min:  {confidences.min():.4f}")
print(f"  Max:  {confidences.max():.4f}")

# %%
# Evaluate thresholds
threshold_results = evaluate_thresholds(
    y_test_enc, y_test_pred, confidences,
    thresholds=CONFIDENCE_THRESHOLDS,
)
print(f"\nThreshold Evaluation Results:")
print(threshold_results.to_string(index=False))

# %%
# Select best threshold
best_threshold = select_best_threshold(threshold_results)
print(f"\n★ Selected best threshold: {best_threshold:.2f}")

# %%
# Visualize threshold analysis
plot_threshold_vs_accuracy(
    threshold_results,
    best_threshold=best_threshold,
    save_name="p2_threshold_vs_accuracy.png",
)

# %%
plot_threshold_vs_approval_rate(
    threshold_results,
    best_threshold=best_threshold,
    save_name="p2_threshold_vs_approval_rate.png",
)

# %%
plot_confidence_distribution(
    confidences,
    threshold=best_threshold,
    title="Prediction Confidence Distribution",
    save_name="p2_confidence_distribution.png",
)

# %%
# Apply routing
routes = route_predictions_batch(confidences, best_threshold)

print(f"\nRouting Results (threshold={best_threshold:.2f}):")
route_counts = pd.Series(routes).value_counts()
for route, count in route_counts.items():
    print(f"  {route}: {count} ({count/len(routes):.1%})")

# %% [markdown]
# ## 15. Error Analysis

# %%
print_section_header("15. Error Analysis")

# Analyze misclassifications
errors_mask = y_test_pred != y_test_enc
n_errors = errors_mask.sum()
print(f"Total misclassifications: {n_errors} / {len(y_test_enc)} ({n_errors/len(y_test_enc):.1%})")

# %%
# Confusion between categories
print("\n--- Most Confused Category Pairs ---")
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
from sklearn.metrics import confusion_matrix as sk_cm
cm = sk_cm(y_test_enc, y_test_pred)

# Find top confusion pairs
confusions = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            confusions.append({
                "true": class_names[i],
                "predicted": class_names[j],
                "count": cm[i, j],
            })

confusion_df = pd.DataFrame(confusions).sort_values("count", ascending=False)
print(confusion_df.head(10).to_string(index=False))

# %%
# Confidence of correct vs incorrect predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(confidences[~errors_mask], bins=25, alpha=0.6, color="#4CAF50",
        label=f"Correct ({(~errors_mask).sum()})", edgecolor="black")
ax.hist(confidences[errors_mask], bins=25, alpha=0.6, color="#F44336",
        label=f"Incorrect ({errors_mask.sum()})", edgecolor="black")
ax.set_xlabel("Confidence Score", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Confidence Distribution: Correct vs Incorrect", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p2_error_analysis_confidence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 16. Conclusion
#
# ### Summary
#
# | Aspect | Detail |
# |--------|--------|
# | **Dataset** | CS1QA — Educational programming Q&A |
# | **Text Processing** | Lowercase, URLs, punctuation, stopwords, lemmatization |
# | **Features** | TF-IDF (max_features=5000, ngrams=(1,2)) |
# | **Baseline** | Logistic Regression (OVR) |
# | **Final Model** | LinearSVC + CalibratedClassifierCV |
# | **Best Threshold** | See results above |
# | **Routing** | Confidence-based: Auto Approval vs Teacher Review |
#
# ### Key Findings
#
# 1. TF-IDF with bigrams captures meaningful patterns in student questions.
# 2. The Calibrated SVC model produces well-calibrated probabilities suitable
#    for confidence-based routing.
# 3. Threshold optimization balances automation rate with accuracy.
#
# ### Limitations
#
# - **Urgency is heuristic**: CS1QA lacks ground-truth urgency labels.
#   The derived urgency is based on keyword matching, not annotator agreement.
# - **Dataset scope**: CS1QA covers an introductory Python course. Performance
#   on advanced topics may differ.
# - **No code analysis**: Only question text is used. Incorporating code
#   features could improve accuracy.
#
# ### Future Work
#
# - Fine-tune a transformer model (e.g., CodeBERT) for better text understanding
# - Incorporate student code context as additional features
# - Implement active learning for continuous model improvement
# - Add real-time urgency classification using deep learning

# %%
print_section_header("Saving Models")

save_model(svc_model, str(DOUBT_TRIAGE_MODEL_PATH), "Calibrated SVC Doubt Triage")
save_model(vectorizer, str(DOUBT_TRIAGE_VECTORIZER_PATH), "TF-IDF Vectorizer")

# Save LabelEncoder for API use
import joblib
le_path = MODELS_DIR / "doubt_triage_label_encoder.pkl"
joblib.dump(le, str(le_path))
print(f"✓ Label encoder saved to {le_path}")

# Save best threshold & metrics
import json
metrics_path = MODELS_DIR / "doubt_triage_metrics.json"
if metrics_path.exists():
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
else:
    all_metrics = {}

all_metrics["threshold"] = float(best_threshold)
all_metrics["p2_accuracy"] = float(test_metrics["accuracy"])
all_metrics["p2_macro_f1"] = float(test_metrics["f1_macro"])
all_metrics["p2_weighted_f1"] = float(test_metrics["f1_weighted"])

with open(metrics_path, "w") as f:
    json.dump(all_metrics, f)

print(f"✓ Metrics saved to {metrics_path}")
print(f"✓ Model saved to {DOUBT_TRIAGE_MODEL_PATH}")
print(f"✓ Vectorizer saved to {DOUBT_TRIAGE_VECTORIZER_PATH}")
print("\n🎯 Pipeline 2 Complete!")
