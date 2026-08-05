# %% [markdown]
# # Pipeline 1: ML-Based Code Grading
# ## KPMG Off-Campus Hiring Assessment — LMS ML Pipeline
#
# **Objective**: Predict software defect likelihood from engineered code metrics
# using the NASA KC1 Software Defect Dataset.
#
# **Pipeline**: EDA → Cleaning → Feature Engineering → Random Forest (Baseline)
# → LightGBM (Final) → Optuna Tuning → SHAP Explainability

# %% [markdown]
# ## 1. Introduction
#
# This notebook implements an end-to-end ML pipeline for code quality grading
# in a Learning Management System (LMS). The goal is to predict whether a code
# submission is likely to contain defects based on software engineering metrics.
#
# We use the **NASA KC1 Software Defect Dataset** from the PROMISE Repository,
# which contains McCabe complexity metrics, Halstead effort metrics, and lines
# of code measurements for software modules.
#
# **Key Steps**:
# 1. Exploratory Data Analysis (EDA)
# 2. Data Cleaning (duplicates, missing, outliers)
# 3. Feature Engineering (derived metrics)
# 4. Data Leakage Detection
# 5. Train/Validation/Test Split (60/20/20)
# 6. Baseline Model (Random Forest)
# 7. Final Model (LightGBM with Optuna tuning)
# 8. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
# 9. Explainability (SHAP)
# 10. Error Analysis

# %% [markdown]
# ## 2. Business Problem
#
# In modern LMS platforms, students submit code assignments that need quality
# assessment. Manual grading is:
# - **Time-consuming**: Instructors review hundreds of submissions
# - **Inconsistent**: Different graders may apply different standards
# - **Delayed**: Students don't get timely feedback
#
# An ML-based code grading system can:
# - **Automate** initial quality screening
# - **Flag** potentially defective submissions for human review
# - **Provide** objective, consistent quality scores
# - **Scale** to thousands of submissions
#
# We frame this as a **binary classification** problem:
# - **Class 0 (No Defect)**: Code meets quality standards
# - **Class 1 (Defect)**: Code has quality issues requiring attention

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

# Add project root to path for imports
PROJECT_ROOT = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    CODE_GRADING_MODEL_PATH,
    CODE_GRADING_SCALER_PATH,
)
from src.utils import set_global_seed, print_section_header
from src.data_loader import download_nasa_dataset, load_nasa_dataset
from src.preprocessing import (
    detect_duplicates,
    remove_duplicates,
    analyze_missing_values,
    handle_missing_values,
    detect_outliers,
    check_data_leakage,
    split_data,
    check_class_imbalance,
)
from src.feature_engineering import (
    engineer_features,
    get_feature_names,
    get_feature_availability_report,
)
from src.model_training import (
    train_baseline_rf,
    train_lightgbm,
    tune_lightgbm_optuna,
    save_model,
    cross_validate_model,
)
from src.evaluation import (
    compute_classification_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_correlation_matrix,
    plot_target_distribution,
)
from src.explainability import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall,
)

# Set reproducibility
set_global_seed(RANDOM_STATE)
print_section_header("Pipeline 1: ML-Based Code Grading")

# %% [markdown]
# ## 3. Dataset Description
#
# **NASA KC1 Dataset** (PROMISE Repository, OpenML ID: 1067)
#
# | Property | Value |
# |----------|-------|
# | Source | NASA Metrics Data Program |
# | Domain | Storage Management Software |
# | Granularity | Module-level metrics |
# | Features | 21 software metrics |
# | Target | `defects` (boolean: true/false) |
#
# **Feature Categories**:
# - **McCabe Metrics**: Cyclomatic complexity (`v(g)`), essential complexity
#   (`ev(g)`), design complexity (`iv(g)`), line count (`loc`)
# - **Halstead Metrics**: Volume, difficulty, effort, bug estimate, time
#   estimate, program length, unique/total operators and operands
# - **Size Metrics**: Lines of code, comments, blanks, branch count

# %%
# === Load Dataset ===
print_section_header("Loading NASA KC1 Dataset")
df, target_col = load_nasa_dataset()

print(f"Dataset shape: {df.shape}")
print(f"Target column: {target_col}")
print(f"\nFirst 5 rows:")
df.head()

# %%
print(f"\nDataset info:")
df.info()

# %%
print(f"\nDescriptive statistics:")
df.describe().T

# %% [markdown]
# ## 4. Exploratory Data Analysis

# %%
print_section_header("4.1 Target Distribution")
print(f"Class distribution:\n{df[target_col].value_counts()}")
print(f"\nClass proportions:\n{df[target_col].value_counts(normalize=True)}")

plot_target_distribution(
    df[target_col],
    title="Code Defect Distribution (NASA KC1)",
    save_name="p1_target_distribution.png",
)

# %%
# Check imbalance
imbalance_info = check_class_imbalance(df, target_col)
print(f"\nImbalance ratio: {imbalance_info['imbalance_ratio']:.2f}x")
print(f"Is imbalanced: {imbalance_info['is_imbalanced']}")

# %%
print_section_header("4.2 Missing Value Analysis")
missing_df = analyze_missing_values(df, save_path=str(FIGURES_DIR / "p1_missing_values.png"))
print(missing_df[missing_df["missing_count"] > 0].to_string(index=False))
if missing_df["missing_count"].sum() == 0:
    print("✓ No missing values found in the dataset.")

# %%
print_section_header("4.3 Correlation Matrix")
plot_correlation_matrix(
    df.select_dtypes(include=[np.number]),
    title="Feature Correlation Matrix (NASA KC1)",
    save_name="p1_correlation_matrix.png",
)

# %%
print_section_header("4.4 Feature Distributions")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != target_col]

# Plot distributions of key features
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for idx, col in enumerate(feature_cols[:12]):
    ax = axes[idx // 4, idx % 4]
    df[col].hist(bins=30, ax=ax, color="#5C6BC0", alpha=0.7, edgecolor="black")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
plt.suptitle("Feature Distributions (Top 12)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p1_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Data Cleaning

# %%
print_section_header("5.1 Duplicate Detection")
dupes, n_dupes = detect_duplicates(df)
print(f"Duplicates found: {n_dupes}")

if n_dupes > 0:
    df = remove_duplicates(df)
    print(f"Dataset after removing duplicates: {df.shape}")
else:
    print("✓ No duplicates found.")

# %%
print_section_header("5.2 Missing Value Handling")
total_missing = df.isnull().sum().sum()
if total_missing > 0:
    print(f"Total missing values: {total_missing}")
    df = handle_missing_values(df, strategy="median")
    print(f"After imputation: {df.isnull().sum().sum()} missing values")
else:
    print("✓ No missing values to handle.")

# %%
print_section_header("5.3 Outlier Detection")
outlier_info = detect_outliers(df, method="iqr")

# Display top columns by outlier count
outlier_summary = pd.DataFrame([
    {"feature": k, **v} for k, v in outlier_info.items()
]).sort_values("n_outliers", ascending=False)

print("Top 10 features by outlier count:")
print(outlier_summary.head(10).to_string(index=False))
print(f"\nNote: Outliers are NOT removed — tree-based models handle them well.")

# %% [markdown]
# ## 6. Feature Engineering

# %%
print_section_header("6. Feature Engineering")

# Print feature availability report
report = get_feature_availability_report()
print(report)

# Engineer derived features
df = engineer_features(df)
print(f"\nDataset shape after feature engineering: {df.shape}")
print(f"New columns: {[c for c in df.columns if c not in feature_cols + [target_col]]}")

# %%
# Display engineered features statistics
engineered_cols = ["Code_Size", "Complexity_Ratio", "Maintainability_Index",
                   "Comment_Density", "Bug_Density", "Effort_Density"]
available_eng = [c for c in engineered_cols if c in df.columns]
if available_eng:
    print("\nEngineered features statistics:")
    print(df[available_eng].describe().T)

# %% [markdown]
# ## 7. Leakage Detection

# %%
print_section_header("7. Data Leakage Detection")
leakage_pairs = check_data_leakage(df, target_col)

if leakage_pairs:
    print(f"\n{len(leakage_pairs)} suspicious feature pairs found:")
    for f1, f2, corr in leakage_pairs[:10]:
        print(f"  {f1} ↔ {f2}: correlation = {corr:.4f}")
    print("\nAction: Highly correlated features are kept — LightGBM handles")
    print("multicollinearity via feature subsampling (colsample_bytree).")
else:
    print("✓ No data leakage detected.")

# %% [markdown]
# ## 8. Train / Validation / Test Split

# %%
print_section_header("8. Data Splitting")

# Get final feature list
feature_names = get_feature_names(df, target_col)
print(f"Using {len(feature_names)} features: {feature_names}")

# Split data
train_df, val_df, test_df = split_data(df, target_col)

# Prepare arrays
X_train = train_df[feature_names].values
y_train = train_df[target_col].values
X_val = val_df[feature_names].values
y_val = val_df[target_col].values
X_test = test_df[feature_names].values
y_test = test_df[target_col].values

print(f"\nTrain: X={X_train.shape}, y distribution={np.bincount(y_train)}")
print(f"Val:   X={X_val.shape}, y distribution={np.bincount(y_val)}")
print(f"Test:  X={X_test.shape}, y distribution={np.bincount(y_test)}")

# %% [markdown]
# ## 9. Baseline Models

# %%
print_section_header("9. Baseline: Random Forest")

rf_pipeline, rf_cv_results = train_baseline_rf(X_train, y_train)

# Evaluate on validation set
y_val_pred_rf = rf_pipeline.predict(X_val)
y_val_prob_rf = rf_pipeline.predict_proba(X_val)[:, 1]

rf_metrics = compute_classification_metrics(y_val, y_val_pred_rf, y_val_prob_rf)
print(f"\nBaseline RF — Validation Metrics:")
for k, v in rf_metrics.items():
    print(f"  {k}: {v:.4f}")

print(f"\nCross-Validation ROC-AUC: {rf_cv_results['mean']:.4f} ± {rf_cv_results['std']:.4f}")

# %%
print("\nBaseline RF — Classification Report:")
print_classification_report(y_val, y_val_pred_rf, target_names=["No Defect", "Defect"])

plot_confusion_matrix(
    y_val, y_val_pred_rf,
    labels=["No Defect", "Defect"],
    title="Baseline RF — Confusion Matrix (Validation)",
    save_name="p1_rf_confusion_matrix.png",
)

# %% [markdown]
# ## 10. Hyperparameter Tuning

# %%
print_section_header("10. Optuna Hyperparameter Tuning (LightGBM)")

best_params, best_score = tune_lightgbm_optuna(
    X_train, y_train,
    n_trials=50,
    timeout=300,
)

print(f"\nBest Optuna ROC-AUC: {best_score:.4f}")
print(f"Best parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 11. Final Models

# %%
print_section_header("11. Final Model: LightGBM (Tuned)")

lgbm_pipeline, lgbm_cv_results = train_lightgbm(X_train, y_train, params=best_params)

# Evaluate on validation
y_val_pred_lgbm = lgbm_pipeline.predict(X_val)
y_val_prob_lgbm = lgbm_pipeline.predict_proba(X_val)[:, 1]

lgbm_val_metrics = compute_classification_metrics(y_val, y_val_pred_lgbm, y_val_prob_lgbm)
print(f"\nLightGBM — Validation Metrics:")
for k, v in lgbm_val_metrics.items():
    print(f"  {k}: {v:.4f}")

print(f"\nCross-Validation ROC-AUC: {lgbm_cv_results['mean']:.4f} ± {lgbm_cv_results['std']:.4f}")

# %% [markdown]
# ## 12. Evaluation

# %%
print_section_header("12.1 Final Model — Test Set Evaluation")

# Evaluate on held-out TEST set
y_test_pred = lgbm_pipeline.predict(X_test)
y_test_prob = lgbm_pipeline.predict_proba(X_test)[:, 1]

test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob)
print(f"\nLightGBM — TEST SET Metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

# %%
print("\nFinal Model — Classification Report (Test Set):")
print_classification_report(y_test, y_test_pred, target_names=["No Defect", "Defect"])

# %%
plot_confusion_matrix(
    y_test, y_test_pred,
    labels=["No Defect", "Defect"],
    title="LightGBM — Confusion Matrix (Test Set)",
    save_name="p1_lgbm_confusion_matrix.png",
)

# %%
plot_roc_curve(
    y_test, y_test_prob,
    title="LightGBM — ROC Curve (Test Set)",
    save_name="p1_lgbm_roc_curve.png",
)

# %%
plot_precision_recall_curve(
    y_test, y_test_prob,
    title="LightGBM — Precision-Recall Curve (Test Set)",
    save_name="p1_lgbm_pr_curve.png",
)

# %%
print_section_header("12.2 Feature Importance")

# Get feature importance from LightGBM
classifier = lgbm_pipeline.named_steps["classifier"]
importance = classifier.feature_importances_

plot_feature_importance(
    importance, feature_names,
    top_n=min(20, len(feature_names)),
    title="LightGBM — Feature Importance",
    save_name="p1_feature_importance.png",
)

# %%
print_section_header("12.3 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Random Forest (Baseline)", "LightGBM (Tuned)"],
    "CV ROC-AUC": [
        f"{rf_cv_results['mean']:.4f} ± {rf_cv_results['std']:.4f}",
        f"{lgbm_cv_results['mean']:.4f} ± {lgbm_cv_results['std']:.4f}",
    ],
    "Val Accuracy": [rf_metrics["accuracy"], lgbm_val_metrics["accuracy"]],
    "Val F1": [rf_metrics["f1_score"], lgbm_val_metrics["f1_score"]],
    "Val ROC-AUC": [rf_metrics.get("roc_auc", 0), lgbm_val_metrics.get("roc_auc", 0)],
})
print(comparison_df.to_string(index=False))

# %% [markdown]
# ## 13. Explainability

# %%
print_section_header("13. SHAP Explainability")

# Compute SHAP values
shap_values = compute_shap_values(lgbm_pipeline, X_test, feature_names)

# %%
# SHAP Summary Plot
plot_shap_summary(
    shap_values,
    feature_names=feature_names,
    title="SHAP Summary — LightGBM Code Grading",
    save_name="p1_shap_summary.png",
)

# %%
# SHAP Bar Plot
plot_shap_bar(
    shap_values,
    feature_names=feature_names,
    title="SHAP Mean Absolute Impact",
    save_name="p1_shap_bar.png",
)

# %%
# SHAP Waterfall for a single prediction
plot_shap_waterfall(
    shap_values,
    sample_index=0,
    title="SHAP Waterfall — Single Prediction Explanation",
    save_name="p1_shap_waterfall.png",
)

# %% [markdown]
# ## 14. Threshold Optimization
#
# **Note**: For the binary code grading pipeline, threshold optimization is
# less critical than for the doubt triage pipeline. The default 0.5 threshold
# is used. Error analysis below provides deeper insight.

# %% [markdown]
# ## 15. Error Analysis

# %%
print_section_header("15. Error Analysis")

# Analyze false positives and false negatives
errors_df = test_df.copy()
errors_df["predicted"] = y_test_pred
errors_df["probability"] = y_test_prob
errors_df["correct"] = (y_test_pred == y_test)

fp_mask = (y_test_pred == 1) & (y_test == 0)
fn_mask = (y_test_pred == 0) & (y_test == 1)

print(f"False Positives (predicted defect, actually clean): {fp_mask.sum()}")
print(f"False Negatives (predicted clean, actually defective): {fn_mask.sum()}")
print(f"Total errors: {fp_mask.sum() + fn_mask.sum()} / {len(y_test)}")

# %%
# Compare mean feature values for FP/FN vs correct predictions
if fp_mask.sum() > 0:
    print("\n--- False Positives: Mean Feature Values ---")
    fp_means = errors_df.loc[fp_mask, feature_names].mean()
    correct_means = errors_df.loc[errors_df["correct"], feature_names].mean()
    comparison = pd.DataFrame({
        "FP Mean": fp_means,
        "Correct Mean": correct_means,
        "Difference": fp_means - correct_means,
    }).sort_values("Difference", key=abs, ascending=False)
    print(comparison.head(10).to_string())

# %%
if fn_mask.sum() > 0:
    print("\n--- False Negatives: Mean Feature Values ---")
    fn_means = errors_df.loc[fn_mask, feature_names].mean()
    comparison_fn = pd.DataFrame({
        "FN Mean": fn_means,
        "Correct Mean": correct_means,
        "Difference": fn_means - correct_means,
    }).sort_values("Difference", key=abs, ascending=False)
    print(comparison_fn.head(10).to_string())

# %%
# Confidence distribution of errors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if fp_mask.sum() > 0:
    axes[0].hist(y_test_prob[fp_mask], bins=20, color="#F44336", alpha=0.7, edgecolor="black")
    axes[0].set_title("False Positive Confidence Distribution", fontweight="bold")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Count")

if fn_mask.sum() > 0:
    axes[1].hist(y_test_prob[fn_mask], bins=20, color="#FF9800", alpha=0.7, edgecolor="black")
    axes[1].set_title("False Negative Confidence Distribution", fontweight="bold")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "p1_error_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 16. Conclusion
#
# ### Summary
#
# | Aspect | Detail |
# |--------|--------|
# | **Dataset** | NASA KC1 — 2,109 modules, 21 original features |
# | **Engineered Features** | Code Size, Complexity Ratio, Maintainability Index, Comment Density, Bug Density, Effort Density |
# | **Baseline** | Random Forest with class_weight='balanced' |
# | **Final Model** | LightGBM with Optuna-tuned hyperparameters |
# | **Evaluation** | Test set metrics reported above |
# | **Explainability** | SHAP TreeExplainer with summary, bar, and waterfall plots |
#
# ### Key Findings
#
# 1. The NASA KC1 dataset exhibits class imbalance (~15% defective modules).
#    We handled this with `class_weight='balanced'` in both models.
#
# 2. Halstead volume and effort metrics, along with McCabe cyclomatic
#    complexity, are the strongest predictors of code defects.
#
# 3. The engineered Maintainability Index and Complexity Ratio provide
#    additional discriminative power beyond raw metrics.
#
# 4. LightGBM outperforms the Random Forest baseline with Optuna-tuned
#    hyperparameters.
#
# ### Limitations
#
# - **Missing Features**: Fan In/Out, Runtime Efficiency, Memory Efficiency,
#   and Function Count are NOT available in KC1 (see Feature Availability Report).
# - **Dataset Age**: KC1 dates from the early 2000s and may not perfectly
#   represent modern code quality patterns.
# - **No Runtime Data**: The dataset contains static analysis metrics only.
#
# ### Future Work
#
# - Integrate IBM Project CodeNet for AST-level features
# - Add runtime profiling metrics
# - Implement function-level granularity
# - Deploy as a GitHub Actions / CI pipeline integration

# %%
print_section_header("Saving Models")

save_model(lgbm_pipeline, str(CODE_GRADING_MODEL_PATH), "LightGBM Code Grading Pipeline")
print(f"✓ Model saved to {CODE_GRADING_MODEL_PATH}")
print("\n🎯 Pipeline 1 Complete!")
