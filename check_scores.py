"""Check current accuracy of both ML model pipelines."""
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from src.data_loader import load_nasa_dataset, load_cs1qa_dataset
from src.feature_engineering import engineer_software_defect_features
from src.text_processing import clean_text_column
from src.preprocessing import split_data

print("=" * 60)
print("  PIPELINE 1: Code Quality Grading (NASA KC1)")
print("=" * 60)

df1, t1 = load_nasa_dataset()
df1 = engineer_software_defect_features(df1)

fcols = [
    "LOC", "CYCLO", "LENGTH", "VOLUME", "DIFFICULTY",
    "INT_FAN_IN", "INT_FAN_OUT", "NUM_OPERATORS", "NUM_OPERANDS",
    "BRANCH_COUNT", "Complexity_per_LOC", "Branch_Density",
    "Fan_Ratio", "Complexity_x_LOC", "Halstead_per_LOC",
]

mapping = {
    "LOC": "loc", "CYCLO": "v(g)", "LENGTH": "l", "VOLUME": "v",
    "DIFFICULTY": "d", "INT_FAN_IN": "iv(g)", "INT_FAN_OUT": "ev(g)",
    "NUM_OPERATORS": "total_Op", "NUM_OPERANDS": "total_Opnd",
    "BRANCH_COUNT": "branchCount",
}
for c in fcols:
    if c not in df1.columns and c in mapping:
        df1[c] = df1.get(mapping[c], 0.0)

loc = df1["LOC"]
df1["Complexity_per_LOC"] = df1["CYCLO"] / (loc + 1e-6)
df1["Branch_Density"] = df1["BRANCH_COUNT"] / (loc + 1e-6)
df1["Fan_Ratio"] = df1["INT_FAN_IN"] / (df1["INT_FAN_OUT"] + 1e-6)
df1["Complexity_x_LOC"] = df1["CYCLO"] * loc
df1["Halstead_per_LOC"] = df1["VOLUME"] / (loc + 1e-6)

_, _, test1 = split_data(df1, t1)
X_te1 = test1[fcols].values
y_te1 = test1[t1].values

m1 = joblib.load("models/code_grading_model.pkl")
y_pred1 = m1.predict(X_te1)
y_prob1 = m1.predict_proba(X_te1)[:, 1]

print(f"  Test Accuracy:  {accuracy_score(y_te1, y_pred1) * 100:.2f}%")
print(f"  ROC-AUC:        {roc_auc_score(y_te1, y_prob1):.4f}")
print(f"  Precision:      {precision_score(y_te1, y_pred1) * 100:.2f}%")
print(f"  Recall:         {recall_score(y_te1, y_pred1) * 100:.2f}%")
print(f"  F1-Score:       {f1_score(y_te1, y_pred1) * 100:.2f}%")

print()
print("=" * 60)
print("  PIPELINE 2: Student Doubt Triage (CS1QA)")
print("=" * 60)

df2, t2 = load_cs1qa_dataset()
df2 = clean_text_column(df2, "question", "cleaned_text")
_, _, test2 = split_data(df2, t2)

vec = joblib.load("models/doubt_triage_vectorizer.pkl")
le = joblib.load("models/doubt_triage_label_encoder.pkl")
m2 = joblib.load("models/doubt_triage_model.pkl")

X_te2 = vec.transform(test2["cleaned_text"])
y_te2 = le.transform(test2[t2])
y_pred2 = m2.predict(X_te2)

print(f"  Test Accuracy:  {accuracy_score(y_te2, y_pred2) * 100:.2f}%")
print(f"  Macro F1:       {f1_score(y_te2, y_pred2, average='macro') * 100:.2f}%")
print(f"  Weighted F1:    {f1_score(y_te2, y_pred2, average='weighted') * 100:.2f}%")

print()
print("=" * 60)
print("  ALL MODELS VERIFIED SUCCESSFULLY!")
print("=" * 60)
