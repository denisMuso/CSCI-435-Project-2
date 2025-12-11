"""
CSCI 435 / DTSC 620 - Project Assignment 2
Denis Musovski
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------
# Load Dataset
# -------------
df = pd.read_csv("spam.data", sep=None, engine="python")
df = df.iloc[1:].reset_index(drop=True)
X = df.iloc[:, :-1].astype(float).values

y_raw = df.iloc[:, -1].astype(str).values
y = np.where(y_raw == "spam", 1, 0)

print("Dataset loaded successfully.")
print("X shape:", X.shape)
print("y distribution:", np.bincount(y))
# ----------------------------
# Helper evaluation function
# ----------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, label):
    """Trains model, predicts, prints metrics, and returns accuracy metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    acc_ham = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc_spam = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n=== {label} ===")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Ham Accuracy:   {acc_ham:.4f}")
    print(f"Spam Accuracy:  {acc_spam:.4f}")

    return acc, acc_ham, acc_spam

# --------------------------------------
# Define Fused Majority-Vote Classifier
# --------------------------------------
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
nb = GaussianNB()
lr = LogisticRegression(max_iter=1000, solver="lbfgs")

fused = VotingClassifier(
    estimators=[("dt", dt), ("nb", nb), ("lr", lr)],
    voting="hard"
)

# ---------------
# AdaBoost Model
# ---------------
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

# ----------------------------------------
# Random Forest Baseline (for fixed split)
# ----------------------------------------
rf = RandomForestClassifier(
    n_estimators=1000,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

# ---------------------------------
# Fixed 1000/3601 split experiment
# ---------------------------------
print("\n==============================")
print(" FIXED TRAINING SPLIT RESULTS ")
print("==============================")

X_train_fixed = X[:1000]
y_train_fixed = y[:1000]
X_test_fixed = X[1000:]
y_test_fixed = y[1000:]

fixed_fused = evaluate_model(fused, X_train_fixed, X_test_fixed, y_train_fixed, y_test_fixed, "Fused (Fixed Split)")
fixed_ada = evaluate_model(ada, X_train_fixed, X_test_fixed, y_train_fixed, y_test_fixed, "AdaBoost (Fixed Split)")
fixed_rf = evaluate_model(rf, X_train_fixed, X_test_fixed, y_train_fixed, y_test_fixed, "Random Forest (Fixed Split)")

# ------------------------------------------------------
# Training Fraction Experiments (0.5, 0.6, 0.7, 0.8)
# ------------------------------------------------------
fractions = [0.5, 0.6, 0.7, 0.8]
results = []

print("\n=====================================")
print(" TRAINING FRACTION EXPERIMENT RESULTS ")
print("=====================================")

for frac in fractions:
    print(f"\n--- Training Fraction = {frac} ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=42
    )

    fused_res = evaluate_model(fused, X_train, X_test, y_train, y_test, f"Fused (frac={frac})")
    ada_res = evaluate_model(ada, X_train, X_test, y_train, y_test, f"AdaBoost (frac={frac})")

    results.append([frac, "Fused", *fused_res])
    results.append([frac, "AdaBoost", *ada_res])

# ------------------------------------------------------
# Save results to CSV for easy use in Report
# ------------------------------------------------------
cols = ["train_frac", "classifier", "accuracy", "acc_ham", "acc_spam"]
df_results = pd.DataFrame(results, columns=cols)
df_results.to_csv("training_fraction_results.csv", index=False)

print("\nAll training-fraction results saved to training_fraction_results.csv")
print("Project 2 experiment execution complete.\n")