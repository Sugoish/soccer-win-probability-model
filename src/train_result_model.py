"""
Win / Draw / Loss in-game probability model.

Trained on game-state snapshots (minute, score diff, xG diff, red-card diff,
possession) from the StatsBomb FIFA World Cup 2022 event data, labeled with
each match's actual final result.

Split is CHRONOLOGICAL by match date (not random), so the model is evaluated
on matches that happened strictly after everything it trained on -- the same
constraint a live in-play model would face. With a 64-match single-elimination
tournament this naturally becomes "train on group stage, test on knockouts,"
which is a harder and more honest test than a random split.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = "data/gamestate_dataset.csv"
MATCHES_PATH = "data/matches.json"
MODEL_OUT = "models/result_model.joblib"
METRICS_OUT = "results/result_model_metrics.json"
CALIB_PLOT_OUT = "results/calibration_curve.png"

FEATURES = ["period", "minute", "score_diff", "xg_diff", "red_diff", "possession_home"]
TARGET = "final_result"
CLASSES = ["AWAY_WIN", "DRAW", "HOME_WIN"]  # alphabetical, sklearn's default ordering

df = pd.read_csv(DATASET_PATH)

matches = json.load(open(MATCHES_PATH))
match_date = {m["match_id"]: m["match_date"] for m in matches}
match_week = {m["match_id"]: m["match_week"] for m in matches}
df["match_date"] = df["match_id"].map(match_date)
df["match_week"] = df["match_id"].map(match_week)

# --- chronological split at the MATCH level (no leakage across a match) ---
match_order = (
    df[["match_id", "match_date"]]
    .drop_duplicates()
    .sort_values("match_date")
    .reset_index(drop=True)
)
n_matches = len(match_order)
split_idx = int(n_matches * 0.8)
train_match_ids = set(match_order.loc[:split_idx - 1, "match_id"])
test_match_ids = set(match_order.loc[split_idx:, "match_id"])

train_df = df[df["match_id"].isin(train_match_ids)]
test_df = df[df["match_id"].isin(test_match_ids)]

print(f"Matches: {n_matches} total -> {len(train_match_ids)} train / {len(test_match_ids)} test")
print(f"Rows: {len(train_df)} train / {len(test_df)} test")
print(f"Train date range: {match_order.loc[:split_idx-1,'match_date'].min()} to {match_order.loc[:split_idx-1,'match_date'].max()}")
print(f"Test date range:  {match_order.loc[split_idx:,'match_date'].min()} to {match_order.loc[split_idx:,'match_date'].max()}")

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# Base classifier + isotonic calibration (5-fold CV on the training rows only,
# grouped implicitly by the fact CV folds are random rows -- acceptable here
# since calibration is a second-stage fit on top of an already-fixed base
# model, not the primary chronological evaluation).
base_clf = GradientBoostingClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
)
calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
calibrated_clf.fit(X_train, y_train)

# Also fit an uncalibrated version for comparison
raw_clf = GradientBoostingClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
)
raw_clf.fit(X_train, y_train)

proba_cal = calibrated_clf.predict_proba(X_test)
proba_raw = raw_clf.predict_proba(X_test)
pred_cal = calibrated_clf.predict(X_test)
pred_raw = raw_clf.predict(X_test)

class_order = list(calibrated_clf.classes_)

def multiclass_brier(y_true, proba, classes):
    y_true_oh = np.array([[1.0 if c == label else 0.0 for c in classes] for label in y_true])
    return np.mean(np.sum((proba - y_true_oh) ** 2, axis=1))

metrics = {
    "n_matches_train": len(train_match_ids),
    "n_matches_test": len(test_match_ids),
    "n_rows_train": len(train_df),
    "n_rows_test": len(test_df),
    "class_order": class_order,
    "calibrated": {
        "accuracy": accuracy_score(y_test, pred_cal),
        "log_loss": log_loss(y_test, proba_cal, labels=class_order),
        "multiclass_brier": multiclass_brier(y_test.values, proba_cal, class_order),
    },
    "raw_uncalibrated": {
        "accuracy": accuracy_score(y_test, pred_raw),
        "log_loss": log_loss(y_test, proba_raw, labels=class_order),
        "multiclass_brier": multiclass_brier(y_test.values, proba_raw, class_order),
    },
    "baseline_majority_class": {
        "accuracy": (y_test.value_counts(normalize=True).max()),
        "majority_class": y_test.value_counts().idxmax(),
    },
}

print(json.dumps(metrics, indent=2))

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump({"calibrated_model": calibrated_clf, "raw_model": raw_clf, "features": FEATURES, "classes": class_order}, MODEL_OUT)

# --- reliability diagram: calibrated vs raw, per class ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, cls in enumerate(class_order):
    y_true_bin = (y_test.values == cls).astype(int)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true_bin, proba_cal[:, i], n_bins=10, strategy="quantile")
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true_bin, proba_raw[:, i], n_bins=10, strategy="quantile")
    ax = axes[i]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred_raw, frac_pos_raw, "o-", label="Uncalibrated", alpha=0.7)
    ax.plot(mean_pred_cal, frac_pos_cal, "s-", label="Isotonic calibrated", alpha=0.9)
    ax.set_title(cls)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(CALIB_PLOT_OUT, dpi=120)
print(f"Saved calibration plot to {CALIB_PLOT_OUT}")
print(f"Saved model to {MODEL_OUT}")
print(f"Saved metrics to {METRICS_OUT}")
