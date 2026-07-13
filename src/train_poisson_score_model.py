"""
Poisson expected-score model (the standard technique for soccer score
prediction, popularized by Dixon-Coles / the Poisson goals model widely used
in football analytics).

Two layers:
  1. Pre-match team strength model: fit attack/defense ratings for every
     World Cup 2022 team from final scores via a Poisson GLM (goals ~ attack
     team + defense opponent + home advantage). This gives a prematch
     expected goal rate (lambda) for each side of any matchup.
  2. Live blending: at any point in a match, blend the prematch lambda
     (regressed to remaining time) with the team's own in-match xG pace,
     to produce a live "expected final score" and, via independent Poisson
     simulation of the remaining goals, live win/draw/loss probabilities --
     a second, methodologically different probability estimate to compare
     against the gradient-boosted classifier from train_result_model.py.

Same chronological match-level train/test split as the classifier, so the
two models are evaluated on identical held-out matches.
"""
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import log_loss, mean_absolute_error

DATASET_PATH = "data/gamestate_dataset.csv"
MATCHES_PATH = "data/matches.json"
MODEL_OUT = "models/poisson_model.joblib"
METRICS_OUT = "results/poisson_model_metrics.json"

matches = json.load(open(MATCHES_PATH))
match_date = {m["match_id"]: m["match_date"] for m in matches}

df = pd.read_csv(DATASET_PATH)
df["match_date"] = df["match_id"].map(match_date)

match_order = (
    df[["match_id", "match_date"]].drop_duplicates().sort_values("match_date").reset_index(drop=True)
)
n_matches = len(match_order)
split_idx = int(n_matches * 0.8)
train_match_ids = set(match_order.loc[:split_idx - 1, "match_id"])
test_match_ids = set(match_order.loc[split_idx:, "match_id"])

# ---------------------------------------------------------------
# 1. Pre-match team attack/defense strength model (Poisson GLM),
#    fit ONLY on final scores of TRAIN matches (no leakage).
# ---------------------------------------------------------------
train_matches = [m for m in matches if m["match_id"] in train_match_ids]

long_rows = []
for m in train_matches:
    home = m["home_team"]["home_team_name"]
    away = m["away_team"]["away_team_name"]
    long_rows.append({"team": home, "opponent": away, "is_home": 1, "goals": m["home_score"]})
    long_rows.append({"team": away, "opponent": home, "is_home": 0, "goals": m["away_score"]})
long_df = pd.DataFrame(long_rows)

poisson_model = smf.glm(
    formula="goals ~ is_home + C(team) + C(opponent)",
    data=long_df,
    family=sm.families.Poisson(),
).fit()

print(poisson_model.summary().tables[0])

# Precompute prematch lambda once per (team, opponent, is_home) combo that
# actually appears in the test matches -- far cheaper than calling .predict()
# once per game-state row, and lets us cleanly fall back for any team/
# opponent the GLM never saw in training (e.g. eliminated in the group
# stage, so it never appears as a TRAIN-period opponent either).
_lambda_cache = {}

def prematch_lambda(team, opponent, is_home):
    key = (team, opponent, is_home)
    if key in _lambda_cache:
        return _lambda_cache[key]
    row = pd.DataFrame([{"team": team, "opponent": opponent, "is_home": is_home}])
    try:
        val = float(poisson_model.predict(row).iloc[0])
    except Exception:
        val = float(long_df["goals"].mean())
    _lambda_cache[key] = val
    return val

# ---------------------------------------------------------------
# 2. Live blending: expected final score at each game-state snapshot.
# ---------------------------------------------------------------
REGULATION_MINUTES = 90.0

def project_expected_score(row):
    minute = row["minute"]
    period = row["period"]
    home_score, away_score = row["home_score"], row["away_score"]
    home_xg, away_xg = row["home_xg"], row["away_xg"]

    elapsed = max(minute, 1.0)  # avoid div by zero right at kickoff
    remaining_frac = max(0.0, (REGULATION_MINUTES - minute) / REGULATION_MINUTES) if period <= 2 else 0.0

    lam_home_pre = prematch_lambda(row["home_team"], row["away_team"], 1)
    lam_away_pre = prematch_lambda(row["away_team"], row["home_team"], 0)

    # In-match observed xG pace, projected forward over remaining time.
    live_rate_home = (home_xg / elapsed) * REGULATION_MINUTES
    live_rate_away = (away_xg / elapsed) * REGULATION_MINUTES

    # Blend prematch prior with live pace; trust the live signal more as
    # the match progresses (more observed data -> less reliance on prior).
    live_weight = min(0.85, elapsed / REGULATION_MINUTES + 0.15)
    prior_weight = 1 - live_weight

    blended_remaining_home = remaining_frac * (
        prior_weight * lam_home_pre + live_weight * live_rate_home
    )
    blended_remaining_away = remaining_frac * (
        prior_weight * lam_away_pre + live_weight * live_rate_away
    )

    exp_final_home = home_score + blended_remaining_home
    exp_final_away = away_score + blended_remaining_away
    return exp_final_home, exp_final_away, blended_remaining_home, blended_remaining_away


def result_probabilities(home_score, away_score, rem_home_lambda, rem_away_lambda, max_goals=8):
    """P(home win / draw / away win) via independent-Poisson convolution of
    remaining goals on top of the current score."""
    rem_home_lambda = max(rem_home_lambda, 1e-6)
    rem_away_lambda = max(rem_away_lambda, 1e-6)
    from scipy.stats import poisson
    ks = np.arange(0, max_goals + 1)
    p_home_goals = poisson.pmf(ks, rem_home_lambda)
    p_away_goals = poisson.pmf(ks, rem_away_lambda)
    p_home_goals[-1] += max(0.0, 1 - p_home_goals.sum())
    p_away_goals[-1] += max(0.0, 1 - p_away_goals.sum())

    joint = np.outer(p_home_goals, p_away_goals)
    final_home = home_score + ks
    final_away = away_score + ks
    p_home_win = joint[np.subtract.outer(final_home, final_away) > 0].sum() if False else None
    # simpler explicit loop (small grid, fine for clarity + correctness)
    p_home_win = p_draw = p_away_win = 0.0
    for i, gh in enumerate(ks):
        for j, ga in enumerate(ks):
            p = joint[i, j]
            fh, fa = home_score + gh, away_score + ga
            if fh > fa:
                p_home_win += p
            elif fh == fa:
                p_draw += p
            else:
                p_away_win += p
    total = p_home_win + p_draw + p_away_win
    return p_home_win / total, p_draw / total, p_away_win / total


test_df = df[df["match_id"].isin(test_match_ids)].copy()

exp_home_list, exp_away_list = [], []
p_home_list, p_draw_list, p_away_list = [], [], []

for _, row in test_df.iterrows():
    exp_final_home, exp_final_away, rem_h, rem_a = project_expected_score(row)
    exp_home_list.append(exp_final_home)
    exp_away_list.append(exp_final_away)
    ph, pd_, pa = result_probabilities(row["home_score"], rov["away_score"], rem_h, rem_a)
    p_home_list.append(ph)
    p_draw_list.append(pd_)
    p_away_list.append(pa)

test_df["exp_final_home"] = exp_home_list
test_df["exp_final_away"] = exp_away_list
test_df["poisson_p_home_win"] = p_home_list
test_df["poisson_p_draw"] = p_draw_list
test_df["poisson_p_away_win"] = p_away_list

mae_home = mean_absolute_error(test_df["final_home_score"], test_df["exp_final_home"])
mae_away = mean_absolute_error(test_df["final_away_score"], test_df["exp_final_away"])

# naive baseline: "final score = current score" (no projection at all)
naive_mae_home = mean_absolute_error(test_df["final_home_score"], test_df["home_score"])
naive_mae_away = mean_absolute_error(test_df["final_away_score"], test_df["away_score"])

# log-loss of the Poisson-derived result probabilities, same test rows/labels
# used for the classifier, for a direct comparison.
class_order = ["AWAY_WIN", "DRAW", "HOME_WIN"]
proba_matrix = test_df[["poisson_p_away_win", "poisson_p_draw", "poisson_p_home_win"]].values
proba_matrix = np.clip(proba_matrix, 1e-6, 1)
proba_matrix = proba_matrix / proba_matrix.sum(axis=1, keepdims=True)
poisson_log_loss = log_loss(test_df["final_result"], proba_matrix, labels=class_order)

metrics = {
    "n_matches_train": len(train_match_ids),
    "n_matches_test": len(test_match_ids),
    "expected_score_mae": {
        "model_home_goals": mae_home,
        "model_away_goals": mae_away,
        "naive_freeze_current_score_home": naive_mae_home,
        "naive_freeze_current_score_away": naive_mae_away,
    },
    "poisson_result_probabilities": {
        "log_loss": poisson_log_loss,
    },
}
print(json.dumps(metrics, indent=2))

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(
    {"poisson_glm": poisson_model, "team_avg_goals": float(long_df["goals"].mean())},
    MODEL_OUT,
)
test_df.to_csv("results/poisson_test_predictions.csv", index=False)
print(f"Saved model to {MODEL_OUT}")
print(f"Saved metrics to {METRICS_OUT}")
print("Saved per-row test predictions to results/poisson_test_predictions.csv")
