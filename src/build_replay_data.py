"""
Build the per-minute replay dataset for one held-out match (the 2022 World
Cup Final, Argentina vs France, match_id 3869685) combining:
  - the calibrated win/draw/loss classifier's live probabilities
  - the Poisson model's live expected score + win/draw/loss probabilities

Output is a single JSON file consumed directly by the replay.html visualization.
This match was in the TEST split for both models (test matches start 2022-12-04),
so every number shown is a genuine out-of-sample prediction, not a fit to
data the models were trained on.
"""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

MATCH_ID = 3869685
DATASET_PATH = "data/gamestate_dataset.csv"
MATCHES_PATH = "data/matches.json"
RESULT_MODEL_PATH = "models/result_model.joblib"
POISSON_MODEL_PATH = "models/poisson_model.joblib"
OUT_PATH = "docs/replay_data.json"

df = pd.read_csv(DATASET_PATH)
match_df = df[df["match_id"] == MATCH_ID].sort_values(["period", "minute"]).reset_index(drop=True)

matches = json.load(open(MATCHES_PATH))
meta = next(m for m in matches if m["match_id"] == MATCH_ID)
home_name = meta["home_team"]["home_team_name"]
away_name = meta["away_team"]["away_team_name"]

# --- classifier ---
result_bundle = joblib.load(RESULT_MODEL_PATH)
clf = result_bundle["calibrated_model"]
FEATURES = result_bundle["features"]
classes = list(clf.classes_)  # [AWAY_WIN, DRAW, HOME_WIN]

clf_proba = clf.predict_proba(match_df[FEATURES])
idx_away = classes.index("AWAY_WIN")
idx_draw = classes.index("DRAW")
idx_home = classes.index("HOME_WIN")

# --- poisson model ---
poisson_bundle = joblib.load(POISSON_MODEL_PATH)
poisson_model = poisson_bundle["poisson_glm"]
team_avg_goals = poisson_bundle["team_avg_goals"]

_lambda_cache = {}
def prematch_lambda(team, opponent, is_home):
    key = (team, opponent, is_home)
    if key in _lambda_cache:
        return _lambda_cache[key]
    row = pd.DataFrame([{"team": team, "opponent": opponent, "is_home": is_home}])
    try:
        val = float(poisson_model.predict(row).iloc[0])
    except Exception:
        val = team_avg_goals
    _lambda_cache[key] = val
    return val

REGULATION_MINUTES = 90.0

def project_expected_score(row):
    minute, period = row["minute"], row["period"]
    home_score, away_score = row["home_score"], row["away_score"]
    home_xg, away_xg = row["home_xg"], row["away_xg"]
    elapsed = max(minute, 1.0)
    remaining_frac = max(0.0, (REGULATION_MINUTES - minute) / REGULATION_MINUTES) if period <= 2 else 0.15

    lam_home_pre = prematch_lambda(home_name, away_name, 1)
    lam_away_pre = prematch_lambda(away_name, home_name, 0)
    live_rate_home = (home_xg / elapsed) * REGULATION_MINUTES
    live_rate_away = (away_xg / elapsed) * REGULATION_MINUTES

    live_weight = min(0.85, elapsed / REGULATION_MINUTES + 0.15)
    prior_weight = 1 - live_weight

    blended_remaining_home = remaining_frac * (prior_weight * lam_home_pre + live_weight * live_rate_home)
    blended_remaining_away = remaining_frac * (prior_weight * lam_away_pre + live_weight * live_rate_away)
    return blended_remaining_home, blended_remaining_away


def result_probabilities(home_score, away_score, rem_home, rem_away, max_goals=8):
    rem_home = max(rem_home, 1e-6)
    rem_away = max(rem_away, 1e-6)
    ks = np.arange(0, max_goals + 1)
    ph = sp_poisson.pmf(ks, rem_home)
    pa = sp_poisson.pmf(ks, rem_away)
    ph[-1] += max(0.0, 1 - ph.sum())
    pa[-1] += max(0.0, 1 - pa.sum())
    joint = np.outer(ph, pa)
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
    tot = p_home_win + p_draw + p_away_win
    return p_home_win / tot, p_draw / tot, p_away_win / tot


records = []
for i, row in match_df.iterrows():
    rem_h, rem_a = project_expected_score(row)
    p_home, p_draw, p_away = result_probabilities(row["home_score"], row["away_score"], rem_h, rem_a)
    records.append({
        "period": int(row["period"]),
        "minute": int(row["minute"]),
        "home_score": int(row["home_score"]),
        "away_score": int(row["away_score"]),
        "home_xg": round(float(row["home_xg"]), 3),
        "away_xg": round(float(row["away_xg"]), 3),
        "home_red": int(row["home_red"]),
        "away_red": int(row["away_red"]),
        "clf_p_home_win": round(float(clf_proba[i, idx_home]), 4),
        "clf_p_draw": round(float(clf_proba[i, idx_draw]), 4),
        "clf_p_away_win": round(float(clf_proba[i, idx_away]), 4),
        "poisson_p_home_win": round(float(p_home), 4),
        "poisson_p_draw": round(float(p_draw), 4),
        "poisson_p_away_win": round(float(p_away), 4),
        "poisson_exp_final_home": round(row["home_score"] + rem_h, 2),
        "poisson_exp_final_away": round(row["away_score"] + rem_a, 2),
    })

output = {
    "match_id": MATCH_ID,
    "home_team": home_name,
    "away_team": away_name,
    "competition_stage": meta["competition_stage"]["name"],
    "match_date": meta["match_date"],
    "final_home_score": meta["home_score"],
    "final_away_score": meta["away_score"],
    "note": "Argentina won 4-2 on penalties after 3-3 after extra time; the models here only predict regulation/extra-time goals (not the shootout), matching how the training labels were defined.",
    "frames": records,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {len(records)} frames to {OUT_PATH}")
print(f"{home_name} {meta['home_score']} - {meta['away_score']} {away_name}")
