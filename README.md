# Soccer Win Probability & Expected Score Model

Two complementary models for predicting how a soccer match ends, trained on real event-level data from the 2022 FIFA World Cup:

1. **A win/draw/loss classifier** — gradient-boosted, probability-calibrated, evaluated with a chronological train/test split.
2. **A Poisson expected-goals model** — the industry-standard technique for soccer score prediction, blending each team's pre-tournament attack/defense strength with live in-match xG pace.

🔗 **[Live replay demo](https://sugoish.github.io/soccer-win-probability-model/)** — scrub through both models' live predictions for the 2022 World Cup Final (Argentina vs. France), a match neither model ever trained on.

## Data

[StatsBomb open data](https://github.com/statsbomb/open-data) — free, real, event-level data (~3,600 events per match: every pass, shot, card, and substitution, each with `statsbomb_xg` for shots) for the entire 2022 FIFA World Cup, 64 matches.

## Methodology

### Feature pipeline (`src/build_features.py`)

For each match, events are replayed in chronological order and a snapshot row is emitted every time the in-match minute advances. Each row captures the running game state — score difference, cumulative xG for each side, red-card difference, a possession proxy — and is labeled with the match's actual final result. This produces 6,555 game-state snapshots across 64 matches.

Running score is reconstructed independently from shot outcomes and own-goal events (penalty shootouts are excluded, matching how the labels are defined) and cross-checked against StatsBomb's official final scores: **64/64 matches reconstruct exactly**, which is the pipeline's own correctness check.

### Win/draw/loss classifier (`src/train_result_model.py`)

A gradient-boosted classifier trained on game-state snapshots, with isotonic probability calibration (5-fold CV).

**Split:** chronological by match date, not random — the model trains only on matches that happened before every match in the test set. With a single-elimination tournament this naturally becomes "train on the group stage, test on the knockout rounds," a harder and more honest test than shuffling rows, since it can't peek at a team's tournament run before predicting it.

| | Accuracy | Log-loss | Multiclass Brier |
|---|---|---|---|
| Calibrated model | 56.4% | 1.022 | 0.594 |
| Uncalibrated model | 58.8% | 1.235 | 0.625 |
| Majority-class baseline (always predict home win) | 46.9% | — | — |

Calibration trades a couple points of raw accuracy for meaningfully better log-loss and Brier score — the probabilities themselves are more trustworthy, which matters more than the top-1 pick for a live win-probability readout. Both beat the always-predict-the-most-common-outcome baseline by a wide margin.

### Poisson expected-score model (`src/train_poisson_score_model.py`)

The standard approach in football analytics: fit each team's attack and defense strength via a Poisson GLM (`goals ~ team + opponent + home advantage`) on final scores from the training matches, giving a pre-match expected goal rate for any matchup. Live in-match predictions blend this prior with the team's own observed xG pace so far, with the blend shifting toward the live signal as the match progresses. Remaining goals are modeled as independent Poisson draws, giving both an expected final score and win/draw/loss probabilities via direct convolution — a second, methodologically different probability estimate to sanity-check against the classifier.

| | MAE (home goals) | MAE (away goals) |
|---|---|---|
| Model (current score + projected remaining goals) | 0.77 | 0.61 |
| Naive baseline (freeze current score, predict no more goals) | 1.00 | 0.57 |

The model meaningfully beats the naive baseline on home goals; on away goals it's roughly on par, since away sides in this dataset scored infrequently enough that "no more goals" was already a strong guess. The team-strength GLM is fit on only 51 matches, so its attack/defense ratings for teams eliminated early carry real uncertainty — a known limitation of small-sample club/country strength models, worth revisiting with pooled historical data in a future iteration.

## Live replay (`docs/index.html`)

A self-contained, static HTML page (Chart.js, no backend) that replays both models' live predictions minute-by-minute through a real match — by default, the 2022 World Cup Final. Hosted for free on GitHub Pages (`docs/` on `main`), so it costs nothing to keep live and needs no server.

## Project structure

```text
soccer-win-probability-model/
├── data/
│   ├── matches.json              # StatsBomb match metadata (64 WC2022 matches)
│   └── gamestate_dataset.csv     # engineered game-state snapshots (output of build_features.py)
├── src/
│   ├── build_features.py         # raw StatsBomb events -> gamestate_dataset.csv
│   ├── train_result_model.py     # win/draw/loss classifier + calibration
│   ├── train_poisson_score_model.py  # Poisson attack/defense + live expected score
│   └── build_replay_data.py      # generates docs/replay_data.json for the live demo
├── models/                       # trained model artifacts (.joblib) -- generated locally, not committed (regenerate via src/)
├── results/                      # metrics + calibration plot
└── docs/
    ├── index.html                # live replay visualization (GitHub Pages)
    └── replay_data.json          # per-minute predictions for the demo match
```

## Running it yourself

```bash
pip install -r requirements.txt

# raw StatsBomb events aren't committed here (too large) -- fetch them first:
git clone --filter=blob:none --sparse https://github.com/statsbomb/open-data.git /tmp/sb_open_data
# then extract data/events/<match_id>.json for the WC2022 match_ids into /tmp/sb_events/

python src/build_features.py
python src/train_result_model.py
python src/train_poisson_score_model.py
python src/build_replay_data.py
```

## Technology

| Component | Choice |
|---|---|
| Data | StatsBomb open data (event-level, free) |
| Classifier | scikit-learn `GradientBoostingClassifier` + `CalibratedClassifierCV` |
| Score model | `statsmodels` Poisson GLM + `scipy.stats.poisson` |
| Visualization | Chart.js (static HTML, no backend) |
| Hosting | GitHub Pages (free) |
