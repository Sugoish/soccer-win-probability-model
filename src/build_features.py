"""
Build a game-state feature dataset from StatsBomb FIFA World Cup 2022 event data.

For every match, walk through events in chronological order, track the running
score / xG / red cards, and emit a snapshot row every time the in-match minute
advances. Each row is labeled with the match's FINAL outcome (result + score),
so the dataset can train models that predict "given the state right now, how
does this match end."
"""
import json
import os

EVENTS_DIR = "/tmp/sb_events"       # data/events/<match_id>.json from statsbomb/open-data, WC2022 (comp 43, season 106)
MATCHES_PATH = "data/matches.json"  # copied alongside data/matches/43/106.json from the same source
OUT_PATH = "data/gamestate_dataset.csv"

matches = json.load(open(MATCHES_PATH))
match_meta = {m["match_id"]: m for m in matches}

rows = []
score_check_ok = 0
score_check_bad = []

for match_id, meta in match_meta.items():
    path = os.path.join(EVENTS_DIR, f"{match_id}.json")
    if not os.path.exists(path):
        continue
    events = json.load(open(path))

    home_name = meta["home_team"]["home_team_name"]
    away_name = meta["away_team"]["away_team_name"]
    final_home = meta["home_score"]
    final_away = meta["away_score"]

    if final_home > final_away:
        result = "HOME_WIN"
    elif final_home < final_away:
        result = "AWAY_WIN"
    else:
        result = "DRAW"

    home_score = away_score = 0
    home_xg = away_xg = 0.0
    home_red = away_red = 0
    home_events = away_events = 0  # crude possession proxy

    last_key = None

    for e in events:
        if e["period"] > 4:
            # Penalty shootout: not part of the normal/extra-time score that
            # matches.json's home_score/away_score (our label) reflects.
            continue

        team_name = e.get("team", {}).get("name")
        is_home = team_name == home_name
        is_away = team_name == away_name

        etype = e["type"]["name"]

        # --- update running state ---
        if etype == "Shot":
            shot = e["shot"]
            xg = shot.get("statsbomb_xg", 0.0) or 0.0
            if is_home:
                home_xg += xg
            elif is_away:
                away_xg += xg
            if shot.get("outcome", {}).get("name") == "Goal":
                if is_home:
                    home_score += 1
                elif is_away:
                    away_score += 1
        elif etype == "Own Goal Against":
            # event's team conceded (own goal against them) -> opponent scores
            if is_home:
                away_score += 1
            elif is_away:
                home_score += 1
        elif etype == "Foul Committed":
            card = e.get("foul_committed", {}).get("card", {}).get("name")
            if card in ("Red Card", "Second Yellow"):
                if is_home:
                    home_red += 1
                elif is_away:
                    away_red += 1
        elif etype == "Bad Behaviour":
            card = e.get("bad_behaviour", {}).get("card", {}).get("name")
            if card in ("Red Card", "Second Yellow"):
                if is_home:
                    home_red += 1
                elif is_away:
                    away_red += 1

        if is_home:
            home_events += 1
        elif is_away:
            away_events += 1

        # --- snapshot once per (period, minute) change ---
        period = e["period"]
        minute = e["minute"]
        if period > 4:
            continue  # skip penalty shootout bookkeeping periods
        key = (period, minute)
        if key != last_key:
            last_key = key
            total_events = home_events + away_events
            possession_home = home_events / total_events if total_events else 0.5
            rows.append({
                "match_id": match_id,
                "home_team": home_name,
                "away_team": away_name,
                "period": period,
                "minute": minute,
                "home_score": home_score,
                "away_score": away_score,
                "score_diff": home_score - away_score,
                "home_xg": round(home_xg, 3),
                "away_xg": round(away_xg, 3),
                "xg_diff": round(home_xg - away_xg, 3),
                "home_red": home_red,
                "away_red": away_red,
                "red_diff": home_red - away_red,
                "possession_home": round(possession_home, 3),
                "final_home_score": final_home,
                "final_away_score": final_away,
                "final_result": result,
            })

    if home_score == final_home and away_score == final_away:
        score_check_ok += 1
    else:
        score_check_bad.append((match_id, home_score, away_score, final_home, final_away))

print(f"Matches processed: {len(match_meta)}")
print(f"Score reconstruction correct: {score_check_ok}/{len(match_meta)}")
if score_check_bad:
    print("MISMATCHES:")
    for b in score_check_bad:
        print("  ", b)

import csv
fieldnames = list(rows[0].keys())
with open(OUT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_PATH}")
