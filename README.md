# NFL PR Historical Model Tournament

Frozen historical tournament for the NFL Power Ratings project.

Models:
- B0: NFLAnalytic-style Elo benchmark
- S0: frozen Simple Core (40% PASS, 25% OL, 20% DEF, 15% PRESS)
- F1: ridge-fitted core using PASS, OL, DEF, PRESS
- F2: F1 plus nonlinear trench-mismatch / dictation interaction

Primary objective: straight-up winner accuracy. Close-game accuracy is the primary subgroup; Brier and log loss are probability-quality guardrails. Market lines are never used as model inputs.

The GitHub Actions workflow downloads public nflverse historical data, executes a chronological 2020–2025 out-of-sample tournament, and uploads the result CSVs as an artifact.