# T1 — TO_Threat Preregistered Design

Preregistered before the first T1 execution.

## Control
F1 remains the historical Core champion and control model: PASS + OL + DEF + PRESS + home indicator. F1's existing state construction, hyperparameter search, and walk-forward framework remain frozen except for one known QA correction made before this T1 comparison: remove full-history centering from `OL_RUN_RAW`; training-fold normalization performs the centering instead.

## Challenger
T1 = F1 + one additional scalar feature, `TO_Threat`.

`TO_Threat` is built only from pregame historical turnover-process evidence:
- defensive interception creation rate;
- defensive forced-fumble creation rate;
- active-QB interception vulnerability;
- team fumble vulnerability.

Raw fumble recoveries are not treated as team skill. Fumble events, not fumbles lost, drive the fumble process input. The fumble component receives a fixed 0.50 possession-change value rather than learning recovery luck.

The raw matchup composite is:

`TO_RAW = INT_MATCHUP + 0.50 * FUMBLE_MATCHUP`

where:
- `INT_MATCHUP = (Home defensive INT creation + Away active-QB INT vulnerability) - (Away defensive INT creation + Home active-QB INT vulnerability)`;
- `FUMBLE_MATCHUP = (Home defensive forced-fumble creation + Away team fumble vulnerability) - (Away defensive forced-fumble creation + Home team fumble vulnerability)`.

All component states are strictly pregame EWMAs. Same-game evidence updates only after prediction. TO state uses the same half-life and offseason shrinkage chosen for F1; no new TO-specific tuning grid is added.

## Double-counting control
Within each outer walk-forward fold, `TO_RAW` is regressed on PASS, OL, DEF, and PRESS using training seasons only. The residual is standardized on training data and becomes the single `TO_Threat` feature used by T1. This is designed to prevent QB/pressure/defense information already present in F1 from receiving another vote.

## Evaluation
Chronological regular-season OOS seasons: 2020–2025.

Primary KPI: straight-up winner accuracy.

Secondary: close-game accuracy, Brier, log loss, margin MAE/RMSE, season stability, TO_Threat coefficient sign/stability, paired season-block bootstrap, and F1-vs-T1 disagreement games.

## Promotion rule
Promote T1 only if one of these is met without materially worsening Brier by more than 0.002:
1. overall SU accuracy improves by at least 0.5 percentage points over F1; or
2. close-game accuracy improves by at least 1.0 percentage point over F1.

Otherwise T1 is HOLD. No model-definition changes are allowed after observing T1 results.