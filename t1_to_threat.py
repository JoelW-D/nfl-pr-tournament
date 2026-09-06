#!/usr/bin/env python3
"""
T1 experiment: frozen F1 historical Core + residualized TO_Threat.

Preregistered design:
- F1 control remains PASS + OL + DEF + PRESS + home indicator.
- TO_Threat uses pregame historical process evidence only:
  defensive interception creation, defensive forced-fumble creation,
  active-QB interception vulnerability, and team fumble vulnerability.
- Fumble recovery outcomes are excluded from the skill input.
- The raw TO composite is residualized against F1's four football pillars
  using training seasons only before T1 is fitted.
- Same F1 half-life, offseason shrinkage, and QB blend are used for TO state;
  no new TO-specific hyperparameter grid is introduced.
- Primary test: 2020-2025 chronological OOS SU accuracy vs F1.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd

import tournament as base

for c in ["qb_hit", "fumble", "fumble_forced", "fumble_not_forced"]:
    if c not in base.PBP_COLS:
        base.PBP_COLS.append(c)

OUT_DIR = Path("NFL_PR_T1_Output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = base.SEED
rng = np.random.default_rng(SEED)


def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)


def build_turnover_observations(pbp: pd.DataFrame):
    c = pbp.copy()
    def zero_or_na(col):
        return c[col].isna() | (_num(c[col]) == 0)

    mask = (
        c.season.isin(base.SOURCE_SEASONS)
        & c.season_type.isin(base.GAME_TYPES)
        & zero_or_na("no_play")
        & zero_or_na("two_point_attempt")
        & zero_or_na("qb_kneel")
        & zero_or_na("qb_spike")
        & c.posteam.notna()
        & c.defteam.notna()
    )
    c = c[mask].copy()
    c["is_db"] = (_num(c.qb_dropback) == 1).astype(int)
    c["is_int"] = (_num(c.interception) == 1).astype(int)
    c["is_fumble"] = (_num(c.fumble) == 1).astype(int)
    c["is_ff"] = (_num(c.fumble_forced) == 1).astype(int)
    c["is_hit"] = (_num(c.qb_hit) == 1).astype(int)
    c["is_sack"] = (_num(c.sack) == 1).astype(int)
    c["pressure_to_event"] = ((c.is_int == 1) | (c.is_ff == 1)) & ((c.is_hit == 1) | (c.is_sack == 1))

    off_rows = []
    for (season, week, gid, team, opp), g in c.groupby(
        ["season", "week", "game_id", "posteam", "defteam"], sort=False
    ):
        off_rows.append({
            "season": season, "week": week, "game_id": gid, "team": team, "opponent": opp,
            "off_plays": len(g), "off_fumbles": int(g.is_fumble.sum()),
        })
    off = pd.DataFrame(off_rows)

    def_rows = []
    for (season, week, gid, team, opp), g in c.groupby(
        ["season", "week", "game_id", "defteam", "posteam"], sort=False
    ):
        db = int(g.is_db.sum())
        def_rows.append({
            "season": season, "week": week, "game_id": gid, "team": team, "opponent": opp,
            "def_plays": len(g), "def_db": db, "def_ints": int(g.is_int.sum()),
            "def_forced_fumbles": int(g.is_ff.sum()),
            "pressure_to_events": int(g.pressure_to_event.sum()),
        })
    de = pd.DataFrame(def_rows)
    team = off.merge(de, on=["season", "week", "game_id", "team", "opponent"], how="outer")
    team["team"] = base.canon_team(team.team.astype(str))
    team["opponent"] = base.canon_team(team.opponent.astype(str))
    team["DEF_INT_RATE_RAW"] = team.def_ints / np.maximum(team.def_db, 1)
    team["DEF_FF_RATE_RAW"] = team.def_forced_fumbles / np.maximum(team.def_plays, 1)
    team["OFF_FUM_RATE_RAW"] = team.off_fumbles / np.maximum(team.off_plays, 1)
    team["PRESSURE_TO_RATE_RAW"] = team.pressure_to_events / np.maximum(team.def_plays, 1)

    qb_rows = []
    qc = c[(c.is_db == 1) & c.passer_player_id.notna()].copy()
    for (season, week, gid, team_name, qid), g in qc.groupby(
        ["season", "week", "game_id", "posteam", "passer_player_id"], dropna=False, sort=False
    ):
        qb_rows.append({
            "season": season, "week": week, "game_id": gid, "team": team_name,
            "qb_id": str(qid), "qb_int_rate_raw": float(g.is_int.mean()), "qb_db": len(g),
        })
    qb = pd.DataFrame(qb_rows)
    return team, qb


def turnover_state_before_game(team_obs, qb_obs, games, half_life, rho, norm_train_seasons):
    decay = math.exp(math.log(.5) / half_life)
    tr = team_obs[team_obs.season.isin(list(norm_train_seasons))]
    if len(tr) < 100:
        tr = team_obs[team_obs.season < min(games.season.max(), min(norm_train_seasons) + 1)]

    cols = ["DEF_INT_RATE_RAW", "DEF_FF_RATE_RAW", "OFF_FUM_RATE_RAW", "PRESSURE_TO_RATE_RAW"]
    norm = {c: (tr[c].mean(skipna=True), base.safe_sd(tr[c])) for c in cols}
    x = team_obs.copy()
    x["dint_obs"] = base.z_apply(x.DEF_INT_RATE_RAW, *norm["DEF_INT_RATE_RAW"])
    x["dff_obs"] = base.z_apply(x.DEF_FF_RATE_RAW, *norm["DEF_FF_RATE_RAW"])
    x["ofum_obs"] = base.z_apply(x.OFF_FUM_RATE_RAW, *norm["OFF_FUM_RATE_RAW"])
    x["pto_obs"] = base.z_apply(x.PRESSURE_TO_RATE_RAW, *norm["PRESSURE_TO_RATE_RAW"])
    xlookup = {(r.game_id, r.team): r for r in x.itertuples()}

    qtr = qb_obs[qb_obs.season.isin(list(norm_train_seasons))]
    qmu = qtr.qb_int_rate_raw.mean(skipna=True)
    qsd = base.safe_sd(qtr.qb_int_rate_raw)
    q = qb_obs.copy()
    q["qbi_obs"] = base.z_apply(q.qb_int_rate_raw, qmu, qsd)
    q_by_game = {gid: g for gid, g in q.groupby("game_id")}

    team_state, team_last = {}, {}
    qb_state, qb_last = {}, {}
    rows = []

    def pull_team(tm, seas):
        st = team_state.get(tm, {"dint": 0., "dff": 0., "ofum": 0., "pto": 0.}).copy()
        ls = team_last.get(tm, seas)
        if seas > ls:
            sh = rho ** (seas - ls)
            for k in st:
                st[k] *= sh
            team_state[tm] = st.copy(); team_last[tm] = seas
        return st

    def pull_qb(qid, seas):
        if qid is None or str(qid) in ["nan", "None", ""]:
            return 0.
        qid = str(qid); st = qb_state.get(qid, 0.); ls = qb_last.get(qid, seas)
        if seas > ls:
            st *= rho ** (seas - ls); qb_state[qid] = st; qb_last[qid] = seas
        return st

    for g in games.itertuples():
        seas = int(g.season)
        hs, aws = pull_team(g.home_team, seas), pull_team(g.away_team, seas)
        hq, aq = pull_qb(g.home_qb_id, seas), pull_qb(g.away_qb_id, seas)
        rows.append({
            "game_id": g.game_id,
            "DINT_H": hs["dint"], "DINT_A": aws["dint"],
            "DFF_H": hs["dff"], "DFF_A": aws["dff"],
            "FUMV_H": hs["ofum"], "FUMV_A": aws["ofum"],
            "PTO_H": hs["pto"], "PTO_A": aws["pto"],
            "QBI_H": hq, "QBI_A": aq,
        })

        for tm in [g.home_team, g.away_team]:
            ob = xlookup.get((g.game_id, tm))
            if ob is not None:
                st = pull_team(tm, seas)
                for k, v in {"dint": ob.dint_obs, "dff": ob.dff_obs, "ofum": ob.ofum_obs, "pto": ob.pto_obs}.items():
                    if np.isfinite(v):
                        st[k] = decay * st[k] + (1 - decay) * v
                team_state[tm] = st; team_last[tm] = seas
        if g.game_id in q_by_game:
            for qr in q_by_game[g.game_id].itertuples():
                qid = str(qr.qb_id)
                if np.isfinite(qr.qbi_obs):
                    old = pull_qb(qid, seas)
                    qb_state[qid] = decay * old + (1 - decay) * qr.qbi_obs
                    qb_last[qid] = seas
    return pd.DataFrame(rows)


def add_to_features(core_ft, to_state):
    z = core_ft.merge(to_state, on="game_id", how="left")
    fill = ["DINT_H","DINT_A","DFF_H","DFF_A","FUMV_H","FUMV_A","PTO_H","PTO_A","QBI_H","QBI_A"]
    z[fill] = z[fill].fillna(0.0)
    z["TO_INT_RAW"] = (z.DINT_H + z.QBI_A) - (z.DINT_A + z.QBI_H)
    z["TO_FUM_RAW"] = (z.DFF_H + z.FUMV_A) - (z.DFF_A + z.FUMV_H)
    z["TO_RAW"] = z.TO_INT_RAW + 0.50 * z.TO_FUM_RAW
    z["PRESSURE_TO_RAW"] = z.PTO_H - z.PTO_A
    return z


def residualize_to_threat(train, test):
    cols = ["PASS", "OL", "DEF", "PRESS"]
    Xtr = np.column_stack([np.ones(len(train)), train[cols].to_numpy(float)])
    Xte = np.column_stack([np.ones(len(test)), test[cols].to_numpy(float)])
    y = train.TO_RAW.to_numpy(float)
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y)
    beta = np.linalg.pinv(Xtr[ok].T @ Xtr[ok]) @ (Xtr[ok].T @ y[ok])
    rtr = y - Xtr @ beta
    rte = test.TO_RAW.to_numpy(float) - Xte @ beta
    mu, sd = np.nanmean(rtr), base.safe_sd(rtr)
    tr, te = train.copy(), test.copy()
    tr["TO_THREAT"] = (rtr - mu) / sd
    te["TO_THREAT"] = (rte - mu) / sd
    diag = {"resid_intercept": beta[0], "resid_mu": mu, "resid_sd": sd}
    for c, b in zip(cols, beta[1:]):
        diag[f"resid_beta_{c}"] = b
        diag[f"raw_corr_{c}"] = train[["TO_RAW", c]].corr().iloc[0, 1]
        diag[f"resid_corr_{c}"] = tr[["TO_THREAT", c]].corr().iloc[0, 1]
    return tr, te, diag


def score(d, model):
    if model == "B0":
        p = d.B0_p_home.to_numpy(float); pm = d.B0_pred_margin_proxy.to_numpy(float)
        point = False; close = np.abs(p - .5) <= .06
    else:
        p = d[f"{model}_p_home"].to_numpy(float); pm = d[f"{model}_pred_margin"].to_numpy(float)
        point = True; close = np.abs(pm) <= 3
    m = d.actual_margin.to_numpy(float); dec = m != 0
    y = np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5))
    acc = np.mean((p[dec] > .5) == (m[dec] > 0))
    c = close & dec
    close_acc = np.mean((p[c] > .5) == (m[c] > 0)) if c.sum() >= 10 else np.nan
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "model": model, "games": len(d), "decided": int(dec.sum()), "su_accuracy": acc,
        "close_games": int(c.sum()), "close_accuracy": close_acc,
        "brier": np.mean((pc - y) ** 2),
        "log_loss": -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
        "margin_mae": np.mean(np.abs(m - pm)) if point else np.nan,
        "margin_rmse": np.sqrt(np.mean((m - pm) ** 2)) if point else np.nan,
    }


def bootstrap_diff(d, a, b="F1", B=5000):
    z = d[d.actual_margin.ne(0)].copy(); m = z.actual_margin.to_numpy()
    def corr(model):
        p = z.B0_p_home.to_numpy() if model == "B0" else z[f"{model}_p_home"].to_numpy()
        return ((p > .5) == (m > 0)).astype(float)
    diff = corr(a) - corr(b); seasons = np.sort(z.season.unique()); vals = []
    for _ in range(B):
        ss = rng.choice(seasons, size=len(seasons), replace=True)
        idx = np.concatenate([np.where(z.season.to_numpy() == s)[0] for s in ss])
        vals.append(diff[idx].mean())
    vals = np.asarray(vals)
    return {
        "model": a, "baseline": b, "paired_accuracy_diff": diff.mean(),
        "ci_low": np.quantile(vals, .025), "ci_high": np.quantile(vals, .975),
        "prob_diff_gt_0": np.mean(vals > 0),
    }


def main():
    print("[1/9] Loading public historical sources")
    schedules, pbp, pfr = base.load_sources()
    print("[2/9] Rebuilding corrected F1 core and TO process observations")
    games, elo_games, tg, qb = base.build_observations(schedules, pbp, pfr)
    tg["OL_RUN_RAW"] = 0.65 * tg.run_epa + 0.35 * tg.off_success
    to_team, to_qb = build_turnover_observations(pbp)

    print("[3/9] Running B0 benchmark")
    ep = base.elo_tournament(elo_games); ep = ep[ep.game_id.isin(games.game_id)]
    all_oos, coefrows, residrows = [], [], []

    print("[4/9] Walk-forward F1 control vs T1")
    for ts in base.OOS_SEASONS:
        print("  OOS", ts)
        hl, rho, qbblend = base.choose_params(ts, tg, games, qb)
        train_years = list(range(base.FULL_FEATURE_START, ts))
        core_state = base.state_before_game(tg, games, qb, hl, rho, qbblend, train_years)
        core_ft = base.make_features(core_state)
        to_state = turnover_state_before_game(to_team, to_qb, games, hl, rho, train_years)
        ft = add_to_features(core_ft, to_state)
        tr = ft[ft.season.isin(train_years)].copy(); te = ft[ft.season.eq(ts)].copy()

        s0tr, s0te, s0cf = base.fit_s0(tr, te)
        te["S0_pred_margin"] = s0te
        te["S0_p_home"] = base.fit_prob_cal(tr.actual_margin.to_numpy(), s0tr, s0te)

        f1tr, f1te, f1cf, l1 = base.fit_ridge(tr, te, ["PASS","OL","DEF","PRESS","home_ind"])
        te["F1_pred_margin"] = f1te
        te["F1_p_home"] = base.fit_prob_cal(tr.actual_margin.to_numpy(), f1tr, f1te)

        tr1, te1, rdiag = residualize_to_threat(tr, te)
        t1tr, t1te, t1cf, lt = base.fit_ridge(tr1, te1, ["PASS","OL","DEF","PRESS","TO_THREAT","home_ind"])
        te1["T1_pred_margin"] = t1te
        te1["T1_p_home"] = base.fit_prob_cal(tr1.actual_margin.to_numpy(), t1tr, t1te)

        te = te1.merge(ep[ep.season.eq(ts)], on=["game_id","season","week"], how="left")
        te["selected_half_life"] = hl; te["selected_rho"] = rho; te["selected_qb_blend"] = qbblend
        all_oos.append(te)
        for model, cf, lam in [("S0",s0cf,np.nan),("F1",f1cf,l1),("T1",t1cf,lt)]:
            for term, val in cf.items():
                coefrows.append({"season":ts,"model":model,"term":term,"coefficient":val,"lambda_":lam,"half_life":hl,"rho":rho,"qb_blend":qbblend})
        rdiag.update({"season":ts,"half_life":hl,"rho":rho,"qb_blend":qbblend})
        residrows.append(rdiag)

    oos = pd.concat(all_oos, ignore_index=True)
    coefs = pd.DataFrame(coefrows); resid_diag = pd.DataFrame(residrows)

    print("[5/9] Scoring")
    summ = pd.DataFrame([score(oos, m) for m in ["B0","S0","F1","T1"]])
    summ = summ.sort_values(["su_accuracy","close_accuracy","brier","log_loss"], ascending=[False,False,True,True]).reset_index(drop=True)
    summ.insert(0, "tournament_rank", np.arange(1, len(summ)+1))
    f1 = summ[summ.model.eq("F1")].iloc[0]
    def status(r):
        if r.model == "F1": return "HISTORICAL CORE CHAMPION / CONTROL"
        if r.model == "T1":
            if r.su_accuracy >= f1.su_accuracy + .005 and r.brier <= f1.brier + .002: return "PROMOTE TO F1"
            if r.close_accuracy >= f1.close_accuracy + .01 and r.brier <= f1.brier + .002: return "CLOSE-GAME PROMOTION CANDIDATE"
            return "HOLD"
        if r.model == "S0": return "Frozen simplicity benchmark"
        return "Benchmark"
    summ["promotion_status"] = summ.apply(status, axis=1)

    by = []
    for sy in sorted(oos.season.unique()):
        for m in ["B0","S0","F1","T1"]:
            r = score(oos[oos.season.eq(sy)], m); r["season"] = sy; by.append(r)
    by = pd.DataFrame(by)
    boot = pd.DataFrame([bootstrap_diff(oos,"T1","F1"), bootstrap_diff(oos,"F1","S0")])

    dec = oos.actual_margin.ne(0)
    disagree = oos[dec & ((oos.F1_p_home > .5) != (oos.T1_p_home > .5))].copy()
    disagree["actual_home_win"] = disagree.actual_margin > 0
    disagree["F1_home_pick"] = disagree.F1_p_home > .5
    disagree["T1_home_pick"] = disagree.T1_p_home > .5
    disagree["F1_correct"] = disagree.F1_home_pick == disagree.actual_home_win
    disagree["T1_correct"] = disagree.T1_home_pick == disagree.actual_home_win
    disagree = disagree[["game_id","season","week","home_team","away_team","actual_margin","PASS","OL","DEF","PRESS","TO_RAW","TO_THREAT","PRESSURE_TO_RAW","F1_pred_margin","T1_pred_margin","F1_p_home","T1_p_home","F1_correct","T1_correct"]]

    print("[6/9] QA")
    resid_cols = [c for c in resid_diag.columns if c.startswith("resid_corr_")]
    qa = pd.DataFrame([{
        "oos_games": len(oos), "oos_decided": int(dec.sum()), "disagreement_games": len(disagree),
        "T1_disagreement_wins": int(disagree.T1_correct.sum()) if len(disagree) else 0,
        "F1_disagreement_wins": int(disagree.F1_correct.sum()) if len(disagree) else 0,
        "mean_abs_train_resid_corr_with_F1": float(np.nanmean(np.abs(resid_diag[resid_cols].to_numpy(float)))),
        "input_uses_fumble_lost_as_TO_skill": False,
    }])
    print(qa.to_string(index=False))

    print("[7/9] Writing outputs")
    summ.to_csv(OUT_DIR/"T1_Tournament_Summary.csv", index=False)
    by.to_csv(OUT_DIR/"T1_Season_By_Season.csv", index=False)
    boot.to_csv(OUT_DIR/"T1_Paired_Bootstrap.csv", index=False)
    coefs.to_csv(OUT_DIR/"T1_WalkForward_Coefficients.csv", index=False)
    resid_diag.to_csv(OUT_DIR/"T1_Residualization_Diagnostics.csv", index=False)
    disagree.to_csv(OUT_DIR/"T1_F1_Disagreements.csv", index=False)
    qa.to_csv(OUT_DIR/"T1_QA.csv", index=False)
    oos.to_csv(OUT_DIR/"T1_OOS_Predictions.csv", index=False)

    t1 = summ[summ.model.eq("T1")].iloc[0]
    receipt = pd.DataFrame([{
        "source_seasons":"2017-2025", "oos_seasons":"2020-2025", "game_types":"REG",
        "control":"F1", "challenger":"T1_F1_plus_residualized_TO_Threat",
        "f1_su_accuracy":f1.su_accuracy, "t1_su_accuracy":t1.su_accuracy,
        "f1_close_accuracy":f1.close_accuracy, "t1_close_accuracy":t1.close_accuracy,
        "f1_brier":f1.brier, "t1_brier":t1.brier, "decision":status(t1),
        "qa_correction":"OL_RUN_RAW full-history centering removed before this comparison; all fold normalization remains training-only.",
        "design":"TO_RAW=(defensive INT creation + opponent active-QB INT vulnerability) + 0.50*(defensive forced-fumble creation + opponent team fumble vulnerability), home-minus-away; recovery outcomes excluded; train-only residualization vs PASS/OL/DEF/PRESS."
    }])
    receipt.to_csv(OUT_DIR/"T1_Receipt.csv", index=False)

    print("[8/9] Scoreboard")
    print(summ.to_string(index=False))
    print("[9/9] COMPLETE", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
