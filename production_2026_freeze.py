#!/usr/bin/env python3
"""Freeze the 2026 F1 numerical fit and Week 1 pregame baseline.

This does NOT change the F1 architecture. It uses the authoritative corrected
OL_RUN_RAW construction, strict minimum-CV ridge lambda, train-only probability
calibration, 2025 posterior -> offseason shrinkage -> expected 2026 personnel.

The personnel bridge is deliberately coefficient-free: each expected starter
carries the already-earned, offseason-shrunk 2025 unit state of his prior NFL
team into the relevant 2026 unit. Rookies/players without a 2025 origin receive
league-mean (0) priors. QB remains player-specific exactly as in F1.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import tournament as t

OUT = Path("NFL_PR_2026_Freeze_Output")
OUT.mkdir(exist_ok=True, parents=True)
TRAIN_YEARS = list(range(2018, 2026))
COLS = ["PASS", "OL", "DEF", "PRESS", "home_ind"]

DEPTH_CSV = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.csv"
DEPTH_PARQUET = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.parquet"
ROSTER_WEEKLY_2025_CSV = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2025.csv"
ROSTER_2025_CSV = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2025.csv"


def canon_team_scalar(x):
    if pd.isna(x):
        return x
    s = str(x)
    return {"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA"}.get(s, s)


def corrected_observations():
    schedules, pbp, pfr = t.load_sources()
    games, elo_games, tg, qb = t.build_observations(schedules, pbp, pfr)
    # AUTHORITATIVE QA FIX: no full-history centering inside OL_RUN_RAW.
    tg = tg.copy()
    tg["OL_RUN_RAW"] = 0.65 * tg["run_epa"] + 0.35 * tg["off_success"]
    return schedules, games, tg, qb


def fit_2026(games, tg, qb):
    hl, rho, qblend = t.choose_params(2026, tg, games, qb)
    st = t.state_before_game(tg, games, qb, hl, rho, qblend, TRAIN_YEARS)
    ft = t.make_features(st)
    tr = ft[ft.season.isin(TRAIN_YEARS)].copy()
    f1tr, _, coef, lam = t.fit_ridge(tr, tr, COLS)

    ok = np.isfinite(tr.actual_margin.to_numpy(float)) & np.isfinite(f1tr) & (tr.actual_margin.to_numpy(float) != 0)
    y = (tr.actual_margin.to_numpy(float)[ok] > 0).astype(int)
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(np.asarray(f1tr)[ok].reshape(-1, 1), y)
    cal_a = float(lr.intercept_[0])
    cal_c = float(lr.coef_[0, 0])

    fit = {
        "architecture": "F1 PASS/QB + OL + DEF + PRESS + fitted HFA",
        "source_seasons": "2017-2025",
        "fit_seasons": "2018-2025",
        "half_life": int(hl),
        "rho": float(rho),
        "qb_blend_q": float(qblend),
        "ridge_lambda": float(lam),
        "b_PASS": float(coef["PASS"]),
        "b_OL": float(coef["OL"]),
        "b_DEF": float(coef["DEF"]),
        "b_PRESS": float(coef["PRESS"]),
        "b_HFA": float(coef["home_ind"]),
        "prob_intercept_a": cal_a,
        "prob_slope_c": cal_c,
        "ol_run_raw": "0.65*RunEPA + 0.35*OffSuccess",
        "lambda_rule": "strict minimum CV loss",
        "market_input": False,
    }
    return fit


def normalized_observations(tg, qb):
    tr = tg[tg.season.isin(TRAIN_YEARS)]
    norm = {}
    for col in ["PASS_ENV_RAW","OL_PROTECT_RAW","OL_RUN_RAW","DEF_PASS_RAW","DEF_RUN_RAW","DEF_STOP_RAW","PRESS_RAW"]:
        norm[col] = (tr[col].mean(skipna=True), t.safe_sd(tr[col]))
    x = tg.copy()
    x["pass_obs"] = t.z_apply(x.PASS_ENV_RAW, *norm["PASS_ENV_RAW"])
    x["prot_obs"] = t.z_apply(x.OL_PROTECT_RAW, *norm["OL_PROTECT_RAW"])
    x["run_obs"] = t.z_apply(x.OL_RUN_RAW, *norm["OL_RUN_RAW"])
    x["defp_obs"] = t.z_apply(x.DEF_PASS_RAW, *norm["DEF_PASS_RAW"])
    x["defr_obs"] = t.z_apply(x.DEF_RUN_RAW, *norm["DEF_RUN_RAW"])
    x["defs_obs"] = t.z_apply(x.DEF_STOP_RAW, *norm["DEF_STOP_RAW"])
    x["press_obs"] = t.z_apply(x.PRESS_RAW, *norm["PRESS_RAW"])
    x["ol_obs"] = .75*x.prot_obs + .25*x.run_obs
    x["def_obs"] = .60*x.defp_obs + .20*x.defr_obs + .20*x.defs_obs

    qtr = qb[qb.season.isin(TRAIN_YEARS)]
    qstats = ["qb_epa_db","qb_cpoe","qb_success","qb_sack_avoid","qb_to_avoid"]
    qmu = {v:qtr[v].mean(skipna=True) for v in qstats}
    qsd = {v:t.safe_sd(qtr[v]) for v in qstats}
    q = qb.copy()
    zs = {v:t.z_apply(q[v], qmu[v], qsd[v]) for v in qstats}
    q["qb_obs"] = .55*zs["qb_epa_db"] + .15*zs["qb_cpoe"] + .10*zs["qb_success"] + .10*zs["qb_sack_avoid"] + .10*zs["qb_to_avoid"]
    return x, q


def end_2025_states(games, tg, qb, fit):
    x, q = normalized_observations(tg, qb)
    xlookup = {(r.game_id, canon_team_scalar(r.team)): r for r in x.itertuples()}
    q_by_game = {gid:g for gid,g in q.groupby("game_id")}
    decay = math.exp(math.log(.5) / fit["half_life"])
    rho = fit["rho"]

    team_state, team_last = {}, {}
    qb_state, qb_last = {}, {}

    def pull_team(tm, seas):
        tm = canon_team_scalar(tm)
        st = team_state.get(tm, {"pass":0.,"ol":0.,"def":0.,"press":0.}).copy()
        ls = team_last.get(tm, seas)
        if seas > ls:
            for k in st:
                st[k] *= rho ** (seas-ls)
            team_state[tm] = st.copy(); team_last[tm] = seas
        return st

    def pull_qb(qid, seas):
        if qid is None or str(qid) in ["nan","None",""]:
            return 0.
        qid = str(qid)
        st = qb_state.get(qid, 0.)
        ls = qb_last.get(qid, seas)
        if seas > ls:
            st *= rho ** (seas-ls)
            qb_state[qid] = st; qb_last[qid] = seas
        return st

    for g in games.itertuples():
        seas = int(g.season)
        home = canon_team_scalar(g.home_team); away = canon_team_scalar(g.away_team)
        pull_team(home, seas); pull_team(away, seas)
        pull_qb(g.home_qb_id, seas); pull_qb(g.away_qb_id, seas)
        for tm in [home, away]:
            ob = xlookup.get((g.game_id, tm))
            if ob is not None:
                st = pull_team(tm, seas)
                vals = {"pass":ob.pass_obs,"ol":ob.ol_obs,"def":ob.def_obs,"press":ob.press_obs}
                for k, v in vals.items():
                    if np.isfinite(v):
                        st[k] = decay*st[k] + (1-decay)*v
                team_state[tm] = st; team_last[tm] = seas
        if g.game_id in q_by_game:
            for qr in q_by_game[g.game_id].itertuples():
                qid = str(qr.qb_id)
                if np.isfinite(qr.qb_obs):
                    old = pull_qb(qid, seas)
                    qb_state[qid] = decay*old + (1-decay)*qr.qb_obs
                    qb_last[qid] = seas

    # Opening-2026 offseason shrink, one season step.
    teams = sorted(set(games.loc[games.season.eq(2025), "home_team"]).union(set(games.loc[games.season.eq(2025), "away_team"])))
    team26 = {}
    for tm in teams:
        tm = canon_team_scalar(tm)
        st = pull_team(tm, 2025)
        team26[tm] = {k: float(v*rho) for k, v in st.items()}
    qb26 = {str(qid): float(v*rho) for qid, v in qb_state.items()}
    return team26, qb26


def read_depth():
    try:
        return pd.read_csv(DEPTH_CSV, low_memory=False)
    except Exception:
        return pd.read_parquet(DEPTH_PARQUET)


def read_roster_2025():
    try:
        r = pd.read_csv(ROSTER_WEEKLY_2025_CSV, low_memory=False)
        if "week" in r.columns:
            r["week_num"] = pd.to_numeric(r["week"], errors="coerce")
            r = r.sort_values("week_num").drop_duplicates("gsis_id", keep="last")
        return r
    except Exception:
        return pd.read_csv(ROSTER_2025_CSV, low_memory=False)


def latest_depth(depth):
    d = depth.copy()
    d["team"] = d["team"].map(canon_team_scalar)
    d["dt_parsed"] = pd.to_datetime(d["dt"], errors="coerce", utc=True)
    chunks = []
    for tm, g in d.groupby("team"):
        mx = g.dt_parsed.max()
        gg = g[g.dt_parsed.eq(mx)].copy() if pd.notna(mx) else g.copy()
        chunks.append(gg)
    return pd.concat(chunks, ignore_index=True)


def build_origin_map(roster):
    r = roster.copy()
    if "team" not in r.columns:
        for candidate in ["recent_team","team_abbr","club"]:
            if candidate in r.columns:
                r["team"] = r[candidate]; break
    r["team"] = r["team"].map(canon_team_scalar)
    if "gsis_id" not in r.columns:
        raise RuntimeError("2025 roster source lacks gsis_id")
    r = r[r.gsis_id.notna() & r.team.notna()].copy()
    return {str(row.gsis_id): row.team for row in r.itertuples()}


def unit_mean(rows, origin_map, team26, key, fallback_team):
    vals, known = [], 0
    for r in rows.itertuples():
        gid = str(getattr(r, "gsis_id", ""))
        origin = origin_map.get(gid)
        if origin in team26:
            vals.append(team26[origin][key]); known += 1
        else:
            vals.append(0.0)
    if not vals:
        return float(team26[fallback_team][key]), 0, 0
    return float(np.mean(vals)), known, len(vals)


def build_week1_board(team26, qb26, fit):
    depth = latest_depth(read_depth())
    roster = read_roster_2025()
    origin_map = build_origin_map(roster)
    depth.to_csv(OUT / "Latest_2026_Depth_Chart_Snapshot.csv", index=False)

    d = depth.copy()
    d["pos_rank_num"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    d["pos_abb_u"] = d["pos_abb"].astype(str).str.upper()
    d["pos_grp_l"] = d["pos_grp"].astype(str).str.lower()
    starters = d[d.pos_rank_num.eq(1)].copy()

    rows = []
    for tm in sorted(team26):
        g = starters[starters.team.eq(tm)].copy()
        qbrow = g[g.pos_abb_u.eq("QB")].sort_values("pos_slot").head(1)
        if len(qbrow):
            qbid = str(qbrow.iloc[0].gsis_id)
            qbname = str(qbrow.iloc[0].player_name)
            qbst = float(qb26.get(qbid, 0.0))
            qb_known = qbid in qb26
        else:
            qbid, qbname, qbst, qb_known = "", "UNRESOLVED", 0.0, False

        skill = g[g.pos_abb_u.isin(["WR","TE","RB","FB"])]
        ol = g[g.pos_abb_u.isin(["LT","LG","C","RG","RT","T","G","OT","OG"])]
        defense = g[g.pos_grp_l.str.contains("def", na=False)]
        front = defense[defense.pos_abb_u.isin(["DE","DT","NT","DL","EDGE","OLB","LB","LOLB","ROLB"])]

        passenv, skill_known, skill_n = unit_mean(skill, origin_map, team26, "pass", tm)
        olst, ol_known, ol_n = unit_mean(ol, origin_map, team26, "ol", tm)
        defst, def_known, def_n = unit_mean(defense, origin_map, team26, "def", tm)
        pressst, front_known, front_n = unit_mean(front, origin_map, team26, "press", tm)

        PASS = fit["qb_blend_q"]*qbst + (1-fit["qb_blend_q"])*passenv
        PR = fit["b_PASS"]*PASS + fit["b_OL"]*olst + fit["b_DEF"]*defst + fit["b_PRESS"]*pressst

        # Qualitative uncertainty only; it never changes PR.
        ol_cov = ol_known/max(ol_n,1)
        def_cov = def_known/max(def_n,1)
        front_cov = front_known/max(front_n,1)
        if (not qb_known) or ol_cov < .60 or def_cov < .60 or front_cov < .50:
            unc = "High"
        elif ol_cov < .85 or def_cov < .80 or front_cov < .75 or skill_known < max(skill_n-1, 0):
            unc = "Medium"
        else:
            unc = "Low"

        rows.append({
            "Team":tm,"PR":PR,"PASS":PASS,"QB":qbst,"OL":olst,"DEF":defst,"PRESS":pressst,
            "QB_Name":qbname,"QB_ID":qbid,"Uncertainty":unc,
            "SkillKnown":skill_known,"SkillN":skill_n,"OLKnown":ol_known,"OLN":ol_n,
            "DEFKnown":def_known,"DEFN":def_n,"FrontKnown":front_known,"FrontN":front_n,
        })
    board = pd.DataFrame(rows).sort_values("PR", ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", np.arange(1, len(board)+1))
    return board


def main():
    print("[1/6] Load frozen historical sources and apply authoritative OL QA correction")
    schedules, games, tg, qb = corrected_observations()
    print("[2/6] Fit/freeze 2026 F1 numerical vector")
    fit = fit_2026(games, tg, qb)
    print(json.dumps(fit, indent=2))
    print("[3/6] Replay through end-2025 and apply offseason shrinkage")
    team26, qb26 = end_2025_states(games, tg, qb, fit)
    pd.DataFrame([{"Team":tm, **st} for tm,st in team26.items()]).to_csv(OUT/"Opening_2026_Shrunk_Team_States.csv", index=False)
    pd.DataFrame([{"QB_ID":qid,"QB_State":v} for qid,v in qb26.items()]).to_csv(OUT/"Opening_2026_Shrunk_QB_States.csv", index=False)
    print("[4/6] Reconstruct Week 1 expected personnel from latest nflverse depth chart")
    board = build_week1_board(team26, qb26, fit)
    print("[5/6] Write authoritative freeze artifacts")
    pd.DataFrame([fit]).to_csv(OUT/"F1_2026_Production_Fit.csv", index=False)
    board.to_csv(OUT/"F1_2026_Week1_Baseline.csv", index=False)
    receipt = {
        "status":"FROZEN",
        "freeze_scope":"2026 F1 numerical production fit + Week 1 baseline",
        "f1_architecture_changed":False,
        "personnel_bridge":"Expected starter carries prior-team offseason-shrunk F1 unit state; rookies/unknown origin=0; QB player-specific",
        "market_used":False,
        "fit":fit,
        "teams":int(len(board)),
    }
    (OUT/"Freeze_Receipt.json").write_text(json.dumps(receipt, indent=2))
    print(board[["Rank","Team","PR","PASS","QB","OL","DEF","PRESS","QB_Name","Uncertainty"]].to_string(index=False))
    print("[6/6] COMPLETE", OUT.resolve())

if __name__ == "__main__":
    main()
