#!/usr/bin/env python3
"""
NFL Power Ratings — Historical Model Tournament v1.1 (Python)

Strict chronological, market-independent comparison:
  B0 = NFLAnalytic-style Elo benchmark
  S0 = frozen Simple Core (PASS/OL/DEF/PRESS; 40/25/20/15)
  F1 = ridge-fitted Core (same four pillars)
  F2 = F1 + nonlinear Trench Mismatch challenger

Designed for an internet-enabled Python environment.
No betting lines are used as football-strength inputs.
"""
from __future__ import annotations

import math, os, json, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

SOURCE_SEASONS = list(range(2017, 2026))
FULL_FEATURE_START = 2018
OOS_SEASONS = list(range(2020, 2026))
GAME_TYPES = {"REG"}
OUT_DIR = Path("NFL_PR_Tournament_Output")
CACHE_DIR = Path("NFL_PR_Tournament_Cache")
OUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

HALF_LIFE_GRID = [4, 6, 8, 10]
RHO_GRID = [0.35, 0.50, 0.65, 0.80]
QB_BLEND_GRID = [0.40, 0.55, 0.70, 0.85]
SIMPLE_WEIGHTS = {"PASS":0.40, "OL":0.25, "DEF":0.20, "PRESS":0.15}
RIDGE_LAMBDAS = np.logspace(-3, 3, 41)
SEED = 20260905
rng = np.random.default_rng(SEED)

ELO_BASE = 1505.0
ELO_K = 20.0
ELO_HFA = 48.0
ELO_REGRESS = 1/3

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
PFR_URL = "https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_week_pass_{season}.csv"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

PBP_COLS = [
    "game_id","season","season_type","week","posteam","defteam","epa","wp",
    "no_play","two_point_attempt","qb_kneel","qb_spike","qb_scramble","qb_dropback",
    "rush_attempt","cpoe","sack","interception","fumble_lost",
    "passer_player_id","passer_player_name"
]


def download(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 100:
        return path
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
    return path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def wmean(x, w):
    x=np.asarray(x,dtype=float); w=np.asarray(w,dtype=float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    return np.sum(x[ok]*w[ok])/np.sum(w[ok]) if ok.any() else np.nan


def safe_sd(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    s=np.std(x,ddof=1) if len(x)>1 else np.nan
    return s if np.isfinite(s) and s>1e-9 else 1.0


def z_apply(x, mu, sd):
    return np.clip((x-mu)/sd, -3, 3)


def canon_team(s):
    mp={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA"}
    return s.map(mp).fillna(s)


def zero_or_na_local(df,col):
    return df[col].isna() | (pd.to_numeric(df[col],errors="coerce")==0)


def load_sources():
    games_path=download(GAMES_URL, CACHE_DIR/"games.csv")
    schedules=pd.read_csv(games_path, low_memory=False)

    pbps=[]
    for s in SOURCE_SEASONS:
        p=download(PBP_URL.format(season=s), CACHE_DIR/f"play_by_play_{s}.parquet")
        try:
            x=pd.read_parquet(p, columns=PBP_COLS)
        except Exception:
            x=pd.read_parquet(p)
            x=x[[c for c in PBP_COLS if c in x.columns]]
        for c in PBP_COLS:
            if c not in x.columns: x[c]=np.nan
        pbps.append(x[PBP_COLS])
    pbp=pd.concat(pbps, ignore_index=True)

    pfrs=[]
    for s in range(FULL_FEATURE_START, max(SOURCE_SEASONS)+1):
        p=download(PFR_URL.format(season=s), CACHE_DIR/f"advstats_week_pass_{s}.csv")
        x=pd.read_csv(p, low_memory=False)
        x["source_season"]=s
        pfrs.append(x)
    pfr=pd.concat(pfrs, ignore_index=True)
    return schedules,pbp,pfr


def build_observations(schedules,pbp,pfr):
    sch=schedules.copy()
    games=sch[(sch.season.isin(SOURCE_SEASONS)) & (sch.game_type.isin(GAME_TYPES)) & sch.home_score.notna() & sch.away_score.notna()].copy()
    keep=["game_id","season","week","gameday","gametime","game_type","home_team","away_team","home_score","away_score","location","home_qb_id","away_qb_id","home_qb_name","away_qb_name"]
    for c in keep:
        if c not in games.columns: games[c]=np.nan
    games=games[keep]
    games["actual_margin"]=games.home_score-games.away_score
    games["neutral"]=(games.location.astype(str).str.lower()!="home").astype(int)
    games["home_ind"]=1-games.neutral
    games=games.sort_values(["gameday","gametime","game_id"]).reset_index(drop=True)

    elo_games=sch[(sch.season.between(1999,2025)) & sch.game_type.isin(["REG","POST"]) & sch.home_score.notna() & sch.away_score.notna()].copy()
    elo_games["actual_margin"]=elo_games.home_score-elo_games.away_score
    elo_games["home_ind"]=(elo_games.location.astype(str).str.lower()=="home").astype(int)
    elo_games=elo_games.sort_values(["gameday","gametime","game_id"]).reset_index(drop=True)

    c=pbp.copy()
    def zero_or_na(col): return c[col].isna() | (pd.to_numeric(c[col],errors="coerce")==0)
    mask=(c.season.isin(SOURCE_SEASONS)) & c.season_type.isin(GAME_TYPES) & zero_or_na("no_play") & zero_or_na("two_point_attempt") & zero_or_na("qb_kneel") & zero_or_na("qb_spike") & c.epa.notna() & c.posteam.notna() & c.defteam.notna()
    c=c[mask].copy()
    wp=pd.to_numeric(c.wp,errors="coerce")
    c["w_comp"]=4*wp*(1-wp)
    c.loc[~np.isfinite(c.w_comp)|(c.w_comp<=0),"w_comp"]=1.0
    c["is_db"]=(pd.to_numeric(c.qb_dropback,errors="coerce")==1).astype(int)
    c["is_des_run"]=((pd.to_numeric(c.rush_attempt,errors="coerce")==1) & zero_or_na_local(c,"qb_scramble") & zero_or_na_local(c,"qb_kneel")).astype(int)
    c["success_epa"]=(c.epa>0).astype(int)

    rows=[]
    for (season,week,gid,team,opp),g in c.groupby(["season","week","game_id","posteam","defteam"], sort=False):
        db=g.is_db.eq(1); rr=g.is_des_run.eq(1)
        rows.append(dict(season=season,week=week,game_id=gid,team=team,opponent=opp,
            pass_epa=wmean(g.loc[db,"epa"],g.loc[db,"w_comp"]),
            run_epa=wmean(g.loc[rr,"epa"],g.loc[rr,"w_comp"]),
            off_success=wmean(g.success_epa,g.w_comp),
            cpoe=wmean(g.loc[db,"cpoe"],g.loc[db,"w_comp"]),
            db=int(db.sum()),des_runs=int(rr.sum()),plays=len(g)))
    off=pd.DataFrame(rows)

    rows=[]
    for (season,week,gid,team,opp),g in c.groupby(["season","week","game_id","defteam","posteam"], sort=False):
        db=g.is_db.eq(1); rr=g.is_des_run.eq(1)
        rows.append(dict(season=season,week=week,game_id=gid,team=team,opponent=opp,
            pass_def=-wmean(g.loc[db,"epa"],g.loc[db,"w_comp"]),
            run_def=-wmean(g.loc[rr,"epa"],g.loc[rr,"w_comp"]),
            stop_rate=1-wmean(g.success_epa,g.w_comp),def_plays=len(g)))
    de=pd.DataFrame(rows)
    tg=off.merge(de,on=["season","week","game_id","team","opponent"],how="outer")

    qrows=[]
    qc=c[(c.is_db==1)&c.passer_player_id.notna()].copy()
    for (season,week,gid,team,qid,qname),g in qc.groupby(["season","week","game_id","posteam","passer_player_id","passer_player_name"], dropna=False, sort=False):
        sack=pd.to_numeric(g.sack,errors="coerce").fillna(0).eq(1)
        intr=pd.to_numeric(g.interception,errors="coerce").fillna(0).eq(1)
        fum=pd.to_numeric(g.fumble_lost,errors="coerce").fillna(0).eq(1)
        qrows.append(dict(season=season,week=week,game_id=gid,team=team,qb_id=str(qid),qb_name=qname,
            qb_epa_db=wmean(g.epa,g.w_comp),qb_cpoe=wmean(g.cpoe,g.w_comp),qb_success=wmean(g.success_epa,g.w_comp),
            qb_sack_avoid=1-sack.mean(),qb_to_avoid=1-(intr|fum).mean(),qb_db=len(g)))
    qb=pd.DataFrame(qrows)

    p=pfr.copy()
    for col in ["team","opponent"]: p[col]=canon_team(p[col].astype(str))
    nums=["times_pressured","times_sacked","times_hurried","times_hit"]
    for n in nums:
        if n not in p.columns: p[n]=0
        p[n]=pd.to_numeric(p[n],errors="coerce").fillna(0)
    pt=p.groupby(["season","week","game_id","team","opponent"],as_index=False)[nums].sum()
    tg["team"]=canon_team(tg.team.astype(str)); tg["opponent"]=canon_team(tg.opponent.astype(str))
    tg=tg.merge(pt,on=["season","week","game_id","team","opponent"],how="left")
    tg["pressure_allowed_rate"]=tg.times_pressured/np.maximum(tg.db,1)
    tg["protection_raw"]=-tg.pressure_allowed_rate
    opp=tg[["season","week","game_id","team","opponent","pressure_allowed_rate"]].copy()
    opp=opp.rename(columns={"team":"opponent2","opponent":"team","pressure_allowed_rate":"pressure_created_rate"})
    opp=opp.rename(columns={"opponent2":"opponent"})
    tg=tg.merge(opp,on=["season","week","game_id","team","opponent"],how="left")
    tg["press_raw"]=tg.pressure_created_rate

    tg["PASS_ENV_RAW"]=tg.pass_epa
    tg["OL_PROTECT_RAW"]=tg.protection_raw
    tg["OL_RUN_RAW"]=0.65*tg.run_epa + 0.35*(tg.off_success-tg.off_success.mean(skipna=True))
    tg["DEF_PASS_RAW"]=tg.pass_def
    tg["DEF_RUN_RAW"]=tg.run_def
    tg["DEF_STOP_RAW"]=tg.stop_rate
    tg["PRESS_RAW"]=tg.press_raw
    return games,elo_games,tg,qb


def state_before_game(obs,games,qb,half_life=6,rho=.5,qb_blend=.7,norm_train_seasons=(2018,2019)):
    decay=math.exp(math.log(.5)/half_life)
    tr=obs[obs.season.isin(list(norm_train_seasons))]
    if len(tr)<100: tr=obs[obs.season<min(games.season.max(), min(norm_train_seasons)+1)]
    norm={}
    for col in ["PASS_ENV_RAW","OL_PROTECT_RAW","OL_RUN_RAW","DEF_PASS_RAW","DEF_RUN_RAW","DEF_STOP_RAW","PRESS_RAW"]:
        norm[col]=(tr[col].mean(skipna=True), safe_sd(tr[col]))
    x=obs.copy()
    x["pass_obs"]=z_apply(x.PASS_ENV_RAW,*norm["PASS_ENV_RAW"])
    x["prot_obs"]=z_apply(x.OL_PROTECT_RAW,*norm["OL_PROTECT_RAW"])
    x["run_obs"]=z_apply(x.OL_RUN_RAW,*norm["OL_RUN_RAW"])
    x["defp_obs"]=z_apply(x.DEF_PASS_RAW,*norm["DEF_PASS_RAW"])
    x["defr_obs"]=z_apply(x.DEF_RUN_RAW,*norm["DEF_RUN_RAW"])
    x["defs_obs"]=z_apply(x.DEF_STOP_RAW,*norm["DEF_STOP_RAW"])
    x["press_obs"]=z_apply(x.PRESS_RAW,*norm["PRESS_RAW"])
    x["ol_obs"]=.75*x.prot_obs+.25*x.run_obs
    x["def_obs"]=.60*x.defp_obs+.20*x.defr_obs+.20*x.defs_obs
    xlookup={(r.game_id,r.team):r for r in x.itertuples()}

    qtr=qb[qb.season.isin(list(norm_train_seasons))]
    qstats=["qb_epa_db","qb_cpoe","qb_success","qb_sack_avoid","qb_to_avoid"]
    qmu={v:qtr[v].mean(skipna=True) for v in qstats}; qsd={v:safe_sd(qtr[v]) for v in qstats}
    q=qb.copy()
    zs={}
    for v in qstats: zs[v]=z_apply(q[v],qmu[v],qsd[v])
    q["qb_obs"]=.55*zs["qb_epa_db"]+.15*zs["qb_cpoe"]+.10*zs["qb_success"]+.10*zs["qb_sack_avoid"]+.10*zs["qb_to_avoid"]
    q_by_game={gid:g for gid,g in q.groupby("game_id")}

    team_state={}; team_last={}; qb_state={}; qb_last={}; out=[]
    def pull_team(tm,seas):
        st=team_state.get(tm, {"pass":0.,"ol":0.,"def":0.,"press":0.}).copy()
        ls=team_last.get(tm,seas)
        if seas>ls:
            for k in st: st[k]*=rho**(seas-ls)
            team_state[tm]=st.copy(); team_last[tm]=seas
        return st
    def pull_qb(qid,seas):
        if qid is None or str(qid) in ["nan","None",""]: return 0.
        qid=str(qid); st=qb_state.get(qid,0.); ls=qb_last.get(qid,seas)
        if seas>ls:
            st*=rho**(seas-ls); qb_state[qid]=st; qb_last[qid]=seas
        return st

    for g in games.itertuples():
        seas=int(g.season); hs=pull_team(g.home_team,seas); a=pull_team(g.away_team,seas)
        hq=pull_qb(g.home_qb_id,seas); aq=pull_qb(g.away_qb_id,seas)
        out.append(dict(game_id=g.game_id,season=seas,week=g.week,home_team=g.home_team,away_team=g.away_team,
            actual_margin=g.actual_margin,home_ind=g.home_ind,home_qb_id=g.home_qb_id,away_qb_id=g.away_qb_id,
            PASS_H=qb_blend*hq+(1-qb_blend)*hs["pass"],PASS_A=qb_blend*aq+(1-qb_blend)*a["pass"],
            OL_H=hs["ol"],OL_A=a["ol"],DEF_H=hs["def"],DEF_A=a["def"],PRESS_H=hs["press"],PRESS_A=a["press"],QB_H=hq,QB_A=aq))
        for tm in [g.home_team,g.away_team]:
            ob=xlookup.get((g.game_id,tm))
            if ob is not None:
                st=pull_team(tm,seas)
                vals={"pass":ob.pass_obs,"ol":ob.ol_obs,"def":ob.def_obs,"press":ob.press_obs}
                for k,v in vals.items():
                    if np.isfinite(v): st[k]=decay*st[k]+(1-decay)*v
                team_state[tm]=st; team_last[tm]=seas
        if g.game_id in q_by_game:
            for qr in q_by_game[g.game_id].itertuples():
                qid=str(qr.qb_id)
                if np.isfinite(qr.qb_obs):
                    old=pull_qb(qid,seas); qb_state[qid]=decay*old+(1-decay)*qr.qb_obs; qb_last[qid]=seas
    return pd.DataFrame(out)


def make_features(d):
    z=d.copy()
    z["PASS"]=z.PASS_H-z.PASS_A; z["OL"]=z.OL_H-z.OL_A; z["DEF"]=z.DEF_H-z.DEF_A; z["PRESS"]=z.PRESS_H-z.PRESS_A; z["QB"]=z.QB_H-z.QB_A
    z["TRENCH_MISMATCH"]=np.maximum(0,z.PRESS_H-z.OL_A)-np.maximum(0,z.PRESS_A-z.OL_H)
    z["S0_SCORE"]=sum(SIMPLE_WEIGHTS[k]*z[k] for k in SIMPLE_WEIGHTS)
    return z


def solve_ridge(X,y,lam,penalty):
    A=X.T@X + lam*np.diag(penalty)
    return np.linalg.pinv(A)@(X.T@y)


def fit_ridge(train,test,cols):
    X=train[cols].to_numpy(float); y=train.actual_margin.to_numpy(float)
    ok=np.isfinite(X).all(axis=1)&np.isfinite(y); X=X[ok]; y=y[ok]
    penalty=np.array([0. if c=="home_ind" else 1. for c in cols])
    nfold=min(10,max(5,len(y)//100)); kf=KFold(n_splits=nfold,shuffle=True,random_state=SEED)
    losses=[]
    for lam in RIDGE_LAMBDAS:
        mse=[]
        for a,b in kf.split(X):
            beta=solve_ridge(X[a],y[a],lam,penalty)
            pred=X[b]@beta; mse.append(np.mean((y[b]-pred)**2))
        losses.append(np.mean(mse))
    best_i=int(np.argmin(losses)); lam=RIDGE_LAMBDAS[best_i]
    beta=solve_ridge(X,y,lam,penalty)
    return train[cols].to_numpy(float)@beta, test[cols].to_numpy(float)@beta, dict(zip(cols,beta)), lam


def fit_s0(train,test):
    X=train[["S0_SCORE","home_ind"]].to_numpy(float); y=train.actual_margin.to_numpy(float)
    beta=np.linalg.pinv(X.T@X)@(X.T@y)
    return X@beta, test[["S0_SCORE","home_ind"]].to_numpy(float)@beta, {"kappa":beta[0],"hfa_points":beta[1]}


def fit_prob_cal(y_margin,train_pm,test_pm):
    ok=np.isfinite(y_margin)&np.isfinite(train_pm)&(y_margin!=0)
    if ok.sum()<200: return 1/(1+np.exp(-np.asarray(test_pm)/6.5))
    y=(np.asarray(y_margin)[ok]>0).astype(int); X=np.asarray(train_pm)[ok].reshape(-1,1)
    lr=LogisticRegression(C=1e6,solver="lbfgs").fit(X,y)
    return lr.predict_proba(np.asarray(test_pm).reshape(-1,1))[:,1]


def elo_tournament(games):
    ratings={}; last_season=None; rows=[]
    for g in games.itertuples():
        if last_season is None: last_season=int(g.season)
        if g.season>last_season:
            for _ in range(last_season+1,int(g.season)+1):
                for tm,r in list(ratings.items()): ratings[tm]=r+ELO_REGRESS*(ELO_BASE-r)
            last_season=int(g.season)
        rh=ratings.get(g.home_team,ELO_BASE); ra=ratings.get(g.away_team,ELO_BASE)
        diff=rh-ra+ELO_HFA*g.home_ind; p=1/(1+10**(-diff/400))
        rows.append(dict(game_id=g.game_id,season=g.season,week=g.week,B0_p_home=p,B0_pred_margin_proxy=diff,B0_home_elo=rh,B0_away_elo=ra))
        m=g.actual_margin
        actual=1 if m>0 else 0 if m<0 else .5
        if m==0: mult=1
        else:
            winner_home=m>0; ew=(rh+ELO_HFA*g.home_ind) if winner_home else ra; el=ra if winner_home else (rh+ELO_HFA*g.home_ind)
            mult=math.log(abs(m)+1)*(2.2/(.001*(ew-el)+2.2))
        change=ELO_K*(actual-p)*mult; ratings[g.home_team]=rh+change; ratings[g.away_team]=ra-change
    return pd.DataFrame(rows)


def choose_params(ts,tg,games,qb):
    if ts==2020: return (6,.50,.70)
    train_years=list(range(FULL_FEATURE_START,ts)); inner_test=max(train_years); inner_train=[y for y in train_years if y<inner_test]
    if not inner_train: return (6,.50,.70)
    best=None
    for hl in HALF_LIFE_GRID:
      for rho in RHO_GRID:
       for qbblend in QB_BLEND_GRID:
        st=state_before_game(tg,games,qb,hl,rho,qbblend,inner_train); ft=make_features(st)
        tr=ft[ft.season.isin(inner_train)]; va=ft[ft.season.eq(inner_test)]
        if len(tr)<200 or len(va)<100: continue
        _,pred,_,_=fit_ridge(tr,va,["PASS","OL","DEF","PRESS","home_ind"])
        dec=va.actual_margin.ne(0); acc=np.mean(np.sign(va.loc[dec,"actual_margin"])==np.sign(pred[dec.to_numpy()]))
        mae=np.mean(np.abs(va.actual_margin-pred)); key=(-acc,mae,hl,rho,qbblend)
        if best is None or key<best[0]: best=(key,(hl,rho,qbblend))
    return best[1] if best else (6,.50,.70)


def score(d,model):
    if model=="B0": p=d.B0_p_home.to_numpy(float); pm=d.B0_pred_margin_proxy.to_numpy(float); point=False; close=np.abs(p-.5)<=.06
    else: p=d[f"{model}_p_home"].to_numpy(float); pm=d[f"{model}_pred_margin"].to_numpy(float); point=True; close=np.abs(pm)<=3
    m=d.actual_margin.to_numpy(float); dec=m!=0; y=(m>0).astype(float)
    acc=np.mean((p[dec]>.5)==(m[dec]>0)); c=close&dec; close_acc=np.mean((p[c]>.5)==(m[c]>0)) if c.sum()>=10 else np.nan
    eps=1e-6; pc=np.clip(p,eps,1-eps)
    brier=np.mean((pc-y)**2); ll=-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))
    return dict(model=model,games=len(d),decided=int(dec.sum()),su_accuracy=acc,close_games=int(c.sum()),close_accuracy=close_acc,brier=brier,log_loss=ll,
                margin_mae=np.mean(np.abs(m-pm)) if point else np.nan,margin_rmse=np.sqrt(np.mean((m-pm)**2)) if point else np.nan)


def bootstrap_diff(d,a,b="S0",B=5000):
    dec=d.actual_margin.ne(0); z=d[dec].copy(); m=z.actual_margin.to_numpy()
    def corr(model):
        p=z.B0_p_home.to_numpy() if model=="B0" else z[f"{model}_p_home"].to_numpy()
        return ((p>.5)==(m>0)).astype(float)
    diff=corr(a)-corr(b); seasons=np.sort(z.season.unique()); vals=[]
    for _ in range(B):
        ss=rng.choice(seasons,size=len(seasons),replace=True); idx=np.concatenate([np.where(z.season.to_numpy()==s)[0] for s in ss]); vals.append(diff[idx].mean())
    vals=np.asarray(vals)
    return dict(model=a,baseline=b,paired_accuracy_diff=diff.mean(),ci_low=np.quantile(vals,.025),ci_high=np.quantile(vals,.975),prob_diff_gt_0=np.mean(vals>0))


def main():
    print("[1/8] Loading public historical sources")
    schedules,pbp,pfr=load_sources()
    print("[2/8] Building causal game/QB/trench observations")
    games,elo_games,tg,qb=build_observations(schedules,pbp,pfr)
    print("[3/8] Running B0 Elo")
    ep=elo_tournament(elo_games); ep=ep[ep.game_id.isin(games.game_id)]
    all_oos=[]; coefrows=[]
    print("[4/8] Outer walk-forward B0/S0/F1/F2")
    for ts in OOS_SEASONS:
        print("  OOS",ts); hl,rho,qbblend=choose_params(ts,tg,games,qb); train_years=list(range(FULL_FEATURE_START,ts))
        st=state_before_game(tg,games,qb,hl,rho,qbblend,train_years); ft=make_features(st); tr=ft[ft.season.isin(train_years)]; te=ft[ft.season.eq(ts)].copy()
        s0tr,s0te,s0cf=fit_s0(tr,te); te["S0_pred_margin"]=s0te; te["S0_p_home"]=fit_prob_cal(tr.actual_margin.to_numpy(),s0tr,s0te)
        f1tr,f1te,f1cf,l1=fit_ridge(tr,te,["PASS","OL","DEF","PRESS","home_ind"]); te["F1_pred_margin"]=f1te; te["F1_p_home"]=fit_prob_cal(tr.actual_margin.to_numpy(),f1tr,f1te)
        f2tr,f2te,f2cf,l2=fit_ridge(tr,te,["PASS","OL","DEF","PRESS","TRENCH_MISMATCH","home_ind"]); te["F2_pred_margin"]=f2te; te["F2_p_home"]=fit_prob_cal(tr.actual_margin.to_numpy(),f2tr,f2te)
        te=te.merge(ep[ep.season.eq(ts)],on=["game_id","season","week"],how="left"); te["selected_half_life"]=hl;te["selected_rho"]=rho;te["selected_qb_blend"]=qbblend
        all_oos.append(te)
        for model,cf,lam in [("S0",s0cf,np.nan),("F1",f1cf,l1),("F2",f2cf,l2)]:
            for term,val in cf.items(): coefrows.append(dict(season=ts,model=model,term=term,coefficient=val,lambda_=lam,half_life=hl,rho=rho,qb_blend=qbblend))
    oos=pd.concat(all_oos,ignore_index=True); coefs=pd.DataFrame(coefrows)
    print("[5/8] Scoring")
    summ=pd.DataFrame([score(oos,m) for m in ["B0","S0","F1","F2"]])
    summ=summ.sort_values(["su_accuracy","close_accuracy","brier","log_loss"],ascending=[False,False,True,True]).reset_index(drop=True); summ.insert(0,"tournament_rank",np.arange(1,len(summ)+1))
    s0=summ[summ.model.eq("S0")].iloc[0]
    def status(r):
        if r.model=="S0": return "Frozen benchmark"
        if r.su_accuracy>=s0.su_accuracy+.005 and r.brier<=s0.brier+.002: return "PROMOTION CANDIDATE"
        if r.close_accuracy>s0.close_accuracy and r.brier<=s0.brier+.002: return "Close-game challenger"
        return "HOLD"
    summ["promotion_status"]=summ.apply(status,axis=1)
    by=[]
    for s in sorted(oos.season.unique()):
        for m in ["B0","S0","F1","F2"]: r=score(oos[oos.season.eq(s)],m); r["season"]=s; by.append(r)
    by=pd.DataFrame(by)
    boot=pd.DataFrame([bootstrap_diff(oos,m,"S0") for m in ["B0","F1","F2"]])
    print("[6/8] Writing artifacts")
    summ.to_csv(OUT_DIR/"Tournament_Summary.csv",index=False); by.to_csv(OUT_DIR/"Season_By_Season.csv",index=False); boot.to_csv(OUT_DIR/"Paired_Bootstrap.csv",index=False); coefs.to_csv(OUT_DIR/"WalkForward_Coefficients.csv",index=False); oos.to_csv(OUT_DIR/"OOS_Predictions.csv",index=False)
    receipt=pd.DataFrame([dict(source_seasons="2017-2025",oos_seasons="2020-2025",game_types="REG",champion=summ.iloc[0].model,champion_su_accuracy=summ.iloc[0].su_accuracy,champion_close_accuracy=summ.iloc[0].close_accuracy,simple_core_su_accuracy=s0.su_accuracy,note="Common-history tournament only; Madden/TO_Threat/ST/coaching/MATCH not promoted by this run.")])
    receipt.to_csv(OUT_DIR/"Tournament_Receipt.csv",index=False)
    hashes=[]
    for p in sorted(CACHE_DIR.glob("*")): hashes.append({"file":p.name,"bytes":p.stat().st_size,"sha256":sha256(p)})
    pd.DataFrame(hashes).to_csv(OUT_DIR/"Source_Hashes.csv",index=False)
    print("[7/8] Scoreboard")
    print(summ.to_string(index=False))
    print("[8/8] COMPLETE ->",OUT_DIR.resolve())

if __name__=="__main__": main()
