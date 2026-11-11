"""Game-level ridge for player impact (net points per 100 by minutes share), then the strength lost to rotation players who did not play.
Walk-forward: each season uses estimates fitted on earlier seasons only. Writes data/model/nba/absence.pkl."""
import sys; sys.path.insert(0,".")
import pandas as pd,numpy as np,scipy.sparse as sp
from sklearn.linear_model import Ridge
from pm_edges import nba_ratings as R
S="data/model/nba/"
rows=R.team_rows(); rows["home"]=rows.home.astype(bool)
p=pd.read_pickle("data/history/box/player_box.pkl"); p["team"]=p.team.replace(R.FIX); p=p[p.team.isin(R.NBA_TEAMS)].copy()
p["played"]=(~p.dnp)&(p["min"].fillna(0)>=1); p["minutes"]=np.where(p.played,p["min"].fillna(0),0.0)
p["date"]=pd.to_datetime(p.date); p=p[p.player_id.notna()]
# game table: home/away, net rating
h=rows[rows.home][["game_id","date","team","off","margin","season"]].rename(columns={"team":"home","off":"h_off"})
a=rows[~rows.home][["game_id","team","off"]].rename(columns={"team":"away","off":"a_off"})
G=h.merge(a,on="game_id"); G["net"]=G.h_off-G.a_off; G=G.dropna(subset=["net"]).sort_values("date").reset_index(drop=True)
# player-game minutes joined with side
pg=p.merge(G[["game_id","home","away"]],on="game_id"); pg["side"]=np.where(pg.team==pg.home,1.0,-1.0)
pg=pg[(pg.team==pg.home)|(pg.team==pg.away)]
pid={x:i for i,x in enumerate(sorted(pg.player_id.unique()))}
gid={x:i for i,x in enumerate(G.game_id)}
print("players",len(pid),"games",len(G))
def fit(train_games,alpha):
    sub=pg[pg.game_id.isin(train_games)]
    X=sp.csr_matrix((sub.side*sub.minutes/48.0,(sub.game_id.map(gid),sub.player_id.map(pid))),shape=(len(G),len(pid)))
    idx=G.index[G.game_id.isin(train_games)].values
    hca=np.ones((len(G),1)); Xf=sp.hstack([X,sp.csr_matrix(hca)]).tocsr()
    m=Ridge(alpha=alpha,fit_intercept=False,solver="sparse_cg",tol=1e-6).fit(Xf[idx],G.net.values[idx])
    return m.coef_[:-1],m.coef_[-1]

def absence_table(beta_by_season: dict) -> pd.DataFrame:
    """Per team-game: strength lost to rotation players who did not play, using that season's walk-forward impact estimates."""
    out=[]
    for team,tg in pg.groupby("team"):
        tg=tg.copy(); tg["sea"]=np.where(tg.date.dt.month>=9,tg.date.dt.year,tg.date.dt.year-1)
        games=tg[["game_id","date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
        M=tg.pivot_table(index="game_id",columns="player_id",values="minutes",aggfunc="sum").reindex(games.game_id).fillna(0.0)
        played=(M>=1).astype(float); rot=(M>=12).astype(float)
        cnt=rot.rolling(10,min_periods=3).sum().shift(1); tot=M.rolling(10,min_periods=3).sum().shift(1); pc=played.rolling(10,min_periods=3).sum().shift(1)
        u=(tot/pc.replace(0,np.nan)).fillna(0)/48.0
        inQ=(cnt>=3).astype(float)
        gsea=games.date.apply(lambda d:d.year if d.month>=9 else d.year-1).values
        res=[]
        for i,gm in enumerate(games.game_id):
            b=beta_by_season.get(gsea[i])
            if b is None: res.append((gm,np.nan,np.nan,np.nan)); continue
            beta=pd.Series(M.columns.map(lambda x:b.get(x,0.0)),index=M.columns)
            val=(beta*u.iloc[i]*inQ.iloc[i])
            full=val.sum(); miss=val[(played.iloc[i]==0)].sum(); nmiss=int(((played.iloc[i]==0)&(inQ.iloc[i]>0)).sum())
            res.append((gm,full,miss,nmiss))
        r=pd.DataFrame(res,columns=["game_id","S_full","S_missing","n_missing"]); r["team"]=team; out.append(r)
    return pd.concat(out)

if __name__ == "__main__":
    G["sea"] = np.where(G.date.dt.month >= 9, G.date.dt.year, G.date.dt.year - 1)
    inv = {i: x for x, i in pid.items()}; betas = {}
    for s in (2022, 2023, 2024, 2025, 2026):                    # impact estimates for season s use only earlier seasons
        b, hca = fit(set(G[G.sea < s].game_id), 20); betas[s] = {inv[i]: b[i] for i in range(len(b))}
    absence_table(betas).to_pickle(S + "absence.pkl")
    print("saved", S + "absence.pkl")
