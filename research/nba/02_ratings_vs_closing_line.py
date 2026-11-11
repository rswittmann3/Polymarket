"""Does the box-score ratings model carry information beyond the closing line? (needs 01 first)"""
import pandas as pd,numpy as np
S="data/model/nba/"
wf=pd.read_pickle(S+"wf_ratings.pkl").dropna(subset=["model_margin"])
g=pd.read_pickle("data/history/games_nba.pkl"); g=g[g.kind.isin(["reg","post","playin"])].dropna(subset=["home_spread_close","total_close"]).copy(); g["date"]=pd.to_datetime(g.date)
g=g[["date","home","away","home_spread_open","home_spread_close","total_close","margin","points"]].rename(columns={"margin":"m2","points":"p2","date":"ldate"})
parts=[]
for off in (0,-1,1):
    w2=wf.copy(); w2["ldate"]=w2.date+pd.Timedelta(days=off); parts.append(w2.merge(g,on=["ldate","home","away"],how="inner"))
m=pd.concat(parts).drop_duplicates("game_id")
print("games with both:",len(m),"of",len(wf)); 
m=m[m.date>="2020-12-01"].copy(); m["line"]=-m.home_spread_close; m["gap"]=m.model_margin-m.line
print("margin check (box vs line table):",float((m.margin-m.m2).abs().mean()))
rm=lambda a:np.sqrt((a**2).mean())
def ols(y,X):
    X=np.c_[np.ones(len(y)),X]; b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b; V=np.linalg.inv(X.T@X)*r.var(ddof=X.shape[1]); return b,np.sqrt(np.diag(V))
print("n",len(m),"| RMSE: closing line %.2f | box-score model %.2f | blend 80/20 %.2f | open line %.2f"%(rm(m.margin-m.line),rm(m.margin-m.model_margin),rm(m.margin-(.8*m.line+.2*m.model_margin)),rm(m.margin+m.home_spread_open)))
b,se=ols((m.margin-m.line).values,m.gap.values); print("(margin - line) ~ gap: slope %+.3f (SE %.3f) t=%.1f"%(b[1],se[1],b[1]/se[1]))
for s,x in m.groupby(m.date.dt.year+(m.date.dt.month>=9)*0):
    pass
m["sea"]=np.where(m.date.dt.month>=9,m.date.dt.year,m.date.dt.year-1)
for s,x in m.groupby("sea"):
    bb,ss=ols((x.margin-x.line).values,x.gap.values); print(f"  {s}-{str(s+1)[2:]}: n={len(x):4d} slope {bb[1]:+.3f} (SE {ss[1]:.3f}) | RMSE line {rm(x.margin-x.line):.2f} model {rm(x.margin-x.model_margin):.2f}")
tb,tse=ols((m.p2-m.total_close).values,(m.model_total-m.total_close).values); print("totals: slope %+.3f (SE %.3f) | RMSE line %.2f model %.2f"%(tb[1],tse[1],rm(m.p2-m.total_close),rm(m.p2-m.model_total)))
for th in (2,3,4):
    big=m[m.gap.abs()>=th]; cov=np.sign(big.gap)*(big.margin-big.line); print("|gap|>=%d: n=%d model-side cover margin %+.2f (SE %.2f) cover %.3f"%(th,len(big),cov.mean(),cov.std()/np.sqrt(len(big)),(cov>0).mean()))
m.to_pickle(S+"merged1.pkl")
