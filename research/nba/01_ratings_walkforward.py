"""Fit the possession-based team ratings walk-forward over every box-scored game (about a minute). Writes data/model/nba/wf_ratings.pkl."""
import sys; sys.path.insert(0, ".")
import numpy as np
from pm_edges import nba_ratings as R

rows = R.team_rows(); rows["home"] = rows.home.astype(bool)
wf = R.walk_forward(rows)
wf.to_pickle("data/model/nba/wf_ratings.pkl")
print("games", len(wf), "| RMSE of model margin vs actual %.2f (predicting 0: %.2f)" % (np.sqrt(((wf.margin - wf.model_margin) ** 2).mean()), np.sqrt((wf.margin ** 2).mean())))
