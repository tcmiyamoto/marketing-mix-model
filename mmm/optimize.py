"""Budget reallocation: hold total weekly budget fixed, move dollars to where the
marginal return is highest, within guardrails (each channel can move +/- a set %)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_budget(model, current_weekly_spend: dict, max_change=0.30):
    chans = list(current_weekly_spend)
    cur = np.array([current_weekly_spend[c] for c in chans], dtype=float)
    total = cur.sum()

    def neg_revenue(x):
        return -sum(model.steady_state_response(c, x[i]) for i, c in enumerate(chans))

    bounds = [(v * (1 - max_change), v * (1 + max_change)) for v in cur]
    cons = [{"type": "eq", "fun": lambda x: x.sum() - total}]
    res = minimize(neg_revenue, cur, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": 500})
    opt = res.x
    out = pd.DataFrame({
        "current_spend": cur,
        "optimized_spend": opt,
        "change_pct": opt / cur - 1,
        "current_revenue": [model.steady_state_response(c, cur[i]) for i, c in enumerate(chans)],
        "optimized_revenue": [model.steady_state_response(c, opt[i]) for i, c in enumerate(chans)],
        "mroi_current": [model.marginal_roi(c, cur[i]) for i, c in enumerate(chans)],
        "mroi_optimized": [model.marginal_roi(c, opt[i]) for i, c in enumerate(chans)],
    }, index=chans)
    return out
