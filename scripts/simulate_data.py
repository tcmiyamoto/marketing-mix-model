"""Simulate 3 years of weekly data for a fictional omnichannel beauty retailer.

Because the data is simulated, the *true* channel effects are known. That lets us
check whether the MMM recovers them, which is impossible with real data.

Built-in trap: an unobserved demand driver (trends, viral products, competitor
stockouts) lifts revenue AND search / affiliate spend at the same time. Search
spend follows query volume and affiliate is commission-based, so both rise when
demand rises on its own. A regression will over-credit them. This is the most
common real-world MMM failure, and the reason lift tests are used to calibrate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmm.transforms import geometric_adstock, hill_saturation  # noqa: E402

rng = np.random.default_rng(42)
N = 156  # 3 years of weeks
weeks = pd.date_range("2023-01-02", periods=N, freq="W-MON")
t = np.arange(N)
woy = weeks.isocalendar().week.to_numpy().astype(int)

# ---- Calendar events -------------------------------------------------------
spring_sale = np.isin(woy, [15, 16]).astype(int)    # spring loyalty sale event
holiday_sale = np.isin(woy, [45, 46]).astype(int)   # fall loyalty sale event
bfcm = (woy == 48).astype(int)                      # Black Friday / Cyber Monday
december = np.isin(woy, [49, 50, 51]).astype(int)   # gifting season

# ---- Baseline demand (no paid media) ---------------------------------------
trend = 1 + 0.18 * t / N
season = 1 + 0.06 * np.sin(2 * np.pi * (t - 8) / 52)
events = 1 + 0.22 * spring_sale + 0.28 * holiday_sale + 0.40 * bfcm + 0.25 * december

# Unobserved demand shock: AR(1), not available to the model
u = np.zeros(N)
for i in range(1, N):
    u[i] = 0.85 * u[i - 1] + rng.normal(0, 0.035)

base_revenue = 3_200_000 * trend * season * events * (1 + u)

# ---- Media spend -----------------------------------------------------------
demand_idx = trend * season * events

def flights(on_prob, min_len=3):
    """On/off campaign flighting for upper-funnel channels."""
    on = np.zeros(N)
    i = 0
    while i < N:
        state = rng.random() < on_prob
        length = rng.integers(min_len, min_len + 5)
        on[i:i + length] = state
        i += length
    return on

spend = pd.DataFrame(index=weeks)
spend["paid_social"] = 175_000 * (0.55 + 0.45 * flights(0.6)) * (1 + 0.25 * (spring_sale + holiday_sale)) * rng.lognormal(0, 0.25, N)
spend["programmatic"] = 140_000 * (0.35 + 0.65 * flights(0.6)) * rng.lognormal(0, 0.2, N)
spend["paid_search"] = 150_000 * demand_idx * (1 + 2.5 * u) * rng.lognormal(0, 0.06, N)
spend["affiliate"] = 70_000 * demand_idx * (1 + 3.0 * u) * rng.lognormal(0, 0.05, N)
spend["online_video"] = 120_000 * flights(0.45, min_len=4) * rng.lognormal(0, 0.2, N)
spend = spend.clip(lower=0).round(0)

# ---- True media effects ----------------------------------------------------
TRUE = {
    #               decay  half_sat shape  target ROI
    "paid_social":  (0.40, 1.00, 1.4, 2.10),
    "programmatic": (0.30, 0.90, 1.2, 1.20),
    "paid_search":  (0.10, 0.80, 1.0, 1.50),
    "affiliate":    (0.05, 1.00, 1.0, 0.80),
    "online_video": (0.70, 0.70, 1.6, 1.70),
}

media_rev = pd.DataFrame(index=weeks)
truth = {}
for ch, (decay, half_sat, shape, roi) in TRUE.items():
    s = spend[ch].to_numpy()
    resp = hill_saturation(geometric_adstock(s / s.mean(), decay), half_sat, shape)
    beta = roi * s.sum() / resp.sum()  # scale so the channel's true ROI equals target
    media_rev[ch] = beta * resp
    truth[ch] = {"decay": decay, "half_sat": half_sat, "shape": shape,
                 "beta": float(beta), "roi": roi,
                 "contribution": float(media_rev[ch].sum())}

noise = rng.normal(0, 0.03, N)
revenue = (base_revenue + media_rev.sum(axis=1).to_numpy()) * (1 + noise)

df = pd.DataFrame({"week": weeks.date})
df[spend.columns] = spend.to_numpy()
df["spring_sale"] = spring_sale
df["holiday_sale"] = holiday_sale
df["bfcm"] = bfcm
df["december"] = december
df["revenue"] = revenue.round(0)

# ---- Lift tests ------------------------------------------------------------
# Experiments measure the true causal effect (plus sampling noise). Results are
# scaled to national-equivalent incremental revenue over the test window.
def lift_test(ch, w0, w1, design, noise_sd=0.06):
    true_window = float(media_rev[ch].iloc[w0:w1].sum())
    return {
        "channel": ch,
        "start_week": str(weeks[w0].date()), "end_week": str(weeks[w1 - 1].date()),
        "start_idx": w0, "end_idx": w1,
        "incremental_revenue": round(true_window * (1 + rng.normal(0, noise_sd)), 0),
        "spend_in_window": float(spend[ch].iloc[w0:w1].sum()),
        "design": design,
        "_true_incremental_revenue": round(true_window, 0),
    }

lift_tests = [
    lift_test("paid_search", 110, 118,
              "Geo holdout: search paused in ~20% of DMAs for 8 weeks, scaled to national."),
    lift_test("affiliate", 128, 136,
              "Publisher holdout: affiliate commissions paused for a randomized 25% of publishers for 8 weeks."),
]
# Spend-scaling geo test: search spend raised 40% in test DMAs for 8 weeks.
# Measures the *marginal* return (incremental revenue / incremental spend), which
# holdouts can't: a holdout tells you the average return of the channel, not the
# return on the next dollar.
w0, w1, inc = 140, 148, 0.40
lvl = float(spend["paid_search"].iloc[w0:w1].mean())
d, k, sh, _ = TRUE["paid_search"]
b = truth["paid_search"]["beta"]
m = spend["paid_search"].mean()
true_mroi = b * (hill_saturation((1 + inc) * lvl / m, k, sh) - hill_saturation(lvl / m, k, sh)) / (inc * lvl)
lift_tests.append({
    "channel": "paid_search", "type": "scale",
    "start_week": str(weeks[w0].date()), "end_week": str(weeks[w1 - 1].date()),
    "start_idx": w0, "end_idx": w1, "spend_level": round(lvl, 0), "increase": inc,
    "incremental_roi": round(float(true_mroi * (1 + rng.normal(0, 0.08))), 3),
    "design": "Geo scale test: search spend +40% in test DMAs for 8 weeks; incremental revenue / incremental spend.",
    "_true_incremental_revenue": round(float(true_mroi), 3),
})

truth_tests = [{k: lt.pop("_true_incremental_revenue") for k in ["_true_incremental_revenue"]} | {"channel": lt["channel"]} for lt in lift_tests]

out = ROOT / "data"
out.mkdir(exist_ok=True)
df.to_csv(out / "weekly_marketing.csv", index=False)
with open(out / "lift_tests.json", "w") as f:
    json.dump(lift_tests, f, indent=2)
with open(out / "ground_truth.json", "w") as f:
    json.dump({"channels": truth, "lift_tests_true_values": truth_tests}, f, indent=2)

print(f"Wrote {len(df)} weeks. Total spend ${spend.to_numpy().sum()/1e6:.1f}M, "
      f"revenue ${df.revenue.sum()/1e6:.0f}M")
print("Spend correlation with hidden demand shock:")
for ch in spend.columns:
    print(f"  {ch:13s} {np.corrcoef(spend[ch], u)[0,1]:+.2f}")
