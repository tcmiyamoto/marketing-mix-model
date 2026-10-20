"""End-to-end MMM run: validate, fit three model versions, calibrate, optimize, report.

Outputs:
  reports/figures/*.png
  reports/results.md      (tables referenced in the README)
  reports/results.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmm.model import MMM  # noqa: E402
from mmm.optimize import optimize_budget  # noqa: E402
from mmm.transforms import half_life  # noqa: E402

CHANNELS = ["paid_social", "programmatic", "paid_search", "affiliate", "online_video"]
CONTROLS = ["spring_sale", "holiday_sale", "bfcm", "december"]
LABELS = {"paid_social": "Paid Social", "programmatic": "Programmatic", "paid_search": "Paid Search",
          "affiliate": "Affiliate", "online_video": "Online Video"}
COLORS = {"paid_social": "#2a78d6", "programmatic": "#eb6834", "paid_search": "#1baf7a",
          "affiliate": "#eda100", "online_video": "#e87ba4", "base": "#c9c8c1"}
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e6e5df"
MODEL_KW = dict(ridge=0.001, rssd_weight=0.05)
MAXITER = 250

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 150,
})

df = pd.read_csv(ROOT / "data/weekly_marketing.csv", parse_dates=["week"])
lift_tests = json.load(open(ROOT / "data/lift_tests.json"))
truth = json.load(open(ROOT / "data/ground_truth.json"))["channels"]
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)


def mape(a, p):
    return float(np.mean(np.abs(a - p) / a))


def r2(a, p):
    return float(1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2))


# 1) Out-of-sample validation: train on first 130 weeks, forecast last 26 ------------
split = 130
train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
val_model = MMM(CHANNELS, CONTROLS, **MODEL_KW).fit(train, lift_tests=lift_tests, maxiter=MAXITER)
# predict holdout: rebuild full series so adstock carries over and trend/season index continue
full_pred = val_model.predict(df)
validation = {
    "train_mape": mape(train.revenue.to_numpy(), full_pred[:split]),
    "test_mape": mape(test.revenue.to_numpy(), full_pred[split:]),
    "train_r2": r2(train.revenue.to_numpy(), full_pred[:split]),
    "test_r2": r2(test.revenue.to_numpy(), full_pred[split:]),
}
print("Validation:", {k: round(v, 3) for k, v in validation.items()})

# 2) Three model versions on full data ------------------------------------------------
versions = {
    "Uncalibrated": [],
    "+ Search holdout": lift_tests[:1],
    "+ Affiliate holdout": lift_tests[:2],
    "+ Search scale test": lift_tests[:3],
}
models, summaries = {}, {}
for name, tests in versions.items():
    m = MMM(CHANNELS, CONTROLS, **MODEL_KW).fit(df, lift_tests=tests, maxiter=MAXITER)
    models[name], summaries[name] = m, m.channel_summary(df)
    print(name, summaries[name]["roi"].round(2).to_dict())
final_name = "+ Search scale test"
final = models[final_name]
fs = summaries[final_name]

roi_table = pd.DataFrame({n: s["roi"] for n, s in summaries.items()})
roi_table["True ROI"] = [truth[c]["roi"] for c in CHANNELS]
roi_table["Spend ($M)"] = fs["spend"] / 1e6

# 3) Budget reallocation (final model), at average weekly spend, +/-30% guardrails -----
current = {c: df[c].mean() for c in CHANNELS}
opt = optimize_budget(final, current, max_change=0.30)
uplift_wk = opt.optimized_revenue.sum() - opt.current_revenue.sum()

# Reality check only possible with simulated data: score the recommended plan on the
# TRUE response curves. Does the reallocation actually make money?
from mmm.transforms import hill_saturation  # noqa: E402
def true_response(c, s):
    t = truth[c]
    return t["beta"] * hill_saturation(s / df[c].mean(), t["half_sat"], t["shape"])
plans = {}
for name, m in models.items():
    plan = optimize_budget(m, current, max_change=0.30)
    modeled = plan.optimized_revenue.sum() - plan.current_revenue.sum()
    true = sum(true_response(c, plan.loc[c, "optimized_spend"]) - true_response(c, plan.loc[c, "current_spend"])
               for c in CHANNELS)
    plans[name] = {"plan": plan, "modeled_wk": modeled, "true_wk": true}
true_uplift_wk = plans[final_name]["true_wk"]
# best achievable under the same guardrails, using the true curves
from scipy.optimize import minimize  # noqa: E402
cur_v = np.array([current[c] for c in CHANNELS])
best = minimize(lambda x: -sum(true_response(c, x[i]) for i, c in enumerate(CHANNELS)), cur_v,
                method="SLSQP", bounds=[(v * 0.7, v * 1.3) for v in cur_v],
                constraints=[{"type": "eq", "fun": lambda x: x.sum() - cur_v.sum()}])
oracle_wk = -best.fun - sum(true_response(c, current[c]) for c in CHANNELS)
for n, p_ in plans.items():
    print(f"{n:22s} modeled {p_['modeled_wk']*52/1e6:+.1f}M/yr  true {p_['true_wk']*52/1e6:+.1f}M/yr")
print(f"Best possible (oracle) {oracle_wk*52/1e6:+.1f}M/yr")

# ---------------------------- Figures ------------------------------------------------
def fmt_m(ax, axis="y"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1e6:.0f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}K")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# Fig 1: actual vs predicted with holdout
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(df.week, df.revenue, color=INK, lw=1.6, label="Actual revenue")
ax.plot(df.week[:split], full_pred[:split], color=COLORS["paid_social"], lw=2, label="Model fit (train)")
ax.plot(df.week[split:], full_pred[split:], color=COLORS["programmatic"], lw=2, ls="--", label="Forecast (holdout)")
ax.axvspan(df.week[split], df.week.iloc[-1], color="#f3f2ee", zorder=0)
ax.text(df.week[split + 1], ax.get_ylim()[1] * 0.985, "26-week holdout", va="top", color=MUTED, fontsize=9)
fmt_m(ax)
ax.set_title(f"Model fit: {validation['train_mape']:.1%} MAPE in-sample, {validation['test_mape']:.1%} on unseen holdout weeks")
ax.legend(loc="upper left", ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "01_fit_and_holdout.png"); plt.close(fig)

# Fig 2: weekly decomposition
dec = final.decompose(df)
fig, ax = plt.subplots(figsize=(10, 4.2))
layers = ["base"] + CHANNELS
ax.stackplot(df.week, *[dec[c].clip(lower=0) for c in layers],
             colors=[COLORS[c] for c in layers], edgecolor="white", linewidth=0.4,
             labels=["Base (organic, seasonality, events)"] + [LABELS[c] for c in CHANNELS])
ax.plot(df.week, df.revenue, color=INK, lw=1.2, label="Actual revenue")
fmt_m(ax)
ax.set_ylim(0, df.revenue.max() * 1.05)
ax.set_title("Weekly revenue decomposition: base demand vs. each paid channel")
ax.legend(loc="upper left", ncol=4, fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "02_decomposition.png"); plt.close(fig)

# Fig 3: ROI by model version vs truth
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(CHANNELS)); w = 0.2
vcols = ["#c9c8c1", COLORS["paid_social"], COLORS["programmatic"], COLORS["paid_search"]]
for i, (name, col) in enumerate(zip(versions, vcols)):
    vals = roi_table[name].to_numpy()
    ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=col, label=name, zorder=2)
for i, c in enumerate(CHANNELS):
    ax.hlines(truth[c]["roi"], i - 2 * w, i + 2 * w, color=INK, lw=2.2, zorder=3,
              label="True ROI (known because data is simulated)" if i == 0 else None)
ax.set_xticks(x, [LABELS[c] for c in CHANNELS])
ax.set_ylabel("Revenue per $1 spent")
ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
ax.set_title("Each lift test pulls confounded channels toward their true ROI (black line)")
fig.tight_layout(); fig.savefig(FIG / "03_roi_calibration.png"); plt.close(fig)

# Fig 4: response curves
fig, ax = plt.subplots(figsize=(10, 4.2))
for c in CHANNELS:
    grid = np.linspace(0, df[c].mean() * 3, 200)
    ax.plot(grid, final.steady_state_response(c, grid), color=COLORS[c], lw=2, label=LABELS[c])
    cur = df[c].mean()
    ax.plot(cur, final.steady_state_response(c, cur), "o", ms=8, color=COLORS[c],
            markeredgecolor="white", markeredgewidth=2, zorder=4)
fmt_m(ax); fmt_m(ax, "x")
ax.set_xlabel("Weekly spend (dot = current average)")
ax.set_ylabel("Weekly incremental revenue")
ax.set_title("Response curves: where each channel sits on its diminishing-returns curve")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "04_response_curves.png"); plt.close(fig)

# Fig 5: budget reallocation
fig, ax = plt.subplots(figsize=(10, 3.8))
y = np.arange(len(CHANNELS))
ax.barh(y + 0.2, opt.current_spend, 0.38, color="#c9c8c1", label="Current", zorder=2)
ax.barh(y - 0.2, opt.optimized_spend, 0.38, color=COLORS["paid_social"], label="Recommended", zorder=2)
for i, c in enumerate(CHANNELS):
    ch = opt.loc[c, "change_pct"]
    ax.text(max(opt.loc[c, "current_spend"], opt.loc[c, "optimized_spend"]) * 1.02, i,
            f"{ch:+.0%}", va="center", color=INK, fontsize=9)
ax.set_yticks(y, [LABELS[c] for c in CHANNELS]); ax.invert_yaxis()
fmt_m(ax, "x")
ax.set_xlabel("Weekly spend")
ax.set_title(f"Same total budget, reallocated: +${uplift_wk*52/1e6:.1f}M modeled annual revenue (±30% guardrails)")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "05_budget_reallocation.png"); plt.close(fig)

# Fig 6: does each model's budget plan actually make money?
fig, ax = plt.subplots(figsize=(10, 3.6))
names = list(plans) + ["Best possible\n(true curves)"]
modeled = [plans[n]["modeled_wk"] * 52 for n in plans] + [np.nan]
true_v = [plans[n]["true_wk"] * 52 for n in plans] + [oracle_wk * 52]
xx = np.arange(len(names))
ax.bar(xx - 0.2, modeled, 0.38, color="#c9c8c1", label="What the model promised", zorder=2)
ax.bar(xx + 0.2, true_v, 0.38, color=COLORS["paid_social"], label="What actually happens (true curves)", zorder=2)
ax.axhline(0, color=INK, lw=1)
for i, v in enumerate(true_v):
    ax.text(i + 0.2, v + (np.sign(v) or 1) * abs(np.nanmax(modeled)) * 0.03, f"{'+' if v >= 0 else '-'}${abs(v)/1e6:.1f}M",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
ax.set_xticks(xx, names, fontsize=9)
fmt_m(ax)
ax.set_ylabel("Annual revenue change")
ax.set_ylim(min(true_v) * 1.4, np.nanmax(modeled) * 1.35)
ax.set_title("Same budget, reallocated: only the fully calibrated model's plan actually makes money")
ax.legend(fontsize=9, loc="upper right", ncol=2)
fig.tight_layout(); fig.savefig(FIG / "06_plan_reality_check.png"); plt.close(fig)

# ---------------------------- Tables -------------------------------------------------
def md(df_):
    return df_.to_markdown()

params = pd.DataFrame({
    "Fitted decay": fs["decay"].to_numpy(), "True decay": [truth[c]["decay"] for c in CHANNELS],
    "Fitted half-life (wks)": [half_life(d) for d in fs["decay"]],
    "Fitted ROI": fs["roi"].to_numpy(), "True ROI": [truth[c]["roi"] for c in CHANNELS],
    "Marginal ROI": fs["mroi"].to_numpy(),
}, index=[LABELS[c] for c in CHANNELS]).round(2)

rt = roi_table.copy(); rt.index = [LABELS[c] for c in CHANNELS]
o = opt.copy(); o.index = [LABELS[c] for c in CHANNELS]
o_out = pd.DataFrame({
    "Current $/wk": o.current_spend.map("${:,.0f}".format),
    "Recommended $/wk": o.optimized_spend.map("${:,.0f}".format),
    "Change": o.change_pct.map("{:+.0%}".format),
    "mROI now": o.mroi_current.round(2), "mROI after": o.mroi_optimized.round(2),
})
media_share = dec[CHANNELS].sum().sum() / dec["predicted"].sum()
true_media = sum(truth[c]["contribution"] for c in CHANNELS) / df.revenue.sum()

with open(ROOT / "reports/results.md", "w") as f:
    f.write("# Results\n\n_Generated by `scripts/run_mmm.py`._\n\n")
    f.write("## Validation\n\n")
    f.write(f"| | MAPE | R² |\n|---|---|---|\n| Train (130 wks) | {validation['train_mape']:.1%} | {validation['train_r2']:.2f} |\n"
            f"| Holdout (26 wks) | {validation['test_mape']:.1%} | {validation['test_r2']:.2f} |\n\n")
    f.write(f"Paid media share of revenue: model {media_share:.1%} vs. true {true_media:.1%}.\n\n")
    f.write("## ROI by model version\n\n" + md(rt.round(2)) + "\n\n")
    f.write("## Final model parameters\n\n" + md(params) + "\n\n")
    f.write(f"## Budget reallocation (same total, ±30% per channel)\n\n{o_out.to_markdown()}\n\n"
            f"Modeled uplift: ${uplift_wk:,.0f}/week, about ${uplift_wk*52/1e6:.1f}M/year "
            f"({uplift_wk/opt.current_revenue.sum():.1%} of paid-media revenue).\n\n"
            "## Reality check: does each model's budget plan actually make money?\n\n"
            "Each model's recommended plan, scored against the true response curves (only possible because the data is simulated).\n\n"
            + pd.DataFrame({"Modeled $/yr": [f"${p_['modeled_wk']*52/1e6:+.1f}M" for p_ in plans.values()] + ["n/a"],
                            "True $/yr": [f"${p_['true_wk']*52/1e6:+.1f}M" for p_ in plans.values()] + [f"${oracle_wk*52/1e6:+.1f}M"],
                            **{LABELS[c]: [f"{p_['plan'].loc[c,'change_pct']:+.0%}" for p_ in plans.values()] + ["n/a"] for c in CHANNELS}},
                           index=list(plans) + ["Best possible (true curves)"]).to_markdown() + "\n")

json.dump({"validation": validation, "roi_by_version": roi_table.round(3).to_dict(),
           "media_share_model": media_share, "media_share_true": true_media,
           "weekly_uplift": uplift_wk, "plans": {n: {"modeled_wk": p_["modeled_wk"], "true_wk": p_["true_wk"]} for n, p_ in plans.items()}, "oracle_wk": oracle_wk, "reallocation": opt.round(3).to_dict()},
          open(ROOT / "reports/results.json", "w"), indent=2, default=float)
print(open(ROOT / "reports/results.md").read())
