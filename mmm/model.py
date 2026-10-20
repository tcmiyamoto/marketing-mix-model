"""A transparent, from-scratch Marketing Mix Model.

revenue_t = base_t + sum_c beta_c * Hill(Adstock(spend_c / mean_c; decay_c); half_sat_c, shape_c) + error

base_t = intercept + trend + Fourier seasonality + calendar events

Fitting uses "variable projection":
  * Outer loop: a global optimizer (differential evolution) searches the nonlinear
    media parameters (decay, half_sat, shape per channel).
  * Inner loop: for any candidate, the remaining coefficients are linear, so they are
    solved exactly with bounded least squares (media betas forced >= 0, since spend
    cannot reduce sales) plus a light ridge penalty for stability.

Calibration (optional): experiment results are added to the objective as a penalty
when the model disagrees with what the experiment measured. Two test types:
  * holdout: incremental revenue the channel drove in the test window (pins average ROI)
  * scale:   incremental ROI of raising spend by X% (pins the marginal ROI / curve shape) Same idea as Meta Robyn's
calibration and the lift-test priors in Google Meridian / PyMC-Marketing.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, lsq_linear

from .transforms import geometric_adstock, hill_saturation

BOUNDS = {"decay": (0.0, 0.9), "half_sat": (0.2, 3.0), "shape": (0.5, 2.0)}


@dataclass
class MMM:
    channels: list
    controls: list
    ridge: float = 0.05
    calibration_weight: float = 1.0
    rssd_weight: float = 0.0
    n_fourier: int = 2
    seed: int = 7
    params_: dict = field(default_factory=dict)

    # ---------- design matrix ------------------------------------------------
    def _base_matrix(self, n, t0=0):
        t = np.arange(t0, t0 + n)
        cols = {"intercept": np.ones(n), "trend": t / 156}
        for k in range(1, self.n_fourier + 1):
            cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52.18)
            cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52.18)
        return pd.DataFrame(cols)

    def _media_response(self, spend, theta):
        """spend: (n, C) array. Returns (n, C) saturated response, each column 0..1."""
        out = np.empty_like(spend, dtype=float)
        for j in range(len(self.channels)):
            d, k, s = theta[3 * j: 3 * j + 3]
            x = geometric_adstock(spend[:, j] / self.spend_mean_[j], d)
            out[:, j] = hill_saturation(x, k, s)
        return out

    def _solve_linear(self, X, y, n_media):
        n_base = X.shape[1] - n_media
        # ridge rows on media coefficients only (scaled to the target)
        R = np.zeros((n_media, X.shape[1]))
        R[:, n_base:] = np.eye(n_media) * np.sqrt(self.ridge * len(y))
        A = np.vstack([X, R])
        b = np.concatenate([y, np.zeros(n_media)])
        lb = np.r_[np.full(n_base, -np.inf), np.zeros(n_media)]
        ub = np.full(X.shape[1], np.inf)
        return lsq_linear(A, b, bounds=(lb, ub), method="bvls").x

    # ---------- fitting ------------------------------------------------------
    def fit(self, df, revenue_col="revenue", lift_tests=None, maxiter=250):
        self.spend_mean_ = df[self.channels].mean().to_numpy()
        self.y_scale_ = df[revenue_col].mean()
        y = df[revenue_col].to_numpy() / self.y_scale_
        spend = df[self.channels].to_numpy(dtype=float)
        base = np.column_stack([self._base_matrix(len(df)).to_numpy(),
                                df[self.controls].to_numpy(dtype=float)])
        sst = np.sum((y - y.mean()) ** 2)
        C = len(self.channels)
        spend_share = spend.sum(axis=0) / spend.sum()
        lift_tests = [lt for lt in (lift_tests or []) if lt["end_idx"] <= len(df)]

        def objective(theta):
            M = self._media_response(spend, theta)
            X = np.column_stack([base, M])
            coef = self._solve_linear(X, y, C)
            loss = np.sum((y - X @ coef) ** 2) / sst
            for lt in lift_tests:
                j = self.channels.index(lt["channel"])
                beta = coef[base.shape[1] + j]
                if lt.get("type", "holdout") == "scale":
                    # marginal ROI of raising spend from spend_level by `increase`
                    d, k, s = theta[3 * j: 3 * j + 3]
                    x = lt["spend_level"] / self.spend_mean_[j]
                    gain = hill_saturation(x * (1 + lt["increase"]), k, s) - hill_saturation(x, k, s)
                    est = beta * gain * self.y_scale_ / (lt["increase"] * lt["spend_level"])
                    target = lt["incremental_roi"]
                else:
                    # holdout: incremental revenue the channel drove inside the test window
                    est = beta * M[lt["start_idx"]:lt["end_idx"], j].sum()
                    target = lt["incremental_revenue"] / self.y_scale_
                loss += self.calibration_weight * ((est - target) / target) ** 2
            if self.rssd_weight:
                # DECOMP.RSSD (Meta Robyn): distance between each channel's share of
                # media effect and its share of spend. Penalizes implausible splits
                # (e.g., one channel getting 70% of credit on 15% of spend).
                eff = coef[base.shape[1]:] * M.sum(axis=0)
                if eff.sum() > 0:
                    rssd = np.sqrt(np.sum((eff / eff.sum() - spend_share) ** 2))
                    loss += self.rssd_weight * rssd
            return loss

        bounds = [BOUNDS[p] for _ in self.channels for p in ("decay", "half_sat", "shape")]
        res = differential_evolution(objective, bounds, seed=self.seed, maxiter=maxiter,
                                     popsize=12, tol=1e-8, polish=True)
        theta = res.x
        M = self._media_response(spend, theta)
        X = np.column_stack([base, M])
        coef = self._solve_linear(X, y, C)

        self.theta_ = theta
        self.coef_ = coef
        self.n_base_ = base.shape[1]
        self.base_names_ = list(self._base_matrix(1).columns) + self.controls
        self.params_ = {
            ch: {"decay": theta[3 * j], "half_sat": theta[3 * j + 1], "shape": theta[3 * j + 2],
                 "beta": coef[self.n_base_ + j] * self.y_scale_}
            for j, ch in enumerate(self.channels)
        }
        self.n_train_ = len(df)
        return self

    # ---------- outputs ------------------------------------------------------
    def decompose(self, df):
        """Weekly revenue split into base components and each channel's contribution."""
        spend = df[self.channels].to_numpy(dtype=float)
        base = np.column_stack([self._base_matrix(len(df)).to_numpy(),
                                df[self.controls].to_numpy(dtype=float)])
        M = self._media_response(spend, self.theta_)
        out = pd.DataFrame(index=df.index)
        out["base"] = base @ self.coef_[: self.n_base_] * self.y_scale_
        for j, ch in enumerate(self.channels):
            out[ch] = M[:, j] * self.coef_[self.n_base_ + j] * self.y_scale_
        out["predicted"] = out.sum(axis=1)
        return out

    def predict(self, df):
        return self.decompose(df)["predicted"].to_numpy()

    def channel_summary(self, df):
        dec = self.decompose(df)
        rows = []
        for ch in self.channels:
            p = self.params_[ch]
            spend = df[ch].sum()
            contrib = dec[ch].sum()
            rows.append({"channel": ch, "spend": spend, "contribution": contrib,
                         "roi": contrib / spend, "mroi": self.marginal_roi(ch, df[ch].mean()),
                         "decay": p["decay"], "half_sat": p["half_sat"], "shape": p["shape"]})
        return pd.DataFrame(rows).set_index("channel")

    def steady_state_response(self, ch, weekly_spend):
        """Long-run weekly revenue if `weekly_spend` were held constant.
        (Normalized adstock converges to the spend level itself.)"""
        p = self.params_[ch]
        j = self.channels.index(ch)
        x = np.asarray(weekly_spend, dtype=float) / self.spend_mean_[j]
        return p["beta"] * hill_saturation(x, p["half_sat"], p["shape"])

    def marginal_roi(self, ch, weekly_spend, step=1000.0):
        """Revenue from the next $1 at a given weekly spend level."""
        r0 = self.steady_state_response(ch, weekly_spend)
        r1 = self.steady_state_response(ch, weekly_spend + step)
        return float((r1 - r0) / step)
