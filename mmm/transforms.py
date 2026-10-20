"""Media transforms used by the MMM.

Two ideas drive every MMM:
  1. Adstock (carryover): ad exposure this week still drives sales in later weeks.
  2. Saturation (diminishing returns): each extra dollar buys less than the last.
"""
import numpy as np
from scipy.signal import lfilter


def geometric_adstock(x, decay):
    """Normalized geometric adstock: a_t = (1 - decay) * x_t + decay * a_{t-1}.

    decay = 0   -> no carryover (all impact lands the week of spend)
    decay = 0.7 -> 30% of impact lands this week, the rest decays over later weeks
    Weights sum to 1, so adstock reshapes *when* impact lands without inflating it.
    Half-life in weeks = ln(0.5) / ln(decay).
    """
    x = np.asarray(x, dtype=float)
    return lfilter([1.0 - decay], [1.0, -decay], x)


def hill_saturation(x, half_sat, shape):
    """Hill curve, scaled 0..1.

    half_sat: the (normalized) spend level where the channel reaches 50% of its max effect.
    shape:    > 1 gives an S-curve (needs a minimum spend to "break through"),
              <= 1 gives a concave curve (diminishing returns from the first dollar).
    """
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    return x**shape / (x**shape + half_sat**shape)


def half_life(decay):
    """Weeks until a dollar of spend has delivered half of its total effect."""
    if decay <= 0:
        return 0.0
    return float(np.log(0.5) / np.log(decay))
