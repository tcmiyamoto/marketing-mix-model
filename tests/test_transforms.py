import numpy as np

from mmm.transforms import geometric_adstock, half_life, hill_saturation


def test_adstock_preserves_total_effect():
    x = np.r_[100.0, np.zeros(200)]
    a = geometric_adstock(x, 0.6)
    assert np.isclose(a.sum(), 100.0)       # normalized: carryover reshapes, never inflates
    assert np.isclose(a[0], 40.0)           # 40% lands in week of spend
    assert np.isclose(a[1], 24.0)           # then decays by 0.6 per week


def test_adstock_zero_decay_is_identity():
    x = np.array([5.0, 0.0, 3.0])
    assert np.allclose(geometric_adstock(x, 0.0), x)


def test_hill_half_saturation_point():
    assert np.isclose(hill_saturation(1.5, half_sat=1.5, shape=2.0), 0.5)
    y = hill_saturation(np.linspace(0, 10, 50), 1.0, 1.0)
    assert np.all(np.diff(y) > 0) and y.max() < 1  # monotonic, bounded


def test_half_life():
    assert np.isclose(half_life(0.5), 1.0)
    assert half_life(0.0) == 0.0
