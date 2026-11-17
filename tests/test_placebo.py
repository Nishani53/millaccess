import numpy as np
import pytest

from millaccess import placebo_draws, placebo_summary, placebo_test

POOL = [f"MI_{i:04d}" for i in range(11)]
YEARS = [2005, 2005, 2006, 2009]


def test_draws_match_original_notebook_rng():
    """Draws are rng.choice of mills, then rng.permutation of years, from one seeded generator."""
    rng = np.random.default_rng(20261108)
    expected = []
    for _ in range(20):
        m = rng.choice(np.array(POOL), size=4, replace=False)
        y = rng.permutation(YEARS)
        expected.append(list(zip(m.tolist(), [int(v) for v in y])))
    assert list(placebo_draws(POOL, YEARS, 20, seed=20261108)) == expected


def test_draws_are_valid():
    for ev in placebo_draws(POOL, YEARS, 50, seed=1):
        mills = [m for m, _ in ev]
        assert len(set(mills)) == 4 and set(mills) <= set(POOL)
        assert sorted(y for _, y in ev) == sorted(YEARS)


def test_placebo_test_and_summary():
    p = placebo_test(lambda ev: {"r0_40": float(len(ev))}, POOL, YEARS, n_draws=10, seed=3)
    assert len(p) == 10 and (p.r0_40 == 4.0).all()
    s = placebo_summary(18.8, np.arange(100))
    assert s["share_larger"] == pytest.approx(0.81)


def test_pool_too_small():
    with pytest.raises(ValueError):
        list(placebo_draws(POOL[:2], YEARS, 1))
