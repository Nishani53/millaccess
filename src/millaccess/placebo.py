"""Placebo (randomization) inference for a small number of events.

With few events, clustered standard errors can mislead (Conley & Taber, 2011). Here
the real event years are assigned at random to mills that never had the event (e.g.
mills that never closed), the same estimator is re-run for each draw, and the real
estimate is compared with the resulting distribution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def placebo_draws(pool: Sequence, event_years: Sequence[int], n_draws: int = 200, seed: int | None = None):
    """Yield placebo event sets: lists of ``(mill_id, year)``.

    Each draw picks ``len(event_years)`` distinct mills from ``pool`` and gives them the
    real event years in random order. With a fixed ``seed`` the draws are identical
    every run (the order of ``pool`` matters).
    """
    pool = np.asarray(pool)
    if len(pool) < len(event_years):
        raise ValueError("the placebo pool is smaller than the number of events")
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        mills = rng.choice(pool, size=len(event_years), replace=False)
        years = rng.permutation(list(event_years))
        yield list(zip(mills.tolist(), [int(y) for y in years]))


def placebo_test(
    estimate: Callable[[list], float | dict],
    pool: Sequence,
    event_years: Sequence[int],
    n_draws: int = 200,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run ``estimate`` on every placebo draw.

    Parameters
    ----------
    estimate : function taking a list of ``(mill_id, year)`` events and returning one
        number or a dict of numbers (e.g. ``{"r0_40": 12.3, "r40_80": 4.5}``).

    Returns
    -------
    One row per draw: ``draw``, ``mills``, ``years`` and the estimate(s).
    """
    rows = []
    for i, ev in enumerate(placebo_draws(pool, event_years, n_draws, seed), 1):
        est = estimate(ev)
        est = est if isinstance(est, dict) else {"estimate": est}
        rows.append(
            dict(draw=i, mills=";".join(map(str, (m for m, _ in ev))), years=";".join(str(y) for _, y in ev), **est)
        )
    return pd.DataFrame(rows)


def placebo_summary(real: float, placebo: Sequence[float]) -> dict:
    """Where the real estimate falls in the placebo distribution."""
    p = np.asarray(placebo, dtype=float)
    return {
        "real": real,
        "placebo_median": float(np.median(p)),
        "placebo_p2_5": float(np.percentile(p, 2.5)),
        "placebo_p97_5": float(np.percentile(p, 97.5)),
        "share_larger": float((p > real).mean()),
        "share_smaller": float((p < real).mean()),
        "p_two_sided": float((np.abs(p) >= abs(real)).mean()),
    }
