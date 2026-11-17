import warnings

import numpy as np
import pandas as pd
import pytest

from millaccess import access_regression, period_means, pooled_did, stack_events, stack_matched


def access_panel(beta=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    ids = [f"c{i}" for i in range(n)]
    fe = rng.normal(1, 0.2, n)
    rows = []
    for t in (2003, 2018, 2023):
        a = rng.uniform(0, 2, n)
        rows.append(
            pd.DataFrame(
                {
                    "origin_id": ids,
                    "period": t,
                    "county": [f"k{i % 40}" for i in range(n)],
                    "access": a,
                    "y": fe + 0.1 * (t == 2018) + beta * a + rng.normal(0, 0.05, n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_access_regression_recovers_coefficient():
    p = access_panel(beta=0.5)
    r = access_regression(p, "y", ["access"], cluster_col="county").iloc[0]
    assert r.coef == pytest.approx(0.5, abs=0.01)
    sd = p.access.groupby(p.period).std().mean()
    mean = p.groupby("period").y.mean().mean()
    assert r.pct_per_sd == pytest.approx(100 * r.coef * sd / mean)
    # user-supplied scaling is used as given
    r2 = access_regression(p, "y", ["access"], sd={"access": 1.0}, mean_outcome=2.0).iloc[0]
    assert r2.pct_per_sd == pytest.approx(100 * r2.coef / 2.0)


def test_period_means():
    out = pd.DataFrame(
        {"origin_id": ["a"] * 4, "year": [2001, 2002, 2016, 2017], "h": [1.0, 3.0, 5.0, 7.0], "county": "k"}
    )
    p = period_means(out, {2003: range(2001, 2006), 2018: range(2016, 2021)}, ["h"], keep_cols=["county"])
    assert p.set_index("period").h.to_dict() == {2003: 2.0, 2018: 6.0}
    assert (p.county == "k").all()


def matched_data(effect=0.1, seed=2):
    rng = np.random.default_rng(seed)
    ids = [f"c{i}" for i in range(240)]
    # c0-c119 near the closed mill X; c120-c239 near the continuing mill Y
    km = np.r_[rng.uniform(0, 80, 120), rng.uniform(0, 80, 120)]
    dist = pd.concat(
        [
            pd.DataFrame({"origin_id": ids[:120], "mill_id": "X", "road_km": km[:120]}),
            pd.DataFrame({"origin_id": ids[120:], "mill_id": "Y", "road_km": km[120:]}),
            pd.DataFrame({"origin_id": ids[120:], "mill_id": "X", "road_km": 150.0}),
        ],
        ignore_index=True,
    )
    fe = rng.normal(0.3, 0.05, 240)
    rows = [
        pd.DataFrame(
            {
                "origin_id": ids,
                "year": y,
                "county": [f"k{i % 24}" for i in range(240)],
                "h": fe + rng.normal(0, 0.01, 240),
            }
        )
        for y in range(2000, 2016)
    ]
    out = pd.concat(rows, ignore_index=True)
    return out, dist


def test_stack_matched_recovers_effect_and_excludes():
    out, dist = matched_data()
    s = stack_matched(out, dist, {"X": ("Closure X", 2006)}, ["Y"], exclude_mills=["X"], exclude_km=80)
    assert set(s.treated) == {0.0, 1.0}
    assert s[s.treated == 1].origin_id.nunique() == 120
    s.loc[(s.treated == 1) & (s.ring == "r0_40") & (s.k >= 1), "h"] += 0.10
    r = pooled_did(s, "h", cluster_col="county").set_index("ring")
    assert r.loc["r0_40", "coef"] == pytest.approx(0.10, abs=0.01)
    assert r.loc["r40_80", "coef"] == pytest.approx(0.0, abs=0.01)
    # pre-period mean is taken from treated locations only
    assert r.loc["r0_40", "n_units"] == s[(s.treated == 1) & (s.ring == "r0_40")].origin_id.nunique()


def test_location_counts_and_empty_groups():
    out, dist = matched_data()
    # the same locations around two events: units count twice, locations once
    dist2 = pd.concat([dist, dist.assign(mill_id=dist.mill_id.str.lower())])
    dist2 = dist2[dist2.mill_id.isin(["X", "x"])]
    dist2.loc[dist2.road_km == 150.0, "road_km"] = 120.0
    s = stack_events(out, dist2, {"X": 2005, "x": 2008})
    r = pooled_did(s, "h").set_index("ring")
    assert r.loc["r0_40", "n_units"] == 2 * r.loc["r0_40", "n_locations"]
    # a group with no locations is reported as NaN instead of failing
    s["grp"] = 0.0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        g = pooled_did(s, "h", by="grp")
    assert len(g) == 2 and g.coef.notna().all()
    s.loc[s.ring == "ref", "grp"] = 1.0  # group 1 has no treated locations
    with pytest.warns(UserWarning):
        g = pooled_did(s, "h", by="grp")
    assert g[g.grp == 1.0].coef.isna().all()
