import numpy as np
import pandas as pd
import pytest

from millaccess import alternative_within, assign_rings, event_study, pooled_did, stack_events


def test_ring_boundaries():
    d = pd.DataFrame({"origin_id": list("abcdefg"), "mill_id": "m", "road_km": [0, 40, 40.1, 80, 120, 160, 170]})
    r = assign_rings(d, "m").set_index("origin_id").ring
    assert r["a"] == "r0_40" and r["b"] == "r0_40"
    assert r["c"] == "r40_80" and r["d"] == "r40_80"
    assert r["e"] == "ref" and r["f"] == "ref"
    assert "g" not in r.index  # beyond the reference ring


def synthetic(effect_near=0.10, effect_alt=None, seed=1):
    """Two closures; origins at known distances; outcome with a planted post-closure effect."""
    rng = np.random.default_rng(seed)
    origins = [f"c{i}" for i in range(300)]
    km = np.r_[rng.uniform(0, 40, 100), rng.uniform(40.5, 80, 100), rng.uniform(80.5, 160, 100)]
    county = [f"k{i % 30}" for i in range(300)]
    dist = pd.concat(
        [pd.DataFrame({"origin_id": origins, "mill_id": m, "road_km": km}) for m in ["mA", "mB"]], ignore_index=True
    )
    events = {"mA": ("Closure A", 2005), "mB": ("Closure B", 2009)}
    fe = rng.normal(0.3, 0.05, 300)
    rows = []
    for y in range(2000, 2016):
        yr = 0.02 * np.sin(y)
        rows.append(
            pd.DataFrame({"origin_id": origins, "year": y, "county": county, "h": fe + yr + rng.normal(0, 0.01, 300)})
        )
    out = pd.concat(rows, ignore_index=True)
    return out, dist, events, dict(zip(origins, km))


def test_pooled_did_recovers_planted_effect():
    out, dist, events, _ = synthetic()
    s = stack_events(out, dist, events)
    # plant +0.10 in the 0-40 km ring after each closure
    s.loc[(s.ring == "r0_40") & (s.k >= 1), "h"] += 0.10
    res = pooled_did(s, "h", cluster_col="county").set_index("ring")
    assert res.loc["r0_40", "coef"] == pytest.approx(0.10, abs=0.01)
    assert res.loc["r40_80", "coef"] == pytest.approx(0.0, abs=0.01)
    assert res.loc["r0_40", "n_units"] == 200  # 100 origins x 2 events


def test_stack_structure_and_window():
    out, dist, events, _ = synthetic()
    s = stack_events(out, dist, events, window=(-5, 8))
    a = s[s.event == "Closure B"]
    assert a.k.min() == -5 and a.k.max() == 6  # outcome data end in 2015
    assert set(s.ring) == {"r0_40", "r40_80", "ref"}
    assert s.stack_unit.nunique() == 2 * 300


def test_event_study_flat_pretrends_and_jump():
    out, dist, events, _ = synthetic()
    s = stack_events(out, dist, events)
    s.loc[(s.ring == "r0_40") & (s.k >= 0), "h"] += 0.10
    es = event_study(s, "h", cluster_col="county")
    near = es[es.ring == "r0_40"].set_index("k").coef
    assert near[-1] == 0.0
    assert near.loc[-5:-2].abs().max() < 0.01
    assert near.loc[0:].mean() == pytest.approx(0.10, abs=0.01)


def test_absorption_split():
    out, dist, events, _km = synthetic()
    s = stack_events(out, dist, events)
    # an alternative mill near the first 50 origins only
    alt_dist = pd.DataFrame({"origin_id": [f"c{i}" for i in range(50)], "mill_id": "mAlt", "road_km": 30.0})
    alt = alternative_within(pd.concat([dist, alt_dist]), events, ["mAlt", "mA"], max_km=80)
    s = s.merge(alt, on=["event", "origin_id"], how="left")
    s["alt_near"] = s.alt_near.fillna(0.0)
    near = (s.ring == "r0_40") & (s.k >= 1)
    s.loc[near & (s.alt_near == 1), "h"] += 0.20
    res = pooled_did(s, "h", cluster_col="county", by="alt_near").set_index(["ring", "alt_near"])
    assert res.loc[("r0_40", 1.0), "coef"] == pytest.approx(0.20, abs=0.02)
    assert res.loc[("r0_40", 0.0), "coef"] == pytest.approx(0.0, abs=0.02)
