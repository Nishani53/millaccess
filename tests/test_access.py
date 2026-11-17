import numpy as np
import pandas as pd
import pytest

from millaccess import access_index, nearest_mill

DIST = pd.DataFrame(
    {
        "origin_id": ["a", "a", "b"],
        "mill_id": ["m1", "m2", "m2"],
        "road_km": [0.0, 50.0, 100.0],
    }
)


def test_access_formula_unweighted():
    a = access_index(DIST, lambda_km=50, origins=["a", "b", "c"]).set_index("origin_id").access
    assert a["a"] == pytest.approx(1 + np.exp(-1))
    assert a["b"] == pytest.approx(np.exp(-2))
    assert a["c"] == 0.0  # no mill nearby -> zero access


def test_access_weights_groups_and_lambdas():
    mills = pd.DataFrame({"mill_id": ["m1", "m2", "m1"], "year": [2003, 2003, 2018], "size_w": [10, 1, 10]})
    a = access_index(DIST, mills, lambda_km=[25, 50], weight_col="size_w", group_cols=["year"])
    row = a[(a.origin_id == "a") & (a.year == 2003) & (a.lambda_km == 50)].access.item()
    assert row == pytest.approx(10 + np.exp(-1))
    # in 2018 only m1 operates; origin b has no mill -> 0
    assert a[(a.origin_id == "b") & (a.year == 2018) & (a.lambda_km == 25)].access.item() == 0.0
    assert len(a) == 2 * 2 * 2  # origins x years x lambdas


def test_nearest_mill():
    n = nearest_mill(DIST, within_km=(50, 100), origins=["a", "b", "c"], none_km=200).set_index("origin_id")
    assert n.loc["a", "nearest_km"] == 0 and n.loc["a", "n_within_50km"] == 2
    assert n.loc["b", "n_within_50km"] == 0 and n.loc["b", "n_within_100km"] == 1
    assert n.loc["c", "nearest_km"] == 200
