"""Market access to mills: distance-weighted access indices and nearest-mill summaries.

The access index for location *c* follows the market-access literature
(Donaldson & Hornbeck, 2016):

    A_c = sum over operating mills m of  w_m * exp(-d_cm / lambda)

where d_cm is the road distance (km), lambda the distance-decay scale (km) and w_m a
mill weight (1 for all mills, or a size weight).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def access_index(
    distances: pd.DataFrame,
    mills: pd.DataFrame | Sequence | None = None,
    lambda_km: float | Iterable[float] = 50.0,
    weight_col: str | None = None,
    origins: Sequence | None = None,
    group_cols: Sequence[str] = (),
    origin_col: str = "origin_id",
    mill_col: str = "mill_id",
    dist_col: str = "road_km",
) -> pd.DataFrame:
    """Distance-weighted access to mills for every origin.

    Parameters
    ----------
    distances : output of :func:`millaccess.road_distances` (origin, mill, road_km).
    mills : the mills that count, e.g. those operating in a given year. Either a list of
        mill ids or a DataFrame with ``mill_col`` plus optional ``weight_col`` and
        ``group_cols`` (e.g. ``year``, ``product_group``). ``None`` uses every mill.
    lambda_km : one or several distance-decay scales (km).
    weight_col : column of ``mills`` holding mill weights (e.g. size weights 1/3/10).
        ``None`` gives every mill a weight of 1.
    origins : all origin ids. Origins with no mill within the distance table get access 0.
        If ``None``, only origins present in ``distances`` are returned.
    group_cols : columns of ``mills`` to compute separate indices for
        (e.g. ``["year", "product_group"]``).

    Returns
    -------
    Long DataFrame: ``origin_col``, group columns, ``lambda_km``, ``access``.
    """
    if mills is None:
        m = pd.DataFrame({mill_col: distances[mill_col].unique()})
    elif isinstance(mills, pd.DataFrame):
        m = mills.copy()
    else:
        m = pd.DataFrame({mill_col: list(mills)})
    m["_w"] = m[weight_col].astype(float).fillna(1.0) if weight_col else 1.0
    group_cols = list(group_cols)
    lams = [lambda_km] if np.isscalar(lambda_km) else list(lambda_km)

    d = distances[[origin_col, mill_col, dist_col]].merge(m[[mill_col, "_w", *group_cols]], on=mill_col)
    if origins is None:
        origins = distances[origin_col].unique()
    origins = pd.Index(origins, name=origin_col)

    def _tup(k):
        return k if isinstance(k, tuple) else (k,)

    if group_cols:
        by_key = {_tup(k): g for k, g in d.groupby(group_cols)}
        keys = [_tup(k) for k in m.groupby(group_cols).size().index]  # groups with no nearby mill still get 0
    else:
        by_key, keys = {(): d}, [()]

    out = []
    for key in keys:
        g = by_key.get(key, d.iloc[0:0])
        for lam in lams:
            s = pd.Series(np.exp(-g[dist_col].values / lam) * g["_w"].values).groupby(g[origin_col].values).sum()
            a = pd.DataFrame({origin_col: origins, "access": s.reindex(origins).fillna(0.0).values})
            for c, v in zip(group_cols, key):
                a[c] = v
            a["lambda_km"] = lam
            out.append(a)
    res = pd.concat(out, ignore_index=True)
    return res[[origin_col, *group_cols, "lambda_km", "access"]]


def nearest_mill(
    distances: pd.DataFrame,
    mills: Sequence | None = None,
    within_km: Sequence[float] = (50, 100),
    origins: Sequence | None = None,
    none_km: float = np.nan,
    origin_col: str = "origin_id",
    mill_col: str = "mill_id",
    dist_col: str = "road_km",
) -> pd.DataFrame:
    """Distance to the nearest mill and number of mills within given road distances.

    Origins with no mill in the distance table get ``nearest_km = none_km`` and counts of 0.
    """
    d = distances if mills is None else distances[distances[mill_col].isin(list(mills))]
    if origins is None:
        origins = distances[origin_col].unique()
    idx = pd.Index(origins, name=origin_col)
    res = pd.DataFrame(index=idx)
    res["nearest_km"] = d.groupby(origin_col)[dist_col].min().reindex(idx).fillna(none_km)
    for w in within_km:
        res[f"n_within_{w:g}km"] = (d[dist_col] <= w).groupby(d[origin_col]).sum().reindex(idx).fillna(0).astype(int)
    return res.reset_index()


def standardize_effect(coef: float, sd_access: float, mean_outcome: float) -> float:
    """Effect of a one-standard-deviation change in access, as % of the mean outcome."""
    return 100.0 * coef * sd_access / mean_outcome


def period_means(
    outcomes: pd.DataFrame,
    periods: Mapping[int, Iterable[int]],
    value_cols: Sequence[str],
    origin_col: str = "origin_id",
    year_col: str = "year",
    keep_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Average annual outcomes over a window of years around each snapshot.

    ``periods`` maps a snapshot label to its years, e.g.
    ``{2003: range(2001, 2006), 2018: range(2016, 2021), 2023: range(2021, 2025)}``.
    ``keep_cols`` are time-invariant origin attributes to carry along (e.g. county).
    Returns one row per origin and snapshot (column ``period``).
    """
    parts = []
    for label, years in periods.items():
        sub = outcomes[outcomes[year_col].isin(list(years))]
        g = sub.groupby(origin_col)[list(value_cols)].mean()
        if keep_cols:
            g = g.join(sub.groupby(origin_col)[list(keep_cols)].first())
        parts.append(g.reset_index().assign(period=label))
    return pd.concat(parts, ignore_index=True)


def access_regression(
    panel: pd.DataFrame,
    outcome: str,
    access_cols: Sequence[str],
    fe_col: str | None = None,
    cluster_col: str | None = None,
    origin_col: str = "origin_id",
    period_col: str = "period",
    sd: Mapping[str, float] | None = None,
    mean_outcome: float | None = None,
) -> pd.DataFrame:
    """Within-location regression of an outcome on mill access across snapshots.

    ``outcome_ct = sum_j b_j * access_jct + location FE + (fe_col) FE + e``, with
    standard errors clustered by ``cluster_col``. Each coefficient is also expressed as
    the effect of a one-standard-deviation increase in access, as % of the mean outcome.

    Parameters
    ----------
    fe_col : column of time-varying fixed effects, e.g. region-by-period. ``None`` uses
        period fixed effects.
    sd, mean_outcome : scaling for the % effects. By default the SD of each access
        column is the average of its cross-sectional SDs across periods, and the mean
        outcome is the average of the period means, both computed from ``panel``. Pass
        full-sample values to scale subsample estimates the same way as the full sample.
    """
    from linearmodels.panel import PanelOLS

    data = panel.copy()
    if fe_col is None:
        fe_col = "_period_fe"
        data[fe_col] = data[period_col].astype(str)
    df = data.set_index([origin_col, period_col])
    model = PanelOLS(
        df[outcome], df[list(access_cols)], entity_effects=True, other_effects=df[[fe_col]], drop_absorbed=True
    )
    if cluster_col is None:
        r = model.fit(cov_type="robust")
    else:
        r = model.fit(cov_type="clustered", clusters=df[[cluster_col]].astype("category").apply(lambda c: c.cat.codes))
    if mean_outcome is None:
        mean_outcome = data.groupby(period_col)[outcome].mean().mean()
    rows = []
    for x in access_cols:
        s = sd[x] if sd is not None and x in sd else data[x].groupby(data[period_col]).std().mean()
        b, se = r.params[x], r.std_errors[x]
        rows.append(
            {
                "var": x,
                "coef": b,
                "se": se,
                "p": r.pvalues[x],
                "sd": s,
                "pct_per_sd": 100 * b * s / mean_outcome,
                "ci_lo_pct": 100 * (b - 1.96 * se) * s / mean_outcome,
                "ci_hi_pct": 100 * (b + 1.96 * se) * s / mean_outcome,
                "n": int(r.nobs),
            }
        )
    return pd.DataFrame(rows)
