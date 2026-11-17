"""Stacked event studies around mill closures (or openings).

For each event (a mill and the year it closed), locations are assigned to distance
rings by road distance from that mill: one or more *treated* rings close to the mill
and a *reference* ring farther away. Each event's sample covers a window of years
around the event; the events are then stacked (Cengiz et al., 2019) and estimated with
location-by-event and event-by-year fixed effects, so treated locations are compared
only with reference locations around the same mill in the same year.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

Ring = tuple[float, float]


def ring_label(ring: Ring) -> str:
    return f"r{ring[0]:g}_{ring[1]:g}"


def assign_rings(
    distances: pd.DataFrame,
    mill_id,
    treated: Sequence[Ring] = ((0, 40), (40, 80)),
    reference: Ring = (80, 160),
    origin_col: str = "origin_id",
    mill_col: str = "mill_id",
    dist_col: str = "road_km",
) -> pd.DataFrame:
    """Assign origins to distance rings around one mill.

    A ring ``(lo, hi)`` holds origins with ``lo < distance <= hi``; the innermost ring
    also includes ``distance == lo``. Origins in no ring (gaps, or beyond the reference
    ring) are dropped. Returns ``origin_col``, ``dist_col``, ``ring`` (``"r0_40"``, ...,
    ``"ref"``).
    """
    d = distances.loc[distances[mill_col] == mill_id, [origin_col, dist_col]].copy()
    x = d[dist_col].values
    labels = np.full(len(d), "", dtype=object)
    lo0 = min(r[0] for r in treated)
    for r in treated:
        inside = (x > r[0]) & (x <= r[1])
        if r[0] == lo0:
            inside |= x == r[0]
        labels[inside & (labels == "")] = ring_label(r)
    inside_ref = (x > reference[0]) & (x <= reference[1])
    labels[inside_ref & (labels == "")] = "ref"
    d["ring"] = labels
    return d[d.ring != ""].reset_index(drop=True)


def alternative_within(
    distances: pd.DataFrame,
    events: Mapping | pd.DataFrame,
    alternative_mills: Sequence | Mapping,
    max_km: float = 80.0,
    origin_col: str = "origin_id",
    mill_col: str = "mill_id",
    dist_col: str = "road_km",
) -> pd.DataFrame:
    """For each event, distance from each origin to the nearest *other* mill still operating.

    Parameters
    ----------
    events : see :func:`stack_events`.
    alternative_mills : ids of mills operating after the closures (one list for all
        events), or a mapping ``event mill id -> list`` of ids. The event mill itself is
        always excluded.
    max_km : origins with an alternative within this road distance get ``alt_near = 1``.

    Returns
    -------
    DataFrame with ``event``, ``origin_col``, ``alt_km`` and ``alt_near`` (0/1). Origins
    with no alternative in the distance table get ``alt_km = inf``.
    """
    ev = _events_table(events)
    rows = []
    for _, e in ev.iterrows():
        alts = alternative_mills[e.mill_id] if isinstance(alternative_mills, Mapping) else alternative_mills
        alts = set(alts) - {e.mill_id}
        m = distances[distances[mill_col].isin(alts)].groupby(origin_col)[dist_col].min()
        rows.append(pd.DataFrame({"event": e.event, origin_col: m.index, "alt_km": m.values}))
    out = pd.concat(rows, ignore_index=True)
    out["alt_near"] = (out.alt_km <= max_km).astype(float)
    return out


def _events_table(events) -> pd.DataFrame:
    """Normalize events to a DataFrame with columns event, mill_id, event_year."""
    if isinstance(events, pd.DataFrame):
        ev = events.copy()
        if "event" not in ev:
            ev["event"] = ev["mill_id"].astype(str)
        return ev[["event", "mill_id", "event_year"]].reset_index(drop=True)
    rows = []
    for mill, val in dict(events).items():
        if isinstance(val, (tuple, list)):
            name, year = val
        else:
            name, year = str(mill), val
        rows.append({"event": name, "mill_id": mill, "event_year": int(year)})
    return pd.DataFrame(rows)


def stack_events(
    outcomes: pd.DataFrame,
    distances: pd.DataFrame,
    events: Mapping | pd.DataFrame,
    treated: Sequence[Ring] = ((0, 40), (40, 80)),
    reference: Ring = (80, 160),
    window: tuple[int, int] = (-5, 8),
    origin_col: str = "origin_id",
    year_col: str = "year",
    mill_col: str = "mill_id",
    dist_col: str = "road_km",
) -> pd.DataFrame:
    """Build a stacked event-study panel.

    Parameters
    ----------
    outcomes : long panel with ``origin_col``, ``year_col`` and one or more outcome
        columns (e.g. annual harvest rates), plus any location attributes such as the
        cluster variable (county).
    distances : origin-to-mill road distances (:func:`millaccess.road_distances`).
    events : ``{mill_id: year}``, ``{mill_id: (name, year)}`` or a DataFrame with
        ``mill_id``, ``event_year`` and optional ``event`` (name).
    treated, reference : distance rings in km (see :func:`assign_rings`).
    window : event-time window ``(first, last)``, inclusive; years outside the outcome
        data are dropped.

    Returns
    -------
    The stacked panel: one row per event x origin x year, with columns ``event``,
    ``event_year``, ``k`` (event time), ``ring``, ``post`` (k >= 1), ``stack_unit``
    (event-by-origin id), ``event_x_year``, plus all columns of ``outcomes``.
    """
    ev = _events_table(events)
    parts = []
    for _, e in ev.iterrows():
        rings = assign_rings(distances, e.mill_id, treated, reference, origin_col, mill_col, dist_col)
        years = range(e.event_year + window[0], e.event_year + window[1] + 1)
        sub = outcomes[outcomes[year_col].isin(years)].merge(rings, on=origin_col, how="inner")
        sub.insert(0, "event", e.event)
        sub.insert(1, "event_year", e.event_year)
        sub["k"] = sub[year_col] - e.event_year
        parts.append(sub)
    s = pd.concat(parts, ignore_index=True)
    s["post"] = (s.k >= 1).astype(float)
    s["stack_unit"] = s.event.astype(str) + "|" + s[origin_col].astype(str)
    s["event_x_year"] = s.event.astype(str) + "_" + s[year_col].astype(str)
    return s


def stack_matched(
    outcomes: pd.DataFrame,
    distances: pd.DataFrame,
    events: Mapping | pd.DataFrame,
    control_mills: Sequence | Mapping,
    rings: Sequence[Ring] = ((0, 40), (40, 80)),
    exclude_mills: Sequence = (),
    exclude_km: float | None = None,
    window: tuple[int, int] = (-5, 8),
    origin_col: str = "origin_id",
    year_col: str = "year",
    mill_col: str = "mill_id",
    dist_col: str = "road_km",
) -> pd.DataFrame:
    """Stacked panel comparing locations near closed mills with locations near continuing mills.

    For each event, *treated* locations lie within ``rings`` of the closed mill and
    *control* locations lie within the same rings of the nearest control mill (e.g.
    mills of the same type that never closed, in the same region). Controls within
    ``exclude_km`` of any mill in ``exclude_mills`` (e.g. all closed mills) are dropped.
    The ``event_x_year`` fixed effect is event-by-year-by-ring, so treated and control
    locations at similar distances from their mill are compared in the same year.

    Parameters
    ----------
    control_mills : one list of control mill ids for all events, or a mapping
        ``event mill id -> list`` (e.g. continuing mills in the same peninsula).

    Returns
    -------
    Stacked panel as :func:`stack_events`, with a 0/1 ``treated`` column. Estimate with
    :func:`pooled_did` or :func:`event_study` and ``treated_col="treated"``.
    """
    ev = _events_table(events)
    excl = set()
    if exclude_km is not None and len(exclude_mills):
        excl = set(
            distances.loc[
                distances[mill_col].isin(list(exclude_mills)) & (distances[dist_col] <= exclude_km), origin_col
            ]
        )
    max_km = max(r[1] for r in rings)
    lo0 = min(r[0] for r in rings)

    def _ring(x):
        lab = np.full(len(x), "", dtype=object)
        for r in rings:
            inside = (x > r[0]) & (x <= r[1])
            if r[0] == lo0:
                inside |= x == r[0]
            lab[inside & (lab == "")] = ring_label(r)
        return lab

    parts = []
    for _, e in ev.iterrows():
        tr = distances.loc[(distances[mill_col] == e.mill_id) & (distances[dist_col] <= max_km), [origin_col, dist_col]]
        tr = tr.assign(treated=1.0)
        cm = control_mills[e.mill_id] if isinstance(control_mills, Mapping) else control_mills
        ct = distances[distances[mill_col].isin(list(cm))].groupby(origin_col)[dist_col].min()
        ct = ct[(ct <= max_km) & ~ct.index.isin(excl)]
        ct = pd.DataFrame({origin_col: ct.index, dist_col: ct.values, "treated": 0.0})
        both = pd.concat([tr, ct], ignore_index=True)
        both["ring"] = _ring(both[dist_col].values)
        both = both[both.ring != ""]
        years = range(e.event_year + window[0], e.event_year + window[1] + 1)
        sub = outcomes[outcomes[year_col].isin(years)].merge(both, on=origin_col, how="inner")
        sub.insert(0, "event", e.event)
        sub.insert(1, "event_year", e.event_year)
        sub["k"] = sub[year_col] - e.event_year
        parts.append(sub)
    s = pd.concat(parts, ignore_index=True)
    s["post"] = (s.k >= 1).astype(float)
    s["stack_unit"] = s.event.astype(str) + "|" + s.treated.astype(int).astype(str) + "|" + s[origin_col].astype(str)
    s["event_x_year"] = s.event.astype(str) + "_" + s[year_col].astype(str) + "_" + s.ring
    return s


def _treated_labels(stacked: pd.DataFrame) -> list[str]:
    labs = [r for r in stacked.ring.unique() if r != "ref"]
    return sorted(labs, key=lambda r: float(r[1:].split("_")[0]))


def _treat(stacked: pd.DataFrame, treated_col: str | None) -> np.ndarray:
    if treated_col is not None and treated_col in stacked:
        return stacked[treated_col].values == 1
    return np.ones(len(stacked), dtype=bool)


def _origin_of(units: pd.Series) -> pd.Series:
    return units.str.rsplit("|", n=1).str[-1]


def _fit(stacked: pd.DataFrame, outcome: str, xcols: list[str], cluster_col: str | None):
    from linearmodels.panel import PanelOLS

    df = stacked.set_index(["stack_unit", "k"])
    model = PanelOLS(
        df[outcome], df[xcols], entity_effects=True, other_effects=df[["event_x_year"]], drop_absorbed=True
    )
    if cluster_col is None:
        return model.fit(cov_type="robust")
    clusters = df[[cluster_col]].astype("category").apply(lambda c: c.cat.codes)
    return model.fit(cov_type="clustered", clusters=clusters)


def _drop_empty(s: pd.DataFrame, xcols: list[str]) -> list[str]:
    """Regressors with no treated observations cannot be estimated; drop them with a warning."""
    keep = [c for c in xcols if s[c].any()]
    if len(keep) < len(xcols):
        warnings.warn(f"no observations for {sorted(set(xcols) - set(keep))}; reported as NaN", stacklevel=3)
    return keep


def event_study(
    stacked: pd.DataFrame,
    outcome: str,
    cluster_col: str | None = None,
    omit: int = -1,
    treated_col: str | None = "treated",
) -> pd.DataFrame:
    """Event-time coefficients for each treated ring relative to the comparison locations.

    Event time ``omit`` (default -1, the year before the event) is the baseline.
    ``treated_col`` is used only for panels from :func:`stack_matched`.
    Returns ``ring``, ``k``, ``coef``, ``se``, ``p`` (baseline row included with 0).
    """
    s = stacked.copy()
    tr = _treat(s, treated_col)
    xcols, meta = [], []
    for ring in _treated_labels(s):
        for k in sorted(s.k.unique()):
            if k == omit:
                continue
            c = f"{ring}_k{k}"
            s[c] = ((s.ring == ring) & (s.k == k) & tr).astype(float)
            xcols.append(c)
            meta.append((ring, k))
    r = _fit(s, outcome, _drop_empty(s, xcols), cluster_col)
    out = pd.DataFrame(
        [
            {
                "ring": ring,
                "k": k,
                "coef": r.params.get(c, np.nan),
                "se": r.std_errors.get(c, np.nan),
                "p": r.pvalues.get(c, np.nan),
            }
            for c, (ring, k) in zip(xcols, meta)
        ]
    )
    base = pd.DataFrame([{"ring": ring, "k": omit, "coef": 0.0, "se": 0.0, "p": np.nan} for ring in _treated_labels(s)])
    return pd.concat([out, base], ignore_index=True).sort_values(["ring", "k"]).reset_index(drop=True)


def pooled_did(
    stacked: pd.DataFrame,
    outcome: str,
    cluster_col: str | None = None,
    by: str | None = None,
    drop_event_year: bool = True,
    treated_col: str | None = "treated",
) -> pd.DataFrame:
    """Pooled post-event change for each treated ring relative to the comparison locations.

    Parameters
    ----------
    by : optional 0/1 column (e.g. ``alt_near`` from :func:`alternative_within`) to let
        the post-event change differ between groups — the absorption test.
    drop_event_year : drop event time 0 (the closure year itself).
    treated_col : used only for panels from :func:`stack_matched`.

    Returns
    -------
    One row per ring (and group): ``coef``, ``se``, ``p``; ``n_units`` (location-event
    pairs: a location near two events counts twice) and ``n_locations`` (distinct
    locations); ``pre_mean`` (mean outcome in the ring before the event) and the change
    as a percentage of the pre-event mean with its 95% confidence interval (``pct``,
    ``ci_lo_pct``, ``ci_hi_pct``). Groups with no observations are reported as NaN.
    """
    s = stacked[stacked.k != 0].copy() if drop_event_year else stacked.copy()
    tr = _treat(s, treated_col)
    rows = []
    groups = [None] if by is None else sorted(s[by].dropna().unique())
    for ring in _treated_labels(s):
        for g in groups:
            c = f"{ring}_post" if g is None else f"{ring}_post_{by}{g:g}"
            sel = (s.ring == ring) & (s.post == 1) & tr
            if g is not None:
                sel &= s[by] == g
            s[c] = sel.astype(float)
            rows.append((ring, g, c))
    xcols = [c for *_, c in rows]
    r = _fit(s, outcome, _drop_empty(s, xcols), cluster_col)
    pre = s[(s.k < 0) & tr]
    out = []
    for ring, g, c in rows:
        m = (pre.ring == ring) if g is None else (pre.ring == ring) & (pre[by] == g)
        pre_mean = pre.loc[m, outcome].mean()
        b, se = r.params.get(c, np.nan), r.std_errors.get(c, np.nan)
        units = pre.loc[m, "stack_unit"]
        row = {
            "ring": ring,
            "coef": b,
            "se": se,
            "p": r.pvalues.get(c, np.nan),
            "n_units": units.nunique(),
            "n_locations": _origin_of(units).nunique(),
            "pre_mean": pre_mean,
        }
        if g is not None:
            row = {by: g, **row}
        row.update(
            pct=100 * b / pre_mean,
            ci_lo_pct=100 * (b - 1.96 * se) / pre_mean,
            ci_hi_pct=100 * (b + 1.96 * se) / pre_mean,
        )
        out.append(row)
    return pd.DataFrame(out)
