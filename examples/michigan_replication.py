"""Reproduce the Michigan mill-closure study (Tables 5-7) with millaccess.

Point the script at the study folder ("Michigan Final Nov 8"). It expects:

    Access/roads_nodes.pkl, Access/roads_edges.pkl          road network (Step 6, Stage A)
    Access/analysis_inputs/cells.csv, mills_off_network.csv  grid cells and off-network mills
    EE Exports Nov 8/MI_cells_harvest_v2_lower.csv, ..._upper.csv, MI_cells_forest_type.csv
    GIS/Layers/MI_Mills_Master.gpkg                         mills (layers mills_master, mills_by_year)

It recomputes road distances and access indices from the network, builds the harvest
panel from the Earth Engine exports, runs every analysis and compares the results with
the published values in ``expected_results.csv``.

    python examples/michigan_replication.py "C:/.../Michigan Final Nov 8"            # ~2 minutes
    python examples/michigan_replication.py "C:/.../Michigan Final Nov 8" --placebo  # + 200 placebo draws (~7 min)

Data are not part of this repository.
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import millaccess as ma

warnings.filterwarnings("ignore")
FINAL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Michigan Final Nov 8")
RUN_PLACEBO = "--placebo" in sys.argv
ACC, INP = FINAL / "Access", FINAL / "Access" / "analysis_inputs"
EE, MILLS = FINAL / "EE Exports Nov 8", FINAL / "GIS" / "Layers" / "MI_Mills_Master.gpkg"
HERE = Path(__file__).parent

EVENTS = {
    "MI_0012": ("Menasha 2005", 2005),
    "MI_0236": ("Sappi Muskegon 2005", 2005),
    "MI_0281": ("Georgia-Pacific Gaylord 2006", 2006),
    "MI_0260": ("Smurfit-Stone Ontonagon 2009", 2009),
}
SIZE_W = {"Small (<=1,000 MBF)": 1, "Medium (1,001-5,000 MBF)": 3, "Large (>5,000 MBF)": 10}
YEARS = range(2001, 2025)
results = []


def record(table, label, ring, pct, lo, hi, p):
    results.append({"table": table, "label": label, "ring": ring, "pct": pct, "ci_lo": lo, "ci_hi": hi, "p": p})


# ---------------- 1. road distances and access indices ----------------
print("1/5 road distances and access indices")
net = ma.RoadNetwork(pd.read_pickle(ACC / "roads_nodes.pkl"), pd.read_pickle(ACC / "roads_edges.pkl"))
cells = pd.read_csv(INP / "cells.csv")
mm = gpd.read_file(MILLS, layer="mills_master")
mx = mm[mm.merged_into.isna()].to_crs(3078).reset_index(drop=True)
dist = ma.road_distances(
    net, cells[["x_m", "y_m"]].values, cells.cell_id.values, np.c_[mx.geometry.x, mx.geometry.y], mx.mill_uid.values
)

off = set(pd.read_csv(INP / "mills_off_network.csv").mill_uid)
by = gpd.read_file(MILLS, layer="mills_by_year")
by = by[
    (by.include_in_access_index == "Yes") & by.product_group.isin(["Bulk fiber", "Sawlog"]) & ~by.mill_uid.isin(off)
]
by = pd.DataFrame(by.drop(columns="geometry"))
by["w_size"] = by.size_class_filled.map(SIZE_W).fillna(1.0)
mills_tab = by[["mill_uid", "year", "product_group", "w_size"]].rename(
    columns={"mill_uid": "mill_id", "product_group": "group"}
)
dist_on = dist[~dist.mill_id.isin(off)]
acc = pd.concat(
    [
        ma.access_index(dist_on, mills_tab, [25, 50, 75], group_cols=["year", "group"], origins=cells.cell_id).assign(
            w="u"
        ),
        ma.access_index(
            dist_on, mills_tab, [25, 50, 75], weight_col="w_size", group_cols=["year", "group"], origins=cells.cell_id
        ).assign(w="w"),
    ]
)

# ---------------- 2. harvest panel (notebook Step 7) ----------------
print("2/5 harvest panel")
h = pd.concat([pd.read_csv(EE / f"MI_cells_harvest_v2_{p}.csv") for p in ("lower", "upper")], ignore_index=True)
d = cells.merge(h.drop(columns=["county", "peninsula", "land_km2"]), on="cell_id")
d = d[(d.forest_ha >= 125) & (d.off_network == "No")].copy()
ft = pd.read_csv(EE / "MI_cells_forest_type.csv")
G = ["aspen_birch", "pine_plantation", "northern_hardwood", "oak", "lowland_conifer"]
cols = [f"ft_{g}_ha" for g in G]
ft[cols] = ft[cols].fillna(0)
ft["dominant"] = ft[cols].idxmax(axis=1).str[3:-3]
ft.loc[ft[cols].sum(axis=1) < 50, "dominant"] = "unclassified"
TYPE = {"aspen_birch": "Clearcut-type", "pine_plantation": "Clearcut-type", "northern_hardwood": "Northern hardwood"}
TYPE.update({"oak": "Oak", "lowland_conifer": "Lowland conifer/swamp", "unclassified": "Unclassified"})
d = d.merge(ft[["cell_id"]].assign(forest_type=ft.dominant.map(TYPE)), on="cell_id", how="left")
long = pd.concat(
    [
        pd.DataFrame(
            {
                "origin_id": d.cell_id,
                "year": y,
                "county": d.county,
                "peninsula": d.peninsula,
                "forest_type": d.forest_type.fillna("Unclassified"),
                "hansen": 100 * d[f"hansen_loss_{y}"] / d.hansen_forest2000_ha.clip(lower=1),
                "lcms": 100 * d[f"lcms_rem_{y}"] / d.forest_ha,
            }
        )
        for y in YEARS
    ],
    ignore_index=True,
)
print(f"    {len(d):,} analysis cells")

# ---------------- 3. Table 5: period comparison ----------------
print("3/5 Table 5")
P = ma.period_means(
    long,
    {2003: range(2001, 2006), 2018: range(2016, 2021), 2023: range(2021, 2025)},
    ["hansen", "lcms"],
    keep_cols=["county", "peninsula"],
)
a = acc.pivot_table(index=["origin_id", "year"], columns=["group", "w", "lambda_km"], values="access")
a.columns = [f"A_{'bulk' if g == 'Bulk fiber' else 'saw'}_{w}{lam}" for g, w, lam in a.columns]
P = P.merge(a.reset_index().rename(columns={"year": "period"}), on=["origin_id", "period"])
P["pen_period"] = P.peninsula + "_" + P.period.astype(str)
P["cty_period"] = P.county + "_" + P.period.astype(str)
SD = {c: P[c].groupby(P.period).std().mean() for c in P if c.startswith("A_")}
MEAN = P.groupby("period")[["hansen", "lcms"]].mean().mean()
t5 = [
    ("Main", "hansen", "u50", "pen_period", P),
    ("LCMS outcome", "lcms", "u50", "pen_period", P),
    ("lambda 25", "hansen", "u25", "pen_period", P),
    ("lambda 75", "hansen", "u75", "pen_period", P),
    ("Size-weighted", "hansen", "w50", "pen_period", P),
    ("County x period FE", "hansen", "u50", "cty_period", P),
    ("2003-2018 only", "hansen", "u50", "pen_period", P[P.period < 2023]),
    ("2018-2023 only", "hansen", "u50", "pen_period", P[P.period > 2003]),
    ("Upper Peninsula", "hansen", "u50", "pen_period", P[P.peninsula == "upper"]),
    ("Lower Peninsula", "hansen", "u50", "pen_period", P[P.peninsula == "lower"]),
]
for label, y, spec, fe, data in t5:
    xs = [f"A_bulk_{spec}", f"A_saw_{spec}"]
    r = ma.access_regression(data, y, xs, fe_col=fe, cluster_col="county", sd=SD, mean_outcome=MEAN[y])
    for _, row in r.iterrows():
        record(
            "T5", label, "bulk" if "bulk" in row["var"] else "saw", row.pct_per_sd, row.ci_lo_pct, row.ci_hi_pct, row.p
        )

# ---------------- 4. Tables 6 and 7: closures ----------------
print("4/5 Tables 6 and 7")
bulk_ids = set(mm.loc[mm.product_group_fixed == "Bulk fiber", "mill_uid"])
dist_b = dist[dist.mill_id.isin(bulk_ids)]


def pooled(events, outcomes=long, y="hansen", ref=(80, 160), win=(-5, 8)):
    s = ma.stack_events(outcomes, dist_b, events, reference=ref, window=win)
    return ma.pooled_did(s, y, cluster_col="county").set_index("ring")


for y, lab in [("hansen", "Pooled (Hansen)"), ("lcms", "Pooled (LCMS)")]:
    for ring, r in pooled(EVENTS, y=y).iterrows():
        record("T6", lab, ring, r.pct, r.ci_lo_pct, r.ci_hi_pct, r.p)
stacked = ma.stack_events(long, dist_b, EVENTS)
for name, _ in EVENTS.values():
    r = ma.pooled_did(stacked[stacked.event == name], "hansen", cluster_col="county").set_index("ring").loc["r0_40"]
    record("T6", name, "r0_40", r.pct, r.ci_lo_pct, r.ci_hi_pct, r.p)

# absorption, as published: any bulk fiber mill operating in 2018 counts as an alternative
alt18 = list(by[(by.product_group == "Bulk fiber") & (by.year == 2018)].mill_uid)
alt = ma.alternative_within(dist_b, EVENTS, alt18, max_km=80)
s3 = stacked.merge(alt, on=["event", "origin_id"], how="left").fillna({"alt_near": 0.0})
for _, r in ma.pooled_did(s3, "hansen", cluster_col="county", by="alt_near").iterrows():
    record("T6", f"Absorption alt_near={int(r.alt_near)}", r.ring, r.pct, r.ci_lo_pct, r.ci_hi_pct, r.p)

specs = [("Ref 100-200", EVENTS, (100, 200), (-5, 8)), ("Ref 80-120", EVENTS, (80, 120), (-5, 8))]
specs += [("Window -3..5", EVENTS, (80, 160), (-3, 5))]
specs += [
    (f"Without {nm}", {k: v for k, v in EVENTS.items() if k != u}, (80, 160), (-5, 8)) for u, (nm, _) in EVENTS.items()
]
specs += [("Without both 2005", {k: v for k, v in EVENTS.items() if v[1] != 2005}, (80, 160), (-5, 8))]
for label, ev, ref, win in specs:
    r = pooled(ev, ref=ref, win=win).loc["r0_40"]
    record("T7A", label, "r0_40", r.pct, r.ci_lo_pct, r.ci_hi_pct, r.p)

cont = mm[
    (mm.product_group_fixed == "Bulk fiber")
    & (mm.include_in_access_index == "Yes")
    & mm.merged_into.isna()
    & (mm.in_2003_corr == 1)
    & (mm.in_2018_corr == 1)
    & (mm.in_2023_corr == 1)
]
closed = {"MI_0281": EVENTS["MI_0281"], "MI_0260": EVENTS["MI_0260"]}
controls = {
    "MI_0281": list(cont[cont.peninsula == "lower"].mill_uid),
    "MI_0260": list(cont[cont.peninsula == "upper"].mill_uid),
}
sm = ma.stack_matched(long, dist_b, closed, controls, exclude_mills=list(EVENTS), exclude_km=80)
for label, data, y in [("Both (Hansen)", sm, "hansen"), ("Both (LCMS)", sm, "lcms")] + [
    (f"{nm} (Hansen)", sm[sm.event == nm], "hansen") for nm, _ in closed.values()
]:
    for _, r in ma.pooled_did(data, y, cluster_col="county").iterrows():
        record("T7C", label, r.ring, r.pct, r.ci_lo_pct, r.ci_hi_pct, r.p)

for tp in ["Clearcut-type", "Northern hardwood", "Oak", "Lowland conifer/swamp"]:
    r = pooled(EVENTS, outcomes=long[long.forest_type == tp]).loc["r0_40"]
    record("T7D", tp, "r0_40", r.pct, r.ci_lo_pct, r.ci_hi_pct, r.p)

# ---------------- 5. placebo closures (optional) ----------------
if RUN_PLACEBO:
    print("5/5 placebo closures (200 draws)")
    pool = cont[cont.mill_uid.isin(dist_b.mill_id.unique())].mill_uid.values

    def estimate(ev):
        return (
            ma.pooled_did(ma.stack_events(long, dist_b, dict(ev)), "hansen", cluster_col="county")
            .set_index("ring")
            .pct.to_dict()
        )

    draws = ma.placebo_test(estimate, pool, [2005, 2005, 2006, 2009], n_draws=200, seed=20261108)
    s = ma.placebo_summary(18.8, draws.r0_40)
    record(
        "T7B", "Placebo median", "r0_40", s["placebo_median"], s["placebo_p2_5"], s["placebo_p97_5"], s["share_larger"]
    )
else:
    print("5/5 placebo skipped (add --placebo)")

# ---------------- compare with the published values ----------------
R = pd.DataFrame(results)
R.to_csv("michigan_replication_results.csv", index=False)
E = pd.read_csv(HERE / "expected_results.csv")
C = E.merge(R, on=["table", "label", "ring"], suffixes=("_published", "_package"))
C["match"] = (C.pct_published - C.pct_package).abs() < 0.05
print(f"\n{C.match.sum()} of {len(C)} published estimates reproduced (to 0.05 percentage points).")
if not C.match.all():
    print(C.loc[~C.match, ["table", "label", "ring", "pct_published", "pct_package"]].to_string(index=False))
