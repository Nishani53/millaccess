# millaccess

**Road-based mill access and mill-closure event studies for forest economics.**

[![tests](https://github.com/Nishani53/millaccess/actions/workflows/tests.yml/badge.svg)](https://github.com/Nishani53/millaccess/actions/workflows/tests.yml)

Forest inventory tools such as [rFIA](https://github.com/doserjef/rFIA) and
[pyFIA](https://github.com/mihiarc/pyfia) estimate *what is in the forest*. `millaccess`
answers the market questions forest economists ask next: *how close is each forest
location to the mills that buy its wood, and what happens nearby when a mill closes or
opens?*

It turns a road network, mill locations and any location-level outcome (satellite-detected
harvest, FIA plot data, stumpage prices) into:

| Step | Function |
|---|---|
| Road network from road lines (e.g. Census TIGER/Line) | `RoadNetwork.from_lines`, `download_tiger_roads` |
| Shortest road distance from every location to every mill | `road_distances` |
| Distance-weighted market access index, Σ w·exp(−d/λ) | `access_index` |
| Nearest mill and mills within X km | `nearest_mill` |
| Outcomes averaged around mill-list snapshots | `period_means` |
| Within-location regression of outcomes on access (effect per SD) | `access_regression` |
| Distance rings around an event mill | `assign_rings` |
| Is another mill still within reach? (absorption) | `alternative_within` |
| Stacked event-study panel (near vs far rings) | `stack_events` |
| Stacked panel, closed vs continuing mills | `stack_matched` |
| Event-time and pooled before/after estimates | `event_study`, `pooled_did` |
| Placebo (randomization) inference for few events | `placebo_test`, `placebo_summary` |

## Installation

```bash
pip install "millaccess[geo] @ git+https://github.com/Nishani53/millaccess"
```

The `[geo]` extra adds geopandas, shapely and requests for building road networks and
downloading TIGER/Line roads.

## Quick start

```python
import numpy as np
import millaccess as ma

# 1. Road network for one state, in a projected CRS in metres (here Maine, UTM zone 19N)
roads = ma.download_tiger_roads(ma.state_county_fips("23", 16), cache_dir="tiger").to_crs(26919)
net = ma.RoadNetwork.from_lines(roads)

# 2. Road distances from grid cells (or FIA plots) to mills, up to 200 km
dist = ma.road_distances(net, cells_xy, cell_ids, mills_xy, mill_ids, max_km=200)

# 3. Access index by year and product group (lambda = 50 km)
acc = ma.access_index(dist, mills_by_year, lambda_km=[25, 50, 75],
                      group_cols=["year", "product_group"], origins=cell_ids)

# 4. Stacked event study around dated closures: 0-40 and 40-80 km vs 80-160 km
events = {"mill_A": ("Closure A", 2006), "mill_B": ("Closure B", 2009)}
stacked = ma.stack_events(harvest_long, dist, events, window=(-5, 8))
coefs = ma.event_study(stacked, "harvest", cluster_col="county")
did = ma.pooled_did(stacked, "harvest", cluster_col="county")

# 5. Placebo closures at mills that never closed
def estimate(ev):
    s = ma.stack_events(harvest_long, dist, dict(ev))
    return ma.pooled_did(s, "harvest", cluster_col="county").set_index("ring").pct.to_dict()

placebo = ma.placebo_test(estimate, never_closed_ids, [2006, 2009], n_draws=200, seed=42)
```

## Status

Early development (`0.1.0.dev0`). Planned:

- [ ] Multi-state road networks (wood flows across state lines)
- [ ] Optional FIA outcomes through pyFIA
- [ ] Documentation site and example notebooks

## Citation

If you use `millaccess`, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

MIT
