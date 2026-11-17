# Changelog

## 0.1.0.dev0 (unreleased)

- Road network from road lines; TIGER/Line download helper
- Road distances from locations to mills (Dijkstra, with off-road legs)
- Distance-weighted access index and nearest-mill summaries
- Distance rings, alternative-mill flag, stacked event-study panel
- Event-time and pooled before/after estimators with clustered SEs
- Placebo (randomization) inference
- Period means and within-location access regressions (effect per SD of access)
- Closed-vs-continuing-mill design (`stack_matched`)
- `pooled_did` reports both location-event pairs (`n_units`) and distinct locations
  (`n_locations`), and reports groups without observations as NaN instead of failing
