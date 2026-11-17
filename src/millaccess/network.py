"""Road networks and shortest-path road distances from locations to mills.

The network is built from road line geometries in a projected coordinate system
(units of metres). Distances are computed with Dijkstra's algorithm from each mill,
and the straight-line "off-road" legs from each location and each mill to its
nearest road node are added.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

#: TIGER/Line road classes used by default: primary, secondary, local/rural roads and ramps.
TIGER_ROAD_CLASSES = ("S1100", "S1200", "S1400", "S1630")


@dataclass
class RoadNetwork:
    """An undirected road network.

    Attributes
    ----------
    nodes : DataFrame with columns ``node``, ``x``, ``y`` (projected metres).
    edges : DataFrame with columns ``u``, ``v``, ``length`` (metres).
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame

    @classmethod
    def from_lines(cls, lines, snap_m: float = 1.0, largest_component: bool = True) -> RoadNetwork:
        """Build a network from road lines.

        Parameters
        ----------
        lines : GeoSeries, GeoDataFrame or array of shapely (Multi)LineStrings in a
            projected CRS with metre units.
        snap_m : vertices closer than this (after rounding) are treated as the same node.
        largest_component : keep only the largest connected piece of the network
            (removes islands and disconnected fragments).

        Notes
        -----
        Road records (e.g. TIGER/Line) can run through intersections, so a node is any
        vertex shared by two or more records, plus every record's two end points.
        Records are split at those nodes.
        """
        import shapely

        geoms = getattr(lines, "geometry", lines)
        if hasattr(geoms, "explode"):
            geoms = geoms.explode(index_parts=False).values
        geoms = np.asarray(shapely.get_parts(np.asarray(geoms)))
        geoms = geoms[shapely.length(geoms) > 0]
        coords, line = shapely.get_coordinates(geoms, return_index=True)

        key = np.round(coords / snap_m, 0) * snap_m
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        pair = np.unique(np.c_[inv, line], axis=0)
        n_lines = np.bincount(pair[:, 0], minlength=len(uniq))
        start = np.r_[True, line[1:] != line[:-1]]
        end = np.r_[line[1:] != line[:-1], True]
        is_node = start | end | (n_lines[inv] > 1)

        seg = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
        seg[line[1:] != line[:-1]] = 0.0
        cum = np.r_[0.0, np.cumsum(seg)]
        p = np.flatnonzero(is_node)
        same = line[p[1:]] == line[p[:-1]]
        u_v, v_v = inv[p[:-1]][same], inv[p[1:]][same]
        length = (cum[p[1:]] - cum[p[:-1]])[same]

        used = np.unique(np.r_[u_v, v_v])
        remap = -np.ones(len(uniq), dtype=np.int64)
        remap[used] = np.arange(len(used))
        nodes = pd.DataFrame({"node": np.arange(len(used)), "x": uniq[used, 0], "y": uniq[used, 1]})
        edges = pd.DataFrame({"u": remap[u_v], "v": remap[v_v], "length": length})
        edges = edges[edges.u != edges.v].reset_index(drop=True)
        net = cls(nodes, edges)
        return net.largest_component() if largest_component else net

    def largest_component(self) -> RoadNetwork:
        """Return the largest connected piece of the network."""
        n = len(self.nodes)
        a = csr_matrix((np.ones(len(self.edges)), (self.edges.u, self.edges.v)), shape=(n, n))
        _, lab = connected_components(a, directed=False)
        keep = lab == np.bincount(lab).argmax()
        remap = -np.ones(n, dtype=np.int64)
        remap[keep] = np.arange(keep.sum())
        edges = self.edges[keep[self.edges.u.values] & keep[self.edges.v.values]].copy()
        edges["u"], edges["v"] = remap[edges.u.values], remap[edges.v.values]
        nodes = self.nodes[keep].copy()
        nodes["node"] = np.arange(len(nodes))
        return RoadNetwork(nodes.reset_index(drop=True), edges.reset_index(drop=True))

    def _graph(self) -> csr_matrix:
        e = pd.DataFrame({"a": self.edges.u.values, "b": self.edges.v.values, "w": self.edges.length.values})
        e = pd.concat([e, e.rename(columns={"a": "b", "b": "a"})])  # roads usable both ways
        e = e.groupby(["a", "b"], as_index=False)["w"].min()  # shortest of parallel edges
        n = len(self.nodes)
        return csr_matrix((e["w"].values, (e["a"].values, e["b"].values)), shape=(n, n))

    def save(self, folder: str | Path) -> None:
        """Save nodes and edges as Parquet (or pickle if pyarrow is missing)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            self.nodes.to_parquet(folder / "nodes.parquet")
            self.edges.to_parquet(folder / "edges.parquet")
        except ImportError:
            self.nodes.to_pickle(folder / "nodes.pkl")
            self.edges.to_pickle(folder / "edges.pkl")

    @classmethod
    def load(cls, folder: str | Path) -> RoadNetwork:
        folder = Path(folder)
        if (folder / "nodes.parquet").exists():
            return cls(pd.read_parquet(folder / "nodes.parquet"), pd.read_parquet(folder / "edges.parquet"))
        return cls(pd.read_pickle(folder / "nodes.pkl"), pd.read_pickle(folder / "edges.pkl"))


def road_distances(
    network: RoadNetwork,
    origins_xy: np.ndarray,
    origin_ids: Sequence,
    mills_xy: np.ndarray,
    mill_ids: Sequence,
    max_km: float = 200.0,
    chunk: int = 25,
    straight_line: bool = True,
) -> pd.DataFrame:
    """Shortest road distance (km) from every origin to every mill within ``max_km``.

    Parameters
    ----------
    network : RoadNetwork built in the same projected CRS (metres) as the points.
    origins_xy, mills_xy : (n, 2) arrays of projected coordinates (metres), e.g. grid
        cell centroids or FIA plot locations, and mill locations.
    origin_ids, mill_ids : identifiers, same length as the coordinate arrays.
    max_km : pairs farther apart than this (by road) are not returned.
    chunk : number of mills routed at once (memory vs speed).
    straight_line : also return the straight-line distance for each pair.

    Returns
    -------
    DataFrame with columns ``origin_id``, ``mill_id``, ``road_km`` (and ``straight_km``).
    Road distance includes the straight-line legs from each point to its nearest road node.
    """
    origins_xy = np.asarray(origins_xy, dtype=float)
    mills_xy = np.asarray(mills_xy, dtype=float)
    origin_ids = np.asarray(origin_ids)
    mill_ids = np.asarray(mill_ids)
    if len(origins_xy) != len(origin_ids) or len(mills_xy) != len(mill_ids):
        raise ValueError("coordinates and ids must have the same length")

    graph = network._graph()
    tree = cKDTree(network.nodes[["x", "y"]].values)
    o_snap, o_node = tree.query(origins_xy)
    m_snap, m_node = tree.query(mills_xy)
    limit = max_km * 1000.0

    parts = []
    for k in range(0, len(mills_xy), chunk):
        sl = slice(k, k + chunk)
        d = dijkstra(graph, directed=False, indices=m_node[sl], limit=limit)
        dc = d[:, o_node] + o_snap[None, :] + m_snap[sl][:, None]
        mi, oi = np.where(dc <= limit)
        parts.append(
            pd.DataFrame(
                {
                    "origin_id": origin_ids[oi],
                    "mill_id": mill_ids[k + mi],
                    "road_km": dc[mi, oi] / 1000.0,
                }
            )
        )
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["origin_id", "mill_id", "road_km"])
    if straight_line and len(out):
        o_pos = pd.Series(np.arange(len(origin_ids)), index=origin_ids)
        m_pos = pd.Series(np.arange(len(mill_ids)), index=mill_ids)
        a = origins_xy[o_pos[out.origin_id].values]
        b = mills_xy[m_pos[out.mill_id].values]
        out["straight_km"] = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]) / 1000.0
    return out


def download_tiger_roads(
    county_fips: Iterable[str],
    cache_dir: str | Path,
    years: Sequence[int] = (2024, 2023),
    road_classes: Sequence[str] = TIGER_ROAD_CLASSES,
):
    """Download U.S. Census TIGER/Line county road files and return one GeoDataFrame.

    Files are cached in ``cache_dir`` and not downloaded again. ``county_fips`` are
    5-digit state+county codes, e.g. ``["26001", "26003"]``. Requires ``requests`` and
    ``geopandas`` (``pip install millaccess[geo]``).
    """
    import geopandas as gpd
    import requests

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in county_fips:
        zp = cache_dir / f"tl_roads_{f}.zip"
        if not zp.exists():
            for yr in years:
                url = f"https://www2.census.gov/geo/tiger/TIGER{yr}/ROADS/tl_{yr}_{f}_roads.zip"
                r = requests.get(url, timeout=300)
                if r.status_code == 200:
                    zp.write_bytes(r.content)
                    break
            else:
                raise RuntimeError(f"Could not download TIGER roads for county {f}")
        g = gpd.read_file(zp)
        frames.append(g[g["MTFCC"].isin(road_classes)][["MTFCC", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def state_county_fips(state_fips: str, n_counties: int) -> list[str]:
    """County FIPS codes for states that number counties 001, 003, ... (odd numbers).

    Maine: ``state_county_fips("23", 16)``. Check your state's list; some states
    have gaps or independent cities.
    """
    return [f"{state_fips}{c:03d}" for c in range(1, 2 * n_counties, 2)]
