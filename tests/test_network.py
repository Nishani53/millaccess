import numpy as np
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString

from millaccess import RoadNetwork, road_distances


def make_network():
    # A horizontal road running *through* an intersection (shared vertex at 10000, 5000),
    # a vertical road crossing it, and a disconnected island road.
    lines = [
        LineString([(0, 5000), (10000, 5000), (20000, 5000)]),
        LineString([(10000, 0), (10000, 5000), (10000, 10000)]),
        LineString([(50000, 50000), (51000, 50000)]),  # island
    ]
    return RoadNetwork.from_lines(np.array(lines, dtype=object))


def test_largest_component_drops_island():
    net = make_network()
    assert net.nodes.x.max() <= 20000  # island removed
    assert len(net.nodes) == 5  # 2 ends of each road + the intersection


def test_distance_routes_through_intersection():
    net = make_network()
    d = road_distances(net, np.array([[0, 5000]]), ["cell1"], np.array([[10000, 0]]), ["millA"])
    assert len(d) == 1
    assert d.road_km.iloc[0] == pytest.approx(15.0)
    assert d.straight_km.iloc[0] == pytest.approx(np.hypot(10, 5))


def test_off_road_legs_are_added():
    net = make_network()
    # origin 300 m north of the road start, mill 400 m east of the vertical road's bottom
    d = road_distances(net, np.array([[0, 5300]]), ["c"], np.array([[10400, 0]]), ["m"])
    assert d.road_km.iloc[0] == pytest.approx(15.0 + 0.3 + 0.4)


def test_max_km_limit():
    net = make_network()
    d = road_distances(net, np.array([[0, 5000]]), ["c"], np.array([[10000, 0]]), ["m"], max_km=10)
    assert d.empty


def test_save_and_load(tmp_path):
    net = make_network()
    net.save(tmp_path)
    net2 = RoadNetwork.load(tmp_path)
    assert len(net2.nodes) == len(net.nodes) and len(net2.edges) == len(net.edges)
