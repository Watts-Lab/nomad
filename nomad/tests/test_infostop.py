import numpy as np
import pandas as pd
import pytest

from nomad.stop_detection.infostop import infostop, infostop_labels
from nomad.stop_detection.sequential_algs import detect_stops_labels


@pytest.fixture
def home_work_home():
    """Two pings each at home, work, then home again, five minutes apart."""
    return pd.DataFrame({
        "x": [0., 0., 100., 100., 0., 0.],
        "y": 0.,
        "timestamp": np.arange(6) * 300,
    })


def test_infostop_labels_separate_visits_to_recurring_location(home_work_home):
    labels = infostop_labels(home_work_home, r1=5, r2=10, return_location_clusters=True)

    assert labels["cluster"].tolist() == [0, 0, 1, 1, 2, 2]
    assert labels["location_cluster"].tolist() == [0, 0, 1, 1, 0, 0]


def test_infostop_stop_table_keeps_location_cluster(home_work_home):
    stops = infostop(home_work_home, r1=5, r2=10, complete_output=True)

    assert stops["cluster"].tolist() == [0, 1, 2]
    assert stops["location_cluster"].tolist() == [0, 1, 0]
    assert stops["location_cluster"].dtype == "Int64"


def test_infostop_supports_custom_location_cluster_column(home_work_home):
    stops = infostop(home_work_home, r1=5, r2=10, traj_cols={"location_cluster": "place"})

    assert stops["place"].tolist() == [0, 1, 0]


def test_radius_graph_joins_adjacent_events(home_work_home):
    home_work_home["x"] = [0., 0., 4., 4., 0., 0.]
    labels = infostop_labels(home_work_home, r1=1, r2=5, return_location_clusters=True)

    # Three stationary events at one location form one uninterrupted visit.
    assert labels["cluster"].tolist() == [0] * 6
    assert labels["location_cluster"].tolist() == [0] * 6


def test_noise_splits_visits_to_same_location(home_work_home):
    home_work_home["x"] = [0., 0., 100., 0., 0., 0.]
    labels = infostop_labels(home_work_home, r1=1, r2=1, return_location_clusters=True)

    assert labels["cluster"].tolist() == [0, 0, -1, 1, 1, 1]
    assert labels["location_cluster"].tolist() == [0, 0, -1, 0, 0, 0]


def test_gap_longer_than_dt_max_splits_visits(home_work_home):
    home_work_home["x"] = 0.
    home_work_home["timestamp"] = [0, 300, 3600, 3900, 7200, 7500]
    labels = infostop_labels(home_work_home, r1=1, r2=1, dt_max=10, return_location_clusters=True)

    assert labels["cluster"].tolist() == [0, 0, 1, 1, 2, 2]
    assert labels["location_cluster"].tolist() == [0] * 6


def test_geographic_radii_are_meters():
    # 0.001 degrees of longitude at 60N is about 56 m.
    data = pd.DataFrame({
        "longitude": [0., 0.0001, 0.001, 0.0011],
        "latitude": 60.,
        "timestamp": np.arange(4) * 300,
    })
    labels = infostop_labels(data, r1=6, r2=75, return_location_clusters=True)

    assert labels["location_cluster"].tolist() == [0, 0, 0, 0]


def test_median_scan_matches_naive_infostop_scan():
    rng = np.random.default_rng(8)
    coords = rng.normal(size=(500, 2)) + np.repeat(np.arange(10) * 10, 50)[:, None]
    times = np.arange(500) * 60
    data = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "timestamp": times})

    labels = detect_stops_labels(data, delta_roam=2, dur_min=5, dt_max=10, method="median")

    expected = np.full(len(coords), -1)
    stop, start = 0, 0
    for end in range(1, len(coords) + 1):
        if end < len(coords) and np.linalg.norm(coords[end] - np.median(coords[start:end], axis=0)) <= 2:
            continue
        if times[end - 1] - times[start] >= 300:
            expected[start:end] = stop
            stop += 1
        start = end
    np.testing.assert_array_equal(labels, expected)


def test_median_scan_does_not_rescan_failed_event():
    # Pings 0 and 1 form a short failed event. Rescanning from ping 1 would
    # yield a stop at pings 1-3; Infostop resumes at the breaking ping 2.
    data = pd.DataFrame({"x": [0., 1.5, 3., 3.], "y": 0., "timestamp": [0, 60, 120, 420]})

    labels = detect_stops_labels(data, delta_roam=2, dur_min=5, dt_max=10, method="median")

    assert labels.tolist() == [-1, -1, 0, 0]


@pytest.fixture
def buildings():
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"building_id": ["home", "work"]},
        geometry=[box(-5, -5, 5, 5), box(95, -5, 105, 5)],
        crs="EPSG:3857",
    )


def test_majority_maps_stop_clusters_to_polygons(buildings):
    from nomad.visit_attribution.visit_attribution import point_in_polygon

    # Cluster 0 has two stops in "home" and one outside; cluster 1 falls in no polygon.
    stops = pd.DataFrame({
        "x": [0., 1., 50., 300.],
        "y": 0.,
        "timestamp": np.arange(4) * 3600,
        "duration": 10,
        "location_cluster": [0, 0, 0, 1],
    })
    location = point_in_polygon(
        stops, buildings, method="majority", cluster_label="location_cluster",
        location_id="building_id", data_crs="EPSG:3857",
    )

    assert location.tolist()[:3] == ["home"] * 3
    assert pd.isna(location.iloc[3])

    with pytest.raises(ValueError, match="cluster_label"):
        point_in_polygon(stops, buildings, method="majority", data_crs="EPSG:3857")


def test_infostop_location_clusters_map_to_buildings(home_work_home, buildings):
    from nomad.visit_attribution.visit_attribution import point_in_polygon

    stops = infostop(home_work_home, r1=5, r2=10, complete_output=True)
    location = point_in_polygon(
        stops, buildings, method="majority", cluster_label="location_cluster",
        location_id="building_id", data_crs="EPSG:3857",
    )

    assert location.tolist() == ["home", "work", "home"]
