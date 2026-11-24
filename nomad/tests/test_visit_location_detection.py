from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest

import nomad.visit_attribution.visit_attribution as visit_attribution
from nomad.visit_attribution.visit_attribution import detect_locations


@pytest.fixture
def cartesian_points():
    return pd.DataFrame(
        {"x": [0.0, 1.0, 20.0], "y": [0.0, 1.0, 20.0]},
        index=[10, 20, 30],
    )


def test_detect_locations_returns_aligned_location_clusters(cartesian_points):
    location_clusters = detect_locations(cartesian_points, epsilon=2)

    assert location_clusters.tolist() == [0, 0, 1]
    assert location_clusters.index.equals(cartesian_points.index)
    assert location_clusters.name == "location_cluster"


def test_detect_locations_accepts_stop_table():
    stops = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0],
            "y": [0.0, 1.0, 20.0],
            "start_timestamp": [0, 600, 1200],
            "duration": [10, 20, 30],
        }
    )

    location_clusters = detect_locations(stops, epsilon=2)

    assert location_clusters.tolist() == [0, 0, 1]


def test_detect_locations_relabels_noise_as_singletons():
    points = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0, 40.0],
            "y": [0.0, 1.0, 20.0, 40.0],
        }
    )

    location_clusters = detect_locations(points, epsilon=2, min_pts=2)

    assert location_clusters.tolist() == [0, 0, 1, 2]


def test_detect_locations_assigns_unique_ids_when_all_rows_are_noise():
    points = pd.DataFrame(
        {"x": [0.0, 20.0, 40.0], "y": [0.0, 20.0, 40.0]}
    )

    location_clusters = detect_locations(points, epsilon=2, min_pts=2)

    assert location_clusters.tolist() == [0, 1, 2]


def test_detect_locations_preserves_duplicate_indices():
    points = pd.DataFrame(
        {"x": [0.0, 1.0, 20.0], "y": [0.0, 1.0, 20.0]},
        index=[5, 5, 9],
    )

    location_clusters = detect_locations(points, epsilon=2)

    assert location_clusters.index.tolist() == [5, 5, 9]
    assert location_clusters.tolist() == [0, 0, 1]


def test_detect_locations_supports_custom_method(cartesian_points):
    def custom_algorithm(data, traj_cols, split_at):
        return pd.Series([0, 0, -1] if split_at == 2 else [0, 0, 0])

    location_clusters = detect_locations(
        cartesian_points,
        method='custom',
        algorithm=custom_algorithm,
        algorithm_kwargs={'split_at': 2},
    )

    assert location_clusters.tolist() == [0, 0, 1]


def test_detect_locations_multi_user_is_not_implemented():
    points = pd.DataFrame(
        {
            'user_id': ['a', 'b'],
            'x': [0.0, 1.0],
            'y': [0.0, 1.0],
        }
    )

    with pytest.raises(NotImplementedError, match='Multi-user'):
        detect_locations(points)


def test_detect_locations_uses_haversine_distance_in_meters():
    points = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.004],
            "latitude": [60.0, 60.0, 60.0],
        }
    )

    location_clusters = detect_locations(points, epsilon=75)

    assert location_clusters.tolist() == [0, 0, 1]


def test_detect_locations_clusters_across_antimeridian():
    points = pd.DataFrame(
        {
            "longitude": [179.9998, -179.9998, 179.99],
            "latitude": [0.0, 0.0, 0.0],
        }
    )

    location_clusters = detect_locations(points, epsilon=50)

    assert location_clusters.tolist() == [0, 0, 1]


def test_detect_locations_supports_custom_column_mappings():
    points = pd.DataFrame(
        {"east": [0.0, 1.0, 20.0], "north": [0.0, 1.0, 20.0]}
    )

    location_clusters = detect_locations(
        points,
        epsilon=2,
        traj_cols={
            "x": "east",
            "y": "north",
            "location_cluster": "destination_id",
        },
    )

    assert location_clusters.tolist() == [0, 0, 1]
    assert location_clusters.name == "destination_id"


def test_detect_locations_optionally_returns_location_geometries(
    cartesian_points,
):
    location_clusters, locations = detect_locations(
        cartesian_points,
        epsilon=2,
        return_locations=True,
    )

    assert location_clusters.tolist() == [0, 0, 1]
    assert locations["location_cluster"].tolist() == [0, 1]
    assert locations["n_stops"].tolist() == [2, 1]
    assert locations.center.x.tolist() == [0.5, 20.0]
    assert locations.center.y.tolist() == [0.5, 20.0]


def test_detect_locations_summarizes_noise_location():
    points = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0],
            "y": [0.0, 1.0, 20.0],
        }
    )

    location_clusters, locations = detect_locations(
        points,
        epsilon=2,
        min_pts=2,
        return_locations=True,
    )
    locations = locations.set_index("location_cluster")

    assert location_clusters.tolist() == [0, 0, 1]
    assert locations["n_stops"].to_dict() == {0: 2, 1: 1}
    assert locations.loc[1, "center"].x == 20.0
    assert locations.loc[1, "center"].y == 20.0
    assert locations.loc[1, "extent"].equals(locations.loc[1, "center"])


def test_detect_locations_rejects_geodataframe():
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0.0], [0.0]))

    with pytest.raises(NotImplementedError, match="GeoDataFrame"):
        detect_locations(points)


def test_detect_locations_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        detect_locations([[0.0, 0.0]])


def test_detect_locations_returns_empty_schema():
    points = pd.DataFrame(columns=["x", "y"])

    location_clusters, locations = detect_locations(points, return_locations=True)

    assert location_clusters.empty
    assert location_clusters.name == "location_cluster"
    assert location_clusters.dtype == "Int64"
    assert locations.empty
    assert locations.columns.tolist() == [
        "location_cluster",
        "n_stops",
        "center",
        "extent",
    ]
    assert locations["location_cluster"].dtype == "Int64"


def test_detect_locations_does_not_mutate_input(cartesian_points):
    original = cartesian_points.copy()

    detect_locations(cartesian_points, epsilon=2)

    pd.testing.assert_frame_equal(cartesian_points, original)


def test_infomap_locations_link_rows_within_epsilon():
    points = pd.DataFrame({"x": [0.0, 1.0, 2.0, 50.0, 51.0, 200.0], "y": 0.0})

    location_clusters = detect_locations(points, epsilon=1.5, method="infomap")

    assert location_clusters.tolist() == [0, 0, 0, 1, 1, 2]


def test_infomap_isolated_rows_keep_separate_locations():
    # Infomap could put a flowless isolated node in any module, so isolated
    # coordinates must not be passed to it. Rows sharing one still share a location.
    points = pd.DataFrame({"x": [0.0, 0.0, 100.0, 101.0, 102.0, 500.0], "y": 0.0})

    location_clusters = detect_locations(points, epsilon=5, method="infomap")

    assert location_clusters.tolist() == [0, 0, 1, 1, 1, 2]


def test_infomap_deduplicates_coordinates_and_weights_by_count(monkeypatch):
    links = []
    network = SimpleNamespace(add_links=lambda edges: links.extend(map(tuple, edges)))
    result = SimpleNamespace(modules=lambda: {0: 1, 1: 1})
    monkeypatch.setattr(visit_attribution.infomap, "Network", lambda: network)
    monkeypatch.setattr(visit_attribution.infomap, "run", lambda network, **options: result)
    points = pd.DataFrame({"x": [0.0, 0.0, 1.0, 20.0], "y": 0.0})

    location_clusters = detect_locations(points, epsilon=2, method="infomap")

    # One edge between the two unique nearby points, weighted by the larger count.
    assert links == [(0.0, 1.0, 2.0)]
    assert location_clusters.tolist() == [0, 0, 0, 1]


def test_infomap_epsilon_is_meters_for_lon_lat():
    # 0.001 degrees of longitude at 60N is about 56 m.
    points = pd.DataFrame({"longitude": [0.0, 0.001], "latitude": 60.0})

    assert detect_locations(points, epsilon=60, method="infomap").nunique() == 1
    assert detect_locations(points, epsilon=50, method="infomap").nunique() == 2
