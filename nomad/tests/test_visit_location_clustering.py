import geopandas as gpd
import pandas as pd
import pytest

from nomad.visit_attribution.visit_attribution import cluster_locations_dbscan


@pytest.fixture
def cartesian_points():
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0],
            "y": [0.0, 1.0, 20.0],
        },
        index=[10, 20, 30],
    )


def test_cluster_locations_dbscan_clusters_coordinate_points(cartesian_points):
    labeled, locations = cluster_locations_dbscan(
        cartesian_points,
        epsilon=2,
        agg_level="dataset",
    )

    assert labeled["location_id"].tolist() == [0, 0, 1]
    assert labeled.index.equals(cartesian_points.index)
    assert locations["location_id"].tolist() == [0, 1]
    assert locations["n_stops"].tolist() == [2, 1]


def test_cluster_locations_dbscan_preserves_stop_table_columns():
    stops = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0],
            "y": [0.0, 1.0, 20.0],
            "start_timestamp": [0, 600, 1200],
            "duration": [10, 20, 30],
        }
    )

    labeled, _ = cluster_locations_dbscan(
        stops,
        epsilon=2,
        agg_level="dataset",
    )

    pd.testing.assert_frame_equal(
        labeled.drop(columns="location_id"),
        stops,
    )
    assert labeled["location_id"].tolist() == [0, 0, 1]


def test_cluster_locations_dbscan_relabels_noise_as_singleton_locations():
    points = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0, 40.0],
            "y": [0.0, 1.0, 20.0, 40.0],
        }
    )

    labeled, locations = cluster_locations_dbscan(
        points,
        epsilon=2,
        min_pts=2,
        agg_level="dataset",
    )

    assert labeled["location_id"].tolist() == [0, 0, 1, 2]
    assert (labeled["location_id"] >= 0).all()
    assert locations["location_id"].tolist() == [0, 1, 2]
    assert locations["n_stops"].tolist() == [2, 1, 1]


def test_cluster_locations_dbscan_assigns_unique_ids_when_all_rows_are_noise():
    points = pd.DataFrame(
        {
            "x": [0.0, 20.0, 40.0],
            "y": [0.0, 20.0, 40.0],
        }
    )

    labeled, _ = cluster_locations_dbscan(
        points,
        epsilon=2,
        min_pts=2,
        agg_level="dataset",
    )

    assert labeled["location_id"].tolist() == [0, 1, 2]


def test_cluster_locations_dbscan_keeps_user_location_ids_disjoint():
    points = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "b"],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 0.0, 1.0],
        }
    )

    labeled, locations = cluster_locations_dbscan(points, epsilon=2)

    assert labeled["location_id"].tolist() == [0, 0, 1, 1]
    assert locations["location_id"].tolist() == [0, 1]
    assert locations["user_id"].tolist() == ["a", "b"]


def test_cluster_locations_dbscan_labels_duplicate_indices_by_position():
    points = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0],
            "y": [0.0, 1.0, 20.0],
        },
        index=[5, 5, 9],
    )

    labeled, _ = cluster_locations_dbscan(
        points,
        epsilon=2,
        agg_level="dataset",
    )

    assert labeled.index.tolist() == [5, 5, 9]
    assert labeled["location_id"].tolist() == [0, 0, 1]


def test_cluster_locations_dbscan_uses_haversine_distance_in_meters():
    points = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.004],
            "latitude": [60.0, 60.0, 60.0],
        }
    )

    labeled, _ = cluster_locations_dbscan(
        points,
        epsilon=75,
        agg_level="dataset",
    )

    assert labeled["location_id"].tolist() == [0, 0, 1]


def test_cluster_locations_dbscan_clusters_across_antimeridian():
    points = pd.DataFrame(
        {
            "longitude": [179.9998, -179.9998, 179.99],
            "latitude": [0.0, 0.0, 0.0],
        }
    )

    labeled, _ = cluster_locations_dbscan(
        points,
        epsilon=50,
        agg_level="dataset",
    )

    assert labeled["location_id"].tolist() == [0, 0, 1]


def test_cluster_locations_dbscan_supports_custom_column_mappings():
    points = pd.DataFrame(
        {
            "east": [0.0, 1.0, 20.0],
            "north": [0.0, 1.0, 20.0],
        }
    )

    labeled, locations = cluster_locations_dbscan(
        points,
        epsilon=2,
        agg_level="dataset",
        traj_cols={
            "x": "east",
            "y": "north",
            "location_id": "destination_id",
        },
    )

    assert labeled["destination_id"].tolist() == [0, 0, 1]
    assert locations["destination_id"].tolist() == [0, 1]


def test_cluster_locations_dbscan_supports_point_geodataframe():
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0, 1.0, 20.0], [0.0, 1.0, 20.0]),
        crs="EPSG:3857",
    )

    labeled, locations = cluster_locations_dbscan(
        points,
        epsilon=2,
        agg_level="dataset",
    )

    assert labeled["location_id"].tolist() == [0, 0, 1]
    assert labeled.crs == points.crs
    assert locations.crs == points.crs


def test_cluster_locations_dbscan_returns_empty_schema():
    points = pd.DataFrame(columns=["x", "y"])

    labeled, locations = cluster_locations_dbscan(
        points,
        agg_level="dataset",
    )

    assert labeled.empty
    assert labeled.columns.tolist() == ["x", "y", "location_id"]
    assert labeled["location_id"].dtype == "int64"
    assert locations.empty
    assert locations.columns.tolist() == [
        "location_id",
        "center",
        "extent",
        "n_stops",
    ]


def test_cluster_locations_dbscan_does_not_mutate_input(cartesian_points):
    original = cartesian_points.copy()

    cluster_locations_dbscan(
        cartesian_points,
        epsilon=2,
        agg_level="dataset",
    )

    pd.testing.assert_frame_equal(cartesian_points, original)


def test_cluster_locations_dbscan_requires_user_column_for_user_level(
    cartesian_points,
):
    with pytest.raises(ValueError, match="requires a user_id column"):
        cluster_locations_dbscan(cartesian_points, agg_level="user")


def test_cluster_locations_dbscan_rejects_unknown_aggregation_level(
    cartesian_points,
):
    with pytest.raises(ValueError, match="agg_level must be"):
        cluster_locations_dbscan(cartesian_points, agg_level="household")
