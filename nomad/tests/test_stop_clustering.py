import pandas as pd
import pytest

from nomad.stop_detection.clustering import dbscan_labels
from nomad.stop_detection.postprocessing import cluster_stops


@pytest.fixture
def cartesian_stops():
    return pd.DataFrame(
        {
            "x": [0.0, 3.0, 20.0],
            "y": [0.0, 4.0, 20.0],
            "duration": [10, 20, 30],
        },
        index=[10, 20, 30],
    )


def test_dbscan_labels_clusters_nearby_cartesian_stops(cartesian_stops):
    labels = dbscan_labels(cartesian_stops, epsilon=5)

    assert labels.tolist() == [0, 0, 1]
    assert labels.index.equals(cartesian_stops.index)
    assert labels.name == "location_id"


def test_dbscan_labels_marks_isolated_stop_as_noise(cartesian_stops):
    labels = dbscan_labels(cartesian_stops, epsilon=5, num_samples=2)

    assert labels.tolist() == [0, 0, -1]


def test_dbscan_labels_clusters_duplicate_stops():
    stops = pd.DataFrame({"x": [2.0, 2.0], "y": [4.0, 4.0]})

    labels = dbscan_labels(stops, epsilon=1, num_samples=2)

    assert labels.tolist() == [0, 0]


def test_dbscan_labels_clusters_single_stop_by_default():
    stops = pd.DataFrame({"x": [2.0], "y": [4.0]}, index=[7])

    labels = dbscan_labels(stops)

    assert labels.tolist() == [0]
    assert labels.index.equals(stops.index)


def test_dbscan_labels_uses_haversine_distance_in_meters():
    stops = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.004],
            "latitude": [60.0, 60.0, 60.0],
        }
    )

    labels = dbscan_labels(stops, epsilon=75)

    assert labels.tolist() == [0, 0, 1]


def test_dbscan_labels_clusters_across_antimeridian():
    stops = pd.DataFrame(
        {
            "longitude": [179.9998, -179.9998, 179.99],
            "latitude": [0.0, 0.0, 0.0],
        }
    )

    labels = dbscan_labels(stops, epsilon=50)

    assert labels.tolist() == [0, 0, 1]


def test_dbscan_labels_supports_custom_coordinate_columns():
    stops = pd.DataFrame(
        {
            "east": [0.0, 1.0, 10.0],
            "north": [0.0, 1.0, 10.0],
        }
    )

    labels = dbscan_labels(
        stops,
        epsilon=2,
        traj_cols={"x": "east", "y": "north"},
    )

    assert labels.tolist() == [0, 0, 1]


def test_dbscan_labels_supports_custom_geographic_columns():
    stops = pd.DataFrame(
        {
            "lon": [0.0, 0.001, 0.004],
            "lat": [60.0, 60.0, 60.0],
        }
    )

    labels = dbscan_labels(
        stops,
        epsilon=75,
        traj_cols={"longitude": "lon", "latitude": "lat"},
    )

    assert labels.tolist() == [0, 0, 1]


def test_dbscan_labels_supports_coordinate_keyword_arguments():
    stops = pd.DataFrame(
        {
            "east": [0.0, 1.0, 10.0],
            "north": [0.0, 1.0, 10.0],
        }
    )

    labels = dbscan_labels(stops, epsilon=2, x="east", y="north")

    assert labels.tolist() == [0, 0, 1]


def test_dbscan_labels_returns_empty_aligned_series():
    stops = pd.DataFrame(columns=["x", "y"])

    labels = dbscan_labels(stops)

    assert labels.empty
    assert labels.index.equals(stops.index)
    assert labels.dtype == "int64"
    assert labels.name == "location_id"


def test_dbscan_labels_requires_spatial_columns():
    stops = pd.DataFrame({"duration": [10, 20]})

    with pytest.raises(ValueError, match="No spatial columns found"):
        dbscan_labels(stops)


def test_cluster_stops_adds_labels_without_mutating_input(cartesian_stops):
    expected_labels = dbscan_labels(cartesian_stops, epsilon=5)

    result = cluster_stops(cartesian_stops, epsilon=5)

    assert "location_id" not in cartesian_stops.columns
    assert result is not cartesian_stops
    pd.testing.assert_frame_equal(
        result.drop(columns="location_id"),
        cartesian_stops,
    )
    pd.testing.assert_series_equal(result["location_id"], expected_labels)


def test_cluster_stops_handles_empty_stop_table():
    stops = pd.DataFrame(columns=["x", "y", "duration"])

    result = cluster_stops(stops)

    assert result.empty
    assert result.columns.tolist() == ["x", "y", "duration", "location_id"]
    assert result["location_id"].dtype == "int64"
