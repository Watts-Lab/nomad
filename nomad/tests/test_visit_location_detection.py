import geopandas as gpd
import pandas as pd
import pytest

from nomad.visit_attribution.visit_attribution import detect_locations


@pytest.fixture
def cartesian_points():
    return pd.DataFrame(
        {"x": [0.0, 1.0, 20.0], "y": [0.0, 1.0, 20.0]},
        index=[10, 20, 30],
    )


def test_detect_locations_returns_aligned_location_ids(cartesian_points):
    location_ids = detect_locations(cartesian_points, epsilon=2)

    assert location_ids.tolist() == [0, 0, 1]
    assert location_ids.index.equals(cartesian_points.index)
    assert location_ids.name == "location_id"


def test_detect_locations_accepts_stop_table():
    stops = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0],
            "y": [0.0, 1.0, 20.0],
            "start_timestamp": [0, 600, 1200],
            "duration": [10, 20, 30],
        }
    )

    location_ids = detect_locations(stops, epsilon=2)

    assert location_ids.tolist() == [0, 0, 1]


def test_detect_locations_relabels_noise_as_singletons():
    points = pd.DataFrame(
        {
            "x": [0.0, 1.0, 20.0, 40.0],
            "y": [0.0, 1.0, 20.0, 40.0],
        }
    )

    location_ids = detect_locations(points, epsilon=2, min_pts=2)

    assert location_ids.tolist() == [0, 0, 1, 2]


def test_detect_locations_assigns_unique_ids_when_all_rows_are_noise():
    points = pd.DataFrame(
        {"x": [0.0, 20.0, 40.0], "y": [0.0, 20.0, 40.0]}
    )

    location_ids = detect_locations(points, epsilon=2, min_pts=2)

    assert location_ids.tolist() == [0, 1, 2]


def test_detect_locations_preserves_duplicate_indices():
    points = pd.DataFrame(
        {"x": [0.0, 1.0, 20.0], "y": [0.0, 1.0, 20.0]},
        index=[5, 5, 9],
    )

    location_ids = detect_locations(points, epsilon=2)

    assert location_ids.index.tolist() == [5, 5, 9]
    assert location_ids.tolist() == [0, 0, 1]


def test_detect_locations_uses_haversine_distance_in_meters():
    points = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.004],
            "latitude": [60.0, 60.0, 60.0],
        }
    )

    location_ids = detect_locations(points, epsilon=75)

    assert location_ids.tolist() == [0, 0, 1]


def test_detect_locations_clusters_across_antimeridian():
    points = pd.DataFrame(
        {
            "longitude": [179.9998, -179.9998, 179.99],
            "latitude": [0.0, 0.0, 0.0],
        }
    )

    location_ids = detect_locations(points, epsilon=50)

    assert location_ids.tolist() == [0, 0, 1]


def test_detect_locations_supports_custom_column_mappings():
    points = pd.DataFrame(
        {"east": [0.0, 1.0, 20.0], "north": [0.0, 1.0, 20.0]}
    )

    location_ids = detect_locations(
        points,
        epsilon=2,
        traj_cols={
            "x": "east",
            "y": "north",
            "location_id": "destination_id",
        },
    )

    assert location_ids.tolist() == [0, 0, 1]
    assert location_ids.name == "destination_id"


def test_detect_locations_optionally_returns_location_geometries(
    cartesian_points,
):
    location_ids, locations = detect_locations(
        cartesian_points,
        epsilon=2,
        return_locations=True,
    )

    assert location_ids.tolist() == [0, 0, 1]
    assert locations["location_id"].tolist() == [0, 1]
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

    location_ids, locations = detect_locations(
        points,
        epsilon=2,
        min_pts=2,
        return_locations=True,
    )
    locations = locations.set_index("location_id")

    assert location_ids.tolist() == [0, 0, 1]
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


def test_detect_locations_rejects_empty_dataframe():
    points = pd.DataFrame(columns=["x", "y"])

    with pytest.raises(ValueError, match="at least one row"):
        detect_locations(points)


def test_detect_locations_does_not_mutate_input(cartesian_points):
    original = cartesian_points.copy()

    detect_locations(cartesian_points, epsilon=2)

    pd.testing.assert_frame_equal(cartesian_points, original)
