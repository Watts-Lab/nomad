import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

import nomad.visit_attribution.visit_attribution as visit_attribution
from nomad.visit_attribution.visit_attribution import detect_locations, poi_map


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


def test_detect_locations_supports_custom_method(cartesian_points):
    def custom_algorithm(data, traj_cols, split_at):
        return pd.Series([0, 0, -1] if split_at == 2 else [0, 0, 0])

    location_ids = detect_locations(
        cartesian_points,
        method='custom',
        algorithm=custom_algorithm,
        algorithm_kwargs={'split_at': 2},
    )

    assert location_ids.tolist() == [0, 0, 1]


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


def test_detect_locations_returns_empty_schema():
    points = pd.DataFrame(columns=["x", "y"])

    location_ids, locations = detect_locations(points, return_locations=True)

    assert location_ids.empty
    assert location_ids.name == "location_id"
    assert location_ids.dtype == "Int64"
    assert locations.empty
    assert locations.columns.tolist() == [
        "location_id",
        "n_stops",
        "center",
        "extent",
    ]
    assert locations["location_id"].dtype == "Int64"


def test_detect_locations_does_not_mutate_input(cartesian_points):
    original = cartesian_points.copy()

    detect_locations(cartesian_points, epsilon=2)

    pd.testing.assert_frame_equal(cartesian_points, original)


def test_poi_map_attributes_unique_h3_cells_and_preserves_alignment(monkeypatch):
    poi_cell = h3.latlng_to_cell(39.95, -75.16, 10)
    adjacent_cell = next(iter(h3.grid_ring(poi_cell, 1)))
    unmatched_cell = next(iter(h3.grid_ring(poi_cell, 2)))
    latitude, longitude = h3.cell_to_latlng(poi_cell)
    stops = pd.DataFrame(
        {"h3_cell": [poi_cell, poi_cell, adjacent_cell, unmatched_cell, pd.NA]},
        index=[3, 3, 7, 9, 11],
    )
    pois = gpd.GeoDataFrame(
        {"building_id": ["library"]},
        geometry=[Point(longitude, latitude).buffer(0.000001)],
        crs="EPSG:4326",
    )
    batch_sizes = []
    grid_disk_distances = visit_attribution.h3ronpy.grid_disk_distances

    def record_batch_size(cells, max_distance):
        batch_sizes.append(len(cells))
        return grid_disk_distances(cells, max_distance)

    monkeypatch.setattr(
        visit_attribution.h3ronpy,
        "grid_disk_distances",
        record_batch_size,
    )

    locations = poi_map(
        stops,
        pois,
        max_distance=1,
        location_id="building_id",
    )

    assert batch_sizes == [3]
    assert locations.index.tolist() == [3, 3, 7, 9, 11]
    assert locations.name == "building_id"
    assert locations.iloc[:3].tolist() == ["library", "library", "library"]
    assert locations.iloc[3:].isna().all()


def test_poi_map_h3_supports_projected_pois_and_column_overrides():
    h3_cell = h3.latlng_to_cell(39.95, -75.16, 10)
    latitude, longitude = h3.cell_to_latlng(h3_cell)
    stops = pd.DataFrame({"containment_area": [h3_cell]})
    pois = gpd.GeoDataFrame(
        {"place": [42]},
        geometry=[Point(longitude, latitude).buffer(0.000001)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    with pytest.warns(UserWarning, match="Reprojecting for H3 attribution"):
        locations = poi_map(
            stops,
            pois,
            location_id="place",
            traj_cols={"h3_cell": "containment_area"},
        )

    assert locations.tolist() == [42]
    assert locations.name == "place"


def test_poi_map_h3_breaks_equidistant_ties_by_poi_order():
    stop_cell = h3.latlng_to_cell(39.95, -75.16, 10)
    poi_cells = list(h3.grid_ring(stop_cell, 1))[:2]
    centers = [h3.cell_to_latlng(cell) for cell in poi_cells]
    pois = gpd.GeoDataFrame(
        {"location_id": ["first", "second"]},
        geometry=[
            Point(longitude, latitude).buffer(0.000001)
            for latitude, longitude in centers
        ],
        crs="EPSG:4326",
    )

    locations = poi_map(
        pd.DataFrame({"h3_cell": [stop_cell]}),
        pois,
        max_distance=1,
        location_id="location_id",
    )

    assert locations.tolist() == ["first"]


def test_poi_map_h3_returns_aligned_empty_result():
    pois = gpd.GeoDataFrame(
        {"location_id": ["unused"]},
        geometry=[Point(-75.16, 39.95).buffer(0.000001)],
        crs="EPSG:4326",
    )
    stops = pd.DataFrame({"cell": pd.Series(dtype="string")})

    locations = poi_map(
        stops,
        pois,
        location_id="location_id",
        traj_cols={"h3_cell": "cell"},
    )

    assert locations.empty
    assert locations.index.equals(stops.index)
    assert locations.name == "location_id"


def test_poi_map_h3_maps_multi_cell_poi_and_falls_back_to_index():
    first_cell = h3.latlng_to_cell(39.95, -75.16, 10)
    second_cell = next(iter(h3.grid_ring(first_cell, 1)))
    centers = [h3.cell_to_latlng(cell) for cell in [first_cell, second_cell]]
    poi = gpd.GeoDataFrame(
        geometry=[
            LineString([
                (longitude, latitude) for latitude, longitude in centers
            ]).buffer(0.000001)
        ],
        index=pd.Index(["building-a"]),
        crs="EPSG:4326",
    )

    with pytest.warns(UserWarning, match="using poi_table.index"):
        locations = poi_map(
            pd.DataFrame({"h3_cell": [first_cell, second_cell]}),
            poi,
        )

    assert locations.tolist() == ["building-a", "building-a"]
    assert locations.name == "location_id"


def test_poi_map_h3_requires_one_resolution_per_call():
    cells = [
        h3.latlng_to_cell(39.95, -75.16, 9),
        h3.latlng_to_cell(39.95, -75.16, 10),
    ]
    poi = gpd.GeoDataFrame(
        geometry=[Point(-75.16, 39.95).buffer(0.000001)],
        crs="EPSG:4326",
    )

    with pytest.warns(UserWarning, match="using poi_table.index"):
        with pytest.raises(ValueError, match="same resolution"):
            poi_map(pd.DataFrame({"h3_cell": cells}), poi)
