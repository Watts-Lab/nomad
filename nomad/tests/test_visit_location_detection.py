import geopandas as gpd
import pandas as pd
import pytest

import nomad.visit_attribution.visit_attribution as visits
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


@pytest.fixture
def infomap_stops():
    pytest.importorskip('infomap')
    return pd.DataFrame({
        'x': [0., 1., 2., 50., 51., 52., 200.], 'y': 0.,
        'start_timestamp': range(0, 4200, 600), 'duration': 5,
    }, index=[5, 5, 9, 2, 3, 8, 4])


def test_infomap_assigns_locations_and_summaries_without_mutation(infomap_stops):
    original = infomap_stops.copy()
    ids, locations = detect_locations(
        infomap_stops, method='infomap', epsilon=5, sigma=2, return_locations=True,
    )
    assert ids.tolist() == [0, 0, 0, 1, 1, 1, 2]
    assert ids.index.equals(infomap_stops.index)
    assert ids.name == 'location_id'
    assert locations.n_stops.tolist() == [3, 3, 1]
    assert locations.center.x.tolist() == [1., 51., 200.]
    pd.testing.assert_frame_equal(infomap_stops, original)
    pd.testing.assert_series_equal(ids, detect_locations(
        infomap_stops, method='infomap', epsilon=5, sigma=2,
    ))


def test_infomap_keeps_users_independent_with_interleaved_duplicate_indices(infomap_stops):
    data = pd.concat([
        infomap_stops.assign(person='a'), infomap_stops.assign(person='b'),
    ]).sort_values('start_timestamp', kind='stable')
    columns = {'user_id': 'person', 'location_id': 'place'}
    ids, locations = detect_locations(
        data, method='infomap', epsilon=5, sigma=2,
        traj_cols=columns, return_locations=True,
    )
    assert ids.index.equals(data.index)
    assert set(ids[data.person == 'a']).isdisjoint(ids[data.person == 'b'])
    assert locations.groupby('person').n_stops.sum().to_dict() == {'a': 7, 'b': 7}
    for user in ['a', 'b']:
        mask = data.person == user
        independent = detect_locations(
            data.loc[mask], method='infomap', epsilon=5, sigma=2, traj_cols=columns,
        )
        assert pd.factorize(ids[mask])[0].tolist() == independent.tolist()


@pytest.mark.parametrize('coords, epsilon, expected', [
    ([(0., 60.), (0.001, 60.), (0.004, 60.)], 75, [0, 0, 1]),
    ([(179.9998, 0.), (-179.9998, 0.), (179.99, 0.)], 50, [0, 0, 1]),
])
def test_infomap_uses_geographic_distance_in_meters(coords, epsilon, expected):
    pytest.importorskip('infomap')
    stops = pd.DataFrame(coords, columns=['longitude', 'latitude']).assign(
        timestamp=[0, 600, 1200], duration=5,
    )
    ids = detect_locations(stops, method='infomap', epsilon=epsilon, sigma=epsilon)
    assert ids.tolist() == expected
    with pytest.raises(NotImplementedError, match='projected'):
        detect_locations(stops, method='infomap', sigma=epsilon, return_locations=True)


@pytest.mark.parametrize('size', [0, 1, 3])
def test_infomap_handles_empty_and_isolated_stops(infomap_stops, size):
    data = infomap_stops.iloc[:size].copy()
    data['x'] = [float(i * 100) for i in range(size)]
    ids, locations = detect_locations(
        data, method='infomap', epsilon=5, sigma=2, return_locations=True,
    )
    assert ids.tolist() == list(range(size))
    assert len(locations) == size
    assert locations.columns.tolist() == ['location_id', 'n_stops', 'center', 'extent']
    if not size:
        assert ids.dtype == 'Int64'


def test_infomap_accepts_coincident_stops_and_end_times(infomap_stops):
    stops = infomap_stops.iloc[:2].drop(columns='duration').assign(
        x=0., end_timestamp=[300, 900],
    ).rename(columns={'x': 'east', 'y': 'north', 'start_timestamp': 'arrival',
                      'end_timestamp': 'departure'})
    ids = detect_locations(
        stops, method='infomap', epsilon=5, sigma=2,
        traj_cols={'x': 'east', 'y': 'north', 'start_timestamp': 'arrival',
                   'end_timestamp': 'departure', 'location_id': 'place'},
    )
    assert ids.tolist() == [0, 0]
    assert ids.name == 'place'


def test_infomap_numbers_an_initial_isolate_before_communities(infomap_stops):
    stops = infomap_stops.iloc[[6, 0, 1]]
    ids = detect_locations(stops, method='infomap', epsilon=5, sigma=2)
    assert ids.tolist() == [0, 1, 1]


def test_infomap_distance_scale_changes_weak_connections(infomap_stops):
    stops = infomap_stops.iloc[:6].assign(x=[0., 1., 2., 8., 9., 10.])
    tight = detect_locations(stops, method='infomap', epsilon=10, sigma=1)
    broad = detect_locations(stops, method='infomap', epsilon=10, sigma=100)
    assert tight.tolist() == [0, 0, 0, 1, 1, 1]
    assert broad.tolist() == [0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize('options, message', [
    ({}, 'sigma'), ({'sigma': 0}, 'sigma'), ({'sigma': float('nan')}, 'sigma'),
    ({'sigma': True}, 'sigma'), ({'sigma': 2, 'epsilon': -1}, 'epsilon'),
    ({'sigma': 2, 'epsilon': float('inf')}, 'epsilon'),
    ({'sigma': 2, 'algorithm_kwargs': {}}, "method='custom'"),
])
def test_infomap_rejects_invalid_options(infomap_stops, options, message):
    with pytest.raises(ValueError, match=message):
        detect_locations(infomap_stops, method='infomap', **options)


def test_infomap_requires_stop_rows(cartesian_points):
    with pytest.raises(ValueError, match='stop table'):
        detect_locations(cartesian_points.assign(timestamp=[0, 600, 1200]),
                         method='infomap', sigma=2)


def test_infomap_dependency_is_optional(cartesian_points, monkeypatch):
    monkeypatch.setattr(visits, 'infomap', None)
    stops = cartesian_points.assign(timestamp=[0, 600, 1200], duration=5)
    with pytest.raises(ImportError, match=r'nomad\[infomap\]'):
        detect_locations(stops, method='infomap', sigma=2)
    assert detect_locations(cartesian_points, epsilon=2).tolist() == [0, 0, 1]


def test_dbscan_is_unaffected_by_infomap_options(cartesian_points):
    expected = detect_locations(cartesian_points, epsilon=2)
    result = detect_locations(
        cartesian_points, epsilon=2, sigma=2, seed=456, num_trials=1,
    )
    pd.testing.assert_series_equal(result, expected)
