"""Infostop event boundaries, spatial graph, and Nomad output contracts."""

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nomad.stop_detection.infostop import (
    _cluster_events, _stationary_events, infostop, infostop_labels,
)


@pytest.fixture
def trajectory():
    return pd.DataFrame({
        'x': [0., 0., 100., 100., 0., 0.], 'y': 0.,
        'timestamp': np.arange(6) * 300,
        'location_id': ['home', 'home', 'work', 'work', 'home', 'home'],
    }, index=[5, 5, 9, 9, 2, 2])


def test_returns_separate_visits_and_recurring_locations_without_mutation(trajectory):
    pytest.importorskip('infomap')
    original = trajectory.copy(deep=True)
    labels = infostop_labels(trajectory, 5, 10, return_location_clusters=True)
    assert labels.cluster.tolist() == [0, 0, 1, 1, 2, 2]
    assert labels.location_cluster.tolist() == [0, 0, 1, 1, 0, 0]
    assert labels.index.equals(trajectory.index)
    pd.testing.assert_series_equal(labels.cluster, infostop_labels(trajectory, 5, 10))
    stops = infostop(trajectory, 5, 10, complete_output=True, passthrough_cols=['location_id'])
    assert stops.cluster.tolist() == [0, 1, 2]
    assert stops.location_cluster.tolist() == [0, 1, 0]
    assert stops.location_id.tolist() == ['home', 'work', 'home']
    assert stops.duration.tolist() == [5, 5, 5]
    assert stops.n_pings.tolist() == [2, 2, 2]
    assert str(stops.location_cluster.dtype) == 'Int64'
    pd.testing.assert_frame_equal(trajectory, original)


def test_radius_graph_joins_distinct_events(trajectory):
    pytest.importorskip('infomap')
    trajectory['x'] = [0., 0., 4., 4., 0., 0.]
    labels = infostop_labels(trajectory, 1, 5, return_location_clusters=True)
    # Three stationary events become one contiguous visit to the same location.
    assert labels.cluster.tolist() == [0] * 6
    assert labels.location_cluster.tolist() == [0] * 6


def test_gap_and_noise_split_visits(trajectory):
    pytest.importorskip('infomap')
    trajectory['x'] = [0., 0., 100., 0., 0., 0.]
    trajectory['timestamp'] = [0, 300, 600, 900, 1200, 1500]
    labels = infostop_labels(trajectory, 1, 1, return_location_clusters=True)
    assert labels.cluster.tolist() == [0, 0, -1, 1, 1, 1]
    assert labels.location_cluster.tolist() == [0, 0, -1, 0, 0, 0]
    trajectory['x'] = 0.
    trajectory['timestamp'] = [0, 300, 3600, 3900, 7200, 7500]
    labels = infostop_labels(trajectory, 1, 1, dt_max=10, return_location_clusters=True)
    assert labels.cluster.tolist() == [0, 0, 1, 1, 2, 2]
    assert labels.location_cluster.tolist() == [0] * 6


def test_graph_deduplicates_events_and_uses_event_counts(monkeypatch):
    infomap = pytest.importorskip('infomap')
    edges = []
    monkeypatch.setattr(infomap, 'Network', lambda: SimpleNamespace(add_link=lambda *edge: edges.append(edge)))
    monkeypatch.setattr(infomap, 'run', lambda network, **kwargs: SimpleNamespace(modules=lambda: {0: 1, 1: 1}))
    labels = _cluster_events(np.array([[0., 0.], [0., 0.], [1., 0.], [20., 0.]]), 2, False, 123, 50)
    assert edges == [(0, 1, 2.)]
    assert labels.tolist() == [0, 0, 0, 1]


def test_running_medians_and_boundaries_match_simple_reference():
    rng = np.random.default_rng(8)
    coords = rng.normal(size=(500, 2)) + np.repeat(np.arange(10) * 10, 50)[:, None]
    times = np.arange(500) * 60.
    centers, events = _stationary_events(coords, times, 2, 5, 10, 3, False)
    reference = np.full(len(coords), -1)
    expected_centers = []
    start = 0
    for end in range(1, len(coords) + 1):
        if end < len(coords) and np.linalg.norm(coords[end] - np.median(coords[start:end], axis=0)) <= 2:
            continue
        if end - start >= 3 and times[end - 1] - times[start] >= 300:
            reference[start:end] = len(expected_centers)
            expected_centers.append(np.median(coords[start:end], axis=0))
        start = end
    np.testing.assert_array_equal(events, reference)
    np.testing.assert_allclose(centers, expected_centers)


def test_inclusive_thresholds_and_final_event():
    coords = np.array([[0., 0.], [2., 0.], [20., 0.], [20., 0.]])
    centers, events = _stationary_events(coords, np.array([0, 300, 600, 900]), 2, 5, 5, 2, False)
    assert centers.tolist() == [[1., 0.], [20., 0.]]
    assert events.tolist() == [0, 0, 1, 1]
    _, rejected = _stationary_events(coords, np.array([0, 299, 600, 899]), 2, 5, 5, 2, False)
    assert rejected.tolist() == [-1] * 4


@pytest.mark.parametrize('size', [0, 1, 6])
def test_empty_and_all_moving_return_standard_schema(trajectory, size):
    data = trajectory.iloc[:size].copy()
    data['x'] = np.arange(size) * 100.
    labels = infostop_labels(data, 1, 1)
    assert labels.tolist() == [-1] * size
    stops = infostop(data, 1, 1, complete_output=True)
    assert stops.empty
    assert stops.columns.tolist() == ['cluster', 'x', 'y', 'timestamp', 'diameter',
                                      'n_pings', 'end_timestamp', 'duration', 'max_gap',
                                      'location_cluster']
    assert str(stops.location_cluster.dtype) == 'Int64'


def test_custom_columns_and_datetime_units(trajectory):
    pytest.importorskip('infomap')
    trajectory['recorded'] = pd.to_datetime(trajectory.pop('timestamp'), unit='s', utc=True)
    data = trajectory.rename(columns={'x': 'east', 'y': 'north'})
    mapping = {'x': 'east', 'y': 'north', 'datetime': 'recorded', 'location_cluster': 'place'}
    stops = infostop(data, 5, 10, traj_cols=mapping, keep_col_names=False, complete_output=True)
    assert stops.place.tolist() == [0, 1, 0]
    assert stops.duration.tolist() == [5, 5, 5]
    assert stops.x.tolist() == [0., 100., 0.]
    assert stops.start_datetime.dtype == data.recorded.dtype


@pytest.mark.parametrize('longitudes,latitude,r1,r2,locations', [
    ([0., 0.0001, 0.001, 0.0011], 60., 6, 75, [0, 0, 0, 0]),
    ([179.9998, -179.9998, 179.9998, -179.9998], 0., 50, 50, [0, 0, 0, 0]),
    ([179.9998, -179.9998, 179.99, 179.99], 0., 50, 50, [0, 0, 1, 1]),
])
def test_geographic_distances_and_antimeridian(longitudes, latitude, r1, r2, locations):
    pytest.importorskip('infomap')
    data = pd.DataFrame({'longitude': longitudes, 'latitude': latitude, 'timestamp': np.arange(4) * 300})
    result = infostop_labels(data, r1, r2, return_location_clusters=True)
    assert result.location_cluster.tolist() == locations


def test_geographic_summary_uses_latitude_for_distance():
    pytest.importorskip('infomap')
    data = pd.DataFrame({'longitude': [0., 0.001], 'latitude': 60., 'timestamp': [0, 300]})
    stops = infostop(data, 60, 60, complete_output=True)
    assert stops.diameter.tolist() == [56]
    assert stops.latitude.tolist() == [60.]
    assert stops.longitude.iloc[0] in data.longitude.to_numpy()


@pytest.mark.parametrize('option,value', [
    ('r1', 0), ('r2', np.nan), ('dur_min', True), ('dt_max', np.inf),
    ('min_pts', 1), ('min_pts', 2.5), ('num_trials', 0), ('seed', False),
])
def test_invalid_parameters(trajectory, option, value):
    params = dict(r1=1, r2=1)
    params[option] = value
    with pytest.raises(ValueError, match=option):
        infostop_labels(trajectory, **params)


@pytest.mark.parametrize('column,values,message', [
    ('timestamp', [0, 300, 200, 900, 1200, 1500], 'chronologically'),
    ('timestamp', [0, 300, np.nan, 900, 1200, 1500], 'finite'),
    ('x', [0, 0, np.inf, 100, 0, 0], 'finite'),
    ('user_id', ['a', 'a', 'b', 'b', 'a', 'a'], 'one user'),
])
def test_invalid_trajectory(trajectory, column, values, message):
    trajectory[column] = values
    with pytest.raises(ValueError, match=message):
        infostop_labels(trajectory, 1, 1)


def test_dependency_is_optional(trajectory, monkeypatch):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == 'infomap':
            raise ModuleNotFoundError("No module named 'infomap'", name='infomap')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(ImportError, match=r'nomad\[infomap\]'):
        infostop(trajectory, 1, 1)
    from nomad.stop_detection.sequential_algs import lachesis
    assert not lachesis(trajectory, delta_roam=1).empty
