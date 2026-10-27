import numpy as np
import pandas as pd
import pytest

from nomad.stop_detection import sequential_algs as sequential
from nomad.stop_detection import utils


@pytest.fixture(params=['sliding', 'centroid', 'lachesis', 'soc'])
def algorithm(request):
    if request.param in {'sliding', 'centroid'}:
        return sequential.detect_stops_labels, {'method': request.param}
    return getattr(sequential, request.param + '_labels'), (
        {'dt_max': 5} if request.param == 'lachesis' else {}
    )


def test_sequential_geographic_labels_and_summary(algorithm):
    label_function, kwargs = algorithm
    data = pd.DataFrame({
        'timestamp': [0, 60, 120, 180],
        'longitude': [10.0, 10.0002, 10.0004, 10.004],
        'latitude': [60.0] * 4,
    })
    labels = label_function(data, delta_roam=30, dur_min=2, **kwargs)
    assert labels.tolist() == [0, 0, 0, -1]

    stop_function = getattr(sequential, label_function.__name__.removesuffix('_labels'))
    stops = stop_function(data, delta_roam=30, dur_min=2, complete_output=True, **kwargs)
    assert len(stops) == 1
    assert stops.iloc[0]['longitude'] == pytest.approx(10.0002)
    assert stops.iloc[0]['latitude'] == 60
    # The public stop schema rounds diameter to whole meters.
    assert stops.iloc[0]['diameter'] == 22
    summary = utils.summarize_stop(data.iloc[:3], complete_output=True)
    assert summary['diameter'] == pytest.approx(22.2389853, abs=0.001)


@pytest.mark.parametrize('include_user', [False, True])
@pytest.mark.parametrize('datetime_input', [False, True])
def test_sequential_mapped_columns_and_empty_input(algorithm, include_user, datetime_input):
    label_function, kwargs = algorithm
    times = pd.Series([0, 60, 120], dtype='Int64')
    time_key = 'datetime' if datetime_input else 'timestamp'
    if datetime_input:
        times = pd.to_datetime(times, unit='s', utc=True)
    data = pd.DataFrame({'time': times, 'east': [0.0] * 3, 'north': [0.0] * 3})
    traj_cols = {time_key: 'time', 'x': 'east', 'y': 'north'}
    if include_user:
        data['person'] = pd.Series(['a'] * 3, dtype='string')
        traj_cols['user_id'] = 'person'

    labels = label_function(data, delta_roam=1, dur_min=2, traj_cols=traj_cols, **kwargs)
    assert labels.tolist() == [0, 0, 0]
    empty = label_function(data.iloc[:0], delta_roam=1, traj_cols=traj_cols, **kwargs)
    assert empty.empty
    assert empty.name == 'cluster'
    assert empty.dtype == labels.dtype


@pytest.mark.parametrize('label_function,kwargs', [
    (sequential.detect_stops_labels, {}),
    (sequential.lachesis_labels, {'dt_max': 5}),
    (sequential.soc_labels, {}),
    (sequential.grid_based_labels, {}),
])
def test_sequential_rejects_multiple_mapped_users(label_function, kwargs):
    data = pd.DataFrame({
        'timestamp': [0, 60], 'x': [0.0, 0.0], 'y': [0.0, 0.0],
        'person': ['a', 'b'], 'location_id': ['home', 'home'],
    })
    with pytest.raises(ValueError, match='Multi-user'):
        label_function(data, delta_roam=1, traj_cols={'user_id': 'person'}, **kwargs)


def _soc_reference(data, radius, duration):
    """Full scalar neighborhood search, independent of the batched expansion."""
    coords = data[['x', 'y']].to_numpy()
    times = data['timestamp'].to_numpy()
    core = np.zeros(len(data), dtype=bool)
    for anchor in range(len(data)):
        first = last = anchor
        while first > 0 and np.linalg.norm(coords[first - 1] - coords[anchor]) <= radius:
            first -= 1
        while last + 1 < len(data) and np.linalg.norm(coords[last + 1] - coords[anchor]) <= radius:
            last += 1
        core[anchor] = times[last] - times[first] >= duration * 60

    labels = np.full(len(data), -1)
    anchors = np.full(len(data), -1)
    active_core = None
    cluster = -1
    for position in range(len(data)):
        reachable = active_core is not None and np.linalg.norm(
            coords[position] - coords[active_core]
        ) <= radius
        if core[position]:
            if not reachable:
                cluster += 1
            active_core = position
        elif not reachable:
            active_core = None
        if active_core is not None:
            labels[position] = cluster
            anchors[position] = active_core
    for stop in range(cluster + 1):
        last_core = np.flatnonzero(core & (labels == stop))[-1]
        labels[(labels == stop) & (np.arange(len(data)) > last_core)] = -1
    return labels, anchors, core


@pytest.mark.parametrize('duration', [0, 0.5, 2, 100])
def test_soc_matches_full_neighborhood_reference(duration):
    rng = np.random.default_rng(440)
    for size in [1, 2, 5, 20, 50]:
        # Discrete coordinates exercise exact radius boundaries and repeated pings.
        data = pd.DataFrame({
            'timestamp': np.cumsum(rng.integers(0, 120, size)),
            'x': rng.integers(-2, 3, size) / 2,
            'y': rng.integers(-2, 3, size) / 2,
        })
        expected, anchors, core = _soc_reference(data, 1, duration)
        result = sequential.soc_labels(data, delta_roam=1, dur_min=duration, return_anchors=True)
        np.testing.assert_array_equal(result['cluster'], expected)
        retained = expected >= 0
        np.testing.assert_array_equal(
            result.loc[retained, 'anchor_time'].to_numpy(dtype=int),
            data['timestamp'].to_numpy()[anchors[retained]],
        )
        assert result.loc[~retained, 'anchor_time'].isna().all()
        assert (expected[core] >= 0).all()


def test_soc_short_sequence_does_not_include_following_gap():
    data = pd.DataFrame({
        'timestamp': [0, 30, 3600], 'x': [0.0, 0.0, 100.0], 'y': [0.0] * 3,
    })
    assert sequential.soc_labels(data, delta_roam=1, dur_min=1).tolist() == [-1, -1, -1]
