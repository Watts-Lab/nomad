import builtins
import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nomad.stop_detection.infostop import (
    _cluster_events, _stationary_events, infostop, infostop_labels,
)

needs_infomap = pytest.mark.skipif(
    importlib.util.find_spec("infomap") is None, reason="requires nomad[infomap]"
)


@pytest.fixture
def home_work_home():
    """Two pings each at home, work, then home again, five minutes apart."""
    return pd.DataFrame({
        "x": [0., 0., 100., 100., 0., 0.],
        "y": 0.,
        "timestamp": np.arange(6) * 300,
    })


@needs_infomap
def test_infostop_labels_separate_visits_to_recurring_location(home_work_home):
    labels = infostop_labels(home_work_home, r1=5, r2=10, return_location_clusters=True)

    assert labels["cluster"].tolist() == [0, 0, 1, 1, 2, 2]
    assert labels["location_cluster"].tolist() == [0, 0, 1, 1, 0, 0]


@needs_infomap
def test_infostop_stop_table_keeps_location_cluster(home_work_home):
    stops = infostop(home_work_home, r1=5, r2=10, complete_output=True)

    assert stops["cluster"].tolist() == [0, 1, 2]
    assert stops["location_cluster"].tolist() == [0, 1, 0]
    assert stops["location_cluster"].dtype == "Int64"


@needs_infomap
def test_infostop_supports_custom_location_cluster_column(home_work_home):
    stops = infostop(home_work_home, r1=5, r2=10, traj_cols={"location_cluster": "place"})

    assert stops["place"].tolist() == [0, 1, 0]


@needs_infomap
def test_radius_graph_joins_adjacent_events(home_work_home):
    home_work_home["x"] = [0., 0., 4., 4., 0., 0.]
    labels = infostop_labels(home_work_home, r1=1, r2=5, return_location_clusters=True)

    # Three stationary events at one location form one uninterrupted visit.
    assert labels["cluster"].tolist() == [0] * 6
    assert labels["location_cluster"].tolist() == [0] * 6


@needs_infomap
def test_noise_splits_visits_to_same_location(home_work_home):
    home_work_home["x"] = [0., 0., 100., 0., 0., 0.]
    labels = infostop_labels(home_work_home, r1=1, r2=1, return_location_clusters=True)

    assert labels["cluster"].tolist() == [0, 0, -1, 1, 1, 1]
    assert labels["location_cluster"].tolist() == [0, 0, -1, 0, 0, 0]


@needs_infomap
def test_gap_longer_than_dt_max_splits_visits(home_work_home):
    home_work_home["x"] = 0.
    home_work_home["timestamp"] = [0, 300, 3600, 3900, 7200, 7500]
    labels = infostop_labels(home_work_home, r1=1, r2=1, dt_max=10, return_location_clusters=True)

    assert labels["cluster"].tolist() == [0, 0, 1, 1, 2, 2]
    assert labels["location_cluster"].tolist() == [0] * 6


@needs_infomap
def test_geographic_radii_are_meters():
    # 0.001 degrees of longitude at 60N is about 56 m.
    data = pd.DataFrame({
        "longitude": [0., 0.0001, 0.001, 0.0011],
        "latitude": 60.,
        "timestamp": np.arange(4) * 300,
    })
    labels = infostop_labels(data, r1=6, r2=75, return_location_clusters=True)

    assert labels["location_cluster"].tolist() == [0, 0, 0, 0]


@needs_infomap
def test_graph_deduplicates_medians_and_weights_by_event_count(monkeypatch):
    import infomap

    edges = []
    monkeypatch.setattr(infomap, "Network", lambda: SimpleNamespace(add_link=lambda *edge: edges.append(edge)))
    monkeypatch.setattr(infomap, "run", lambda network, **kwargs: SimpleNamespace(modules=lambda: {0: 1, 1: 1}))
    centers = np.array([[0., 0.], [0., 0.], [1., 0.], [20., 0.]])

    labels = _cluster_events(centers, 2, False, seed=123, num_trials=50)

    assert edges == [(0, 1, 2.)]
    assert labels.tolist() == [0, 0, 0, 1]


def test_stationary_events_match_naive_median_scan():
    rng = np.random.default_rng(8)
    coords = rng.normal(size=(500, 2)) + np.repeat(np.arange(10) * 10, 50)[:, None]
    times = np.arange(500) * 60.

    centers, events = _stationary_events(coords, times, 2, 5, 10, 3, False)

    expected_events = np.full(len(coords), -1)
    expected_centers = []
    start = 0
    for end in range(1, len(coords) + 1):
        if end < len(coords) and np.linalg.norm(coords[end] - np.median(coords[start:end], axis=0)) <= 2:
            continue
        if end - start >= 3 and times[end - 1] - times[start] >= 300:
            expected_events[start:end] = len(expected_centers)
            expected_centers.append(np.median(coords[start:end], axis=0))
        start = end
    np.testing.assert_array_equal(events, expected_events)
    np.testing.assert_allclose(centers, expected_centers)


def test_stationary_event_thresholds_are_inclusive():
    coords = np.array([[0., 0.], [2., 0.], [20., 0.], [20., 0.]])

    centers, events = _stationary_events(coords, np.array([0, 300, 600, 900]), 2, 5, 5, 2, False)

    assert centers.tolist() == [[1., 0.], [20., 0.]]
    assert events.tolist() == [0, 0, 1, 1]


def test_stationary_events_shorter_than_dur_min_are_noise():
    coords = np.array([[0., 0.], [2., 0.], [20., 0.], [20., 0.]])

    _, events = _stationary_events(coords, np.array([0, 299, 600, 899]), 2, 5, 5, 2, False)

    assert events.tolist() == [-1] * 4


def test_stationary_event_median_crosses_antimeridian():
    # Pings 44 m apart on either side of 180 degrees longitude form one event.
    coords = np.radians([[0., 179.9998], [0., -179.9998]] * 2)

    centers, events = _stationary_events(coords, np.arange(4) * 300., 50, 5, 60, 2, True)

    assert events.tolist() == [0] * 4
    assert abs(np.degrees(centers[0, 1])) == pytest.approx(180)


def test_infomap_dependency_is_optional(home_work_home, monkeypatch):
    original_import = builtins.__import__

    def missing_infomap(name, *args, **kwargs):
        if name == "infomap":
            raise ModuleNotFoundError("No module named 'infomap'", name="infomap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_infomap)
    with pytest.raises(ImportError, match=r"nomad\[infomap\]"):
        infostop(home_work_home, r1=5, r2=10)
