"""Median-based stationary events clustered into recurring locations with Infomap."""

from heapq import heappop, heappush
from math import hypot
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import nomad.io.base as loader
from nomad.constants import EARTH_RADIUS_METERS
from nomad.stop_detection import utils


class _RunningMedian:
    """Insertion-only median with O(log n) updates and O(1) lookup."""

    def __init__(self):
        self.lower = []
        self.upper = []

    def add(self, value):
        heappush(self.lower, -value)
        heappush(self.upper, -heappop(self.lower))
        if len(self.upper) > len(self.lower):
            heappush(self.lower, -heappop(self.upper))

    @property
    def value(self):
        if len(self.lower) == len(self.upper):
            return (-self.lower[0] + self.upper[0]) / 2
        return -self.lower[0]


class _EventMedian:
    """Coordinate medians, choosing the shorter longitude span across +/-pi."""

    def __init__(self, geographic):
        self.first = _RunningMedian()
        self.second = _RunningMedian()
        self.shifted = _RunningMedian() if geographic else None
        self.bounds = [np.inf, -np.inf, np.inf, -np.inf]

    def add(self, point):
        first, second = point
        self.first.add(first)
        self.second.add(second)
        if self.shifted is not None:
            shifted = second % (2 * np.pi)
            self.shifted.add(shifted)
            lo, hi, shifted_lo, shifted_hi = self.bounds
            self.bounds = [min(lo, second), max(hi, second),
                           min(shifted_lo, shifted), max(shifted_hi, shifted)]

    @property
    def center(self):
        second = self.second.value
        if self.shifted is not None:
            lo, hi, shifted_lo, shifted_hi = self.bounds
            if hi - lo > np.pi and shifted_hi - shifted_lo < hi - lo:
                second = self.shifted.value
            second = (second + np.pi) % (2 * np.pi) - np.pi
        return np.array([self.first.value, second])


def _stationary_events(coords, times, r1, dur_min, dt_max, min_pts, geographic):
    """Return event centers and positional ping-to-event IDs; geo inputs are radians."""
    events = np.full(len(coords), -1, dtype=np.int64)
    centers = []
    if not len(coords):
        return np.empty((0, 2)), events
    start = 0
    median = _EventMedian(geographic)
    median.add(coords[0])
    center = median.center
    for end in range(1, len(coords) + 1):
        if end < len(coords) and times[end] - times[end - 1] <= dt_max * 60:
            distance = (utils._haversine_distance(coords[end], center)
                        if geographic else hypot(*(coords[end] - center)))
            if distance <= r1:
                median.add(coords[end])
                center = median.center
                continue
        if end - start >= min_pts and times[end - 1] - times[start] >= dur_min * 60:
            events[start:end] = len(centers)
            centers.append(center)
        if end < len(coords):
            start = end
            median = _EventMedian(geographic)
            median.add(coords[end])
            center = median.center
    return np.asarray(centers).reshape(-1, 2), events


def _cluster_events(centers, r2, geographic, seed, num_trials):
    """Cluster unique event coordinates, weighting edges by endpoint event counts."""
    try:
        import infomap
    except ModuleNotFoundError as error:
        if error.name != 'infomap':
            raise
        raise ImportError('Infostop requires: pip install "nomad[infomap]"') from error

    points, inverse, counts = np.unique(
        centers, axis=0, return_inverse=True, return_counts=True,
    )
    tree = BallTree(points, metric='haversine' if geographic else 'euclidean')
    radius = r2 / EARTH_RADIUS_METERS if geographic else r2
    network = infomap.Network()
    connected = np.zeros(len(points), dtype=bool)
    # Limit temporary neighbor storage; the graph itself still requires O(edges).
    for start in range(0, len(points), 4096):
        neighbors = tree.query_radius(points[start:start + 4096], r=radius)
        for source, targets in enumerate(neighbors, start):
            for target in targets[targets > source]:
                network.add_link(source, int(target), float(max(counts[source], counts[target])))
                connected[source] = connected[target] = True
    labels = np.arange(len(points), dtype=np.int64)
    if connected.any():
        result = infomap.run(
            network, directed=False, two_level=True,
            options=infomap.Options(seed=int(seed), num_trials=int(num_trials)),
        )
        for node, module in result.modules().items():
            labels[node] = module + len(points)
    return pd.factorize(labels[inverse], sort=False)[0]


def infostop_labels(
    data, r1, r2, dur_min=5, dt_max=60, min_pts=2, traj_cols=None, *,
    seed=123, num_trials=50, return_location_clusters=False, **kwargs,
):
    """Detect stationary events, cluster their medians, and label chronological visits.

    Parameters
    ----------
    data : pandas.DataFrame
        A single user's trajectory in chronological order; equal timestamps are allowed.
    r1 : float
        Maximum distance from a new ping to the median of the current event's preceding pings.
    r2 : float
        Maximum distance between event medians for a spatial graph edge.
        Both radii use projected coordinate units, or meters for longitude/latitude.
    dur_min : float, default 5
        Minimum stationary-event duration in minutes.
    dt_max : float, default 60
        Maximum consecutive-ping gap in minutes, including within a visit.
    min_pts : int, default 2
        Minimum number of pings in a stationary event.
    traj_cols : dict, optional
        Nomad column mappings, including location_cluster for recurring locations.
    seed : int, default 123
        Infomap random seed.
    num_trials : int, default 50
        Number of two-level Infomap searches; retain the shortest description.
    return_location_clusters : bool, default False
        Also return recurring location membership as a location_cluster column.
    **kwargs
        Additional trajectory column mappings.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Aligned chronological visit labels named cluster, or a table with cluster
        and location_cluster. Noise is -1 in both columns. Separate visits can
        share a location_cluster. Isolated stationary locations are retained.

    Notes
    -----
    Each unique stationary-event median is a node. Edges are undirected, have no
    self-links, and have weight max(event_count_at_source, event_count_at_target).
    There is no Gaussian distance weighting. Locations and visits are numbered
    by first occurrence. The optional nomad[infomap] extra supplies Infomap.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    for name, value in [('r1', r1), ('r2', r2), ('dur_min', dur_min), ('dt_max', dt_max)]:
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive number')
    for name, value, minimum in [('min_pts', min_pts, 2), ('num_trials', num_trials, 1), ('seed', seed, 1)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    t_key, key1, key2, use_datetime, geographic = utils._fallback_st_cols(data.columns, traj_cols, kwargs)
    cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
    user_col = cols['user_id']
    if user_col in data and data[user_col].nunique(dropna=False) > 1:
        raise ValueError('Infostop expects one user; group input by user_id before calling')
    if cols['location_cluster'] == 'cluster':
        raise ValueError('location_cluster must differ from the visit label column cluster')
    coords = data[[cols[key1], cols[key2]]].to_numpy(dtype=float)
    time_col = data[cols[t_key]]
    if time_col.isna().any():
        raise ValueError('timestamps must be finite and chronologically ordered')
    times = ((time_col - time_col.iloc[0]).dt.total_seconds().to_numpy()
             if use_datetime and len(data) else time_col.to_numpy(dtype=float))
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('timestamps must be finite and chronologically ordered')
    if not np.isfinite(coords).all():
        raise ValueError('coordinates must be finite')
    if geographic:
        # Public column parsing returns longitude, latitude; BallTree expects lat, lon.
        coords = coords[:, [1, 0]]
        if (np.abs(coords) > [90, 180]).any():
            raise ValueError('latitude and longitude must be within geographic bounds')
        coords[:, 1] = (coords[:, 1] + 180) % 360 - 180
        coords[np.abs(coords[:, 0]) == 90, 1] = 0
        coords = np.radians(coords)
    centers, events = _stationary_events(coords, times, r1, dur_min, dt_max, min_pts, geographic)
    locations = np.full(len(data), -1, dtype=np.int64)
    valid = events >= 0
    if len(centers):
        locations[valid] = _cluster_events(centers, r2, geographic, seed, num_trials)[events[valid]]
    visits = np.full(len(data), -1, dtype=np.int64)
    if valid.any():
        starts = np.r_[True, (locations[1:] != locations[:-1]) | (np.diff(times) > dt_max * 60)]
        visits[valid] = np.cumsum(starts & valid)[valid] - 1
    labels = pd.Series(visits, index=data.index, name='cluster')
    if return_location_clusters:
        return pd.DataFrame({'cluster': visits, cols['location_cluster']: locations}, index=data.index)
    return labels


def infostop(
    data, r1, r2, dur_min=5, dt_max=60, min_pts=2, complete_output=False,
    passthrough_cols=None, keep_col_names=True, traj_cols=None, *,
    seed=123, num_trials=50, **kwargs,
):
    """Return one standard Nomad stop row per Infostop visit.

    Detection parameters and units match :func:`infostop_labels`. Standard
    summary options are passed to ``utils.summarize_stops``. Recurring location
    membership is retained in ``location_cluster`` (customizable in ``traj_cols``);
    polygon ``location_id`` is only retained when requested in ``passthrough_cols``.
    """
    labels = infostop_labels(
        data, r1, r2, dur_min, dt_max, min_pts, traj_cols,
        seed=seed, num_trials=num_trials, return_location_clusters=True, **kwargs,
    ).reset_index(drop=True)
    cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
    location_col = cols['location_cluster']
    # Summarization joins on index; normalize it to avoid duplicate-index expansion.
    pings = data.reset_index(drop=True).copy()
    pings[location_col] = pd.array(labels[location_col], dtype='Int64')
    passthrough = list(passthrough_cols or [])
    for column in (location_col, cols['user_id']):
        if column in pings and column not in passthrough:
            passthrough.append(column)
    return utils.summarize_stops(
        pings, labels['cluster'], complete_output=complete_output,
        passthrough_cols=passthrough, keep_col_names=keep_col_names,
        traj_cols=traj_cols, **kwargs,
    )


__all__ = ['infostop_labels', 'infostop']
