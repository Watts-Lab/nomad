"""Infostop: stationary events clustered into recurring locations with Infomap."""

from heapq import heappop, heappush

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import nomad.io.base as loader
from nomad.constants import EARTH_RADIUS_METERS
from nomad.filters import to_timestamp
from nomad.stop_detection import utils


class _RunningMedian:
    """Coordinate-wise median of a growing point set with O(log n) insertion."""

    def __init__(self, point):
        self.heaps = [([], []) for _ in point]
        self.add(point)

    def add(self, point):
        for (lower, upper), value in zip(self.heaps, point):
            heappush(lower, -value)
            heappush(upper, -heappop(lower))
            if len(upper) > len(lower):
                heappush(lower, -heappop(upper))

    @property
    def value(self):
        return np.array([
            -lower[0] if len(lower) > len(upper) else (upper[0] - lower[0]) / 2
            for lower, upper in self.heaps
        ])


def _wrap_longitude(lon):
    """Wrap a longitude in radians into [-pi, pi)."""
    return (lon + np.pi) % (2 * np.pi) - np.pi


def _stationary_events(coords, times, r1, dur_min, dt_max, min_pts, use_lon_lat):
    """Return event medians and per-ping event ids (-1 outside events).

    Geographic coordinates must be [lat, lon] in radians.
    """
    events = np.full(len(coords), -1, dtype=int)
    centers = []
    start = 0
    for end in range(len(coords) + 1):
        # The current event, started at `start`, grows while pings stay near its median.
        if 0 < end < len(coords) and times[end] - times[end - 1] <= dt_max * 60:
            point = coords[end]
            if use_lon_lat:
                # Unwrap longitude next to the event's first ping, so medians of
                # events crossing the antimeridian stay circular as in Infostop.
                point = np.array([point[0], coords[start, 1] + _wrap_longitude(point[1] - coords[start, 1])])
                dist = utils._haversine_distance(point, median.value)
            else:
                dist = np.linalg.norm(point - median.value)
            if dist <= r1:
                median.add(point)
                continue
        if end > 0 and end - start >= min_pts and times[end - 1] - times[start] >= dur_min * 60:
            events[start:end] = len(centers)
            center = median.value
            if use_lon_lat:
                center[1] = _wrap_longitude(center[1])
            centers.append(center)
        if end < len(coords):
            start, median = end, _RunningMedian(coords[end])
    return np.array(centers).reshape(-1, 2), events


def _cluster_events(centers, r2, use_lon_lat, seed, num_trials):
    """Cluster unique event medians with Infomap; edges join medians within r2."""
    try:
        import infomap
    except ImportError as error:
        raise ImportError('Infostop requires: pip install "nomad[infomap]"') from error

    points, inverse, counts = np.unique(centers, axis=0, return_inverse=True, return_counts=True)
    tree = BallTree(points, metric='haversine' if use_lon_lat else 'euclidean')
    radius = r2 / EARTH_RADIUS_METERS if use_lon_lat else r2

    # Isolated medians keep their node id; linked ones take their module id,
    # offset past the node ids so the two cannot collide.
    labels = np.arange(len(points))
    network = infomap.Network()
    has_links = False
    for source, targets in enumerate(tree.query_radius(points, r=radius)):
        for target in targets[targets > source]:
            network.add_link(source, int(target), float(max(counts[source], counts[target])))
            has_links = True
    if has_links:
        result = infomap.run(
            network, directed=False, two_level=True,
            options=infomap.Options(seed=seed, num_trials=num_trials),
        )
        for node, module in result.modules().items():
            labels[node] = len(points) + module
    return pd.factorize(labels[inverse])[0]


def infostop_labels(
    data,
    r1,
    r2,
    dur_min=5,
    dt_max=60,
    min_pts=2,
    traj_cols=None,
    return_location_clusters=False,
    seed=123,
    num_trials=50,
    **kwargs
):
    """
    Scan a trajectory into stationary events, cluster them into recurring
    locations with Infomap, and label each ping with a visit index or -1 for noise.

    Parameters
    ----------
    data : pd.DataFrame or GeoDataFrame
        Input trajectory with spatial and temporal columns.
    r1 : float
        Maximum distance from a ping to the running median of its event.
    r2 : float
        Maximum distance between event medians for an Infomap graph edge.
    dur_min : int, default 5
        Minimum duration in minutes for a stationary event.
    dt_max : int, default 60
        Maximum allowed gap in minutes between consecutive pings in a visit.
    min_pts : int, default 2
        Minimum number of pings in a stationary event.
    traj_cols : dict, optional
        Mapping for 'x', 'y', 'longitude', 'latitude', 'timestamp', 'datetime',
        or 'location_cluster'.
    return_location_clusters : bool, default False
        Also return each ping's recurring location as ``location_cluster``.
    seed : int, default 123
        Infomap random seed.
    num_trials : int, default 50
        Number of Infomap trials; the best partition is kept.
    **kwargs
        Passed along to the column-detection helper.

    Returns
    -------
    pd.Series or pd.DataFrame
        Visit labels, or labels with aligned ``location_cluster``. Separate
        visits can share a location cluster.

    Notes
    -----
    Distances are in projected units, or meters for longitude/latitude. Each
    unique event median is a graph node; undirected edges are weighted by the
    larger endpoint event count. A visit ends at noise, a change of location,
    or a gap longer than ``dt_max``. Requires ``pip install "nomad[infomap]"``.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
    user_col = traj_cols['user_id']
    if user_col in data.columns and data[user_col].nunique(dropna=False) > 1:
        raise ValueError("Multi-user data? Use infostop_labels_per_user instead.")

    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    coords = data[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy(dtype='float64')
    if use_lon_lat:
        coords = np.radians(coords[:, ::-1])  # haversine helpers expect [lat, lon]
    time_col = data[traj_cols[t_key]]
    times = (to_timestamp(time_col) if use_datetime else time_col).to_numpy(dtype='float64')

    centers, events = _stationary_events(coords, times, r1, dur_min, dt_max, min_pts, use_lon_lat)
    in_event = events >= 0
    locations = np.full(len(data), -1)
    if len(centers):
        locations[in_event] = _cluster_events(centers, r2, use_lon_lat, seed, num_trials)[events[in_event]]

    new_visit = np.r_[True, (locations[1:] != locations[:-1]) | (np.diff(times) > dt_max * 60)]
    visits = np.where(in_event, np.cumsum(new_visit & in_event) - 1, -1)

    result = pd.Series(visits, index=data.index, name='cluster')
    if return_location_clusters:
        return pd.DataFrame({'cluster': result, traj_cols['location_cluster']: locations}, index=data.index)
    return result


def infostop(
    data,
    r1,
    r2,
    dur_min=5,
    dt_max=60,
    min_pts=2,
    complete_output=False,
    passthrough_cols=None,
    keep_col_names=True,
    traj_cols=None,
    seed=123,
    num_trials=50,
    **kwargs
):
    """
    Infostop stop detection with summarization.

    Parameters
    ----------
    data : pd.DataFrame or GeoDataFrame
        Input trajectory with spatial and temporal columns.
    r1, r2, dur_min, dt_max, min_pts, seed, num_trials
        See :func:`infostop_labels`.
    complete_output : bool, default False
        If True, include additional summary statistics in output.
    passthrough_cols : list, optional
        Columns to retain (and summarize/propagate) per stop.
    traj_cols : dict, optional
        Mapping for column names.
    **kwargs
        Passed along to the column-detection helper.

    Returns
    -------
    pd.DataFrame
        One row per visit, including its ``location_cluster``.

    Raises
    ------
    ValueError if multiple users found; use infostop_per_user instead.
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
    location_col = traj_cols_temp['location_cluster']
    labels = infostop_labels(
        data=data,
        r1=r1,
        r2=r2,
        dur_min=dur_min,
        dt_max=dt_max,
        min_pts=min_pts,
        return_location_clusters=True,
        seed=seed,
        num_trials=num_trials,
        traj_cols=traj_cols,
        **kwargs
    )
    data = data.assign(**{location_col: labels[location_col].astype('Int64')})
    for col in (location_col, traj_cols_temp['user_id']):
        if col in data.columns and col not in passthrough_cols:
            passthrough_cols = passthrough_cols + [col]
    return utils.summarize_stops(
        data,
        labels['cluster'],
        complete_output=complete_output,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
        **kwargs,
    )


def infostop_per_user(
    data,
    r1,
    r2,
    dur_min=5,
    dt_max=60,
    min_pts=2,
    complete_output=False,
    passthrough_cols=None,
    keep_col_names=True,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run infostop on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("infostop_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        infostop,
        {
            "r1": r1,
            "r2": r2,
            "dur_min": dur_min,
            "dt_max": dt_max,
            "min_pts": min_pts,
            "complete_output": complete_output,
            "passthrough_cols": pt_cols,
            "keep_col_names": keep_col_names,
            "traj_cols": traj_cols,
            **kwargs,
        },
        reset_index=True,
        n_jobs=n_jobs,
        print_progress=print_progress
    )
    return pd.concat(results, ignore_index=True)


def infostop_labels_per_user(
    data,
    r1,
    r2,
    dur_min=5,
    dt_max=60,
    min_pts=2,
    traj_cols=None,
    return_location_clusters=False,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run infostop_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("infostop_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        infostop_labels,
        {
            "r1": r1,
            "r2": r2,
            "dur_min": dur_min,
            "dt_max": dt_max,
            "min_pts": min_pts,
            "return_location_clusters": return_location_clusters,
            "traj_cols": traj_cols,
            **kwargs,
        },
        n_jobs=n_jobs,
        print_progress=print_progress,
    )
    return pd.concat(results).reindex(data.index)


__all__ = [
    "infostop_labels",
    "infostop",
    "infostop_per_user",
    "infostop_labels_per_user",
]
