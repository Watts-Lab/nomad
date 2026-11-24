"""Infostop: running-median stops grouped into recurring locations with Infomap."""

import geopandas as gpd
import pandas as pd

import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.stop_detection.sequential_algs import detect_stops_labels, grid_based_labels
from nomad.visit_attribution.visit_attribution import _detect_locations


def infostop_labels(
    data,
    r1,
    r2,
    dur_min=5,
    dt_max=60,
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
    Stationary events come from ``detect_stops_labels(method='median')``.
    Their medians are clustered by ``detect_locations(method='infomap')``,
    and ``grid_based_labels`` splits the located pings into visits at noise,
    a change of location, or a gap longer than ``dt_max``. Distances are in
    projected units, or meters for longitude/latitude.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")

    t_key, _, _, _, _ = utils._fallback_st_cols(data.columns, traj_cols, kwargs)
    cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
    user_col = cols['user_id']
    if user_col in data.columns and data[user_col].nunique(dropna=False) > 1:
        raise ValueError("Multi-user data? Use infostop_labels_per_user instead.")
    location_col = cols['location_cluster']

    events = detect_stops_labels(
        data, delta_roam=r1, dt_max=dt_max, dur_min=dur_min,
        method='median', traj_cols=traj_cols, **kwargs,
    )
    locations = pd.Series(pd.NA, index=data.index, dtype='Int64')
    if (events >= 0).any():
        stops = utils.summarize_stops(data, events, method='median', traj_cols=traj_cols, **kwargs)
        stop_locations = _detect_locations(
            stops, epsilon=r2, method='infomap',
            algorithm_kwargs={'seed': seed, 'num_trials': num_trials},
            traj_cols=traj_cols, **kwargs,
        )
        locations = events.map(stop_locations.set_axis(stops['cluster'])).astype('Int64')

    time_col = cols[t_key]
    visits = grid_based_labels(
        pd.DataFrame({time_col: data[time_col], location_col: locations}),
        time_thresh=dt_max,
        traj_cols={t_key: time_col, 'location_id': location_col},
    )

    if return_location_clusters:
        return pd.DataFrame({'cluster': visits, location_col: locations.fillna(-1).astype(int)}, index=data.index)
    return visits


def infostop(
    data,
    r1,
    r2,
    dur_min=5,
    dt_max=60,
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
    r1, r2, dur_min, dt_max, seed, num_trials
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
