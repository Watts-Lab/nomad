import numpy as np
import pandas as pd

import nomad.io.base as loader


def fill_timestamp_gaps(first_time, last_time, stop_table):
    """Add unassigned intervals between timestamp-based stops."""
    if stop_table.empty:
        return stop_table.copy()

    stops = stop_table.sort_values('start_timestamp').reset_index(drop=True)
    starts = stops['start_timestamp'].to_numpy()
    ends = starts + stops['duration'].to_numpy() * 60
    gap_starts = np.concatenate(([first_time], ends))
    gap_ends = np.concatenate((starts, [last_time]))
    has_gap = gap_starts < gap_ends

    gaps = pd.DataFrame(
        {
            'start_timestamp': gap_starts[has_gap],
            'duration': (gap_ends[has_gap] - gap_starts[has_gap]) // 60,
            'building_id': 'None',
        }
    ).reindex(columns=stops.columns)

    return (
        pd.concat([stops, gaps], ignore_index=True)
        .sort_values('start_timestamp')
        .reset_index(drop=True)
    )


def _merge_stops_from_agg(
    stops,
    traj_cols,
    t_key,
    use_datetime,
    time_thresh=60,
    return_relabeling=False,
):
    """Merge a chronological single-user stop table or return its relabeling."""
    start_col = traj_cols[t_key]
    end_key = 'end_datetime' if use_datetime else 'end_timestamp'
    end_col = traj_cols[end_key]
    end_col_present = end_col in stops.columns
    duration_col_present = traj_cols['duration'] in stops.columns
    stop_rows = stops.reset_index(drop=True)

    if not end_col_present:
        if use_datetime:
            stop_rows[end_col] = stop_rows[start_col] + pd.to_timedelta(
                stop_rows[traj_cols['duration']], unit='min'
            )
        else:
            stop_rows[end_col] = (
                stop_rows[start_col] + stop_rows[traj_cols['duration']] * 60
            )

    gap_threshold = (
        pd.to_timedelta(time_thresh, unit='min')
        if use_datetime else time_thresh * 60
    )
    location_col = traj_cols['location_id']
    location_codes = pd.Series(pd.factorize(stop_rows[location_col], sort=False)[0])
    same_location = location_codes.eq(location_codes.shift()) & location_codes.ne(-1)
    sequence_id = (~same_location).cumsum()
    # A running maximum handles intervals contained inside a longer stop.
    previous_end = stop_rows.groupby(sequence_id, sort=False)[end_col].cummax().shift()
    new_visit = (
        ~same_location
        | stop_rows[start_col].sub(previous_end).gt(gap_threshold)
    ).fillna(True)
    visit_id = new_visit.cumsum()
    if return_relabeling:
        relabeling = pd.Series(
            visit_id.to_numpy() - 1,
            index=stop_rows['cluster'],
            name='cluster',
        )
        relabeling.loc[-1] = -1
        return relabeling

    user_col = traj_cols['user_id'] if traj_cols['user_id'] in stop_rows.columns else None
    aggregations = {
        start_col: 'first',
        end_col: 'max',
        location_col: 'first',
    }
    if user_col is not None:
        aggregations[user_col] = 'first'

    merged = stop_rows.groupby(visit_id, sort=False).agg(aggregations)
    if duration_col_present:
        duration = merged[end_col] - merged[start_col]
        if use_datetime:
            duration = duration.dt.total_seconds()
        merged[traj_cols['duration']] = (duration / 60).astype(int)

    merged.index = stops.index[new_visit.to_numpy()]
    merged.index.name = stops.index.name
    if not end_col_present:
        merged = merged.drop(columns=end_col)
    merged.insert(0, 'cluster', np.arange(len(merged), dtype='int64'))
    return merged


def merge_stops(
    stops,
    time_thresh=60,
    location_col=None,
    method='grid_based',
    algorithm=None,
    algorithm_kwargs=None,
    traj_cols=None,
    **kwargs
):
    """
    Merge consecutive stops at the same location.

    Parameters
    ----------
    stops : pd.DataFrame
        Ping or stop table containing time and location columns. Stop tables
        additionally contain an end time or duration. Rows must be ordered by
        their temporal column.
    time_thresh : int, default 60
        Largest gap in minutes between one stop's end and the next stop's start
        that can belong to the same visit. For custom ping-table methods, a
        value in ``algorithm_kwargs`` takes precedence.
    location_col : str, optional
        Location identifier column. Defaults to the ``location_id`` mapping in
        ``traj_cols``.
    method : {'grid_based', 'custom'}, default 'grid_based'
        Stop-detection method used for a location-labeled ping table. This is
        ignored when ``stops`` is already a stop table.
    algorithm : callable, optional
        Stop-detection callable required when ``method='custom'``.
    algorithm_kwargs : dict, optional
        Arguments for a custom stop-detection callable.
    traj_cols : dict, optional
        Trajectory-column mappings.
    **kwargs
        Additional trajectory-column mappings.

    Returns
    -------
    pd.DataFrame
        One row per uninterrupted visit.
    """
    if not isinstance(stops, pd.DataFrame):
        raise TypeError("stops must be a pandas DataFrame")

    traj_cols = loader._parse_traj_cols(
        stops.columns, traj_cols, kwargs, warn=False
    )
    location_col = location_col or traj_cols['location_id']
    traj_cols['location_id'] = location_col
    if location_col not in stops.columns:
        raise ValueError(f"Location column '{location_col}' not found in stops DataFrame")
    user_col = traj_cols['user_id'] if traj_cols['user_id'] in stops.columns else None
    if user_col is not None and stops[user_col].nunique(dropna=False) > 1:
        raise ValueError(
            "merge_stops expects one user per call; group the input by user_id "
            "and call merge_stops for each group."
        )
    if isinstance(time_thresh, bool) or not isinstance(time_thresh, (int, np.integer)):
        raise TypeError("time_thresh must be an integer number of minutes")
    if time_thresh < 0:
        raise ValueError("time_thresh must be nonnegative")

    t_key, use_datetime = loader._fallback_time_cols_dt(
        stops.columns, traj_cols, kwargs
    )
    time_col = traj_cols[t_key]
    if not stops[time_col].is_monotonic_increasing:
        raise ValueError(f"'{time_col}' must be monotonically increasing")

    end_col_present = loader._has_end_cols(stops.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stops.columns, traj_cols)
    if end_col_present or duration_col_present:
        if stops.empty:
            merged = stops.copy()
            if 'cluster' not in merged.columns:
                merged.insert(
                    0,
                    'cluster',
                    pd.Series(index=merged.index, dtype='Int64'),
                )
            return merged
        return _merge_stops_from_agg(
            stops,
            traj_cols=traj_cols,
            t_key=t_key,
            use_datetime=use_datetime,
            time_thresh=time_thresh,
        )

    if t_key in ('start_timestamp', 'start_datetime'):
        raise ValueError("Stops must contain either end time or duration columns")

    if method == 'grid_based':
        if algorithm is not None or algorithm_kwargs is not None:
            raise ValueError(
                "algorithm and algorithm_kwargs require method='custom'"
            )
        # sequential_algs imports the aggregation helper used by Lachesis.
        from nomad.stop_detection.sequential_algs import grid_based

        return grid_based(
            stops,
            time_thresh=time_thresh,
            min_cluster_size=1,
            dur_min=0,
            traj_cols=traj_cols,
        )
    if method != 'custom':
        raise NotImplementedError(f"merge_stops method '{method}' is not implemented")
    if algorithm is None:
        raise ValueError("algorithm is required when method='custom'")

    options = {'time_thresh': time_thresh}
    options.update(algorithm_kwargs or {})
    options['traj_cols'] = traj_cols
    return algorithm(data=stops, **options)
