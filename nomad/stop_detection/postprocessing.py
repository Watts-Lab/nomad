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


def merge_stops(
    stops,
    time_thresh=10,
    location_col=None,
    agg=None,
    traj_cols=None,
    **kwargs
):
    """
    Merge consecutive stops at the same location.

    Parameters
    ----------
    stops : pd.DataFrame
        Ping or stop table containing time and location columns. Stop tables
        additionally contain an end time or duration.
    time_thresh : int, default 10
        Largest gap in minutes between one stop's end and the next stop's start
        that can belong to the same visit.
    location_col : str, optional
        Location identifier column. Defaults to the ``location_id`` mapping in
        ``traj_cols``.
    agg : dict, optional
        Additional column aggregations for each merged visit in a stop table.
    traj_cols : dict, optional
        Trajectory-column mappings.
    **kwargs
        Additional trajectory-column mappings.

    Returns
    -------
    pd.DataFrame
        One row per uninterrupted visit. Ping tables are summarized through the
        grid-based algorithm; stop tables are merged using their intervals.
    """
    traj_cols = loader._parse_traj_cols(
        stops.columns, traj_cols, kwargs, warn=False
    )
    location_col = location_col or traj_cols['location_id']
    traj_cols['location_id'] = location_col
    if location_col not in stops.columns:
        raise ValueError(f"Location column '{location_col}' not found in stops DataFrame")
    if isinstance(time_thresh, bool) or not isinstance(time_thresh, (int, np.integer)):
        raise TypeError("time_thresh must be an integer number of minutes")
    if time_thresh < 0:
        raise ValueError("time_thresh must be nonnegative")
    if stops.empty:
        return stops.copy()

    user_col = traj_cols['user_id'] if traj_cols['user_id'] in stops.columns else None
    if user_col is not None and stops[user_col].nunique(dropna=False) > 1:
        raise ValueError(
            "merge_stops expects one user per call; group the input by user_id "
            "and call merge_stops for each group."
        )

    end_col_present = loader._has_end_cols(stops.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stops.columns, traj_cols)
    if not end_col_present and not duration_col_present:
        # Imported here because sequential_algs embeds merge_stops as an optional
        # postprocessing step and therefore imports this module.
        from nomad.stop_detection.sequential_algs import grid_based

        t_key, _ = loader._fallback_time_cols_dt(
            stops.columns, traj_cols, kwargs
        )
        if t_key in ('start_timestamp', 'start_datetime'):
            raise ValueError("Stops must contain either end time or duration columns")
        return grid_based(
            stops,
            time_thresh=time_thresh,
            min_cluster_size=1,
            dur_min=0,
            traj_cols=traj_cols,
            **kwargs,
        )

    t_key, use_datetime = loader._fallback_time_cols_dt(
        stops.columns, traj_cols, kwargs
    )
    start_col = traj_cols[t_key]
    end_key = 'end_datetime' if use_datetime else 'end_timestamp'

    order = (
        stops.reset_index(drop=True)
        .sort_values(start_col, kind='stable')
        .index.to_numpy()
    )
    input_index = stops.index.take(order)
    ordered = stops.iloc[order].reset_index(drop=True)

    if end_col_present:
        end_col = traj_cols[end_key]
    else:
        end_col = end_key
        if use_datetime:
            ordered[end_col] = ordered[start_col] + pd.to_timedelta(
                ordered[traj_cols['duration']], unit='min'
            )
        else:
            ordered[end_col] = (
                ordered[start_col] + ordered[traj_cols['duration']] * 60
            )

    gap_threshold = (
        pd.to_timedelta(time_thresh, unit='min')
        if use_datetime else time_thresh * 60
    )
    location_codes = pd.Series(pd.factorize(ordered[location_col], sort=False)[0])
    same_location = location_codes.eq(location_codes.shift()) & location_codes.ne(-1)
    sequence_id = (~same_location).cumsum()
    # The running maximum handles stops nested inside a longer stop.
    previous_end = ordered.groupby(sequence_id, sort=False)[end_col].cummax().shift()
    new_visit = ~same_location | ordered[start_col].sub(previous_end).gt(gap_threshold)
    visit_id = new_visit.fillna(True).cumsum()

    agg_dict = {
        start_col: 'first',
        end_col: 'max',
        location_col: 'first',
    }
    if user_col is not None:
        agg_dict[user_col] = 'first'
    if agg:
        agg_dict.update(agg)

    merged = ordered.groupby(visit_id, sort=False).agg(agg_dict)
    if duration_col_present:
        duration = merged[end_col] - merged[start_col]
        if use_datetime:
            duration = duration.dt.total_seconds()
        merged[traj_cols['duration']] = (duration / 60).astype(int)

    first_positions = ordered.groupby(visit_id, sort=False).head(1).index
    merged.index = input_index.take(first_positions)
    merged.index.name = stops.index.name
    if not end_col_present:
        merged = merged.drop(columns=end_col)

    return merged
