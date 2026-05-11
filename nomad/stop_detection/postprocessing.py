import pandas as pd
import numpy as np
import nomad.io.base as loader

def fill_timestamp_gaps(first_time, last_time, stop_table):

    # if stop_table is empty, return empty DataFrame with same columns
    if stop_table.empty:
        return pd.DataFrame(columns=stop_table.columns)

    new_rows = []

    # fill initial gap
    if first_time < stop_table.loc[0, 'start_timestamp']:
        gap = (stop_table.loc[0, 'start_timestamp'] - first_time) // 60
        new_rows.append({
            'cluster': None,
            'x': None,
            'y': None,
            'start_timestamp': first_time,
            'duration': gap,
            'building_id': "None"
        })

    # fill intermediate gaps
    for i in range(len(stop_table) - 1):
        end_time = stop_table.loc[i, 'start_timestamp'] + stop_table.loc[i, 'duration'] * 60
        next_start = stop_table.loc[i + 1, 'start_timestamp']
        
        if end_time < next_start:
            gap = (next_start - end_time) // 60
            new_rows.append({
                'cluster': None,
                'x': None,
                'y': None,
                'start_timestamp': end_time,
                'duration': gap,
                'building_id': "None"
            })

    # fill final gap
    last_end = stop_table.iloc[-1]['start_timestamp'] + stop_table.iloc[-1]['duration'] * 60
    if last_end < last_time:
        gap = (last_time - last_end) // 60
        new_rows.append({
            'cluster': None,
            'x': None,
            'y': None,
            'start_timestamp': last_end,
            'duration': gap,
            'building_id': "None"
        })

    df_full = pd.concat([stop_table, pd.DataFrame(new_rows)], ignore_index=True)
    df_full = df_full.sort_values('start_timestamp').reset_index(drop=True)

    return df_full


def merge_stops(stops, max_time_gap="10min", location_col="loc_id", agg=None, traj_cols=None, **kwargs):
    """
    Merge consecutive stops at the same location within a time threshold.

    This function aggregates stops that are:
    - At the same location (same location_col value)
    - Consecutive in time (gap between stops <= max_time_gap)
    - From the same user

    Parameters
    ----------
    stops : pd.DataFrame
        Stop table with temporal and location columns.
        Must contain columns for start time, end time (or duration), and location.

    max_time_gap : str or pd.Timedelta, default "10min"
        Maximum duration between consecutive stops to still be merged.
        If str, must be parsable by pd.to_timedelta (e.g., "10min", "1h", "30s").
        If pd.Timedelta, used directly.

    location_col : str, default "loc_id"
        Name of the column containing location identifiers.
        Stops are only merged if they have the same value in this column.

    agg : dict, optional
        Dictionary to aggregate columns after merging stops.
        Keys are column names, values are aggregation functions.
        If None or empty, only required columns (user_id, start times, end times)
        are aggregated and returned.
        Example: {"geometry": "first", "n_pings": "sum"}

    traj_cols : dict, optional
        Column name mappings. Supported keys:
        - 'user_id': user identifier column
        - 'timestamp' or 'datetime': start time column
        - 'end_timestamp' or 'end_datetime': end time column
        - 'duration': duration column (used if end time not present)

    **kwargs
        Additional keyword arguments for column name specification.

    Returns
    -------
    pd.DataFrame
        Merged stops table with the same structure as input but with consecutive
        stops at the same location merged into single rows.

    Notes
    -----
    - The index of the returned DataFrame corresponds to the first stop in each
      merged group.
    - If a stop has no consecutive neighbor at the same location (within max_time_gap),
      it remains unchanged.
    - Aggregation for required columns:
        - start time: first (earliest start)
        - end time: last (latest end)
        - location: first (same for all in group)
        - user_id: first (same for all in group)

    Examples
    --------
    >>> # Basic usage
    >>> merged = merge_stops(stops, max_time_gap="15min", location_col="location_id")

    >>> # With custom aggregation to preserve geometry and sum n_pings
    >>> merged = merge_stops(
    ...     stops,
    ...     max_time_gap="30min",
    ...     location_col="building_id",
    ...     agg={"geometry": "first", "n_pings": "sum"}
    ... )
    """
    if agg is None:
        agg = {}

    # Convert max_time_gap to Timedelta
    if isinstance(max_time_gap, str):
        max_time_gap = pd.to_timedelta(max_time_gap)
    elif not isinstance(max_time_gap, pd.Timedelta):
        raise TypeError("Parameter max_time_gap must be either of type str or pd.Timedelta!")

    # Validate location column exists
    if location_col not in stops.columns:
        raise ValueError(f"Location column '{location_col}' not found in stops DataFrame")

    if stops.empty:
        return stops.copy()

    # Parse column names
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)

    # Determine temporal columns
    t_key, use_datetime = loader._fallback_time_cols_dt(stops.columns, traj_cols, kwargs)
    end_t_key = 'end_datetime' if use_datetime else 'end_timestamp'

    # Check for required temporal columns
    end_col_present = loader._has_end_cols(stops.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stops.columns, traj_cols)

    if not (end_col_present or duration_col_present):
        raise ValueError("Stops must contain either end time or duration columns")

    # Get user_id column if present
    user_col = None
    if 'user_id' in traj_cols and traj_cols['user_id'] in stops.columns:
        user_col = traj_cols['user_id']

    # Work with a copy
    stops_merge = stops.copy()
    index_name = stops.index.name if stops.index.name else 'id'

    # Compute end time if not present
    if not end_col_present:
        dur_mins = stops_merge[traj_cols['duration']]
        if use_datetime:
            stops_merge[end_t_key] = stops_merge[traj_cols[t_key]] + pd.to_timedelta(dur_mins, unit='m')
        else:
            stops_merge[end_t_key] = stops_merge[traj_cols[t_key]] + dur_mins * 60
    else:
        end_t_key = traj_cols[end_t_key]

    # Reset index and preserve it
    stops_merge = stops_merge.reset_index()
    stops_merge["index_temp"] = stops_merge[index_name]

    # Sort by user and time
    if user_col:
        stops_merge = stops_merge.sort_values(by=[user_col, traj_cols[t_key]])
    else:
        stops_merge = stops_merge.sort_values(by=traj_cols[t_key])

    # Get next row information
    shift_cols = [traj_cols[t_key], location_col]
    shift_names = ["next_started_at", "next_location"]

    if user_col:
        shift_cols.insert(0, user_col)
        shift_names.insert(0, "next_user_id")

    stops_merge[shift_names] = stops_merge[shift_cols].shift(-1)
    stops_merge["next_id"] = stops_merge["index_temp"].shift(-1)

    # Iteratively merge stops
    cond = pd.Series(data=False, index=stops_merge.index)
    cond_old = pd.Series(data=True, index=stops_merge.index)

    while np.sum(cond != cond_old) >= 1:
        # Build merge conditions
        conditions = []

        # Same user (if user_col exists)
        if user_col:
            conditions.append(stops_merge["next_user_id"] == stops_merge[user_col])

        # Time gap within threshold
        conditions.append(
            stops_merge["next_started_at"] - stops_merge[end_t_key] <= max_time_gap
        )

        # Same location
        conditions.append(stops_merge[location_col] == stops_merge["next_location"])

        # Not already merged
        conditions.append(stops_merge["index_temp"] != stops_merge["next_id"])

        # Combine all conditions
        cond = pd.Series(data=True, index=stops_merge.index)
        for c in conditions:
            cond = cond & c

        # Assign merged index
        stops_merge.loc[cond, "index_temp"] = stops_merge.loc[cond, "next_id"]

        # Update for next iteration
        cond_diff = cond != cond_old
        cond_old = cond.copy()

        if np.sum(cond_diff) == 0:
            break

    # Define aggregation dictionary
    agg_dict = {
        index_name: "first",
        traj_cols[t_key]: "first",  # earliest start time
        end_t_key: "last",           # latest end time
        location_col: "first",       # same location for all in group
    }

    if user_col:
        agg_dict[user_col] = "first"

    # Add user-defined aggregations
    agg_dict.update(agg)

    # Group and aggregate
    stops_merged = stops_merge.groupby(by="index_temp").agg(agg_dict)

    # Recompute duration if it was in the original
    if duration_col_present:
        if use_datetime:
            stops_merged[traj_cols['duration']] = (
                (stops_merged[end_t_key] - stops_merged[traj_cols[t_key]])
                .dt.total_seconds() / 60
            ).astype(int)
        else:
            stops_merged[traj_cols['duration']] = (
                (stops_merged[end_t_key] - stops_merged[traj_cols[t_key]]) / 60
            ).astype(int)

    # Clean up: set index and sort
    stops_merged = stops_merged.set_index(index_name)

    if user_col:
        stops_merged = stops_merged.sort_values(by=[user_col, traj_cols[t_key]])
    else:
        stops_merged = stops_merged.sort_values(by=traj_cols[t_key])

    # Remove the computed end_t_key if it wasn't in the original
    if not end_col_present and end_t_key in stops_merged.columns:
        stops_merged = stops_merged.drop(columns=[end_t_key])

    return stops_merged
