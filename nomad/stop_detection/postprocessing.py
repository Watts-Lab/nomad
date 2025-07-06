import pandas as pd
import numpy as np
from functools import partial
import nomad.stop_detection.utils as utils
import nomad.stop_detection.grid_based as GRID_BASED 
import nomad.io.base as loader

def remove_overlaps(pred, time_thresh, dur_min, min_pts, method = 'polygon', traj_cols = None, **kwargs):
        pred = pred.copy()
        # load kwarg and traj_col args onto lean defaults
        traj_cols = loader._parse_traj_cols(
            pred.columns,
            traj_cols,
            kwargs,
            defaults={'location_id':'location_id'},
            warn=False) 
        
        summarize_stops_with_loc = partial(
            utils.summarize_stop,
            x=traj_cols['x'], # to do: what if it is lat, lon?
            y=traj_cols['y'],
            keep_col_names=False,
            complete_output=True,
            passthrough_cols = [traj_cols['location_id']])
    
        if  method == 'polygon':
            if traj_cols['location_id'] not in pred.columns:
                raise KeyError(
                        f"Missing required `location_id` column for method `polygon`."
                         " pass a column name for `location_id` in keyword arguments or traj_cols"
                         " or use another method."
                    )
                
            pred['temp_building_id'] = pred[traj_cols['location_id']].fillna("None-"+pred.cluster.astype(str))
            traj_cols['location_id'] = 'temp_building_id'
            
            labels = GRID_BASED.grid_based_labels(
                                    data=pred.loc[pred.cluster!=-1],
                                    time_thresh=time_thresh,
                                    dur_min=dur_min,
                                    min_cluster_size=min_pts,
                                    traj_cols=traj_cols)
                    
            pred.loc[pred.cluster!=-1, 'cluster'] = labels
            pred = pred.drop('temp_building_id', axis=1)
            # Consider returning just cluster labels, same as the input! 
            stops = pred.loc[pred.cluster!=-1].groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)
    
        elif method == 'cluster':
            traj_cols['location_id'] = 'cluster'
            labels = GRID_BASED.grid_based_labels(
                                    data=pred.loc[pred.cluster!=-1],
                                    time_thresh=time_thresh,
                                    dur_min=dur_min,
                                    min_cluster_size=min_pts,
                                    traj_cols=traj_cols)
            
            pred.loc[pred.cluster!=-1, 'cluster'] = labels
            stops = pred.loc[pred.cluster!=-1].groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)
        
        elif method == 'recurse':
            raise ValueError("Method `recurse` not implemented yet.")
    
        return stops

def invalid_stops(stop_data, traj_cols=None, print_stops=False, **kwargs):
    """
    Detect any overlapping stops in a stop-detection table.

    Parameters
    ----------
    stop_data : pd.DataFrame
        Output of a stop-detection algorithm containing start/end columns.
    traj_cols : dict, optional
        Mapping for column names.  Only two keys matter:
        - 'start_timestamp' or 'start_datetime'
        - 'end_timestamp'   or 'end_datetime'

    Returns
    -------
    False
        When no overlaps are found.

    Raises
    ------
    ValueError
        If at least one pair of stops overlaps.  The message shows the
        first offending pair.
    """
    # determine start-time key and whether it's datetime
    t_key, use_datetime = utils._fallback_time_cols(stop_data.columns, traj_cols, kwargs)
    end_t_key = 'end_datetime' if use_datetime else 'end_timestamp'

    # canonical column mapping
    traj_cols = loader._parse_traj_cols(stop_data.columns, traj_cols, kwargs, warn=False)
    end_col_present  = loader._has_end_cols(stop_data.columns, traj_cols)
    duration_col_present  = loader._has_duration_cols(stop_data.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        raise ValueError("Missing required (end or duration) temporal columns for stop_table dataframe.")

    # compute a uniform '_end_time' column
    if not end_col_present:
        dur_mins = stop_data[traj_cols['duration']]
        end_col = stop_data[traj_cols[t_key]] + pd.to_timedelta(dur_mins, unit='m')
        if use_datetime:
            end_col = stop_data[traj_cols[t_key]] + pd.to_timedelta(dur_mins, unit='m')
        else:
            end_col = stop_data[traj_cols[t_key]] + dur_mins*60
    else:
        end_col = stop_data[traj_cols[end_t_key]]


    # single scan for overlap
    for i in range(1, len(stop_data)):
        prev, curr = stop_data.iloc[i-1], stop_data.iloc[i]
        prev_end, curr_end = end_col.iloc[i-1], end_col.iloc[i] #scalar
        
        if curr[traj_cols[t_key]] < prev_end:
            if print_stops:
                print(
                    f"Overlapping stops spanning:",
                    f"({prev[traj_cols[t_key]]}–{prev_end}) and ",
                    f"({curr[traj_cols[t_key]]}–{curr_end})"
                    )
            return True

    return False

def fill_timestamp_gaps(first_time, last_time, stop_table):
    # If the stop table is empty, the entire duration is a gap.
    if stop_table.empty:
        gap = (last_time - first_time) // 60
        return pd.DataFrame([{
            'start_timestamp': first_time,
            'duration': gap,
            'building_id': "Street"
        }])

    new_rows = []
    # Use "Street" instead of "None" for clarity
    gap_building_id = "Street"

    # fill initial gap
    if first_time < stop_table.loc[0, 'start_timestamp']:
        gap = (stop_table.loc[0, 'start_timestamp'] - first_time) // 60
        new_rows.append({'start_timestamp': first_time, 'duration': gap, 'building_id': gap_building_id})

    # fill intermediate gaps
    for i in range(len(stop_table) - 1):
        end_time = stop_table.loc[i, 'start_timestamp'] + stop_table.loc[i, 'duration'] * 60
        next_start = stop_table.loc[i + 1, 'start_timestamp']
        if end_time < next_start:
            gap = (next_start - end_time) // 60
            new_rows.append({'start_timestamp': end_time, 'duration': gap, 'building_id': gap_building_id})

    # fill final gap
    last_end = stop_table.iloc[-1]['start_timestamp'] + stop_table.iloc[-1]['duration'] * 60
    if last_end < last_time:
        gap = (last_time - last_end) // 60
        new_rows.append({'start_timestamp': last_end, 'duration': gap, 'building_id': gap_building_id})

    # Combine and sort, filling other columns with NaN automatically
    df_full = pd.concat([stop_table, pd.DataFrame(new_rows)], ignore_index=True)
    df_full = df_full.sort_values('start_timestamp').reset_index(drop=True)
    return df_full