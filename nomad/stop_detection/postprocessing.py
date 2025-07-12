import pandas as pd
import numpy as np
from functools import partial
import nomad.stop_detection.utils as utils
import nomad.stop_detection.grid_based as GRID_BASED 
import nomad.io.base as loader
from nomad.stop_detection.dbscan import _find_neighbors
from nomad.filters import to_timestamp
import nomad.stop_detection.hdbscan as HDBSCAN
import nomad.stop_detection.lachesis as LACHESIS
import nomad.stop_detection.dbscan as TADBSCAN

def remove_overlaps(pred, time_thresh=None, dur_min=None, min_pts=None, dist_thresh=None, method = 'polygon', traj_cols = None, **kwargs):
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
        stops = pred.groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)
    
    elif method == 'recurse':
        t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(pred.columns, traj_cols, kwargs)
        times = pred[traj_cols[t_key]]
        times = to_timestamp(times).values if use_datetime else times.values

        data_temp = pred.copy()
        data_temp.index = times
        stops = pd.DataFrame({'cluster': -1, 'core': -1}, index=times)
        _process_clusters(data_temp, time_thresh, dist_thresh, min_pts, stops, use_lon_lat, use_datetime, traj_cols, dur_min=dur_min)
        stops.index = pred.index
    
    return stops

def _process_clusters(data, time_thresh, dist_thresh, min_pts, output, use_lon_lat, use_datetime, traj_cols, 
                     cluster_df=None, neighbor_dict=None, dur_min=5):
    """
    Recursively process spatiotemporal clusters from trajectory data to identify and refine valid clusters.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing spatial and temporal information.
    time_thresh : int
        Time threshold in minutes for identifying neighbors.
    dist_thresh : float
        Distance threshold for identifying neighbors.
    min_pts : int
        Minimum number of points required to form a dense region (core point).
    output : pandas.DataFrame
        Output DataFrame to store cluster and core labels for valid clusters.
    use_lon_lat : bool
        Whether to use longitude/latitude coordinates.
    use_datetime : bool
        Whether to cluster using time columns of type pandas.datetime64 if available.
    traj_cols : dict
        Dictionary mapping column names for trajectory attributes.
    cluster_df : pandas.DataFrame, optional
        DataFrame containing cluster and core labels from DBSCAN. If not provided,
        it will be computed.
    neighbor_dict : dict, optional
        Precomputed dictionary of neighbors. If not provided, it will be computed.
    dur_min : int, optional
        Minimum duration (in minutes) required for a cluster to be considered valid (default is 4).
    
    Returns
    -------
    bool
        True if at least one valid cluster is identified and processed, otherwise False.
    """
    if not neighbor_dict:
        neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, use_lon_lat, use_datetime, traj_cols)
    
    if cluster_df is None:
        cluster_df = TADBSCAN.dbscan(data, time_thresh, dist_thresh, min_pts, use_lon_lat, use_datetime, traj_cols, neighbor_dict=neighbor_dict)
    
    if len(cluster_df) < min_pts:
        return False

    cluster_df = cluster_df[cluster_df['cluster'] != -1]  # Remove noise pings

    # All pings are in the same cluster
    if len(cluster_df['cluster'].unique()) == 1:
        # We rerun dbscan because possibly these points no longer hold their own
        x = TADBSCAN.dbscan(data = data.loc[cluster_df.index], time_thresh = time_thresh, dist_thresh = dist_thresh,
                   min_pts = min_pts, use_lon_lat = use_lon_lat, use_datetime = use_datetime, traj_cols = traj_cols, neighbor_dict = neighbor_dict)
        
        y = x.loc[x['cluster'] != -1]
        z = x.loc[x['core'] != -1]

        if len(y) > 0:
            duration = int((y.index.max() - y.index.min()) // 60)

            if duration > dur_min:
                cid = max(output['cluster']) + 1 # Create new cluster id
                output.loc[y.index, 'cluster'] = cid
                output.loc[z.index, 'core'] = cid
            
            return True
        elif len(y) == 0: # The points in df, despite originally being part of a cluster, no longer hold their own
            return False

    # There are no clusters
    elif len(cluster_df['cluster'].unique()) == 0:
        return False
   
    # There is more than one cluster
    elif len(cluster_df['cluster'].unique()) > 1:
        i, j = _extract_middle(cluster_df)  # Indices of the "middle" of the cluster
        
        # Recursively processes clusters
        if _process_clusters(data, time_thresh, dist_thresh, min_pts, output, use_lon_lat, use_datetime, traj_cols, cluster_df = cluster_df[i:j], dur_min=dur_min):  # Valid cluster in the middle
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             use_lon_lat, use_datetime, traj_cols, cluster_df = cluster_df[:i], dur_min=dur_min)  # Process the initial stub
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             use_lon_lat, use_datetime, traj_cols, cluster_df = cluster_df[j:], dur_min=dur_min)  # Process the "tail"
            return True
        else:  # No valid cluster in the middle
            return _process_clusters(data, time_thresh, dist_thresh, min_pts, output, use_lon_lat, use_datetime, traj_cols, pd.concat([cluster_df[:i], cluster_df[j:]]), dur_min=dur_min)

def _extract_middle(data):
    """
    Extract the middle segment of a cluster within the provided data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data containing a 'cluster' column.
    
    Returns
    -------
    tuple (i, j)
        Indices marking the start (`i`) and end (`j`) of the middle segment
        of the cluster. If the cluster does not reappear, returns indices marking
        the tail of the data.
    """
    current = data.iloc[0]['cluster']
    x = (data.cluster != current).values
    if len(np.where(x)[0]) == 0:  # There is no inbetween
        return (len(data), len(data))
    else:
        i = np.where(x)[0][
            0]  # First index where the cluster is not the value of the first entry's cluster
    if len(np.where(~x[i:])[
               0]) == 0:  # There is no current again (i.e., the first cluster does not reappear, so the middle is actually the tail)
        return (i, len(data))
    else:  # Current reappears
        j = i + np.where(~x[i:])[0][0]
    return (i, j)

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