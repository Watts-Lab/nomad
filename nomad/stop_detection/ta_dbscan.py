import pandas as pd
import numpy as np
from collections import defaultdict
import pdb
import warnings
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils
from nomad.filters import to_timestamp

##########################################
########        TA-DBSCAN         ########
##########################################
def _find_neighbors(data, time_thresh, dist_thresh, use_lon_lat, use_datetime, traj_cols):
    """
    Compute neighbors within specified time and distance thresholds for a trajectory dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing spatial and temporal information.
    traj_cols : dict
        Dictionary mapping column names for trajectory attributes.
    time_thresh : int
        Time threshold in minutes for considering neighboring points.
    dist_thresh : float
        Distance threshold for considering neighboring points.
    long_lat : bool, optional
        Whether to use longitude/latitude coordinates.
    datetime : bool, optional
        Whether to process timestamps as datetime objects.
    
    Returns
    -------
    dict
        A dictionary where keys are timestamps, and values are sets of neighboring
        timestamps that satisfy both time and distance thresholds.
    """
    # getting coordinates based on whether they are geographic coordinates (lon, lat) or catesian (x,y)
    if use_lon_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values) # TC: O(n)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values # TC: O(n)
    
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    times = to_timestamp(times).values if use_datetime else times.values
      
    # Pairwise time differences
    time_diffs = np.abs(times[:, np.newaxis] - times)
    time_diffs = time_diffs.astype(int)
  
    # Filter by time threshold
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1)
    time_pairs = np.where(within_time_thresh)
  
    # Distance calculation
    if use_lon_lat:
        distances = np.array([utils._haversine_distance(coords[i], coords[j]) for i, j in zip(*time_pairs)])
    else:
        distances_sq = (coords[time_pairs[0], 0] - coords[time_pairs[1], 0])**2 + (coords[time_pairs[0], 1] - coords[time_pairs[1], 1])**2
        distances = np.sqrt(distances_sq)

    # Filter by distance threshold
    neighbor_pairs = distances < dist_thresh
  
    # Building the neighbor dictionary
    neighbor_dict = defaultdict(set)
  
    for i, j in zip(time_pairs[0][neighbor_pairs], time_pairs[1][neighbor_pairs]):
        neighbor_dict[times[i]].add(times[j])
        neighbor_dict[times[j]].add(times[i])

    return neighbor_dict

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

def dbscan(data, times, time_thresh, dist_thresh, min_pts, use_lon_lat, use_datetime, traj_cols, neighbor_dict=None):
    """
    Perform DBSCAN on a trajectory dataset with spatiotemporal constraints.
    
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
    long_lat : bool
        Whether to use longitude/latitude coordinates.
    datetime : bool
        Whether to process timestamps as datetime objects.
    traj_cols : dict
        Dictionary mapping column names for trajectory attributes.
    neighbor_dict : dict, optional
        Precomputed dictionary of neighbors. If not provided, it will be computed.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - 'cluster': Cluster labels assigned to each point. Noise points are labeled as -1.
        - 'core': Core point labels for each point. Non-core points are labeled as -3.
    """
    if not neighbor_dict:
        neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, use_lon_lat, use_datetime, traj_cols)
    else:
        neighbor_dict = defaultdict(set,
                                    {k: v.intersection(times) for k, v in
                                     neighbor_dict.items() if
                                     k in times})

    cluster_df = pd.Series(-2, index=times, name='cluster')
    core_df = pd.Series(-3, index=times, name='core')
    
    # Initialize cluster label
    cid = -1
    
    for i, cluster in cluster_df.items():
        if cluster < 0:
            if len(neighbor_dict[i]) < min_pts:
                # Mark as noise if below min_pts
                cluster_df[i] = -1
            else:
                cid += 1
                cluster_df[i] = cid  # Assign new cluster label
                core_df[i] = cid  # Assign new core label
                S = list(neighbor_dict[i])  # Initialize stack with neighbors
                while S:
                    j = S.pop()
                    if cluster_df[j] < 0:  # Process if not yet in a cluster
                        cluster_df[j] = cid
                        if len(neighbor_dict[j]) >= min_pts:
                            core_df[j] = cid  # Assign core label
                            for k in neighbor_dict[j]:
                                if cluster_df[k] < 0:
                                    S.append(k)  # Add new neighbors

    return pd.DataFrame({'cluster': cluster_df, 'core': core_df})


def _process_clusters(data, time_thresh, dist_thresh, min_pts, output, use_lon_lat, use_datetime, traj_cols, 
                     cluster_df=None, neighbor_dict=None, min_duration=5):
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
        Whether to process timestamps as datetime objects.
    traj_cols : dict
        Dictionary mapping column names for trajectory attributes.
    cluster_df : pandas.DataFrame, optional
        DataFrame containing cluster and core labels from DBSCAN. If not provided,
        it will be computed.
    neighbor_dict : dict, optional
        Precomputed dictionary of neighbors. If not provided, it will be computed.
    min_duration : int, optional
        Minimum duration (in minutes) required for a cluster to be considered valid (default is 4).
    
    Returns
    -------
    bool
        True if at least one valid cluster is identified and processed, otherwise False.
    """
    if not neighbor_dict:
        neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, use_lon_lat, use_datetime, traj_cols)
    if cluster_df is None:
        cluster_df = dbscan(data=data,
                            times=data.index,
                            time_thresh=time_thresh,
                            dist_thresh=dist_thresh,
                            min_pts=min_pts,
                            use_lon_lat=use_lon_lat,
                            use_datetime=use_datetime,
                            traj_cols=traj_cols,
                            neighbor_dict=neighbor_dict)
    if len(cluster_df) < min_pts:
        return False

    cluster_df = cluster_df[cluster_df['cluster'] != -1]  # Remove noise pings

    # All pings are in the same cluster
    if len(cluster_df['cluster'].unique()) == 1:
        # We rerun dbscan because possibly these points no longer hold their own
        x = dbscan(data=data.loc[cluster_df.index],
                   times=cluster_df.index,
                   time_thresh = time_thresh,
                   dist_thresh = dist_thresh,
                   min_pts=min_pts,
                   use_lon_lat=use_lon_lat,
                   use_datetime=use_datetime,
                   traj_cols=traj_cols,
                   neighbor_dict=neighbor_dict)
        
        y = x.loc[x['cluster'] != -1]
        z = x.loc[x['core'] != -1]

        if len(y) > 0:
            duration = int((y.index.max() - y.index.min()) // 60)

            if duration > min_duration:
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
        if _process_clusters(data, time_thresh, dist_thresh, min_pts, output, use_lon_lat, use_datetime, traj_cols, cluster_df = cluster_df[i:j]):  # Valid cluster in the middle
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             use_lon_lat, use_datetime, traj_cols, cluster_df = cluster_df[:i])  # Process the initial stub
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             use_lon_lat, use_datetime, traj_cols, cluster_df = cluster_df[j:])  # Process the "tail"
            return True
        else:  # No valid cluster in the middle
            return _process_clusters(data, time_thresh, dist_thresh, min_pts, output, use_lon_lat, use_datetime, traj_cols, pd.concat([cluster_df[:i], cluster_df[j:]]))


def temporal_dbscan(data, time_thresh, dist_thresh, min_pts, traj_cols=None, complete_output=False, **kwargs):
    labels_tadbscan = _temporal_dbscan_labels(data, time_thresh, dist_thresh, min_pts, False, traj_cols, **kwargs) # see patch on 424

    merged = data.join(labels_tadbscan)
    merged = merged[merged.cluster != -1]

    stop_table = merged.groupby('cluster', as_index=False).apply(lambda grp: utils.summarize_stop(grp,
                                                                                                  complete_output=complete_output,
                                                                                                  traj_cols=traj_cols,
                                                                                                  keep_col_names=False,
                                                                                                  **kwargs),
                        include_groups=False)
    return stop_table

def _temporal_dbscan_labels(data, time_thresh, dist_thresh, min_pts, return_cores=False, traj_cols=None, **kwargs):
    # Check if user wants long and lat and datetime
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)
    # Load default col names
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    
    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Parse if necessary
    if use_datetime:
        # Use to_timestamp to convert datetime series to UNIX timestamps (seconds)
        times = to_timestamp(data[traj_cols[t_key]])
    else:
        first_val = data[traj_cols[t_key]].iloc[0]
        s_val = str(first_val)
        if len(s_val) == 13:
            warnings.warn(f"The '{traj_cols[t_key]}' column appears to be in milliseconds. Converting to seconds.")
            times = data[traj_cols[t_key]] // 10**3
        elif len(s_val) == 19:
            warnings.warn(f"The '{traj_cols[t_key]}' column appears to be in nanoseconds. Converting to seconds.")
            times = data[traj_cols[t_key]] // 10**9
        elif len(s_val) > 10: 
            warnings.warn(f"The '{traj_cols[t_key]}' column does not appear to be in seconds, with {len(s_val)} digits.")
            times = data[traj_cols[t_key]]
        else:
            times = data[traj_cols[t_key]]

    data_temp = data.copy()
    data_temp.index = times        

    output = pd.DataFrame({'cluster': -1, 'core': -1}, index=times)

    _process_clusters(data_temp, time_thresh, dist_thresh, min_pts, output, use_lon_lat, use_datetime, traj_cols, min_duration=5)

    if return_cores:
        output.index = list(data[traj_cols[t_key]])
        return output
    else:
        labels = output.cluster
        return labels.set_axis(data.index)