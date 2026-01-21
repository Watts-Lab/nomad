import pandas as pd
import numpy as np
from collections import defaultdict
import nomad.io.base as loader
import warnings
import geopandas as gpd
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.stop_detection.postprocessing import remove_overlaps
from nomad.filters import to_timestamp
from nomad.stop_detection.preprocessing import _find_neighbors
import pdb

##########################################
########         DBSCAN           ########
##########################################

def ta_dbscan_labels(data, dist_thresh, min_pts, time_thresh, dur_min=5, return_cores=False, remove_overlaps=False, traj_cols=None, **kwargs):
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    if data.empty:
        return pd.DataFrame()

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    valid_times = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]
    
    neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, use_lon_lat, use_datetime, traj_cols)

    cluster_df = pd.Series(-2, index=valid_times, name='cluster')
    core_df = pd.Series(-3, index=valid_times, name='core')
    # Initialize cluster label
    cid = -1

    # replace with connected components of core points?
    
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
                        if remove_overlaps:
                            cluster_df[j] = -1
                        else:
                            cluster_df[j] = cid  # delay adding border points, set to -1 (visited)
                        if len(neighbor_dict[j]) >= min_pts: # if core point reachable from i
                            core_df[j] = cid  # Add to i's cluster, assigning cid label
                            for k in neighbor_dict[j]:
                                if remove_overlaps:
                                    if cluster_df[k] < -1: #unvisited
                                        S.append(k)  # Add new neighbors
                                else:
                                    if cluster_df[k] < 0: #unassigned
                                        S.append(k)
    
    ### Remove overlaps (optional) and attach clusters ###
    if remove_overlaps:
        new_label = cid
        i = 0
        start_time = cluster_df.index[0] if not cluster_df.empty else -1

        while i<len(cluster_df): #i is end time
            if cluster_df.index[i] < start_time:
                i = i+1
                continue
            
            if len(core_df[(core_df.index>=start_time)&(core_df>0)])>0:
                label = core_df[(core_df.index>=start_time)&(core_df>0)].iloc[0]
            else:
                break
            
            # how do I find the cutoff? 
            next_labels = core_df.loc[(core_df>0)&(core_df!=label)&(core_df.index>start_time)]
            if len(next_labels)>0:
                end_time = next_labels.index[0]
            else:
                end_time = np.inf
                
            if len(core_df.loc[(core_df==label)&(core_df.index>end_time)])>0: # cluster "label" is split
                new_label = new_label + 1
                core_df.loc[(core_df==label)&(core_df.index>end_time)] = new_label
                
            # attach border points all neighbors of "label" cores within time window
            set_list = [neighbor_dict[idx] for idx in core_df.loc[core_df==label].index]
            set_list = set_list + [set(core_df.loc[core_df==label].index)]
            label_neighbors = sorted(set().union(*set_list))

            mask = (cluster_df.index>=start_time)&(cluster_df.index<end_time)&(cluster_df.index.isin(label_neighbors))
            
            filtered = cluster_df[mask]
                
            if filtered.index.max() - filtered.index.min() >= dur_min*60:
                start_time = filtered.index.max()
                cluster_df.loc[mask] = label
                
            else:
                cluster_df.loc[mask] = -1
                core_df.loc[mask] = -3

            i = i+1

    output = pd.DataFrame({'cluster': cluster_df, 'core': core_df})

    if return_cores:
        return output.set_axis(data.index)
    else:
        labels = output.cluster
        return labels.set_axis(data.index)

def ta_dbscan(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    complete_output=False,
    passthrough_cols=[],
    keep_col_names=True,
    traj_cols=None,
    **kwargs
):
    """
    Temporal-augmented DBSCAN stop detection with summarization.

    Parameters
    ----------
    data : pd.DataFrame
        Input trajectory with spatial and temporal columns.
    time_thresh : int
        Max time gap (minutes) for neighbors.
    dist_thresh : float
        Max spatial distance for neighbors.
    min_pts : int
        Minimum number of neighbors for a core point.
    dur_min : int, optional
        Minimum duration (minutes) for a stop (default: 5).
    complete_output : bool, optional
        Include extra stats if True (default: False).
    passthrough_cols : list, optional
        Columns to retain per stop.
    traj_cols : dict, optional
        Mapping for column names.
    **kwargs
        Passed to internal helpers.

    Returns
    -------
    pd.DataFrame
        One row per stop with medoid/centroid, duration, and optionally extra columns.

    Raises
    ------
    ValueError if multi-user data detected; use ta_dbscan_per_user instead.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        first = arr[0]
        if any(x != first for x in arr[1:]):
            raise ValueError("Multi-user data? Use ta_dbscan_per_user instead.")
        passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]

    labels = ta_dbscan_labels(
        data=data,
        dist_thresh=dist_thresh,
        min_pts=min_pts,
        time_thresh=time_thresh,
        dur_min=5,
        return_cores=False,
        remove_overlaps=True,
        traj_cols=traj_cols,
        **kwargs
    )
    merged = data.join(labels)

    # if len(merged.cluster.unique())>2:
    #     # Get adjusted cluster labels (not summary table)
    #     adjusted_labels = remove_overlaps(
    #         merged,
    #         dist_thresh=dist_thresh,
    #         min_pts=min_pts,
    #         time_thresh=time_thresh,
    #         method="cluster",
    #         traj_cols=traj_cols,
    #         summarize_stops=False,  # Return cluster labels, not summary table
    #         **kwargs)
        
    #     # Update the cluster column with adjusted labels
    #     merged['cluster'] = adjusted_labels
    
    # Filter out noise points after overlap removal
    merged = merged[merged.cluster != -1]

    if merged.empty:
        # Get column names by calling summarize function on dummy data
        cols = utils._get_empty_stop_columns(
            data.columns, complete_output, passthrough_cols, traj_cols, 
            keep_col_names=keep_col_names, is_grid_based=False, **kwargs
        )
        return pd.DataFrame(columns=cols, dtype=object)

    stop_table = merged.groupby('cluster', as_index=False, sort=False).apply(
        lambda grp: utils.summarize_stop(
            grp,
            complete_output=complete_output,
            traj_cols=traj_cols,
            keep_col_names=keep_col_names,
            passthrough_cols=passthrough_cols,
            **kwargs
        ),
        include_groups=False
    )
    return stop_table

def ta_dbscan_per_user(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    complete_output=False,
    passthrough_cols=[],
    traj_cols=None,
    **kwargs
):
    """
    Run ta_dbscan on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("ta_dbscan_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols + [uid]

    results = [
        ta_dbscan(
            data=group,
            dist_thresh=dist_thresh,
            min_pts=min_pts,
            time_thresh=time_thresh,
            dur_min=dur_min,
            complete_output=complete_output,
            passthrough_cols=pt_cols,
            traj_cols=traj_cols,
            **kwargs
        )
        for _, group in data.groupby(uid, sort=False)
    ]
    return pd.concat(results, ignore_index=True)
