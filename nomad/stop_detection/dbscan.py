import pandas as pd
import numpy as np
from collections import defaultdict
import nomad.io.base as loader
import warnings
import geopandas as gpd
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.filters import to_timestamp
from nomad.stop_detection.preprocessing import _find_neighbors

##########################################
########         DBSCAN           ########
##########################################

def ta_dbscan_labels(data,
                     dist_thresh,
                     min_pts,
                     time_thresh,
                     return_cores=False,
                     remove_overlaps=True,
                     traj_cols=None,
                     **kwargs):
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    if data.empty:
        return pd.DataFrame()

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)
    
    G = _find_neighbors(data, time_thresh, traj_cols, dist_thresh,
                False, use_datetime, use_lon_lat, return_trees=False, relabel_nodes=True)
    
    cluster_df = pd.Series(-2, index=G, name='cluster')
    core_df = pd.Series(-2, index=G, name='core')
    # Initialize cluster label
    cid = -1

    for i, cluster in cluster_df.items():
        if cluster < 0:
            if len(G[i]) < min_pts:
                # Mark as noise if below min_pts
                cluster_df[i] = -1
            else:
                cid += 1
                cluster_df[i] = cid  # Assign new cluster label
                core_df[i] = cid  # Assign new core label
                S = list(G[i])  # Initialize stack with neighbors
                while S:
                    j = S.pop()
                    if cluster_df[j] < 0:  # Process if not yet in a cluster
                        cluster_df[j] = cid
                        if len(G[j]) >= min_pts:
                            core_df[j] = cid  # Assign core label
                            for k in G[j]:
                                if cluster_df[k] < 0:
                                    S.append(k)  # Add new neighbors
                                    
    ### Remove overlaps (optional) reassign all border points
    if remove_overlaps and (core_df >= 0).any():
        next_label = cid + 1
    
        relabel_dict = {}   # raw_label -> assigned_label (raw until first split, then new id)
        active = None      # active assigned label
    
        for t in core_df.index[core_df >= 0]:
            raw = int(core_df.at[t])

            seen = raw in relabel_dict
            if not seen:
                relabel_dict[raw] = raw
    
            assigned = relabel_dict[raw]    
                
            if active is not None and assigned != active:
                if seen:
                    relabel_dict[raw] = next_label
                    next_label += 1
                    assigned = relabel_dict[raw]
    
            active = assigned
            core_df.at[t] = assigned
            cluster_df.at[t] = assigned
        
        ### Reassign border points to non-overlapping core points
        cluster_df.loc[core_df < 0] = -1  
        prev_run_end = -np.inf                           # left bound (exclusive)
        
        run_label = None
        run_end = None
        run_neighbors = set()                            # union of neighbors of cores in current run
        
        for t in core_df.index[core_df >= 0]:
            lab = core_df.at[t]
        
            if run_label is None:
                run_label = lab
                run_end = t
                run_neighbors.clear()
                run_neighbors.update(G[t])
                continue
        
            if lab == run_label:
                run_end = t
                run_neighbors.update(G[t])
                continue
        
            # label changed => t is the start of the next run, so flush current run now
            next_run_start = t
            max_assigned = prev_run_end
        
            for nb in run_neighbors:
                if prev_run_end < nb < next_run_start and cluster_df.at[nb] == -1:
                    cluster_df.at[nb] = run_label
                    if nb > max_assigned:
                        max_assigned = nb
        
            # advance left bound for the next run:
            # at least to the last core of the run, and also to the latest border we just assigned
            if run_end > max_assigned:
                max_assigned = run_end
            prev_run_end = max_assigned
        
            # start new run
            run_label = lab
            run_end = t
            run_neighbors.clear()
            run_neighbors.update(G[t])
        
        # flush last run to +inf
        next_run_start = np.inf
        max_assigned = prev_run_end
        
        for nb in run_neighbors:
            if prev_run_end < nb < next_run_start and cluster_df.at[nb] == -1:
                cluster_df.at[nb] = run_label
                if nb > max_assigned:
                    max_assigned = nb
        
        if run_end is not None and run_end > max_assigned:
            max_assigned = run_end
        prev_run_end = max_assigned
            
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
    remove_overlaps=True,
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
    if data.empty:
        cols = utils._get_empty_stop_columns(
            data.columns,
            complete_output,
            passthrough_cols,
            traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
            **kwargs,
        )
        col_dtypes = utils._get_empty_stop_column_dtypes(
            data.columns,
            complete_output,
            passthrough_cols,
            traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
            **kwargs,
        )
        return pd.DataFrame({col: pd.Series(dtype=col_dtypes[col]) for col in cols})

    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        first = arr[0]
        if any(x != first for x in arr[1:]):
            raise ValueError("Multi-user data? Use ta_dbscan_per_user instead.")
        if traj_cols_temp['user_id'] not in passthrough_cols:
            passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]

    labels = ta_dbscan_labels(
        data=data,
        dist_thresh=dist_thresh,
        min_pts=min_pts,
        time_thresh=time_thresh,
        return_cores=False,
        remove_overlaps=remove_overlaps,
        traj_cols=traj_cols,
        **kwargs
    )
    merged = data.join(labels)
    
    # Filter out noise points after overlap removal
    merged = merged[merged.cluster != -1]

    if merged.empty:
        # Get column names by calling summarize function on dummy data
        cols = utils._get_empty_stop_columns(
            data.columns, complete_output, passthrough_cols, traj_cols, 
            keep_col_names=keep_col_names, is_grid_based=False, **kwargs
        )
        col_dtypes = utils._get_empty_stop_column_dtypes(
            data.columns,
            complete_output,
            passthrough_cols,
            traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
            **kwargs,
        )
        return pd.DataFrame({col: pd.Series(dtype=col_dtypes[col]) for col in cols})

    stop_table = merged.groupby('cluster', as_index=False, sort=False).apply(
        lambda grp: utils.summarize_stop(
            grp,
            complete_output=complete_output,
            traj_cols=traj_cols,
            dur_min=dur_min,
            keep_col_names=keep_col_names,
            passthrough_cols=passthrough_cols,
            **kwargs
        ),
        include_groups=False
    ).reset_index(drop=True)
    
    return stop_table.loc[stop_table['duration']>=dur_min]

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

    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]

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
