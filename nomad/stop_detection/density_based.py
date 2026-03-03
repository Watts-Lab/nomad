import numpy as np
import pandas as pd
import geopandas as gpd
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.stop_detection.preprocessing import _find_neighbors
from nomad.filters import to_timestamp


def seqscan_labels(
    data,
    dist_thresh,
    dur_min=5,
    time_thresh=90,
    min_pts=3,
    user_id=None,
    return_cores=False,
    traj_cols=None,
    back_merge=False,
    **kwargs
):
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    if data.empty:
        return pd.DataFrame()

    if user_id is not None:
        data = data.loc[data["user_id"] == user_id].copy()

    t_key, _, _, use_datetime, use_lon_lat = utils._fallback_st_cols(
        data.columns, traj_cols, kwargs
    )        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    valid_times = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]
    neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, use_lon_lat, use_datetime, traj_cols)
    
    cluster_df = pd.Series(-1, index=valid_times, name='cluster')
    core_df = pd.Series(-1, index=valid_times, name='core')

    # SeqScan main loop start
    start = valid_times.iloc[0]        # current time context start
    end = valid_times.iloc[0] - 1        # current candidate to cut time context
    #find cluster routine start
    temp_neighbor_dict = {}
    active_cid = -1
    # thus active_cid - 1 is the preceeding cluster id
    temp_cid = 100000 # for internal dbscan until new active cluster is found

    def findCluster(start_time, t):
        nonlocal temp_neighbor_dict, temp_cid, active_cid, start, end
        temp_neighbor_dict = {idx:nbs for idx, nbs in temp_neighbor_dict.items() if idx >= start_time}
        temp_neighbor_dict[t] = {nb for nb in neighbor_dict[t] if t > nb >= start_time}
        curr_is_core = len(temp_neighbor_dict[t]) >= min_pts
        if curr_is_core:
            temp_cid += 1
            core_df[t] = temp_cid
            cluster_df[t] = temp_cid

        for s in temp_neighbor_dict[t]:
            temp_neighbor_dict[s].add(t)
            if len(temp_neighbor_dict[s]) >= min_pts:
                if core_df[s] >= 0:
                    # already was a temporary core point
                    # just merge "through" t
                    # if t is core, relabel s and its neighbors (relabeling for merge)
                    if curr_is_core:
                        relabel_mask = (cluster_df.index >= start_time) & (cluster_df.index <= t) & (core_df == core_df[t])
                        cluster_df.loc[relabel_mask] = core_df[s]
                        core_df.loc[relabel_mask] = core_df[s]
                    
                elif cluster_df[s] >= 0:
                    core_df[s] = cluster_df[s]
                    if curr_is_core:
                        relabel_mask = (cluster_df.index >= start_time) & (cluster_df.index <= t) & (core_df == core_df[t])
                        cluster_df.loc[relabel_mask] = core_df[s]
                        core_df.loc[relabel_mask] = core_df[s]
                    for nb in temp_neighbor_dict[s]:
                        if core_df[nb] >= 0:
                            # TODO: border points might be neighbors of 2 disconnected core points
                            # every core point and every neighor of core point should be relabeled
                            nb_labs = set()
                            for nb in temp_neighbor_dict[s]:
                                if core_df[nb] >= 0:
                                    # for later relabel of entire connected component
                                    nb_labs.add(core_df[nb])
                                else:
                                    cluster_df[nb] = core_df[s] # (re) assign border point                            
                            # bulk relabel and merge of all affected connected components
                            for lab in nb_labs:
                                relabel_core = (cluster_df.index >= start_time) & (cluster_df.index <= t) & (core_df == lab)
                                relabel_cluster = (cluster_df.index >= start_time) & (cluster_df.index <= t) & (cluster_df == lab)
                                core_df.loc[relabel_core] = core_df[s]
                                cluster_df.loc[relabel_cluster] = core_df[s]
                
                elif cluster_df[s] == -1: # is a new cluster
                    if curr_is_core:
                        core_df[s] = core_df[t]
                        cluster_df[s] = cluster_df[t]
                    else:
                        temp_cid += 1 # increase temporary label counter
                        core_df[s] = temp_cid
                        cluster_df[s] = temp_cid
                        # propagate label
                        for nb in temp_neighbor_dict[s]:
                            # there is no case in which nb is core point: otherwise cluster_df[s] would have had a label
                            cluster_df[nb] = core_df[s]

        window_mask = (cluster_df.index >= start_time) & (cluster_df.index <= t)
        cand = cluster_df.loc[window_mask & (cluster_df >= 0)]
        
        if not cand.empty:
            spans = cand.index.to_series().groupby(cand, sort=False).agg(["first", "last"])
            eligible = spans[(spans["last"] - spans["first"]) >= (dur_min * 60)]
        
            if not eligible.empty:
                c = eligible.index[0]
                
                active_cid += 1
                keep = window_mask & (cluster_df == c)
                drop = window_mask & ~keep

                first = spans.loc[c, "first"]
                prev_border = window_mask & (cluster_df == active_cid-1) & (cluster_df.index <= first)
                
                cluster_df.loc[drop] = -1
                core_df.loc[drop] = -1
                cluster_df.loc[prev_border] = active_cid - 1
                cluster_df.loc[keep] = active_cid
                core_df.loc[keep] = active_cid
        
                end = spans.loc[c, "last"]
                start = start_time
                temp_cid = 100000
                return True
                # vars changed: temp_neighbors_df, core_df, cluster_df, active_cid, end, temp_cid
        # vars changed: temp_neighbors_df, core_df, cluster_df
        return False



    for curr_time in valid_times:
        if active_cid == -1:
            findCluster(start, curr_time)
        else:
            temp_neighbor_dict[curr_time] = {nb for nb in neighbor_dict[curr_time] if curr_time > nb >= start}
            curr_is_core = len(temp_neighbor_dict[curr_time]) >= min_pts
            is_reachable = False
            for nb in temp_neighbor_dict[curr_time]:
                if core_df[nb] == active_cid:
                    is_reachable = True
                    cluster_df[curr_time] = active_cid
                    break
                    
            if curr_is_core and is_reachable:
                core_df[curr_time] = active_cid
                end = curr_time
                if back_merge and active_cid > 0:
                    prev_lab = active_cid - 1
                    for nb in core_df[core_df == prev_lab].index:
                        if curr_time in neighbor_dict[nb]:
                            cluster_df[cluster_df == (active_cid - 1)] = active_cid
                            core_df[core_df == (active_cid - 1)] = active_cid
                            break
            else:
                findCluster(end + 1, curr_time)
    # TODO: find better way to avoid collisions    
    cluster_df.loc[cluster_df > 100000] = -1
    core_df.loc[core_df > 100000] = -1
    output = pd.DataFrame({'cluster': cluster_df, 'core': core_df})

    if return_cores:
        return output.set_axis(data.index)
    else:
        labels = output.cluster
        return labels.set_axis(data.index)
    
def seqscan(
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

    labels = seqscan_labels(
        data=data,
        dist_thresh=dist_thresh,
        min_pts=min_pts,
        time_thresh=time_thresh,
        dur_min=dur_min,
        return_cores=False,
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

