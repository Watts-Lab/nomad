import numpy as np
import pandas as pd
import geopandas as gpd
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.stop_detection.preprocessing import _find_neighbors
from nomad.filters import to_timestamp
import networkx as nx

def window_graph(G, lo, hi):
    return nx.subgraph_view(G, filter_node=lambda n, lo=lo, hi=hi: lo <= n <= hi)

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

    G = _find_neighbors(data, time_thresh, traj_cols, dist_thresh,
                False, use_datetime, use_lon_lat, return_trees=False, relabel_nodes=True)
    cluster_df = pd.Series(-2, index=G, name='cluster')
    core_df = pd.Series(-2, index=G, name='core')

    # SeqScan main loop start
    start = next(iter(G))      # current time context start
    end = start                  # current candidate to cut time context
    #find cluster routine start
    temp_G = nx.subgraph_view(G, filter_node=lambda n: start <= n <= end)
    
    active_cid = -1
    # thus active_cid - 1 is the preceeding cluster id
    temp_cid = active_cid  # temporary labels are always > active_cid

    def findCluster(start_time, t):
        nonlocal temp_G, temp_cid, active_cid, start, end
        window = slice(start_time, t)
        temp_G = window_graph(G, start_time, t)

        curr_is_core = len(temp_G[t]) >= min_pts
        if curr_is_core:
            temp_cid += 1
            core_df[t] = temp_cid
            cluster_df[t] = temp_cid

        for s in temp_G[t]:
            if len(temp_G[s]) >= min_pts:
                if core_df[s] >= 0:
                    if curr_is_core:
                        core_win = core_df.loc[window]
                        relabel_idxs = core_win.index[core_win.isin([core_df[s], core_df[t]])]
                        merged_label = min(core_df[s], core_df[t])

                        cluster_df.loc[relabel_idxs] = merged_label
                        core_df.loc[relabel_idxs] = merged_label
                    
                elif cluster_df[s] >= 0:
                    core_df[s] = cluster_df[s]
                    if curr_is_core:
                        core_win = core_df.loc[window]
                        relabel_idxs = core_win.index[core_win.isin([core_df[s], core_df[t]])]
                        merged_label = min(core_df[s], core_df[t])

                        cluster_df.loc[relabel_idxs] = merged_label
                        core_df.loc[relabel_idxs] = merged_label

                    nb_labs = {core_df.loc[s]}

                    for nb in temp_G[s]:
                        if core_df[nb] >= 0:
                            nb_labs.add(core_df[nb])
                        else:
                            cluster_df[nb] = core_df[s]

                    merged_label = min(nb_labs)
                    core_win = core_df.loc[window]
                    clu_win = cluster_df.loc[window]

                    core_df.loc[core_win.index[core_win.isin(nb_labs)]] = merged_label
                    cluster_df.loc[clu_win.index[clu_win.isin(nb_labs)]] = merged_label
                
                elif cluster_df[s] == -1:
                    if curr_is_core:
                        core_df[s] = core_df[t]
                        cluster_df[s] = cluster_df[t]
                    else:
                        temp_cid += 1
                        core_df[s] = temp_cid
                        cluster_df[s] = temp_cid

                        for nb in temp_G[s]:
                            cluster_df[nb] = core_df[s]
            else:
                for nb in reversed(list(temp_G[s])):
                    if core_df[nb] >= 0:
                        cluster_df[t] = core_df[nb]
                        break

        clu_win = cluster_df.loc[window]
        cand = clu_win[clu_win >= 0]

        if cand.empty:
            # vars changed: temp_neighbors_df, core_df, cluster_df
            return False
        else:
            spans = cand.index.to_series().groupby(cand, sort=False).agg(["first", "last"])
            eligible = spans[(spans["last"] - spans["first"]) >= (dur_min * 60)]
            
            if eligible.empty:
                return False
            else:
                c = eligible.index[0]
                clu_win = cluster_df.loc[window]
                # indices in the window that belong to label c
                keep_idx = clu_win.index[clu_win == c]
                
                end = spans.at[c, "last"]
                new_cluster = (c != active_cid)

                if new_cluster:
                    if active_cid != -1:
                        first = spans.at[c, "first"]
                        prev_border_idx = clu_win.index[(clu_win == active_cid) & (clu_win.index <= first)]
                            
                    active_cid += 1
                    start = start_time

                # cleanup of labels in (start_time, t); then restore the new active cluster labels
                cluster_df.loc[window] = -1
                core_df.loc[window] = -1
                
                if new_cluster and active_cid>0:
                    cluster_df.loc[prev_border_idx] = active_cid - 1
                
                cluster_df.loc[keep_idx] = active_cid
                core_df.loc[keep_idx] = active_cid

                temp_cid = active_cid
                return True
                # vars changed: temp_neighbors_df, core_df, cluster_df, active_cid, end, temp_cid
        ###### End of def find_cluster

    for curr_time in G:
        # mark as visited. core relabeling happens later.
        cluster_df.at[curr_time] = -1
        core_df.at[curr_time] = -1
        if active_cid == -1:
            findCluster(start, curr_time)
        else:
            temp_G = window_graph(G, start, curr_time)
            curr_is_core = len(temp_G[curr_time]) >= min_pts
            is_reachable = False
            for nb in temp_G[curr_time]:
                if core_df[nb] == active_cid:
                    is_reachable = True
                    cluster_df[curr_time] = active_cid
                    break
                    
            if curr_is_core and is_reachable:
                core_df[curr_time] = active_cid
                end = curr_time
                if back_merge and active_cid > 0:
                    prev_lab = active_cid - 1
                    for nb in reversed(core_df[core_df == prev_lab].index):
                        if curr_time in G[nb]:
                            cluster_df[cluster_df == (active_cid - 1)] = active_cid
                            core_df[core_df == (active_cid - 1)] = active_cid
                            break
            else:
                findCluster(end + 1, curr_time)

    # temporary labels are above active_cid; clear them before returning
    cluster_df.loc[cluster_df > active_cid] = -1
    core_df.loc[core_df > active_cid] = -1
    output = pd.DataFrame({'cluster': cluster_df, 'core': core_df}).set_axis(data.index)

    if return_cores:
        return output
    else:
        labels = output.cluster
        return labels
    
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
            keep_col_names=keep_col_names,
            passthrough_cols=passthrough_cols,
            **kwargs
        ),
        include_groups=False
    ).reset_index(drop=True)
    return stop_table

