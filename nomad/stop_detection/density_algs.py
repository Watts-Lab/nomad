from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import nomad.constants as constants
import nomad.io.base as loader
from nomad.stop_detection import utils
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
    """
    Return temporal DBSCAN labels.

    Parameters
    ----------
    return_cores : bool, default False
        Return core labels and ``promotion_time`` with cluster labels. Core
        pings use their own time; border pings use their propagating core time.

    Notes
    -----
    ``promotion_time`` records approximate final-membership propagation time.
    For plotting, accent a ping at ``max(ping_time, promotion_time)``. Its raw
    value can show propagation edges from cores, including to later pings.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    if data.empty:
        return utils._get_empty_aux_df(data[traj_cols[t_key]], return_cores=return_cores)

    G = _find_neighbors(data, time_thresh, traj_cols, dist_thresh,
                False, use_datetime, use_lon_lat, return_trees=False, relabel_nodes=True)
    
    cluster_df = pd.Series(-2, index=G, name='cluster')
    core_df = pd.Series(-2, index=G, name='core')
    if return_cores:
        promotion_time = pd.Series(np.nan, index=G, name='promotion_time')
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
                if return_cores:
                    promotion_time[i] = i
                S = [(neighbor, i) for neighbor in G[i]]
                while S:
                    j, promoting_core = S.pop()
                    if cluster_df[j] < 0:  # Process if not yet in a cluster
                        cluster_df[j] = cid
                        if return_cores:
                            promotion_time[j] = promoting_core
                        if len(G[j]) >= min_pts:
                            core_df[j] = cid  # Assign core label
                            if return_cores:
                                promotion_time[j] = j
                            for k in G[j]:
                                if cluster_df[k] < 0:
                                    S.append((k, j))
                                    
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
        border_mask = core_df < 0
        cluster_df.loc[border_mask] = -1
        if return_cores:
            promotion_time.loc[border_mask] = np.nan
        prev_run_end = -np.inf                           # left bound (exclusive)
        
        run_label = None
        run_end = None
        run_neighbors = set()                            # union of neighbors of cores in current run
        run_neighbor_promoters = {}
        
        for t in core_df.index[core_df >= 0]:
            lab = core_df.at[t]
        
            if run_label is None:
                run_label = lab
                run_end = t
                run_neighbors.clear()
                run_neighbors.update(G[t])
                run_neighbor_promoters = {nb: t for nb in G[t]}
                continue
        
            if lab == run_label:
                run_end = t
                run_neighbors.update(G[t])
                for nb in G[t]:
                    run_neighbor_promoters.setdefault(nb, t)
                continue
        
            # label changed => t is the start of the next run, so flush current run now
            next_run_start = t
            max_assigned = prev_run_end
        
            for nb in run_neighbors:
                if prev_run_end < nb < next_run_start and cluster_df.at[nb] == -1:
                    cluster_df.at[nb] = run_label
                    if return_cores:
                        promotion_time.at[nb] = run_neighbor_promoters[nb]
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
            run_neighbor_promoters = {nb: t for nb in G[t]}
        
        # flush last run to +inf
        next_run_start = np.inf
        max_assigned = prev_run_end
        
        for nb in run_neighbors:
            if prev_run_end < nb < next_run_start and cluster_df.at[nb] == -1:
                cluster_df.at[nb] = run_label
                if return_cores:
                    promotion_time.at[nb] = run_neighbor_promoters[nb]
                if nb > max_assigned:
                    max_assigned = nb
        
        if run_end is not None and run_end > max_assigned:
            max_assigned = run_end
        prev_run_end = max_assigned
            
    if not return_cores:
        return cluster_df.set_axis(data.index)

    original_times = pd.Series(data[traj_cols[t_key]].to_numpy(), index=G)
    promotion_time = promotion_time.map(original_times)
    if not use_datetime:
        promotion_time = promotion_time.astype('Int64')
    output = pd.DataFrame(
        {'cluster': cluster_df, 'core': core_df, 'promotion_time': promotion_time}
    ).set_axis(data.index)
    return output
       
def ta_dbscan(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    remove_overlaps=True,
    complete_output=False,
    passthrough_cols=None,
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
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    if data.empty:
        return utils._get_empty_stop_df(
            data,
            complete_output,
            passthrough_cols,
            traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
            **kwargs,
        )

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
    return utils.summarize_stops(
        data,
        labels,
        complete_output=complete_output,
        dur_min=dur_min,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
        **kwargs,
    )

def ta_dbscan_per_user(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run ta_dbscan on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("ta_dbscan_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]

    def process_user_group(group):
        return ta_dbscan(
            data=group[1].reset_index(drop=True),
            dist_thresh=dist_thresh,
            min_pts=min_pts,
            time_thresh=time_thresh,
            dur_min=dur_min,
            complete_output=complete_output,
            passthrough_cols=pt_cols,
            traj_cols=traj_cols,
            **kwargs
        )

    grouped = data.groupby(uid, sort=False, as_index=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )
    return pd.concat(results, ignore_index=True)


def ta_dbscan_labels_per_user(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    return_cores=False,
    remove_overlaps=True,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run ta_dbscan_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("ta_dbscan_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    def process_user_group(group):
        return ta_dbscan_labels(
            data=group[1],
            dist_thresh=dist_thresh,
            min_pts=min_pts,
            time_thresh=time_thresh,
            return_cores=return_cores,
            remove_overlaps=remove_overlaps,
            traj_cols=traj_cols,
            **kwargs,
        )

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )

    if return_cores:
        return pd.concat(results).reindex(data.index)
    return pd.concat(results).reindex(data.index)

def dbstop_labels(data,
                 dist_thresh,
                 min_pts,
                 time_thresh,
                 return_cores=False,
                 traj_cols=None,
                 **kwargs):
    """
    Return density-based stop labels.

    Parameters
    ----------
    return_cores : bool, default False
        Return core labels and ``promotion_time`` with cluster labels.
        ``promotion_time`` is the sweep time that propagates final membership.

    Notes
    -----
    ``promotion_time`` records approximate final-membership propagation time.
    For plotting, accent a ping at ``max(ping_time, promotion_time)``. Its raw
    value can show propagation edges from cores, including to later pings.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    if data.empty:
        return utils._get_empty_aux_df(data[traj_cols[t_key]], return_cores=return_cores)

    G, t_tree, s_tree = _find_neighbors(data,  time_thresh,  traj_cols,  dist_thresh, False,  use_datetime,  use_lon_lat,  return_trees=True, relabel_nodes=True)
    node_times = np.asarray(list(G), dtype=np.float64)

    cluster_df = pd.Series(-2, index=G, name='cluster')
    core_df = pd.Series(-2, index=G, name='core')
    if return_cores:
        promotion_time = pd.Series(np.nan, index=G, name='promotion_time')
    past_cutoff = next(iter(G))  # for querying and relabeling neighbors
    candidate_cutoff = past_cutoff  # useful for splitting border points when a new cluster is formed
    prev_core = -1
    active_cid = -1

    def _expand_active_cluster(seed_time, cutoff_time):
        seed_was_core = core_df.at[seed_time] >= 0
        cluster_df.at[seed_time] = active_cid
        core_df.at[seed_time] = active_cid
        if not seed_was_core:
            if return_cores:
                promotion_time.at[seed_time] = seed_time
        for nb in G[seed_time]:
            if cutoff_time <= nb:
                was_clustered = cluster_df.at[nb] >= 0
                was_core = core_df.at[nb] >= 0
                cluster_df.at[nb] = active_cid
                if return_cores and not was_clustered:
                    promotion_time.at[nb] = max(nb, seed_time)
                if len(G[nb]) >= min_pts:
                    core_df.at[nb] = active_cid
                    if return_cores and not was_core:
                        promotion_time.at[nb] = max(nb, seed_time)

    for curr_time in G:
        curr_is_core = (core_df.at[curr_time] >= 0) or (len(G[curr_time]) >= min_pts)
        if not curr_is_core:
            reachable = (cluster_df.at[curr_time] == active_cid)
            core_df.at[curr_time] = -1
            if reachable:
                candidate_cutoff = curr_time
            else:  # previous labels not reachable, so it is noise
                cluster_df.at[curr_time] = -1
                if return_cores:
                    promotion_time.at[curr_time] = np.nan
        else:
            # Future-labeled neighbors can keep continuity for A-C-B style orderings.
            reachable = (active_cid >= 0 and core_df.at[curr_time] == active_cid)
            if not reachable and active_cid >= 0:
                for nb in G[curr_time]:
                    if nb > curr_time and core_df.at[nb] == active_cid:
                        reachable = True
                        break

            new_active_cluster = False
            if reachable:
                candidate_cutoff = curr_time
                prev_core = curr_time
                _expand_active_cluster(curr_time, past_cutoff)

            elif active_cid > -1:
                # compare observed core-time radius to an interpolated continuity baseline
                future_core = core_df[(core_df.index > curr_time) & (core_df == active_cid)].index.min()
                if pd.notna(future_core):
                    core_time_range = sorted(abs(nb - curr_time) for nb in G[curr_time])[min_pts - 1]
                    prev_pos, future_pos = np.searchsorted(node_times, [prev_core, future_core])
                    coord_cols = [traj_cols[coord_key1], traj_cols[coord_key2]]
                    prev_coords = data[coord_cols].iloc[prev_pos].to_numpy(dtype=np.float64)
                    future_coords = data[coord_cols].iloc[future_pos].to_numpy(dtype=np.float64)
                    counterfactual_coords = prev_coords + ((curr_time - prev_core) / (future_core - prev_core)) * (
                        future_coords - prev_coords
                    )

                    if use_lon_lat:
                        spatial_nb_idx = s_tree.query_radius(
                            # _find_neighbors builds BallTree in [lat, lon] radians.
                            np.radians(counterfactual_coords[[1, 0]]).reshape(1, -1),
                            r=dist_thresh / constants.EARTH_RADIUS_METERS,
                        )[0]
                    else:
                        spatial_nb_idx = s_tree.query_radius(
                            np.asarray(counterfactual_coords).reshape(1, -1),
                            r=dist_thresh,
                        )[0]

                    if len(spatial_nb_idx) >= min_pts:
                        counterfactual_time_range = np.sort(np.abs(node_times[spatial_nb_idx] - curr_time))[min_pts - 1]
                        new_active_cluster = (core_time_range <= counterfactual_time_range)
                    else:
                        new_active_cluster = False
                else:
                    new_active_cluster = True
            else:  # not reachable, and first core point
                new_active_cluster = True

            if new_active_cluster:
                # new active cluster branch
                past_cutoff = candidate_cutoff
                candidate_cutoff = curr_time
                active_cid = active_cid + 1
                prev_core = curr_time
                _expand_active_cluster(curr_time, past_cutoff)
            else:
                if not reachable:
                    core_df.at[curr_time] = -1
                    cluster_df.at[curr_time] = -1
                    if return_cores:
                        promotion_time.at[curr_time] = np.nan

    if not return_cores:
        return cluster_df.set_axis(data.index)

    original_times = pd.Series(data[traj_cols[t_key]].to_numpy(), index=G)
    promotion_time = promotion_time.map(original_times)
    if not use_datetime:
        promotion_time = promotion_time.astype('Int64')
    output = pd.DataFrame(
        {'cluster': cluster_df, 'core': core_df, 'promotion_time': promotion_time}
    ).set_axis(data.index)
    return output
       
def dbstop(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
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
    ValueError if multi-user data detected; use dbstop_per_user instead.
    """
    # Use a fresh list per call so future in-place additions cannot leak across calls.
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    if data.empty:
        return utils._get_empty_stop_df(
            data,
            complete_output,
            passthrough_cols,
            traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
            **kwargs,
        )

    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        first = arr[0]
        if any(x != first for x in arr[1:]):
            raise ValueError("Multi-user data? Use dbstop_per_user instead.")
        if traj_cols_temp['user_id'] not in passthrough_cols:
            passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]

    labels = dbstop_labels(
        data=data,
        dist_thresh=dist_thresh,
        min_pts=min_pts,
        time_thresh=time_thresh,
        return_cores=False,
        traj_cols=traj_cols,
        **kwargs
    )
    return utils.summarize_stops(
        data,
        labels,
        complete_output=complete_output,
        dur_min=dur_min,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
        **kwargs,
    )

def dbstop_per_user(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    keep_col_names=True,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run dbstop on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("dbstop_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]

    def process_user_group(group):
        return dbstop(
            data=group[1].reset_index(drop=True),
            dist_thresh=dist_thresh,
            min_pts=min_pts,
            time_thresh=time_thresh,
            dur_min=dur_min,
            complete_output=complete_output,
            passthrough_cols=pt_cols,
            keep_col_names=keep_col_names,
            traj_cols=traj_cols,
            **kwargs
        )

    grouped = data.groupby(uid, sort=False, as_index=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress
    )
    return pd.concat(results, ignore_index=True)


def dbstop_labels_per_user(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    return_cores=False,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run dbstop_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("dbstop_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    def process_user_group(group):
        return dbstop_labels(
            group[1],
            dist_thresh=dist_thresh,
            min_pts=min_pts,
            time_thresh=time_thresh,
            return_cores=return_cores,
            traj_cols=traj_cols,
            **kwargs
        )

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress
    )

    if return_cores:
        return pd.concat(results).reindex(data.index)
    return pd.concat(results).reindex(data.index)
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
    """
    Return SeqScan labels.

    Parameters
    ----------
    return_cores : bool, default False
        Return core labels and ``promotion_time`` with cluster labels.
        ``promotion_time`` is the scan time when final membership is retained.

    Notes
    -----
    ``promotion_time`` records approximate final-membership propagation time.
    For plotting, accent a ping at ``max(ping_time, promotion_time)``. Its raw
    value can show propagation edges from cores, including to later pings.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")

    if user_id is not None:
        data = data.loc[data["user_id"] == user_id].copy()

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(
        data.columns, traj_cols, kwargs
    )        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    if data.empty:
        return utils._get_empty_aux_df(data[traj_cols[t_key]], return_cores=return_cores)

    G = _find_neighbors(data, time_thresh, traj_cols, dist_thresh,
                False, use_datetime, use_lon_lat, return_trees=False, relabel_nodes=True)
    cluster_df = pd.Series(-2, index=G, name='cluster')
    core_df = pd.Series(-2, index=G, name='core')
    if return_cores:
        promotion_time = pd.Series(np.nan, index=G, name='promotion_time')

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
            if return_cores:
                promotion_time[t] = t

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
                    if return_cores:
                        promotion_time[s] = t
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
                            was_clustered = cluster_df[nb] >= 0
                            cluster_df[nb] = core_df[s]
                            if return_cores and not was_clustered:
                                promotion_time[nb] = t

                    merged_label = min(nb_labs)
                    core_win = core_df.loc[window]
                    clu_win = cluster_df.loc[window]

                    core_df.loc[core_win.index[core_win.isin(nb_labs)]] = merged_label
                    cluster_df.loc[clu_win.index[clu_win.isin(nb_labs)]] = merged_label
                
                elif cluster_df[s] == -1:
                    if curr_is_core:
                        core_df[s] = core_df[t]
                        cluster_df[s] = cluster_df[t]
                        if return_cores:
                            promotion_time[s] = t
                    else:
                        temp_cid += 1
                        core_df[s] = temp_cid
                        cluster_df[s] = temp_cid
                        if return_cores:
                            promotion_time[s] = t

                        for nb in temp_G[s]:
                            was_clustered = cluster_df[nb] >= 0
                            cluster_df[nb] = core_df[s]
                            if return_cores and not was_clustered:
                                promotion_time[nb] = t
            else:
                for nb in reversed(list(temp_G[s])):
                    if core_df[nb] >= 0:
                        cluster_df[t] = core_df[nb]
                        if return_cores:
                            promotion_time[t] = t
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
                keep_core_idx = keep_idx[core_df.loc[keep_idx] >= 0]
                if return_cores:
                    keep_promotion_time = promotion_time.loc[keep_idx].copy()
                
                end = spans.at[c, "last"]
                new_cluster = (c != active_cid)

                if new_cluster:
                    if active_cid != -1:
                        first = spans.at[c, "first"]
                        prev_border_idx = clu_win.index[(clu_win == active_cid) & (clu_win.index <= first)]
                        if return_cores:
                            prev_border_promotion_time = promotion_time.loc[prev_border_idx].copy()
                            
                    active_cid += 1
                    start = start_time

                # cleanup of labels in (start_time, t); then restore the new active cluster labels
                cluster_df.loc[window] = -1
                core_df.loc[window] = -1
                if return_cores:
                    promotion_time.loc[window] = np.nan
                
                if new_cluster and active_cid>0:
                    cluster_df.loc[prev_border_idx] = active_cid - 1
                    if return_cores:
                        promotion_time.loc[prev_border_idx] = prev_border_promotion_time
                
                cluster_df.loc[keep_idx] = active_cid
                core_df.loc[keep_core_idx] = active_cid
                if return_cores:
                    promotion_time.loc[keep_idx] = keep_promotion_time

                temp_cid = active_cid
                return True
                # vars changed: temp_neighbors_df, core_df, cluster_df, active_cid, end, temp_cid
        ###### End of def find_cluster

    for curr_time in G:
        # mark as visited. core relabeling happens later.
        cluster_df.at[curr_time] = -1
        core_df.at[curr_time] = -1
        if return_cores:
            promotion_time.at[curr_time] = np.nan
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
                    if return_cores:
                        promotion_time[curr_time] = curr_time
                    break
                    
            if curr_is_core and is_reachable:
                core_df[curr_time] = active_cid
                if return_cores:
                    promotion_time[curr_time] = curr_time
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
    temporary_mask = cluster_df > active_cid
    cluster_df.loc[temporary_mask] = -1
    core_df.loc[core_df > active_cid] = -1
    if return_cores:
        promotion_time.loc[temporary_mask] = np.nan

    if not return_cores:
        return cluster_df.set_axis(data.index)

    original_times = pd.Series(data[traj_cols[t_key]].to_numpy(), index=G)
    promotion_time = promotion_time.map(original_times)
    if not use_datetime:
        promotion_time = promotion_time.astype('Int64')
    output = pd.DataFrame(
        {'cluster': cluster_df, 'core': core_df, 'promotion_time': promotion_time}
    ).set_axis(data.index)
    return output
    
def seqscan(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
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
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    if data.empty:
        return utils._get_empty_stop_df(
            data,
            complete_output,
            passthrough_cols,
            traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
            **kwargs,
        )

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
    return utils.summarize_stops(
        data,
        labels,
        complete_output=complete_output,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
        **kwargs,
    )


def seqscan_per_user(
    data,
    dist_thresh,
    min_pts,
    time_thresh,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    keep_col_names=True,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run seqscan on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("seqscan_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]

    def process_user_group(group):
        return seqscan(
            data=group[1].reset_index(drop=True),
            dist_thresh=dist_thresh,
            min_pts=min_pts,
            time_thresh=time_thresh,
            dur_min=dur_min,
            complete_output=complete_output,
            passthrough_cols=pt_cols,
            keep_col_names=keep_col_names,
            traj_cols=traj_cols,
            **kwargs,
        )

    grouped = data.groupby(uid, sort=False, as_index=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )
    return pd.concat(results, ignore_index=True)


def seqscan_labels_per_user(
    data,
    dist_thresh,
    dur_min=5,
    time_thresh=90,
    min_pts=3,
    return_cores=False,
    traj_cols=None,
    back_merge=False,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run seqscan_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("seqscan_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    def process_user_group(group):
        return seqscan_labels(
            data=group[1],
            dist_thresh=dist_thresh,
            dur_min=dur_min,
            time_thresh=time_thresh,
            min_pts=min_pts,
            return_cores=return_cores,
            traj_cols=traj_cols,
            back_merge=back_merge,
            **kwargs,
        )

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )

    if return_cores:
        return pd.concat(results).reindex(data.index)
    return pd.concat(results).reindex(data.index)
def _compute_core_distance(G, min_pts):
    result = {}
    
    for node in G.nodes():
        edges = sorted(G.edges(node, data='weight'), key=lambda e: e[2])
        result[node] = edges[min_pts - 1][2] if len(edges) >= min_pts else np.inf
    
    core_distances = pd.Series(result)
    core_distances.index.name = 'time'
    return core_distances
   
# pass non core points of parent
# each parent has some border points at any given time
# upon splitting, might have new border points in the same way as dbstop, assigns labels of border points
# once written as permanent labels in hierarchy, they can be used for stability calculation and logged to
# then correspond to their own children, ensuring at each level, nodes are disjoint -- non-overlap 
def _borders_from_cores(scale, core_set, core_distances, G, parent_borders=None):
    """
    Assign each non-core node to its nearest core (by edge weight) within `scale`,
    checking only the temporally adjacent predecessor and successor core.
    Returns {core_ts: set(border_ts)}.

    parent_borders : set or None
        If provided, only these timestamps are considered as candidate border points.
        None means unrestricted (all non-core points at this scale are candidates).
    """
    cores = np.asarray(sorted(core_set))
    if parent_borders is None:
        border_ts = core_distances.index[core_distances > scale].to_numpy()
    else:
        border_ts = np.asarray(sorted(parent_borders))

    if cores.size == 0 or border_ts.size == 0:
        return defaultdict(set)

    border_to_best = {}
    for b in border_ts:
        pos = np.searchsorted(cores, b)
        candidates = []
        if pos > 0:
            candidates.append(cores[pos - 1])
        if pos < cores.size:
            candidates.append(cores[pos])
        for c in candidates:
            if not G.has_edge(b, c):
                continue
            w = np.round(G.edges[b, c]['weight'] * 4) / 4
            if w <= scale and (b not in border_to_best or w < border_to_best[b][1]):
                border_to_best[b] = (c, w)

    result = defaultdict(set)
    for b, (c, _) in border_to_best.items():
        result[c].add(b)
    return result

def cluster_hierarchy(edges_sorted, core_distances, G, H, min_cluster_size,
                      data, coord_col1, coord_col2, use_lon_lat, s_tree, node_times,
                      dist_thresh, dur_min=5, min_pts=2):
    """
    Builds a cluster hierarchy from a pre-computed Minimum Spanning Tree.

    Iteratively removes edges from the MST (from largest to smallest weight)
    to form a hierarchy of clusters. Uses a chronological border-point
    assignment strategy to test for cluster spuriousness.

    Parameters
    ----------
    edges_sorted : pd.Series
        The MST with self-loops, indexed by ('from', 'to') and sorted
        descending by weight (which represents distance/scale).
    core_distances : pd.Series
        Sorted Series mapping each timestamp to its core distance.
    G : nx.Graph
        Weighted graph of distances between temporally-close points.
    H : nx.Graph
        Precomputed hierarchy graph (MST plus self-loops).
    min_cluster_size : int
        Minimum number of core points for a cluster to be considered valid.
    dur_min : int
        Minimum duration in minutes for a cluster to be considered valid.

    Returns
    -------
    tuple
        (label_history_df, hierarchy_df)
    """
    hierarchy = []
    label_history = []

    # Build full ping index once and reuse it for label snapshots.
    all_pings = pd.Index(G.nodes(), name='time')
    nx.set_node_attributes(H, 0, 'cluster_id')
    nx.set_node_attributes(H, -1, 'temp_cluster_id')

    # Initial state is known: all nodes belong to cluster 0.
    label_history.append(pd.DataFrame({
        'time': all_pings,
        'cluster_id': 0,
        'dendogram_scale': np.nan,
    }))

    current_label_id = 1

    # Per-cluster border state:
    #   _cluster_border_map  : cluster_id -> set(border_ts)
    #   _cluster_birth_scale : cluster_id -> float (scale at which cluster was born)
    #
    # When a cluster splits, each child inherits only the borders whose nearest core
    # fell inside that child's component.
    _cluster_border_map   = {0: set()}
    _cluster_birth_scale  = {0: np.inf}  # root has no meaningful birth scale

    def _is_non_spurious(component_nodes, border_nodes=None):
        if not component_nodes:
            return False
        border_nodes = set() if border_nodes is None else set(border_nodes)
        span_nodes = set(component_nodes) | border_nodes
        return (
            (max(span_nodes) - min(span_nodes)) >= dur_min * 60
            and len(component_nodes) >= min_cluster_size
        )

    initial_components = [set(component) for component in nx.connected_components(H)]
    if len(initial_components) > 1:
        new_ids = []
        nodes_to_drop = set()
        for component in initial_components:
            core_nodes = {node for node in component if np.isfinite(core_distances.at[node])}
            border_nodes = component - core_nodes
            if _is_non_spurious(core_nodes, border_nodes):
                child_id = current_label_id
                current_label_id += 1
                for node in component:
                    H.nodes[node]['cluster_id'] = child_id
                _cluster_birth_scale[child_id] = np.inf
                _cluster_border_map[child_id] = set(border_nodes)
                new_ids.append(child_id)
            else:
                nodes_to_drop.update(component)

        if nodes_to_drop:
            H.remove_nodes_from(nodes_to_drop)

        if new_ids:
            hierarchy.append((np.inf, 0, new_ids))

        cluster_ids = pd.Series(nx.get_node_attributes(H, 'cluster_id'))
        cluster_ids = cluster_ids.reindex(all_pings, fill_value=-1)
        label_history.append(pd.DataFrame({
            'time': cluster_ids.index,
            'cluster_id': cluster_ids.values,
            'dendogram_scale': np.inf,
        }))
 
    # Iteratively process pruning events grouped by weight (scale)
    for scale, edges_to_remove in edges_sorted.groupby(edges_sorted, sort=False):
        edges_batch = list(edges_to_remove.index)
        idx_from = edges_to_remove.index.get_level_values('from')
        idx_to = edges_to_remove.index.get_level_values('to')
        event_nodes = idx_from.union(idx_to)

        # Remove both regular edges and self-loops at this scale.
        H.remove_edges_from(edges_batch)
        # Remove edges, everything has temp_cluster_id of -1
        nx.set_node_attributes(H, -1, 'temp_cluster_id')

        _split_entries = {}
        _parent_comp_count = defaultdict(int)

        # Assigns a temp_id from 0 to k for each component, where k is the number of children of a single parent_id
        # Drops non-cores
        for seed in event_nodes:
            if not H.has_node(seed):
                continue

            if H.nodes[seed]['temp_cluster_id'] != -1:
                continue

            parent_id = H.nodes[seed]['cluster_id']
            component_nodes = nx.node_connected_component(H, seed)

            if len(component_nodes) == 1:
                node = next(iter(component_nodes))
                if not H.has_edge(node, node):
                    H.remove_node(node)
                    continue

            temp_id = _parent_comp_count[parent_id]
            # default dict guards against missing keys, so this is safe even if parent_id is new (defaults to 0).
            _parent_comp_count[parent_id] += 1
            for node in component_nodes:
                H.nodes[node]['temp_cluster_id'] = temp_id
                _split_entries[node] = (parent_id, temp_id)

        # split_df: DataFrame indexed by sorted timestamp, columns = (parent_id, temp_id).
        # parent_id identifies the cluster being split; temp_id identifies which sub-component the node belongs to.
        split_df = pd.DataFrame.from_dict(
            _split_entries, orient='index', columns=['parent_id', 'temp_id']
        ).sort_index()

        all_cores_at_scale = core_distances.index[core_distances <= scale]
        raw_at_scale = _borders_from_cores(scale, all_cores_at_scale, core_distances, G)
        root_core_set = set(split_df.index[split_df['parent_id'] == 0])
        _cluster_border_map[0] = set().union(*(raw_at_scale.get(ts, set()) for ts in root_core_set))

        for parent_id in split_df['parent_id'].unique():
            # once we have one parent id, subset to only the children of that parent
            # then do the remove overlaps logic only with sorted timestamps and temporary label and neighbors
            children_df = split_df[split_df['parent_id'] == parent_id]
            components = [set(grp.index) for _, grp in children_df.groupby('temp_id')]

            parent_core_set = set(children_df.index)
            parent_borders = _cluster_border_map[parent_id]

            # union of core timestamps and border timestamps for this parent cluster
            all_ts = sorted(parent_core_set | parent_borders)

            # initialize to -1
            cluster_df = pd.Series(-1, index=all_ts, name='cluster')
            cluster_df.loc[children_df.index] = parent_id

            if len(components) >= 2:
                # Iterate chronologically; 'active_temp_id' is the current main thread.
                # The check fires on the first step away from the active component
                # (prev_temp_id == active but curr_temp_id != active).
                # Separating "decide" (top if) from "advance window" (bottom if) avoids
                # duplicating the coordinate update in both the normal and switch branches.
                active_temp_id = children_df.iloc[0]['temp_id']
                prev_core = None; prev_coords = None
                # prev_prev is needed because prev_core == check_time (the point under evaluation).
                # Using check_time as the interpolation anchor would make alpha=0 and collapse the
                # counterfactual to the actual position, making the test trivially pass every time.
                # prev_prev_core is the preceding active-cluster core point, giving a real anchor.
                prev_prev_core = None; prev_prev_coords = None
                prev_temp_id = None

                for curr_time in children_df.index:
                    curr_temp_id = children_df.at[curr_time, 'temp_id']

                    if curr_temp_id != active_temp_id and prev_temp_id == active_temp_id:
                        # First step away from active: evaluate the transition.
                        check_time  = prev_core
                        future_core = curr_time

                        # now we're confident that everything in children_df is a core point - no need to do checks

                        core_time_range = sorted(abs(nb - check_time) for nb in G[check_time])[min_pts - 1]

                        anchor        = prev_prev_core   if prev_prev_core   is not None else check_time
                        anchor_coords = prev_prev_coords if prev_prev_coords is not None else prev_coords

                        # should already be easy to find? we already have sorted df and active id. 
                        # think about this - should future_pos just be the next_one
                        future_pos    = np.searchsorted(node_times, future_core)
                        future_coords = data[[coord_col1, coord_col2]].iloc[future_pos].to_numpy(dtype=np.float64)

                        denom = future_core - anchor
                        alpha = (check_time - anchor) / denom if denom != 0 else 0.0
                        counterfactual_coords = anchor_coords + alpha * (future_coords - anchor_coords)

                        if dist_thresh is None:
                            new_active_cluster = False
                        else:
                            if use_lon_lat:
                                spatial_nb_idx = s_tree.query_radius(
                                    # _find_neighbors builds BallTree in [lat, lon] radians.
                                    np.radians(counterfactual_coords[[1, 0]]).reshape(1, -1),
                                    r=dist_thresh / constants.EARTH_RADIUS_METERS,
                                )[0]
                            else:
                                spatial_nb_idx = s_tree.query_radius(
                                    np.asarray(counterfactual_coords).reshape(1, -1),
                                    r=dist_thresh,
                                )[0]

                            if len(spatial_nb_idx) >= min_pts:
                                counterfactual_time_range = np.sort(np.abs(node_times[spatial_nb_idx] - check_time))[min_pts - 1]
                                new_active_cluster = (core_time_range <= counterfactual_time_range)
                            else:
                                new_active_cluster = False

                        if not new_active_cluster:
                            # is split_df the equivalent to core_df?
                            # do we drop from H?
                            split_df.at[check_time, 'parent_id'] = -1
                            cluster_df.at[check_time] = -1
                        else:
                            # No explicit expand needed: all core points in children_df are already
                            # labeled parent_id (set before this loop), and border points are
                            # assigned in the post-loop block below. Switching active_temp_id is
                            # the only action required, equivalent to incrementing active_cid in dbstop.
                            active_temp_id = curr_temp_id

                    if curr_temp_id == active_temp_id:
                        prev_prev_core, prev_prev_coords = prev_core, prev_coords
                        prev_core = curr_time
                        _pos = np.searchsorted(node_times, curr_time)
                        prev_coords = data[[coord_col1, coord_col2]].iloc[_pos].to_numpy(dtype=np.float64)

                    prev_temp_id = curr_temp_id

            non_spurious = []
            nodes_to_drop = set()

            for component_nodes in components:
                border_nodes = set()
                for core_ts in component_nodes:
                    border_nodes.update(raw_at_scale.get(core_ts, set()))

                comp_min = min(component_nodes)
                comp_max = max(component_nodes)
                if border_nodes:
                    comp_min = min(comp_min, min(border_nodes))
                    comp_max = max(comp_max, max(border_nodes))

                if _is_non_spurious(component_nodes, border_nodes):
                    non_spurious.append(component_nodes)
                else:
                    nodes_to_drop.update(component_nodes)

            if nodes_to_drop:
                H.remove_nodes_from(nodes_to_drop)

            if len(non_spurious) == 0:
                continue

            if len(non_spurious) == 1:
                # Remaining child already has parent_id.
                continue

            # around here, we are querying cluster_df and change to new labels
            new_ids = []
            for component in non_spurious:
                for node in component:
                    if H.has_node(node):
                        H.nodes[node]['cluster_id'] = current_label_id

                new_ids.append(current_label_id)
                current_label_id += 1
            
            # Partition the parent's border set among the newly minted children.
            # Each child inherits the subset of the parent's borders reachable from its core points,
            # equivalent to dbstop's _expand_active_cluster labeling non-core neighbors: the
            # non-core points that fall within a child's core neighborhood become that child's borders.
            for component, child_id in zip(non_spurious, new_ids):
                core_set = set(component)
                child_borders = set()
                for core_ts in core_set:
                    candidate = raw_at_scale.get(core_ts, set())
                    child_borders.update(candidate & parent_borders)
                _cluster_birth_scale[child_id] = scale
                _cluster_border_map[child_id] = child_borders

            hierarchy.append((scale, parent_id, new_ids))

        # O(N) per scale: get_node_attributes returns {node: cluster_id}.
        cluster_ids = pd.Series(nx.get_node_attributes(H, 'cluster_id'))
        cluster_ids = cluster_ids.reindex(all_pings, fill_value=-1)

        label_history.append(pd.DataFrame({
            'time': cluster_ids.index,
            'cluster_id': cluster_ids.values,
            'dendogram_scale': scale,
        }))

    # combine label history into one DataFrame
    label_history_df = pd.concat(label_history, ignore_index=True)
    # build cluster lineage for all clusters
    hierarchy_df = _build_cluster_lineage(hierarchy)
    return label_history_df, hierarchy_df


def _build_cluster_lineage(hierarchy):
    """
    Returns a DataFrame with columns: child, parent, scale
    """
    lineage = []
    for scale, parent, children in hierarchy:
        for child in children:
            lineage.append({
                "child": child,
                "parent": parent,
                "scale": scale
            })
    return pd.DataFrame(lineage)


def _base_cdf(eps):
    """
    The standard HDBSCAN stability CDF, equivalent to (1 - 1/eps).
    Handles edge cases for eps=inf (returns 1) and eps=0 (returns 0).
    """
    eps = np.asarray(eps)
    # Create a result array of floats
    res = np.zeros_like(eps, dtype=float)

    # Where eps is not infinite and greater than 0
    valid_mask = (eps != np.inf) & (eps > 0)
    res[valid_mask] = 1.0 - (1.0 / eps[valid_mask])

    # Where eps is infinite, the CDF is 1
    res[eps == np.inf] = 1.0

    # Where eps is 0 or invalid, the CDF is 0
    # This is already handled by np.zeros_like initialization

    return res

def compute_cluster_stability(label_history_df, cdf_function=_base_cdf):
    """
    Computes cluster stability using a vectorized approach and a provided CDF.

    This method is significantly faster than iterative approaches by avoiding
    Python loops in favor of pandas' optimized, C-backend operations.

    Parameters
    ----------
    label_history_df : pd.DataFrame
        DataFrame containing the cluster label history for each point at each scale.
        Must have columns ['time', 'cluster_id', 'dendogram_scale'].
    cdf_function : callable, optional
        A function that computes the Cumulative Distribution Function for a given
        epsilon (scale). It should accept a NumPy array and return an array of
        the same shape. Defaults to the standard HDBSCAN stability CDF (1 - 1/eps).

    Returns
    -------
    pd.DataFrame
        A DataFrame with ['cluster_id', 'cluster_stability'] for each valid cluster.
    """

    df = label_history_df[
        (label_history_df['cluster_id'] > 0) &
        (label_history_df['dendogram_scale'].notna())
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=['cluster_id', 'cluster_stability'])

    # 2. For each point-cluster pair, find eps_max (birth scale of the cluster).
    eps_max_map = df.groupby('cluster_id')['dendogram_scale'].max()
    df['eps_max'] = df['cluster_id'].map(eps_max_map)

    # 3. For each 'time', we go from largest scale (birth) to smallest.
    df.sort_values(['time', 'dendogram_scale'], ascending=[True, False], inplace=True)

    # 4. For each point, find the cluster of the *next* step in its timeline.
    # This allows us to detect when a point "exits" a cluster.
    df['next_cluster_id'] = df.groupby('time')['cluster_id'].shift(-1)
    
    # 5. An exit occurs where the cluster_id changes to a different *valid* cluster.
    # Excluding NaN next_cluster_id here prevents double-counting with never_exited below:
    # pandas treats (x != NaN) as True, so without the notna() guard, points that drop to
    # noise would appear in both exit_events and never_exited.
    exit_events = df[
        (df['cluster_id'] != df['next_cluster_id']) & df['next_cluster_id'].notna()
    ].copy()
    exit_events.rename(columns={'dendogram_scale': 'eps_min'}, inplace=True)
    
    # 6. For points that never exit a cluster, eps_min is the last scale at which they appear
    # (the smallest scale the cluster reached). Using inf would make cdf(inf)=1 and give a
    # negative stability term, incorrectly penalising persistent points.
    last_state = df.drop_duplicates(subset='time', keep='last')
    never_exited = last_state[last_state['next_cluster_id'].isna()]

    # Combine the two types of stability events
    stability_points = pd.concat([
        exit_events[['time', 'cluster_id', 'eps_min', 'eps_max']],
        never_exited[['time', 'cluster_id', 'dendogram_scale', 'eps_max']].rename(
            columns={'dendogram_scale': 'eps_min'}
        ),
    ])
    
    # 7. Apply the provided CDF to calculate the stability contribution of each point.
    stability_points['stability_term'] = (
        cdf_function(stability_points['eps_max']) - cdf_function(stability_points['eps_min'])
    )

    # 8. Sum the contributions for each cluster to get the final stability score.
    final_stability = stability_points.groupby('cluster_id')['stability_term'].sum()

    return final_stability.reset_index().rename(columns={'stability_term': 'cluster_stability'})

def select_most_stable_clusters(hierarchy_df, cluster_stability_df):
    # handles error of not finding any parent in the data: returns empty set of selected clusters
    if 'parent' not in hierarchy_df.columns or 'child' not in hierarchy_df.columns or 'scale' not in hierarchy_df.columns:
        return set()
    
    hierarchy = [
        (group['scale'].iloc[0], parent, list(group['child']))
        for parent, group in hierarchy_df.groupby('parent')
    ]

    # Build tree of clusters
    children = defaultdict(list)
    parent = {}
    for _, parent_id, child_ids in hierarchy:
        for child_id in child_ids:
            children[parent_id].append(child_id)
            parent[child_id] = parent_id

    # Stability lookup
    stability_map = dict(zip(cluster_stability_df['cluster_id'], cluster_stability_df['cluster_stability']))
    
    selected_clusters = set()
    best_stability = {}

    # Get descendants of cluster
    def get_descendants(cid):
        descendants = set()
        stack = [cid]
        while stack:
            node = stack.pop()
            for child in children.get(node, []):
                descendants.add(child)
                stack.append(child)
        return descendants

    # DFS
    def dfs(cid):
        if cid not in children:
            best_stability[cid] = stability_map.get(cid, 0.0)
            selected_clusters.add(cid)
            return best_stability[cid]

        sum_children = sum(dfs(child) for child in children[cid])
        own_stab = stability_map.get(cid, 0.0)

        if own_stab >= sum_children:
            best_stability[cid] = own_stab
            # removes elements from the current set that are also present in another iterable
            selected_clusters.difference_update(get_descendants(cid))
            selected_clusters.add(cid)
        else:
            best_stability[cid] = sum_children

        return best_stability[cid]

    # Start from root children
    for cid in children.get(0, []):
        dfs(cid)

    return selected_clusters

def select_clusters_by_epsilon(hierarchy_df, label_history_df, epsilon):
    """
    Select clusters by performing a flat cut at a specific scale in the dendrogram.
    
    Instead of using stability to choose clusters, this method returns all clusters
    that exist at the specified scale threshold.
    
    Parameters
    ----------
    hierarchy_df : pd.DataFrame
        Cluster hierarchy with columns ['child', 'parent', 'scale'].
    label_history_df : pd.DataFrame
        Full label history with columns ['time', 'cluster_id', 'dendogram_scale'].
    cut_scale : float
        The scale at which to cut the dendrogram. All clusters alive at this
        scale will be selected.
    
    Returns
    -------
    set
        Set of cluster IDs that are active at the cut_scale.
    
    Examples
    --------
    >>> # Get clusters at scale 50 meters
    >>> selected = select_clusters_by_epsilon(hierarchy_df, label_history_df, epsilon=50.0)
    """
    if hierarchy_df.empty or label_history_df.empty:
        return set()
    
    if 'parent' not in hierarchy_df.columns or 'child' not in hierarchy_df.columns:
        return set()
    
    # Filter label history to the specified scale
    # Find the closest scale that exists in the data
    available_scales = label_history_df['dendogram_scale'].dropna().unique()
    if len(available_scales) == 0:
        return set()

    # Find the next smallest scale ≤ epsilon
    smaller_scales = available_scales[available_scales <= epsilon]
    
    if len(smaller_scales) == 0:
        # No scales below epsilon
        return set()
    else:
        # largest scale that is ≤ epsilon
        closest_scale = smaller_scales.max()

    # Get all clusters that exist at this scale
    clusters_at_scale = label_history_df[
        (label_history_df['dendogram_scale'] == closest_scale) &
        (label_history_df['cluster_id'] > 0)
    ]['cluster_id'].unique()
    
    return set(clusters_at_scale)

def _build_hdbscan_graphs(G, core_dist):
    """
    Computes all graphs required for the HDBSCAN algorithm in one pass.
    Uses precomputed edge weights from G instead of recomputing distances.

    Returns
    -------
    H : nx.Graph
        Hierarchy graph with mutual-reachability MST edges and core-distance
        self-loops.
    edges_sorted_df : pd.Series
        H sorted descending by weight, MultiIndex (from, to).
    """
    G_copy = G.copy()
    for u, v, data in G_copy.edges(data=True):
        d = np.round(data["weight"] * 4) / 4
        data["weight"] = max(core_dist.at[u], core_dist.at[v], d)

    H = nx.minimum_spanning_tree(G_copy)

    H.add_edges_from((node, node, {'weight': weight}) for node, weight in core_dist.items())

    all_edges = nx.to_pandas_edgelist(H, source='from', target='to')
    all_edges.sort_values('weight', ascending=False, inplace=True)

    all_edges.set_index(['from', 'to'], inplace=True)
    return H, all_edges['weight']

def hdbscan_labels(data,
                   time_thresh,
                   min_pts = 2,
                   min_cluster_size = 1,
                   dur_min=5,
                   delta_roam=None,
                   dist_thresh=None,
                   return_cores=False,
                   traj_cols=None, **kwargs):
    """
    Compute HDBSCAN cluster labels for trajectory data, with core/border assignment.

    Parameters
    ----------
    data : pd.DataFrame
        Input trajectory data.
    time_thresh : int
        Maximum allowed time gap (minutes) for temporal neighbors.
    min_pts : int, optional
        Minimum neighbors for a core point (default: 2).
    min_cluster_size : int, optional
        Minimum cluster size for a valid stop (default: 1).
    dur_min : int, optional
        Minimum duration (minutes) for a stop (default: 5).
    return_cores : bool, default False
        Return core labels and ``promotion_time`` with cluster labels. Core
        pings use their own time; border pings use their propagating core time.
    traj_cols : dict, optional
        Mapping for key columns.
    **kwargs
        Passed to internal helpers.

    Returns
    -------
    pd.Series or pd.DataFrame
        Cluster labels, or ``cluster``, ``core``, and ``promotion_time`` when
        ``return_cores`` is true.

    Notes
    -----
    ``promotion_time`` records approximate final-membership propagation time.
    For plotting, accent a ping at ``max(ping_time, promotion_time)``. Its raw
    value can show propagation edges from cores, including to later pings.
    """
    # Check if user wants long and lat and datetime
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)
    # Load default col names
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    
    if traj_cols['user_id'] in data.columns:
        uid_col = data[traj_cols['user_id']]
        arr = uid_col.values
        if len(arr) > 0:
            first = arr[0]
            if any(x != first for x in arr[1:]):
                raise ValueError("Multi-user data? Groupby or use hdbscan_per_user instead.")
    
    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    if data.empty:
        return utils._get_empty_aux_df(data[traj_cols[t_key]], return_cores=return_cores)

    G, t_tree, s_tree = _find_neighbors(data, time_thresh, traj_cols, dist_thresh,
                    weighted=True, use_datetime=use_datetime, use_lon_lat=use_lon_lat,
                    return_trees=True, relabel_nodes=True)
    node_times = np.asarray(list(G), dtype=np.float64)

    core_distances = _compute_core_distance(G, min_pts)

    H, edges_sorted = _build_hdbscan_graphs(G, core_distances)

    label_history_df, hierarchy_df = cluster_hierarchy(
        edges_sorted=edges_sorted,
        core_distances=core_distances,
        G=G,
        H=H,
        min_cluster_size=min_cluster_size,
        min_pts=min_pts,
        data=data,
        coord_col1=traj_cols[coord_key1],
        coord_col2=traj_cols[coord_key2],
        use_lon_lat=use_lon_lat,
        s_tree=s_tree,
        node_times=node_times,
        dist_thresh=dist_thresh,
        dur_min=dur_min,
    )

    if delta_roam is None:
        cluster_stability_df = compute_cluster_stability(label_history_df)
        selected_clusters = select_most_stable_clusters(hierarchy_df, cluster_stability_df)
    else:
        selected_clusters = select_clusters_by_epsilon(hierarchy_df, label_history_df, epsilon=delta_roam)

    final_labels = pd.Series(-1, index=core_distances.index, name='cluster', dtype=int)
    if return_cores:
        core_labels = pd.Series(-1, index=core_distances.index, name='core', dtype=int)
        promotion_time = pd.Series(np.nan, index=core_distances.index, name='promotion_time')
        
    # keep only info of selected clusters and their birthscales, sort from denser to less dense
    cluster_info_df = label_history_df[label_history_df['cluster_id'].isin(selected_clusters)]
    birth_scales = cluster_info_df.groupby('cluster_id')['dendogram_scale'].max()
    cluster_info = birth_scales.sort_values(ascending=True).reset_index().rename(columns={'dendogram_scale': 'scale'})

    claimed_points = set()
    for _, row in cluster_info.iterrows():
        cid, scale = row['cluster_id'], row['scale']
        
        # 1. Identify core members for this cluster at its birth scale
        # These are points that are part of the cluster and have not been claimed by a denser cluster
        core_mask = (label_history_df['cluster_id'] == cid) & \
                    (label_history_df['dendogram_scale'] == scale)
        core_members = set(label_history_df.loc[core_mask, 'time'].unique())
        
        # Exclude points already claimed by a denser cluster (should be rare for cores, but good practice)
        unclaimed_cores = core_members - claimed_points

        # 2. Find border points for these unclaimed cores at this scale
        all_cores_at_scale = core_distances.index[core_distances <= scale]
        border_map = _borders_from_cores(scale, all_cores_at_scale, core_distances, G)
        potential_borders = set().union(*(border_map.get(ts, set()) for ts in unclaimed_cores))
        unclaimed_borders = potential_borders - claimed_points

        # 3. Assign labels and update claimed set
        all_new_members = unclaimed_cores.union(unclaimed_borders)
        
        if all_new_members:
            sorted_cores = sorted(unclaimed_cores)
            if return_cores:
                core_labels.loc[sorted_cores] = cid
                promotion_time.loc[sorted_cores] = sorted_cores
                for core_ts in sorted_cores:
                    borders = sorted(border_map.get(core_ts, set()) & unclaimed_borders)
                    if borders:
                        promotion_time.loc[borders] = core_ts
            final_labels.loc[list(all_new_members)] = cid
            claimed_points.update(all_new_members)

    if not return_cores:
        return final_labels.set_axis(data.index)

    original_times = pd.Series(data[traj_cols[t_key]].to_numpy(), index=G)
    promotion_time = promotion_time.map(original_times)
    if not use_datetime:
        promotion_time = promotion_time.astype('Int64')
    output = pd.DataFrame(
        {
            'cluster': final_labels,
            'core': core_labels,
            'promotion_time': promotion_time,
        }
    ).set_axis(data.index)

    return output

def st_hdbscan(
    data,
    time_thresh,
    min_pts=2,
    min_cluster_size=1,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    traj_cols=None,
    **kwargs
):
    """
    HDBSCAN-based stop detection.

    Parameters
    ----------
    data : pd.DataFrame
        Input trajectory data.
    time_thresh : int
        Maximum allowed time gap (minutes) for temporal neighbors.
    min_pts : int, optional
        Minimum neighbors for a core point (default: 2).
    min_cluster_size : int, optional
        Minimum cluster size for a valid stop (default: 1).
    dur_min : int, optional
        Minimum duration (minutes) for a stop (default: 5).
    complete_output : bool, optional
        If True, include extra stats.
    passthrough_cols : list, optional
        Columns to passthrough to final stop table
    traj_cols : dict, optional
        Mapping for key columns.
    **kwargs
        Passed to internal helpers.

    Returns
    -------
    pd.DataFrame
        Stop table
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        if len(arr) > 0:
            first = arr[0]
            if any(x != first for x in arr[1:]):
                raise ValueError("Multi-user data? Use hdbscan_per_user instead.")
            if traj_cols_temp['user_id'] not in passthrough_cols:
                passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]
    else:
        uid_col = None
        
    labels = hdbscan_labels(
        data=data,
        time_thresh=time_thresh,
        min_pts=min_pts,
        min_cluster_size=min_cluster_size,
        dur_min=dur_min,
        passthrough_cols=passthrough_cols,
        traj_cols=traj_cols,
        **kwargs
    )
    return utils.summarize_stops(
        data,
        labels,
        complete_output=complete_output,
        passthrough_cols=passthrough_cols,
        keep_col_names=True,
        traj_cols=traj_cols,
        **kwargs,
    )

def st_hdbscan_per_user(
    data,
    time_thresh,
    min_pts=2,
    min_cluster_size=1,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run HDBSCAN-based stop detection on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """

    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("st_hdbscan_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]

    def process_user_group(group):
        return st_hdbscan(
            data=group[1].reset_index(drop=True),
            time_thresh=time_thresh,
            min_pts=min_pts,
            min_cluster_size=min_cluster_size,
            dur_min=dur_min,
            complete_output=complete_output,
            passthrough_cols=pt_cols,
            traj_cols=traj_cols,
            **kwargs
        )

    grouped = data.groupby(uid, sort=False, as_index=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )
    return pd.concat(results, ignore_index=True)


def hdbscan_labels_per_user(
    data,
    time_thresh,
    min_pts=2,
    min_cluster_size=1,
    dur_min=5,
    delta_roam=None,
    return_cores=False,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run hdbscan_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("hdbscan_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    def process_user_group(group):
        return hdbscan_labels(
            data=group[1],
            time_thresh=time_thresh,
            min_pts=min_pts,
            min_cluster_size=min_cluster_size,
            dur_min=dur_min,
            delta_roam=delta_roam,
            return_cores=return_cores,
            traj_cols=traj_cols,
            **kwargs,
        )

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )

    # With return_cores, columns retain the cluster/core/promotion_time semantics above.
    return pd.concat(results).reindex(data.index)


__all__ = [
    "ta_dbscan_labels",
    "ta_dbscan",
    "ta_dbscan_per_user",
    "ta_dbscan_labels_per_user",
    "dbstop_labels",
    "dbstop",
    "dbstop_per_user",
    "dbstop_labels_per_user",
    "window_graph",
    "seqscan_labels",
    "seqscan",
    "seqscan_per_user",
    "seqscan_labels_per_user",
    "hdbscan_labels",
    "st_hdbscan",
    "st_hdbscan_per_user",
    "hdbscan_labels_per_user",
]
