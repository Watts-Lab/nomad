import pandas as pd
import numpy as np
import nomad.io.base as loader
import geopandas as gpd
from nomad.stop_detection import utils
from nomad.stop_detection.preprocessing import _find_neighbors

def dbstop_labels(data,
                 dist_thresh,
                 min_pts,
                 time_thresh,
                 return_cores=False,
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
    
    G, t_tree, s_tree = _find_neighbors(data,  time_thresh,  traj_cols,  dist_thresh, False,  use_datetime,  use_lon_lat,  return_trees=True, relabel_nodes=True)
    node_times = np.asarray(list(G), dtype=np.float64)

    cluster_df = pd.Series(-2, index=G, name='cluster')
    core_df = pd.Series(-2, index=G, name='core')
    past_cutoff = next(iter(G))  # for querying and relabeling neighbors
    candidate_cutoff = past_cutoff  # useful for splitting border points when a new cluster is formed
    prev_core = -1
    active_cid = -1

    def _expand_active_cluster(seed_time, cutoff_time):
        cluster_df.at[seed_time] = active_cid
        core_df.at[seed_time] = active_cid
        for nb in G[seed_time]:
            if cutoff_time <= nb:
                cluster_df.at[nb] = active_cid
                if len(G[nb]) >= min_pts:
                    core_df.at[nb] = active_cid

    for curr_time in G:
        curr_is_core = (core_df.at[curr_time] >= 0) or (len(G[curr_time]) >= min_pts)
        if not curr_is_core:
            reachable = (cluster_df.at[curr_time] == active_cid)
            core_df.at[curr_time] = -1
            if reachable:
                candidate_cutoff = curr_time
            else:  # previous labels not reachable, so it is noise
                cluster_df.at[curr_time] = -1
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
                    prev_coords = data.iloc[prev_pos][[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy()
                    future_coords = data.iloc[future_pos][[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy()
                    counterfactual_coords = prev_coords + ((curr_time - prev_core) / (future_core - prev_core)) * (
                        future_coords - prev_coords
                    )

                    if use_lon_lat:
                        spatial_nb_idx = s_tree.query_radius(
                            # _find_neighbors builds BallTree in [lat, lon] radians.
                            np.radians(counterfactual_coords[[1, 0]]).reshape(1, -1),
                            r=dist_thresh / 6_371_000,
                        )[0]
                    else:
                        spatial_nb_idx = s_tree.query_ball_point(counterfactual_coords, r=dist_thresh)

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

    output = pd.DataFrame({'cluster': cluster_df, 'core': core_df})

    if return_cores:
        return output.set_axis(data.index)
    else:
        labels = output.cluster
        return labels.set_axis(data.index)
       
def dbstop(
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
    ValueError if multi-user data detected; use dbstop_per_user instead.
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
        return pd.DataFrame(columns=cols, dtype=object)

    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        first = arr[0]
        if any(x != first for x in arr[1:]):
            raise ValueError("Multi-user data? Use dbstop_per_user instead.")
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
    merged = data.join(labels)
    
    # Filter out noise points
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
            dur_min=dur_min,
            keep_col_names=keep_col_names,
            passthrough_cols=passthrough_cols,
            **kwargs
        ),
        include_groups=False
    )
    
    return stop_table.loc[stop_table['duration']>=dur_min]

def dbstop_per_user(
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
    Run dbstop on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("dbstop_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols + [uid]

    results = [
        dbstop(
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
