import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import pdb
import geopandas as gpd


def seqscan_labels(
    data,
    dist_thresh,
    min_dur, # min time to count as stay in minutes
    min_pts=3,
    user_id=None,
    x_col="x",
    y_col="y",
    t_col="timestamp",
):
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    if data.empty:
        return pd.DataFrame()

    if user_id is not None:
        data = data.loc[data["user_id"] == user_id].copy()

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(
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
    start = valid_times[0]        # current time context start
    end = valid_times[0] - 1        # current candidate to cut time context
    #find cluster routine start
    temp_neighbor_dict = {} # in time context
    active_cid = -1  # or -1, which DBSCAN cluster id is curr stay
    # thus active_cid - 1 is the preceeding cluster id
    #TODO: global?
    temp_cid = 100000 # for internal dbscan until new active cluster is found

    def findCluster(start_time, t):
        # should ensure only have cluster from time context, should never be older than t
        # want time indices to be >= start
        temp_neighbor_dict = {idx:nbs for idx, nbs in temp_neighbor_dict.items() if idx >= start}
        temp_neighbor_dict[t] = {nb for nb in neighbor_dict[t] if t > nb >= start}
        t_is_core = len(temp_neighbor_dict[t]) >= min_pts
        if t_is_core:
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
                    if t_is_core:
                        cluster_df.loc[core_df==core_df[t]] = core_df[s]
                        core_df.loc[core_df==core_df[t]] = core_df[s]
                    
                elif cluster_df[s] >= 0:
                    core_df[s] = cluster_df[s]
                    if t_is_core:
                        cluster_df.loc[core_df==core_df[t]] = core_df[s]
                        core_df.loc[core_df==core_df[t]] = core_df[s]
                    for nb in temp_neighbor_dict[s]:
                        if core_df[nb]:
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
                                core_df.loc[core_df == lab] = core_df[s]
                                cluster_df.loc[cluster_df == lab] = core_df[s]
                
                elif cluster_df[s] == -1: # is a new cluster
                    if t_is_core:
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


        window = cluster_df.loc[(cluster_df.index >= start_time) & (cluster_df.index <= t)]
        for c in window.loc[window >= 0].unique():
            cluster = window.loc[window == c]
            if (cluster.index.max() - cluster.index.min()) >= (min_dur * 60):
                # return clusterid and end time
                # get rid?
                active_cid = active_cid + 1
                # disregard all other temporary labels
                cluster_df.loc[start, t].loc[lambda s: s != c] = -1
                core_df.loc[start, t].loc[lambda s: s != c] = -1
                # log permanent label for active cluster
                cluster_df.loc[start, t].loc[lambda s: s == c] = active_cid
                core_df.loc[start, t].loc[lambda s: s == c] = active_cid
                end = cluster.index.max()
                start = start_time
                temp_cid = 100000
                # vars changed: temp_neighbors_df, core_df, cluster_df, active_cid, end, temp_cid
                return True
        # vars changed: temp_neighbors_df, core_df, cluster_df
        return False



    for t in valid_times:
        if active_cid == -1:
            findCluster(start, t)
        else:
            t_nb = {nb for nb in neighbor_dict[t] if t > nb >= start}
            t_is_core = len(t_nb) >= min_pts
            if t_is_core:
                for nb in core_df[core_df == active_cid].index:
                    if t in neighbor_dict[nb]:
                        core_df[t] = active_cid
                        end = t
                    else:
                        continue
            else:
                findCluster(end + 1, t)
    cluster_df.loc[cluster_df > 100000] = -1
    core_df.loc[core_df > 100000] = -1
    output = pd.DataFrame({'cluster': cluster_df, 'core': core_df})

    if return_cores:
        return output.set_axis(data.index)
    else:
        labels = output.cluster
        return labels.set_axis(data.index)