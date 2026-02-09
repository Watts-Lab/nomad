import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import pdb
import geopandas as gpd


def seqscan_labels(
    data,
    dist_thresh,
    min_dur, # presence treshold: min time to count as stay
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

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)        
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
    active_cid = None  # or -1, which DBSCAN cluster id is curr stay
    # thus active_cid - 1 is the preceeding cluster id
    temp_c_id = -1 # for internal dbscan until new active cluster is found
    for t in valid_times:
        # current timeContext is (start, t]        
        curr_dur = 0

        if active_cid is None:
            # add point with available neighbors
            temp_neighbor_dict[t] = {nb for nb in neighbor_dict[t] if t > nb >= start}
            t_is_core = len(temp_neighbor_dict[t])>=min_pts
            if t_is_core:
                temp_c_id = temp_c_id + 1
                core_df[t] = temp_c_id
                cluster_df[t] = temp_c_id
                
            # update neighbors and merge
            for s in temp_neighbor_dict[t]:
                temp_neighbor_dict[s].add(t)
                if len(temp_neighbor_dict[s])>=k:
                    if core_df[s] >= 0:
                        # already was a temporary core point
                        # just merge "through" t
                        if t_is_core:
                            cluster_df.loc[core_df==core_df[t]] = core_df[s]
                            core_df.loc[core_df==core_df[t]] = core_df[s]
                        
                    elif cluster_df[s] >= 0:
                        # was already a core's neighbor
                        core_df[s] = cluster_df[s]
                        # propagate label
                        nb_labs = set()
                        for nb in temp_neighbor_dict[s]:
                            if core_df[nb] >= 0:
                                # for later relabel of entire connected component
                                nb_labs.add(core_df[s])
                            else:
                                cluster_df[nb] = core_df[s] # (re) assign border point                            
                        # bulk relabel and merge of all affected connected components
                        core_df.loc[core_df.isin(nb_labs)] = core_df[s]
                        cluster_df.loc[core_df==core_df[s]] = core_df[s]
                        
                    elif cluster_df[s] == -1: # is a new cluster
                        temp_c_id = temp_c_id + 1 # increase temporary label counter
                        core_df[s] = temp_c_id
                        # propagate label
                        for nb in temp_neighbor_dict[s]:
                            # there is no case in which nb is core point: otherwise cluster_df[s] would have had a label
                            cluster_df[nb] = core_df[s]

            # Check for first non-spurious cluster regardless of overlaps, discard anything within it, promote to active cluster
            for c in cluster_df.loc[start, t].loc[lambda s: s >= 0].unique(): # <<< preserves order of appearance 
                clus_ = cluster_df.loc[start, t].loc[lambda s: s == c]
                if (clus_.index.max() - clus_.index.min()) >= min_dur * 60:
                    # c is the next active cluster
                    break
                #find cluster routine end

            active_cid = active_cid + 1
            # disregard all other temporary labels
            cluster_df.loc[start, t].loc[lambda s: s != c] = -1
            core_df.loc[start, t].loc[lambda s: s != c] = -1
            # log permanent label for active cluster
            cluster_df.loc[start, t].loc[lambda s: s == c] = active_cid
            core_df.loc[start, t].loc[lambda s: s == c] = active_cid
            end = clus_.index.max()

        #TODO: when active_cluster, if next iter can be expand as a core point and neighbor to active_cluster, move to end and continue
        #TODO: when not core point, do find_cluster on tail from end to t
        else: # active_id is not None
            # try to attach + merge with previous cluster
            return None
                
        return cluster_df
        
        #TODO: do we want immediately previous cluster to be mergeable with curr cluster if already cut

    