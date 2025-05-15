import pandas as pd
import numpy as np
import heapq
import datetime as dt
from datetime import timedelta
import itertools
from collections import defaultdict
import sys
import os
import pdb
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils

##########################################
########        HDBSCAN           ########
##########################################
def _find_bursts(times, time_thresh):
    """
    Find timestamp pairs that are within time threshold.

    Parameters
    ----------
    times : array of timestamps.
    time_thresh : time threshold for finding what timestamps are close in time.

    Returns
    -------
    time_pairs : list of tuples of timestamps [(t1, t2), ...] that are close in time given time_thresh.
    """
    #TC: O(n^2)
    times = np.array(times)
    
    # Pairwise time differences
    # times[:, np.newaxis]: from shape (n,) -> to shape (n, 1) – a column vector
    time_diffs = np.abs(times[:, np.newaxis] - times)
    time_diffs = time_diffs.astype(int)
    
    # Filter by time threshold
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1) # keep upper triangle
    i_idx, j_idx = np.where(within_time_thresh)
    
    # Return a list of (timestamp1, timestamp2) tuples
    time_pairs = [(times[i], times[j]) for i, j in zip(i_idx, j_idx)]
    
    return time_pairs

def _compute_core_distance(traj, time_pairs, min_samples = 2):
    """
    Calculate the core distance for each ping in traj.

    Parameters
    ----------
    traj : dataframe

    time_pairs : tuples of timestamps that are close in time given time_thresh
    
    min_samples : int
        used to calculate the core distance of a point p, where core distance of a point p 
        is defined as the distance from p to its min_samples-th smallest nearest neighbor
        (including itself).

    Returns
    -------
    core_distances : dictionary of timestamps
        {timestamp_1: core_distance_1, ..., timestamp_n: core_distance_n}
    """
    # it gives local density estimate: small core distance → high local density.
    coords = traj[['x', 'y']].to_numpy() # TC: O(n)
    timestamps = traj['timestamp'].to_numpy() # TC: O(n)
    n = len(coords)
    # get the index of timestamp in the arrays (for accessing their value later)
    timestamp_indices = {ts: idx for idx, ts in enumerate(timestamps)} # TC: O(n)

    # Build neighbor map from time_pairs
    neighbors = {ts: set() for ts in timestamps}  # TC: O(n)
    for t1, t2 in time_pairs:
        neighbors[t1].add(t2)
        neighbors[t2].add(t1)
    
    core_distances = {}
    
    for i in range(n): # TC: O(n^2 log n)
        u = timestamps[i]
        allowed_neighbors = neighbors[u]
        dists = np.full(n, 1e9) # use a very large number e.g. 1e9 (not infinity) for edges between points not temporally close
        dists[i] = 0  # distance to itself
        
        for v in allowed_neighbors:
            j = timestamp_indices.get(v)
            if j is not None:
                dists[j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
        
        sorted_dists = np.sort(dists) # TC: O(nlogn)
        core_distances[u] = sorted_dists[min_samples - 1]

    return core_distances

def _compute_mrd_graph(traj, core_distances):
    """
    Mutual Reachability Distance (MRD) of point p and point q is the maximum of: 
        cordistance of p, core distance of q, and the distance between p and q.
    MRD graph will have O(n^2) edges, one between every pair of points.
    
    Explanation:
    - Edges between dense regions are favored, and sparse regions have larger edge weights.
    - MRD will inflate distances so that sparse areas are less likely to form clusters.
    - Even if two points are physically close they may live in very different local densities.

    Parameters
    ----------
    traj : dataframe
    
    core_distances : dict
        Keys are (timestamp1, timestamp2), values are mutual reachability distances.

    Returns
    -------
    mrd_graph : dictionary of timestamp pairs
        {(timestamp1, timestamp2): mrd_value}.
    """
    coords = traj[['x', 'y']].to_numpy()
    timestamps = traj['timestamp'].to_numpy()
    n = len(coords)
    mrd_graph = {}

    # TC: 0(n^2)
    # perhaps use time_pairs to reduce time complexity to O(n+m)
    for i in range(n):
        for j in range(i + 1, n):
            # euclidean distance between point i and j
            dist = np.sqrt(np.sum((coords[j] - coords[i]) ** 2))
            core_i = core_distances[timestamps[i]]
            core_j = core_distances[timestamps[j]]

            mrd_graph[(timestamps[i], timestamps[j])] = max(core_i, core_j, dist)

    return mrd_graph

def _mst(mrd_graph):
    """
    Compute the MST using Prim's algorithm from mutual reachability distances.
    The MST retains the minimum set of edges that connect all points with the lowest maximum distances.
    The MST contains only (n - 1) edges.
    
    Parameters
    ----------
    mrd : dict
        Keys are (timestamp1, timestamp2), values are mutual reachability distances.

    Returns
    -------
    mst : list of tuples
        (t1, t2, mrd_value) for the MST.
    """
    # {t1: (t2, mrd(t1,t2)),
    #  t2: (t1, mrd(t1,t2)), ...}
    graph = defaultdict(list)
    for (t1, t2), mrd in mrd_graph.items():
        graph[t1].append((t2, mrd))
        graph[t2].append((t1, mrd))

    visited = set()
    mst = []
    start_node = next(iter(graph)) # gets the first key from the graph (starting node)
    min_heap = [(0, start_node, start_node)]  # (mrd, from, to)

    while min_heap and len(visited) < len(graph):
        mrd, u, v = heapq.heappop(min_heap)
        
        if v in visited:
            continue
        
        visited.add(v)
        
        if u != v:
            mst.append((u, v, mrd))
        
        for neighbor, next_weight in graph[v]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (next_weight, v, neighbor))

    return mst

def mst_ext(mst, core_distances):
    """
    Add self-edges to MST with weights equal to each point's core distance,
    and return as a sorted DataFrame.

    Parameters
    ----------
    mst : list of (from, to, weight)
    core_distances : dict {timestamp: core_distance}

    Returns
    -------
    pd.DataFrame with columns ['from', 'to', 'weight'], sorted descending
    """
    self_edges = [(ts, ts, core_distances[ts]) for ts in core_distances]

    mst_ext_edges = mst + self_edges

    return pd.DataFrame(mst_ext_edges, columns=["from", "to", "weight"])

def hdbscan(mst_ext_df, min_cluster_size):
    """
    Extracts HDBSCAN hierarchy from the extended MST DataFrame.

    Parameters
    ----------
    mst_ext_df : pd.DataFrame
        DataFrame with columns ['from', 'to', 'weight'], including MST + self-edges.
    min_cluster_size : int
        Minimum number of points required for a valid cluster.

    Returns
    -------
    label_map : dict
        Final flat cluster labels {timestamp: label}.
    hierarchy : list
        List of (scale, parent_cluster_id, [child_cluster_ids]) for dendrogram.
    cluster_memberships : dict
        {cluster_id: set of member timestamps}.
    """
    hierarchy = []
    cluster_memberships = defaultdict(set)
    
    # 4.1 For the root of the tree assign all objects the same label (single “cluster”).
    all_pings = set(mst_ext_df["from"]).union(set(mst_ext_df["to"]))
    label_map = {ts: 0 for ts in all_pings} # {'t1':0, 't2':0, 't3':0}
    active_clusters = {0: set(all_pings)} # e.g. { 0: {'t1', 't2', 't3'} }
    cluster_memberships[0] = set(all_pings)
    
    # sort edges in decreasing order of weight
    mst_ext_df = mst_ext_df.sort_values("weight", ascending=False)
    
    # group edges by weight
    dendrogram_scales = defaultdict(list)
    for u, v, w in mst_ext_sorted:
        dendrogram_scales[w].append((u, v))
    
    current_label_id = max(label_map.values()) + 1

    # Iteratively remove all edges from MSText in decreasing order of weights
    # 4.2.1 Before each removal, set the dendrogram scale value of the current hierarchical level as the weight of the edge(s) to be removed.
    for scale, edges in dendrogram_scales.items():
        affected_clusters = set()
        edges_to_remove = []
        
        for u, v in edges:
            if label_map.get(u) != label_map.get(v): # if labels are different continue
                continue
            cluster_id = label_map[u]
            affected_clusters.add(cluster_id)
            edges_to_remove.append((u, v))
            
        # 4.2.2: For each affected cluster, reassign components
        for cluster_id in affected_clusters:
            if cluster_id == -1 or cluster_id not in active_clusters:
                continue  # skip noise or already removed clusters
            
            members = active_clusters[cluster_id]
    
            # build connectivity graph (excluding removed edges)
            G = _build_graph(members, mst_ext, edges_to_remove)
            components = _connected_components(G)
            non_spurious = [c for c in components if len(c) >= min_cluster_size] # duration < 5 mins

            # cluster has disappeared
            if not non_spurious:
                for ts in members:
                    label_map[ts] = -1  # noise
                del active_clusters[cluster_id]
            # cluster has just shrunk
            elif len(non_spurious) == 1:
                remaining = non_spurious[0]
                for ts in members:
                    label_map[ts] = cluster_id if ts in remaining else -1
                active_clusters[cluster_id] = remaining
                cluster_memberships[cluster_id] = set(remaining)
            # true cluster split: multiple valid subclusters
            elif len(non_spurious) > 1:
                new_ids = []
                del active_clusters[cluster_id]
                for component in non_spurious:
                    for ts in component:
                        label_map[ts] = current_label_id
                    active_clusters[current_label_id] = set(component)
                    cluster_memberships[current_label_id] = set(component)
                    new_ids.append(current_label_id)
                    current_label_id += 1
                
                hierarchy.append((scale, cluster_id, new_ids))

    return label_map, hierarchy, dict(cluster_memberships)

def _build_graph(nodes, mst_edges, removed_edges):
    '''
    Builds a new graph given the MST edges and the edges to remove.
    
    Parameters
    ----------
    nodes : set of timestamps {t1,t2,t3} 
    mst_edges : list of (u, v, w) tuples
    removed_edges : list of (u,v) tuples
    '''
    graph = defaultdict(set)
    removed_set = set(frozenset(e) for e in removed_edges)
    for u, v, _ in mst_edges:
        if frozenset((u, v)) in removed_set: # to check (u,v) or (v,u) are in set
            continue
        if u in nodes and v in nodes and u != v:
            graph[u].add(v)
            graph[v].add(u)
    return graph

def _connected_components(graph):
    '''
    Finds the connected components for a graph with removed edges.
    
    Parameters
    ----------
    graph : graph derived from MST after removing edges.
    '''
    seen = set()
    components = []

    for node in graph:
        if node in seen:
            continue
        stack = [node]
        comp = []
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                comp.append(n)
                stack.extend(graph[n] - seen)
        components.append(comp)

    return components


# class UnionFind:
#     def __init__(self, size):
#         self.parent = list(range(size))
    
#     def find(self, x):
#         if self.parent[x] != x:
#             self.parent[x] = self.find(self.parent[x])  # path compression
#         return self.parent[x]
    
#     def union(self, x, y):
#         self.parent[self.find(x)] = self.find(y)

# def _find_bursts(times, time_thresh, burst_col=False):
#     '''
#     TC: O(n^2)
#     '''
#     if isinstance(times, pd.Series):
#         times = times.values
#     elif not isinstance(times, np.ndarray):
#         times = np.array(times)
    
#     n = len(times)
    
#     #TC: O(n^2)
#     # Pairwise time differences
#     time_diffs = np.abs(times[:, np.newaxis] - times).astype(int)
    
#     # Identify (i, j) pairs within the threshold
#     within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1)
#     i_idx, j_idx = np.where(within_time_thresh)
    
#     # Union-find to group connected timestamps
#     uf = UnionFind(n)
#     for i, j in zip(i_idx, j_idx):
#         uf.union(i, j)

#     # Assign unique burst labels to roots
#     root_to_label = {}
#     labels = []
#     label_counter = 0

#     # TC: O(log n)
#     for i in range(n):
#         root = uf.find(i)
#         if root not in root_to_label:
#             root_to_label[root] = label_counter
#             label_counter += 1
#         labels.append(root_to_label[root])

#     if burst_col:
#         return pd.DataFrame({"timestamp": times, "burst_label": labels})
#     else:
#         bursts = {}
#         for ts, label in zip(times, labels):
#             if label not in bursts:
#                 bursts[label] = []
#             bursts[label].append(ts)
#         return bursts

#     return labeled_bursts

# Basic compute_core_distance
# def compute_core_distance(traj, min_samples = 2):
#     """
#     Calculate the core distance for each ping in traj.

#     Parameters
#     ----------
#     traj : dataframe
    
#     min_samples : int
#         used to calculate the core distance of a point p, where core distance of a point p 
#         is defined as the distance from p to its min_samples-th smallest nearest neighbor
#         (including itself).

#     Returns
#     -------
#     core_distances : dictionary of timestamps
#         {timestamp_1: core_distance_1, ..., timestamp_n: core_distance_n}
#     """
#     # it gives local density estimate: small core distance → high local density.
#     coords = traj[['x', 'y']].to_numpy()
#     timestamps = traj['timestamp'].to_numpy()
#     n = len(coords)
#     core_distances = {}

#     # TC: 0(n^2)
#     for i in range(n):
#         # pairwise euclidean distances
#         dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
#         # sort distances and get min_samples-th smallest
#         sorted_dists = np.sort(dists)
#         core_distance = sorted_dists[min_samples - 1]
#         core_distances[timestamps[i]] = core_distance

#     return core_distances

# def hierarchical_temporal_dbscan(traj, time_thresh, dist_thresh, min_cluster_size = 10, traj_cols=None, complete_output=False, **kwargs):
#     data = traj.copy()
    
#     # Check if user wants long and lat
#     long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in data.columns and kwargs['longitude'] in data.columns

#     # Check if user wants datetime
#     datetime = 'datetime' in kwargs and kwargs['datetime'] in data.columns

#     # Set initial schema
#     if not traj_cols:
#         traj_cols = {}

#     traj_cols = loader._update_schema(traj_cols, kwargs)
#     traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

#     # Ensure required spatial and temporal columns exist
#     loader._has_spatial_cols(data.columns, traj_cols)
#     loader._has_time_cols(data.columns, traj_cols)

#     # Determine if we are using lat/lon or cartesian (meters)
#     if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns:
#         long_lat = False
#     else:
#         long_lat = True

#     if traj_cols['timestamp'] in data.columns:
#         datetime = False
#     else:
#         datetime = True

#     # Timestamp handling
#     if datetime:
#         time_col_name = traj_cols['datetime']
#         # times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
#         data['mapped_time'] = data[traj_cols['datetime']].astype('datetime64[s]').astype(int)
#     else:
#         first_timestamp = data[traj_cols['timestamp']].iloc[0]
#         timestamp_length = len(str(first_timestamp))

#         if timestamp_length > 10:
#             if timestamp_length == 13:
#                 warnings.warn(
#                     f"The '{data[traj_cols['timestamp']]}' column appears to be in milliseconds. "
#                     "This may lead to inconsistencies."
#                 )
#                 time_col_name = traj_cols['timestamp']
#                 data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64') // 10**3
#                 # times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
#             elif timestamp_length == 19:
#                 warnings.warn(
#                     f"The '{time_col_name}' column appears to be in nanoseconds. "
#                     "This may lead to inconsistencies."
#                 )
#                 time_col_name = traj_cols['timestamp']
#                 data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64') // 10**9
#                 # times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9
#         else:
#             time_col_name = traj_cols['timestamp']
#             data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64')
#             # times = data[traj_cols['timestamp']].values

#     # Build weighted adjacency matrix
#     adjacency_matrix = _find_neighbors_hdbscan(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols, alpha=0.5)

#     # Construct MST
#     mst_edges = _construct_mst(adjacency_matrix)

#     # Extract hierarchical tree structure
#     cluster_tree, clusters = _extract_hierarchy(mst_edges, set(data['mapped_time'].unique()))

#     # Find the optimal threshold for flat cutting
#     optimal_threshold = _find_optimal_threshold(cluster_tree, clusters, min_cluster_size)

#     # Apply flat cut at optimal threshold
#     final_clusters = _flat_cut(cluster_tree, clusters, optimal_threshold, min_cluster_size)

#     # Assign cluster labels based on `mapped_time`
#     data['cluster'] = data['mapped_time'].map(final_clusters).fillna(-1).astype(int)
    
#     hdbscan_labels = data[[time_col_name, 'cluster']]
#     hdbscan_labels = hdbscan_labels.set_index(traj_cols['datetime'])
#     hdbscan_labels.index.name = None

#     # Remove noise (-1 clusters)
#     data = data[data['cluster'] != -1]

#     # Compute stop metrics per cluster
#     stop_table = data.groupby('cluster').apply(
#         lambda group: SD._stop_metrics(group, long_lat, datetime, traj_cols, complete_output), include_groups=False)
    
#     stop_table.index.name = None
#     # Return stop metrics + timestamps with final cluster labels
#     return stop_table, hdbscan_labels