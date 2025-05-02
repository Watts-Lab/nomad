import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import math
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
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

def _find_bursts(times, time_thresh, burst_col = False):
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
        dists = np.full(n, np.inf)
        dists[i] = 0  # distance to itself
        
        for v in allowed_neighbors:
            j = timestamp_indices.get(v)
            if j is not None:
                dists[j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
        
        sorted_dists = np.sort(dists) # TC: O(nlogn)
        core_distances[u] = sorted_dists[min_samples - 1]

    return core_distances

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

def _compute_mrd(traj, core_distances):
    """
    Mutual Reachability Distance (MRD) of point p and point q is the maximum of: 
        cordistance of p, core distance of q, and the distance between p and q.
    MRD graph will have O(n^2) edges, one between every pair of points.

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
    # edges between dense regions are favored, and sparse regions have larger edge weights
    # MRD will inflate distances so that sparse areas are less likely to form clusters
    # Even if two points are physically close, we shouldn’t treat them as equally “reachable”
    # because they live in very different local densities
    coords = traj[['x', 'y']].to_numpy()
    timestamps = traj['timestamp'].to_numpy()
    n = len(coords)
    mrd_graph = {}

    # TC: 0(n^2)
    for i in range(n):
        for j in range(i + 1, n):
            # euclidean distance between point i and j
            dist = np.sqrt(np.sum((coords[j] - coords[i]) ** 2))
            core_i = core_distances[timestamps[i]]
            core_j = core_distances[timestamps[j]]

            mrd_graph[(timestamps[i], timestamps[j])] = max(core_i, core_j, dist)

    return mrd_graph


def mst(mrd_graph):
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
    Add self-edges to MST with weights equal to each point's core distance.
    """
    self_edges = [(ts, ts, core_distances[ts]) for ts in core_distances]
    return mst + self_edges


def hierarchical_temporal_dbscan(traj, time_thresh, dist_thresh, min_cluster_size = 10, traj_cols=None, complete_output=False, **kwargs):
    data = traj.copy()
    
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in data.columns and kwargs['longitude'] in data.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in data.columns

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Ensure required spatial and temporal columns exist
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Determine if we are using lat/lon or cartesian (meters)
    if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns:
        long_lat = False
    else:
        long_lat = True

    if traj_cols['timestamp'] in data.columns:
        datetime = False
    else:
        datetime = True

    # Timestamp handling
    if datetime:
        time_col_name = traj_cols['datetime']
        # times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
        data['mapped_time'] = data[traj_cols['datetime']].astype('datetime64[s]').astype(int)
    else:
        first_timestamp = data[traj_cols['timestamp']].iloc[0]
        timestamp_length = len(str(first_timestamp))

        if timestamp_length > 10:
            if timestamp_length == 13:
                warnings.warn(
                    f"The '{data[traj_cols['timestamp']]}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies."
                )
                time_col_name = traj_cols['timestamp']
                data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64') // 10**3
                # times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{time_col_name}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                time_col_name = traj_cols['timestamp']
                data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64') // 10**9
                # times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9
        else:
            time_col_name = traj_cols['timestamp']
            data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64')
            # times = data[traj_cols['timestamp']].values

    # Build weighted adjacency matrix
    adjacency_matrix = _find_neighbors_hdbscan(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols, alpha=0.5)

    # Construct MST
    mst_edges = _construct_mst(adjacency_matrix)

    # Extract hierarchical tree structure
    cluster_tree, clusters = _extract_hierarchy(mst_edges, set(data['mapped_time'].unique()))

    # Find the optimal threshold for flat cutting
    optimal_threshold = _find_optimal_threshold(cluster_tree, clusters, min_cluster_size)

    # Apply flat cut at optimal threshold
    final_clusters = _flat_cut(cluster_tree, clusters, optimal_threshold, min_cluster_size)

    # Assign cluster labels based on `mapped_time`
    data['cluster'] = data['mapped_time'].map(final_clusters).fillna(-1).astype(int)
    
    hdbscan_labels = data[[time_col_name, 'cluster']]
    hdbscan_labels = hdbscan_labels.set_index(traj_cols['datetime'])
    hdbscan_labels.index.name = None

    # Remove noise (-1 clusters)
    data = data[data['cluster'] != -1]

    # Compute stop metrics per cluster
    stop_table = data.groupby('cluster').apply(
        lambda group: SD._stop_metrics(group, long_lat, datetime, traj_cols, complete_output), include_groups=False)
    
    stop_table.index.name = None
    # Return stop metrics + timestamps with final cluster labels
    return stop_table, hdbscan_labels