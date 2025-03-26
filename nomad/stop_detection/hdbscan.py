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
def _find_neighbors_hdbscan(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols, alpha=0.5):
    """
    - Computes pairwise distances between points based on space & time.
    - If two points are connected, the matrix stores the weighted distance.
    - Returns a weighted adjacency matrix, where edges represent spatiotemporal proximity.
    """
    if long_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values

    if datetime:
        times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
    else:
        # if timestamps, we change the values to seconds
        first_timestamp = data[traj_cols['timestamp']].iloc[0]
        timestamp_length = len(str(first_timestamp))

        if timestamp_length > 10:
            if timestamp_length == 13:
                times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
            elif timestamp_length == 19:
                times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9   
        else:
            times = data[traj_cols['timestamp']].values

    n = len(data)

    adjacency_matrix = {t: {} for t in times}
    
    # Pairwise time differences (convert seconds to minutes)
    time_diffs = np.abs(times[:, np.newaxis] - times) / 60.0

    # Compute spatial distances
    if long_lat:
        distances = np.array([[_haversine_distance(coords[i], coords[j]) for j in range(n)] for i in range(n)])
    else:
        distances = np.sqrt((coords[:, 0][:, np.newaxis] - coords[:, 0])**2 + 
                            (coords[:, 1][:, np.newaxis] - coords[:, 1])**2)

    # Normalize distances
    max_dist = np.max(distances) if np.max(distances) > 0 else 1
    max_time = np.max(time_diffs) if np.max(time_diffs) > 0 else 1

    distances /= max_dist
    time_diffs /= max_time

    # compute weighted combination
    weighted_distances = alpha * distances + (1 - alpha) * time_diffs

    # apply a threshold to keep only valid neighbors
    threshold = alpha * dist_thresh / max_dist + (1 - alpha) * time_thresh / max_time
    
    for i, t1 in enumerate(times):
        for j, t2 in enumerate(times):
            if weighted_distances[i, j] < threshold and i != j:
                adjacency_matrix[t1][t2] = weighted_distances[i, j]
    
    return adjacency_matrix

def _construct_mst(adjacency_matrix):
    """
    - Uses Prim’s algorithm to construct a Minimum Spanning Tree (MST).
    - Works with a dictionary-based adjacency matrix indexed by timestamps.
    - The edges retain spatiotemporal weights computed earlier.
    - Ensures that every point remains part of the same structure.

    Parameters
    ----------
    adjacency_matrix : dict of dict
        A dictionary where keys are timestamps and values are dictionaries of {neighbor_timestamp: weight}.

    Returns
    -------
    list of tuples
        A list of MST edges in the format: (weight, timestamp1, timestamp2).
    """
    timestamps = list(adjacency_matrix.keys())
    mst_edges = []
    visited = {t: False for t in timestamps}
    pq = []

    def find_next_start():
        for t in timestamps:
            if not visited[t] and adjacency_matrix[t]:
                return t
        return None

    start_node = find_next_start()

    while start_node is not None:
        visited[start_node] = True
        for neighbor, weight in adjacency_matrix[start_node].items():
            heappush(pq, (weight, start_node, neighbor))

        while pq:
            weight, u, v = heappop(pq)

            # only skip edges where both nodes are visited
            if visited[u] and visited[v]:
                continue

            mst_edges.append((weight, u, v))

            # visit the unvisited node
            next_node = v if not visited[v] else u
            visited[next_node] = True

            for neighbor, edge_weight in adjacency_matrix[next_node].items():
                if not visited[neighbor]:
                    heappush(pq, (edge_weight, next_node, neighbor))

        start_node = find_next_start()

    return mst_edges

def _extract_hierarchy(mst_edges, all_timestamps):
    """
    - Sorts MST edges by decreasing weight (weakest connections are removed first).
    - Start with individual points as their own clusters.
    - Iteratively merges clusters, creating a tree-like structure (dendrogram).
    - Assigns `-1` to completely isolated timestamps (i.e., those missing from MST).
    
    Parameters
    ----------
    mst_edges : list of tuples
        A list of edges in the format: (weight, timestamp1, timestamp2).
    all_timestamps : set
        The full set of timestamps that should be included in the clustering process.
    
    Returns
    -------
    np.ndarray
        A hierarchical linkage matrix representing the clustering process.
    dict
        A dictionary mapping timestamps to their cluster labels (including `-1` for noise).
    """

    # sort edges by descending weight (removing weak connections first)
    sorted_edges = sorted(mst_edges, reverse=True, key=lambda x: x[0])

    # initialize all timestamps as their own cluster
    clusters = {t: t for t in all_timestamps}
    cluster_tree = []
    cluster_count = max(all_timestamps) + 1

    for weight, u, v in sorted_edges:
        root_u, root_v = clusters[u], clusters[v]

        # only merge if they belong to different clusters
        if root_u != root_v:
            new_cluster = cluster_count
            cluster_count += 1

            # update all points in both clusters to the new cluster ID
            for key in clusters.keys():
                if clusters[key] in (root_u, root_v):
                    clusters[key] = new_cluster

            # compute cluster size
            cluster_size = sum(1 for x in clusters.values() if x == new_cluster)

            # append to the cluster tree
            cluster_tree.append((root_u, root_v, weight, cluster_size))

    # identify timestamps that were never merged (completely isolated)
    timestamps_in_mst = set()
    for _, u, v in mst_edges:
        timestamps_in_mst.update([u, v])

    isolated_timestamps = all_timestamps - timestamps_in_mst

    for t in isolated_timestamps:
        clusters[t] = -1

    return np.array(cluster_tree), clusters

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