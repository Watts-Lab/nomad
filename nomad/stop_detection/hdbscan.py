import pandas as pd
import numpy as np
import heapq
from collections import defaultdict
import warnings
from collections import defaultdict
from scipy.stats import norm

from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree as KDTree

import matplotlib.pyplot as plt


import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils
from nomad.filters import to_timestamp
from nomad.constants import DEFAULT_SCHEMA

import pdb

def _find_temp_neighbors(times, time_thresh, use_datetime):
    """
    Find timestamp pairs that are within time threshold.

    Parameters
    ----------
    times : array of timestamps.
    time_thresh : time threshold for finding what timestamps are close in time.
    use_datetime : Whether to process timestamps as datetime objects.

    Returns
    -------
    time_pairs : list of tuples of timestamps [(t1, t2), ...] that are close in time given time_thresh.

    TC: O(n^2)
    """
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    times = to_timestamp(times).values if use_datetime else times.values
        
    # Pairwise time differences
    # times[:, np.newaxis]: from shape (n,) -> to shape (n, 1) – a column vector
    time_diffs = np.abs(times[:, np.newaxis] - times)
    time_diffs = time_diffs.astype(int)
    
    # Filter by time threshold
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1) # keep upper triangle
    i_idx, j_idx = np.where(within_time_thresh)
    
    # Return a list of (timestamp1, timestamp2) tuples
    time_pairs = [(times[i], times[j]) for i, j in zip(i_idx, j_idx)]
    
    return time_pairs, times

def _build_neighbor_graph(time_pairs, times):
    # Build neighbor map from time_pairs
    neighbors = {ts: set() for ts in times}  # TC: O(n)
    for t1, t2 in time_pairs:
        neighbors[t1].add(t2)
        neighbors[t2].add(t1)
    
    return neighbors

def _compute_core_distance(traj, time_pairs, times, use_lon_lat, traj_cols, min_pts = 2):
    """
    Calculate the core distance for each ping in traj.
    It gives local density estimate: small core distance → high local density.

    Parameters
    ----------
    traj : dataframe

    time_pairs : tuples of timestamps that are close in time given time_thresh
    
    min_pts : int
        used to calculate the core distance of a point p, where core distance of a point p 
        is defined as the distance from p to its min_pts-th smallest nearest neighbor
        (including itself).

    Returns
    -------
    core_distances : dictionary of timestamps
        {timestamp_1: core_distance_1, ..., timestamp_n: core_distance_n} distances are quantized
    """
    # getting coordinates based on whether they are geographic coordinates (lon, lat) or catesian (x,y)
    if use_lon_lat:
        coords = np.radians(traj[[traj_cols['latitude'], traj_cols['longitude']]].values) # TC: O(n)
    else:
        coords = traj[[traj_cols['x'], traj_cols['y']]].values # TC: O(n)
    
    n = len(coords)
    # get the index of timestamp in the arrays (for accessing their value later)
    ts_indices = {ts: idx for idx, ts in enumerate(times)} # TC: O(n)

    # Build neighbor map from time_pairs
    neighbors = _build_neighbor_graph(time_pairs, times)

    D_INF = np.pi * 6_371_000  # max distance on earth
    core_distances = {}

    for i in range(n): # TC: O(n+m (mlogm)) 
        u = times[i]
        allowed_neighbors = neighbors[u]
        dists = [0.0]  # distance to itself

        for v in allowed_neighbors:
            j = ts_indices.get(v)
            if j is not None:
                if use_lon_lat:
                    dist = utils._haversine_distance(coords[i], coords[j])
                else:
                    dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                
                dists.append(np.round(dist * 4) / 4)

        # pad with large numbers if not enough neighbors
        while len(dists) < min_pts:
            dists.append(D_INF) # use a very large number e.g. infinity for edges between points not temporally close

        sorted_dists = np.sort(dists) # TC: O(nlogn)
        core_distances[u] = np.round(sorted_dists[min_pts - 1] * 4)/4
    return core_distances, coords

def _compute_mrd_graph(coords, times, time_pairs, core_distances, use_lon_lat):
    """
    Mutual Reachability Distance (MRD) of point p and point q is the maximum of: 
        core distance of p, core distance of q, and the distance between p and q.
    An edge is stored only for temporally admissible pairs (given by time_pairs).
    
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

    ts_idx = {ts: i for i, ts in enumerate(times)}
    neighbours = _build_neighbor_graph(time_pairs, times)
    mrd_graph = {}

    for u in times:
        i = ts_idx[u]
        for v in neighbours[u]:
            if u >= v:        # keep each undirected edge once
                continue
            j = ts_idx[v]
            dist = (utils._haversine_distance(coords[i], coords[j])
                    if use_lon_lat
                    else np.linalg.norm(coords[j] - coords[i]))

            dist   = np.round(dist * 4) / 4
            mrd_graph[(u, v)] = max(core_distances[u], core_distances[v], dist)

    return mrd_graph

def _mst(mrd_graph):
    """
    Compute the MST using Prim's algorithm from mutual reachability distances.
    Adapted to compute the MST of each connected component, since temporally 
    unadmissible pairs can never be connected (distance is infinite).
    
    Parameters
    ----------
    mrd : dict
        Keys are (timestamp1, timestamp2), values are mutual reachability distances.

    Returns
    -------
    mst : list of tuples
        (t1, t2, mrd_value) for the MST.
    """
    graph = defaultdict(list)
    for (u, v), w in mrd_graph.items():
        graph[u].append((v, w))
        graph[v].append((u, w))

    visited, mst, heap = set(), [], []

    def seed(start):
        heapq.heappush(heap, (0.0, start, start))

    seed(next(iter(graph)))
    while len(visited) < len(graph):
        if not heap:                              # start next component
            seed(next(t for t in graph if t not in visited))
            continue
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        if u != v:
            mst.append((u, v, w))
        for nbr, wt in graph[v]:
            if nbr not in visited:
                heapq.heappush(heap, (wt, v, nbr))
                
    return np.array(mst,
                    dtype=[('from', 'int64'),
                           ('to', 'int64'),
                           ('weight', 'float64')])


def mst_ext(mst_arr, core_distances):
    """
    Add self-edges to MST with weights equal to each point's core distance,
    and return sorted.

    Parameters
    ----------
    mst : (n_edges,3) structured array (from, to, weight)
    core_distances : dict {timestamp: core_distance}
    Returns
    -------
    (n_edges,3) structured array sorted by descending weight
    """
    dtype = mst_arr.dtype
    self_edges = np.array(
        [(ts, ts, w) for ts, w in core_distances.items()],
        dtype=dtype
    )
    edges = np.concatenate((mst_arr, self_edges))
    order = np.argsort(edges['weight'])[::-1]          # big → small
    return edges[order]

def hdbscan(edges_sorted, min_cluster_size):
    hierarchy = []
    cluster_memberships = defaultdict(set)
    label_history = []

    # 4.1 For the root of the tree assign all objects the same label (single “cluster”), label = 0
    all_pings = set(edges_sorted['from']).union(edges_sorted['to'])
   
    label_map = {ts: 0 for ts in all_pings}
    active_clusters = {0: set(all_pings)}
    cluster_memberships[0] = set(all_pings)

    # Log initial labels
    label_history.append(pd.DataFrame({"time": list(label_map.keys()), "cluster_id": list(label_map.values()), "dendrogram_scale": np.nan}))

    # edges_sorted is already ordered by descending weight
    weights = edges_sorted['weight']
    # find split points where weight changes
    split_idx = np.flatnonzero(np.diff(weights)) + 1
    # batch edges of equal weight
    batches = np.split(edges_sorted, split_idx)
    sorted_scales = [batch[0]['weight'] for batch in batches]

    # next label after label 0
    current_label_id = 1

    # Iteratively remove all edges from MSText in decreasing order of weights
    # 4.2.1 Before each removal, set the dendrogram scale value of the current hierarchical level as the weight of the edge(s) to be removed.
    for scale in sorted_scales:
        # pop next batch of edges sharing this scale
        edges = batches.pop(0)
        affected_clusters = set()
        edges_to_remove = []

        for u, v, _ in edges:
            # if labels between two timestamps are different, continue
            if label_map.get(u) != label_map.get(v):
                continue
            cluster_id = label_map[u]
            affected_clusters.add(cluster_id)
            edges_to_remove.append((u, v))

        # print("# of edges to rmv: ", len(edges_to_remove))
        # print("edges to rmv: ", edges_to_remove)

        # 4.2.2: For each affected cluster, reassign components
        for cluster_id in affected_clusters:
            # print("affected cluster: ", cluster_id)
            if cluster_id == -1 or cluster_id not in active_clusters: 
                continue # skip noise or already removed clusters

            members = active_clusters[cluster_id]
            G = _build_graph(members, edges_sorted, edges_to_remove)
            # visualize_adjacency_dict(G)

            components = _connected_components(G)
            # print("number of components: ", len(components))
            # print("components: ", components)
            non_spurious = [c for c in components if len(c) >= min_cluster_size]

            # print("label map: ", label_map)
            # print("active clusters:", active_clusters)
                
            # cluster has disappeared
            if not non_spurious:
                # print("cluster has disappeared")
                for ts in members:
                    label_map[ts] = -1 # noise
                del active_clusters[cluster_id]

            # cluster has just shrunk
            elif len(non_spurious) == 1:
                # print("cluster has just shrunk")
                remaining_pings = non_spurious[0]
                # print("remaining pings:", remaining_pings)
                # print(len(remaining_pings))
                for ts in members:
                    label_map[ts] = cluster_id if ts in remaining_pings else -1
                active_clusters[cluster_id] = remaining_pings
                cluster_memberships[cluster_id] = set(remaining_pings)
            # true cluster split: multiple valid subclusters
            else:
                # print("true cluster split")
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

            # print("active clusters:", active_clusters.keys())
        
        # log label map after this scale
        label_history.append(pd.DataFrame({"time": list(label_map.keys()), "cluster_id": list(label_map.values()), "dendrogram_scale": scale}))

    # combine label history into one DataFrame
    label_history_df = pd.concat(label_history, ignore_index=True)
    # build cluster lineage for all clusters
    hierarchy_df = _build_cluster_lineage(hierarchy)

    return label_history_df, hierarchy_df

def compute_cluster_duration(cluster):
    max_time = max(cluster)
    min_time = min(cluster)
    return (max_time - min_time) * 60

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

def _build_graph(nodes, edges_arr, removed_edges):
    """
    Build the adjacency map for the given cluster, skipping any edges
    scheduled for removal.  No DataFrame row objects are created.
    """
    nodes_set    = set(nodes)                       # O( |cluster| )
    removed_set  = set(map(frozenset, removed_edges))
    graph        = defaultdict(set)

    for u, v, _ in edges_arr:
        if u not in nodes_set or v not in nodes_set or u == v:
            continue
        if frozenset((u, v)) in removed_set:
            continue
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
                comp.append(int(n))
                stack.extend(graph[n] - seen)
        components.append(comp)

    return components


def _get_eps(label_history_df, target_cluster_id):
    eps_df = []
    
    # ε_max(Ci): maximum ε value at which Ci exisits
    eps_max = label_history_df[label_history_df['cluster_id']== target_cluster_id]['dendrogram_scale'].max()
    timestamps = label_history_df[label_history_df['cluster_id'] == target_cluster_id]['time'].unique()

    for ts in timestamps:
        # print(ts)
        history = label_history_df[label_history_df['time'] == ts].copy()
        # print(history)
        history = history[~history['dendrogram_scale'].isna()]
        history = history.sort_values('dendrogram_scale', ascending=False)
        # print(history)

        in_cluster = history[history['cluster_id'] == target_cluster_id]
        # print("in cluster:")
        # print(in_cluster)
        out_cluster = history[history['cluster_id'] != target_cluster_id]
        # print("out cluster:")
        # print(out_cluster)

        if not in_cluster.empty:
            last_in_cluster_scale = in_cluster['dendrogram_scale'].min()  # since descending
            # print("scale last in cluster", last_in_cluster_scale)
            exited_after = out_cluster[out_cluster['dendrogram_scale'] < last_in_cluster_scale]

            if not exited_after.empty:
                # ε_min(xj , Ci): ε value (scale) beyond which object xj no longer belongs to cluster Ci
                eps_min = exited_after['dendrogram_scale'].max()  # highest scale after exit
            else:
                eps_min = np.inf  # never exited
        else:
            eps_min = np.nan  # never entered

        eps_df.append({
            "time": ts,
            "eps_min": eps_min,
            "eps_max": eps_max
        })

        # print("-----")

    return pd.DataFrame(eps_df)

def custom_CDF(eps):
    x = np.asarray(eps)
    y = np.zeros_like(x, dtype=float)

    m = (x >= 5)  & (x <= 20)
    y[m] = 2 * (x[m] - 5)

    m = (x > 20) & (x <= 80)
    y[m] = 30 + 1.5 * (x[m] - 20)

    m = (x > 80) & (x <= 200)
    y[m] = 120 + (x[m] - 80)

    y[x > 200] = 1
    return y

def custom_cluster_stability(label_history_df):
    # restrict to real clusters and non-NaN scales
    df = label_history_df[
        (label_history_df['cluster_id'] != 0) &
        (label_history_df['cluster_id'] != -1) &
        label_history_df['dendrogram_scale'].notna()
    ]

    # build lookup: time → (scales_desc, cluster_ids_at_scales)
    history_by_time = {}
    for ts, grp in df.groupby('time', sort=False):
        sorted_grp = grp.sort_values('dendrogram_scale', ascending=False)
        history_by_time[ts] = (
            sorted_grp['dendrogram_scale'].to_numpy(),
            sorted_grp['cluster_id'].to_numpy()
        )

    # precompute ε_max per cluster
    eps_max = df.groupby('cluster_id')['dendrogram_scale'].max().to_dict()

    out = []
    for cluster_id, eps_max_c in eps_max.items():
        total_stab = 0.0
        for scales, clusters in history_by_time.values():
            # find all positions where this time was still in cluster:
            in_mask = (clusters == cluster_id)
            if not in_mask.any():
                continue
            # last scale at which it was in Ci
            last_in = scales[in_mask].min()
            # among scales < last_in where cluster != Ci, take max (or ∞)
            out_mask = (clusters != cluster_id) & (scales < last_in)
            eps_min = scales[out_mask].max() if out_mask.any() else np.inf

            total_stab += custom_CDF([eps_min])[0] - custom_CDF([eps_max_c])[0]

        out.append({'cluster_id': cluster_id, 'cluster_stability': total_stab})

    return pd.DataFrame(out)


def compute_cluster_stability(label_history_df):
    # Get clusters that are not root (0) or noise (-1)
    clusters = label_history_df.loc[~label_history_df['cluster_id'].isin([0, -1]), 'cluster_id'].unique()

    cluster_stability_df = []

    for cluster in clusters:
        eps_df = _get_eps(label_history_df, target_cluster_id=cluster)
        
        # Avoid division by zero or NaNs
        eps_df = eps_df.replace({'eps_min': {0: np.nan}, 'eps_max': {0: np.nan}})
        eps_df['stability_term'] = (1 / eps_df['eps_min']) - (1 / eps_df['eps_max'])
        total_stability = eps_df['stability_term'].sum(skipna=True)

        cluster_stability_df.append({
            "cluster_id": cluster,
            "cluster_stability": total_stability
        })

    return pd.DataFrame(cluster_stability_df)

def select_most_stable_clusters(hierarchy_df, cluster_stability_df):
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

def hdbscan_labels(traj, time_thresh, min_pts = 2, min_cluster_size = 10, traj_cols=None, **kwargs):
    # Check if user wants long and lat and datetime
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(traj.columns, traj_cols, kwargs)
    # Load default col names
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)
    
    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    time_pairs, times = _find_temp_neighbors(traj[traj_cols[t_key]], time_thresh, use_datetime)

    core_distances, coords = _compute_core_distance(traj, time_pairs, times, use_lon_lat, traj_cols, min_pts)

    mrd = _compute_mrd_graph(coords, times, time_pairs, core_distances, use_lon_lat)

    mst_edges = _mst(mrd)
    
    edges_sorted   = mst_ext(mst_edges, core_distances) # << with self-loops

    label_history_df, hierarchy_df = hdbscan(edges_sorted, min_cluster_size)
    
    #cluster_stability_df = compute_cluster_stability(label_history_df)
    cluster_stability_df = custom_cluster_stability(label_history_df)

    selected_clusters = select_most_stable_clusters(hierarchy_df, cluster_stability_df)

    all_timestamps = set(label_history_df['time'])
    assigned_timestamps = set()
    rows = []

    for cid in selected_clusters:
        # Filter to rows where cluster_id == cid
        cluster_rows = label_history_df[label_history_df['cluster_id'] == cid]

        if cluster_rows.empty:
            continue  # skip if no rows (just in case)
        
        # Find the smallest scale before this cluster disappears/splits into subclusters
        # min_scale = cluster_rows['dendrogram_scale'].min()
        max_scale = cluster_rows['dendrogram_scale'].max()
        
        # Get the timestamps assigned to this cluster at that scale
        # members = set(cluster_rows[cluster_rows['dendrogram_scale'] == min_scale]['time'])
        members = set(cluster_rows[cluster_rows['dendrogram_scale'] == max_scale]['time'])

        for ts in members:
            rows.append({"time": ts, "cluster": cid})
        assigned_timestamps.update(members)
    
    # Add noise cluster (-1) for unassigned timestamps
    for ts in all_timestamps - assigned_timestamps:
        rows.append({"time": ts, "cluster": -1})
    
    hdbscan_labels_df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    labels_hdbscan = hdbscan_labels_df.cluster.set_axis(traj.index)
    labels_hdbscan.name = 'cluster'
    return labels_hdbscan

def st_hdbscan(traj, time_thresh, min_pts = 2, min_cluster_size = 10, complete_output = False, traj_cols=None, **kwargs):
    labels_hdbscan = hdbscan_labels(traj=traj, time_thresh=time_thresh, min_pts = min_pts,
                                        min_cluster_size = min_cluster_size, traj_cols=traj_cols, **kwargs)

    merged_data_hdbscan = traj.join(labels_hdbscan)
    merged_data_hdbscan = merged_data_hdbscan[merged_data_hdbscan.cluster != -1]

    stop_table = merged_data_hdbscan.groupby('cluster', as_index=False).apply(
                    lambda g: utils.summarize_stop(g, complete_output=complete_output, traj_cols=traj_cols, **kwargs),
                    include_groups=False)
                
    # return stop_table
    return labels_hdbscan, stop_table