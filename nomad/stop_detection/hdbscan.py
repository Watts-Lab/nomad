import pandas as pd
import numpy as np
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils

##########################################
########        HDBSCAN           ########
##########################################
def _find_temp_neighbors(times, time_thresh, is_datetime):
    """
    Find timestamp pairs that are within time threshold.

    Parameters
    ----------
    times : array of timestamps.
    time_thresh : time threshold for finding what timestamps are close in time.
    is_datetime : Whether to process timestamps as datetime objects.

    Returns
    -------
    time_pairs : list of tuples of timestamps [(t1, t2), ...] that are close in time given time_thresh.

    TC: O(n^2)
    """
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    if is_datetime:
        times = pd.to_datetime(times)
        times = times.dt.tz_convert('UTC').dt.tz_localize(None)
        times = times.astype('int64') // 10**9
        times = times.values
    else:
        # if timestamps, we change the values to seconds
        first_timestamp = times.iloc[0]
        timestamp_length = len(str(first_timestamp))
    
        if timestamp_length > 10:
            if timestamp_length == 13:
                times = times.values.view('int64') // 10 ** 3
            elif timestamp_length == 19:
                times = times.values.view('int64') // 10 ** 9   
        else:
            times = times.values
    
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

def _compute_core_distance(traj, time_pairs, times, is_long_lat, traj_cols, min_pts = 2):
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
        {timestamp_1: core_distance_1, ..., timestamp_n: core_distance_n}
    """
    # getting coordinates based on whether they are geographic coordinates (lon, lat) or catesian (x,y)
    if is_long_lat:
        coords = np.radians(traj[[traj_cols['latitude'], traj_cols['longitude']]].values) # TC: O(n)
    else:
        coords = traj[[traj_cols['x'], traj_cols['y']]].values # TC: O(n)
    
    n = len(coords)
    # get the index of timestamp in the arrays (for accessing their value later)
    ts_indices = {ts: idx for idx, ts in enumerate(times)} # TC: O(n)

    # Build neighbor map from time_pairs
    neighbors = _build_neighbor_graph(time_pairs, times)
    
    core_distances = {}

    for i in range(n): # TC: O(n+m (mlogm)) < O(n^2 log n)
        u = times[i]
        allowed_neighbors = neighbors[u]
        dists = [0.0]  # distance to itself

        for v in allowed_neighbors:
            j = ts_indices.get(v)
            if j is not None:
                if is_long_lat:
                    dist = utils._haversine_distance(coords[i], coords[j])
                    dists.append(dist)
                else:
                    dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                    dists.append(dist)

        # pad with large numbers if not enough neighbors
        while len(dists) < min_pts:
            dists.append(np.inf) # use a very large number e.g. infinity for edges between points not temporally close

        sorted_dists = np.sort(dists) # TC: O(nlogn)
        core_distances[u] = sorted_dists[min_pts - 1]

    return core_distances, coords

def _compute_mrd_graph(coords, times, time_pairs, core_distances, is_long_lat):
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
    n = len(coords)
    neighbors = _build_neighbor_graph(time_pairs, times)
    mrd_graph = {}

    # TC: O(n+m)
    for i in range(n):
        for j in range(i + 1, n):
            u = times[i]
            v = times[j]
            if v not in neighbors[u]:
                mrd_graph[(u, v)] = np.inf
            else:
                if is_long_lat:
                    # haversine distance between point i and j
                    dist = utils._haversine_distance(coords[i], coords[j])
                    core_i = core_distances[u]
                    core_j = core_distances[v]
                else:
                    # euclidean distance between point i and j
                    dist = np.sqrt(np.sum((coords[j] - coords[i]) ** 2))
                    core_i = core_distances[u]
                    core_j = core_distances[v]

                mrd_graph[(u, v)] = max(core_i, core_j, dist)

    # TC: 0(n^2)
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if is_long_lat:
    #             # haversine distance between point i and j
    #             dist = utils._haversine_distance(coords[i], coords[j])
    #             core_i = core_distances[times[i]]
    #             core_j = core_distances[times[j]]
    #         else:
    #             # euclidean distance between point i and j
    #             dist = np.sqrt(np.sum((coords[j] - coords[i]) ** 2))
    #             core_i = core_distances[times[i]]
    #             core_j = core_distances[times[j]]

    #         mrd_graph[(times[i], times[j])] = max(core_i, core_j, dist)

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
    pd.DataFrame with columns ['from', 'to', 'weight']
    """
    self_edges = [(ts, ts, core_distances[ts]) for ts in core_distances]

    mst_ext_edges = mst + self_edges

    return pd.DataFrame(mst_ext_edges, columns=["from", "to", "weight"])

def hdbscan(mst_ext_df, min_cluster_size):
    hierarchy = []
    cluster_memberships = defaultdict(set)
    label_history = []

    # 4.1 For the root of the tree assign all objects the same label (single “cluster”), label = 0
    all_pings = set(mst_ext_df['from']).union(set(mst_ext_df['to']))
    label_map = {ts: 0 for ts in all_pings}
    active_clusters = {0: set(all_pings)}
    cluster_memberships[0] = set(all_pings)

    # Log initial labels
    label_history.append(pd.DataFrame({"time": list(label_map.keys()), "cluster_id": list(label_map.values()), "dendrogram_scale": np.nan}))

    # group edges by weight
    dendrogram_scales = {
        w: list(zip(group['from'], group['to']))
        for w, group in mst_ext_df.groupby('weight', sort=False)
    }

    # sort edges in decreasing order of weight
    sorted_scales = sorted(dendrogram_scales.keys(), reverse=True)

    # next label after label 0
    current_label_id = 1

    # Iteratively remove all edges from MSText in decreasing order of weights
    # 4.2.1 Before each removal, set the dendrogram scale value of the current hierarchical level as the weight of the edge(s) to be removed.
    for scale in sorted_scales:
        # print("---------")
        # print("scale:", scale)
        
        edges = dendrogram_scales[scale]
        affected_clusters = set()
        edges_to_remove = []

        for u, v in edges:
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
            G = _build_graph(members, mst_ext_df, edges_to_remove)
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

def _build_graph(nodes, mst_df, removed_edges):
    """
    Build a connectivity graph from MST edges (DataFrame), excluding removed edges.

    Parameters
    ----------
    nodes : set
        Nodes in the current cluster.
    mst_df : pd.DataFrame
        DataFrame with columns ['from', 'to', 'weight'].
    removed_edges : list of (u, v) tuples
        Edges to be removed (regardless of direction).

    Returns
    -------
    graph : dict of sets
        Adjacency list representation: {u: {v1, v2, ...}, ...}
    """
    graph = defaultdict(set)
    removed_set = set(frozenset(e) for e in removed_edges)

    for _, row in mst_df.iterrows():
        u, v = row['from'], row['to']
        if frozenset((u, v)) in removed_set:
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

def hdbscan_labels(label_history_df, selected_clusters):
    all_timestamps = set(label_history_df['time'])
    assigned_timestamps = set()
    rows = []

    for cid in selected_clusters:
        # Filter to rows where cluster_id == cid
        cluster_rows = label_history_df[label_history_df['cluster_id'] == cid]

        if cluster_rows.empty:
            continue  # skip if no rows (just in case)
        
        # Find the smallest scale before this cluster disappears/splits into subclusters
        min_scale = cluster_rows['dendrogram_scale'].min()
        
        # Get the timestamps assigned to this cluster at that scale
        members = set(cluster_rows[cluster_rows['dendrogram_scale'] == min_scale]['time'])
        for ts in members:
            rows.append({"time": ts, "cluster": cid})
        assigned_timestamps.update(members)
    
    # Add noise cluster (-1) for unassigned timestamps
    for ts in all_timestamps - assigned_timestamps:
        rows.append({"time": ts, "cluster": -1})
    
    hdbscan_labels_df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    return hdbscan_labels_df

def st_hdbscan(traj, is_long_lat, is_datetime, traj_cols, complete_output, time_thresh, min_pts = 2, min_cluster_size = 10, **kwargs):
    # Check if user wants long and lat
    is_long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    # Check if user wants datetime
    is_datetime = 'datetime' in kwargs and kwargs['datetime'] in traj.columns

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns and not is_long_lat:
        is_long_lat = False
    else:
        is_long_lat = True

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in traj.columns and not is_datetime:
        is_datetime = False
    else:
        is_datetime = True

    if is_datetime:
        time_col_name = traj_cols['datetime']
    else:
        first_timestamp = traj[traj_cols['timestamp']].iloc[0]
        timestamp_length = len(str(first_timestamp))
        
        if timestamp_length > 10:
            if timestamp_length == 13:
                warnings.warn(
                    f"The '{traj[traj_cols['timestamp']]}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies."
                )
                
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{traj[traj_cols['timestamp']]}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
        
        time_col_name = traj_cols['timestamp']

    time_pairs, times = _find_temp_neighbors(traj[time_col_name], time_thresh, is_datetime)
    core_distances, coords = _compute_core_distance(traj, time_pairs, times, is_long_lat, traj_cols, min_pts)
    mrd = _compute_mrd_graph(coords, times, time_pairs, core_distances, is_long_lat)
    mst_edges = _mst(mrd)
    mstext_edges = mst_ext(mst_edges, core_distances)
    label_history_df, hierarchy_df = hdbscan(mstext_edges, min_cluster_size)
    cluster_stability_df = compute_cluster_stability(label_history_df)
    selected_clusters = select_most_stable_clusters(hierarchy_df, cluster_stability_df)
    sample_labels_hdbscan = hdbscan_labels(label_history_df, selected_clusters)
    merged_data_hdbscan = traj.merge(sample_labels_hdbscan, left_on=time_col_name, right_on='time')
    stop_table = merged_data_hdbscan.groupby('cluster').apply(lambda group: _stop_metrics(group, is_long_lat, is_datetime, traj_cols, complete_output), include_groups=False)
    # remove noise pings from stop table
    stop_table = stop_table[stop_table.index != -1]
    return stop_table

def _stop_metrics(grouped_data, is_long_lat, is_datetime, traj_cols, complete_output):
    # Coordinates array and distance metrics
    if is_long_lat:
        coords = grouped_data[[traj_cols['longitude'], traj_cols['latitude']]].to_numpy()
        stop_medoid = utils._medoid(coords, metric='haversine')
        diameter_m = utils._diameter(coords, metric='haversine')
    else:
        coords = grouped_data[[traj_cols['x'], traj_cols['y']]].to_numpy()
        stop_medoid = utils._medoid(coords, metric='euclidean')
        diameter_m = utils._diameter(coords, metric='euclidean')

    # Compute duration and start and end time of stop
    if is_datetime:
        start_time = grouped_data[traj_cols['datetime']].min()
        end_time = grouped_data[traj_cols['datetime']].max()
        duration = (end_time - start_time).total_seconds() / 60.0
    else:
        start_time = grouped_data[traj_cols['timestamp']].min()
        # print("start time:", start_time)
        end_time = grouped_data[traj_cols['timestamp']].max()
        # print("end time:", end_time)
        timestamp_length = len(str(start_time))

        if timestamp_length > 10:
            if timestamp_length == 13:
                duration = ((end_time // 10 ** 3) - (start_time // 10 ** 3)) / 60.0
            elif timestamp_length == 19:
                duration = ((end_time // 10 ** 9) - (start_time // 10 ** 9)) / 60.0
        else:
            duration = (end_time - start_time) / 60.0
                            
    # Number of pings in stop
    n_pings = len(grouped_data)
    
    # Compute max_gap between consecutive pings (in minutes)
    if is_datetime:
        times = pd.to_datetime(grouped_data[traj_cols['datetime']]).sort_values()
        time_diffs = times.diff().dropna()
        max_gap = int(time_diffs.max().total_seconds() / 60) if not time_diffs.empty else 0
    else:
        times = grouped_data[traj_cols['timestamp']].sort_values()
        timestamp_length = len(str(times.iloc[0]))
        
        if timestamp_length == 13:
            time_diffs = np.diff(times.values) // 1000
        elif timestamp_length == 19:  # nanoseconds
            time_diffs = np.diff(times.values) // 10**9
        else:
            time_diffs = np.diff(times.values)
        
        max_gap = int(np.max(time_diffs) / 60) if len(time_diffs) > 0 else 0

    # Prepare data for the Series
    if is_long_lat:
        if complete_output:
            stop_attr = {
                'start_time': start_time,
                'end_time': end_time,
                traj_cols['longitude']: stop_medoid[0],
                traj_cols['latitude']: stop_medoid[1],
                'diameter': diameter_m,
                'n_pings': n_pings,
                'duration': duration,
                'max_gap': max_gap
            }
        else:
            stop_attr = {
                'start_time': start_time,
                'duration': duration,
                traj_cols['longitude']: stop_medoid[0],
                traj_cols['latitude']: stop_medoid[1]
            }
    else:
        if complete_output:
            stop_attr = {
                'start_time': start_time,
                'end_time': end_time,
                traj_cols['x']: stop_medoid[0],
                traj_cols['y']: stop_medoid[1],
                'diameter': diameter_m,
                'n_pings': n_pings,
                'duration': duration,
                'max_gap': max_gap
            }
        else:
            stop_attr = {
                'start_time': start_time,
                'duration': duration,
                traj_cols['x']: stop_medoid[0],
                traj_cols['y']: stop_medoid[1]
            }

    return pd.Series(stop_attr)


# def visualize_mst(mst_df):
#     G = nx.Graph()

#     for u, v, w in mst_df.itertuples(index=False):
#         w_val = 1e6 if np.isinf(w) else w
#         inv_w = 1.0 / w_val if w_val > 0 else 0.001
#         label = "∞" if np.isinf(w) else f"{w:.2f}"
#         G.add_edge(u, v, weight=w_val, inv_weight=inv_w, label=label)

#     # Use inverse weights for layout
#     pos = nx.spring_layout(G, weight='inv_weight')

#     # Retrieve edge labels from edge attributes
#     edge_labels = nx.get_edge_attributes(G, 'label')

#     # Plot
#     plt.figure(figsize=(6, 6))
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.title("MST Extended Graph with Edge Weights")
#     plt.axis('off')
#     plt.show()

# def visualize_adjacency_dict(G_dict):
#     """
#     Visualize an adjacency dictionary as a NetworkX graph.

#     Parameters
#     ----------
#     G_dict : dict
#         Output of _build_graph() — {node: set(neighbors)}
#     """
#     # Convert dict-of-sets into a networkx Graph
#     G = nx.Graph()
#     for u, neighbors in G_dict.items():
#         for v in neighbors:
#             G.add_edge(u, v)

#     pos = nx.spring_layout(G, seed=42)

#     # Draw graph
#     nx.draw(
#         G, pos,
#         with_labels=True,
#         node_color='lightgreen',
#         edge_color='gray',
#         node_size=200,
#         font_size=10
#     )
#     plt.title("Connected Component After Edge Removal")
#     plt.tight_layout()
#     plt.show()