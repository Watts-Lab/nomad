import pandas as pd
import numpy as np
import heapq
from collections import defaultdict
import warnings
from scipy.stats import norm
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

def _compute_core_distance(data, time_pairs, times, use_lon_lat, traj_cols, min_pts = 2):
    """
    Calculate the core distance for each ping in data.
    It gives local density estimate: small core distance → high local density.

    Parameters
    ----------
    data : dataframe

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
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values) # TC: O(n)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values # TC: O(n)
    
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

def _build_border_map(scale, core_distances, d_graph):
    """
    For a given threshold `scale`, assign each non-core point to its nearest
    core neighbor (preceding or succeeding in time) within distance <= scale.
    
    Parameters
    ----------
    scale : float
        Distance threshold at this dendogram level.
    core_distances : pd.Series
        Indexed by timestamp; contains each point's “core distance.”
        Must be sorted by index (timestamp).
    d_graph : pd.Series
        MultiIndex (border, core) → actual distance between points.
    
    Returns
    -------
    defaultdict(set)
        Mapping {core_ts: set(border_ts, ...)} of border points for each core.
    """
    # Identify which timestamps are core vs border
    is_core = core_distances <= scale
    cores   = core_distances.index[is_core].to_numpy()
    borders = core_distances.index[~is_core].to_numpy()
    
    if cores.size == 0 or borders.size == 0:
        return defaultdict(set)
    
    # Locate insertion points of each border among cores
    succ_pos = np.searchsorted(cores, borders, side="left")
    pred_pos = succ_pos - 1
    
    # Build pairs (border, candidate_core) for existing neighbors
    pairs = []
    # predecessor candidates
    valid_pred = pred_pos >= 0
    for b, p in zip(borders[valid_pred], cores[pred_pos[valid_pred]]):
        pairs.append((b, p))
    # successor candidates
    valid_succ = succ_pos < cores.size
    for b, s in zip(borders[valid_succ], cores[succ_pos[valid_succ]]):
        pairs.append((b, s))
    
    # Look up distances, drop ones beyond scale
    idx = pd.MultiIndex.from_tuples(pairs, names=["border", "core"])
    dists = d_graph.reindex(idx).dropna()
    dists = dists[dists <= scale]
    
    if dists.empty:
        return defaultdict(set)
    
    # For each border, pick the core with minimal distance (ties break naturally)
    best = dists.groupby(level="border").idxmin().values
    core_to_border = defaultdict(set)
    for border, core in best:
        core_to_border[core].add(border)
    
    return core_to_border

def hdbscan(edges_sorted_df, core_distances, d_graph, min_cluster_size, dur_min=5):
    """
    Builds a cluster hierarchy from a pre-computed Minimum Spanning Tree.

    Iteratively removes edges from the MST (from largest to smallest weight)
    to form a hierarchy of clusters. Uses a chronological border-point
    assignment strategy to test for cluster spuriousness.

    Parameters
    ----------
    edges_sorted_df : pd.Series
        The MST with self-loops, indexed by ('from', 'to') and sorted
        descending by weight (which represents distance/scale).
    core_distances : pd.Series
        Sorted Series mapping each timestamp to its core distance.
    d_graph : pd.Series
        Symmetric graph of raw distances between all temporally-close points.
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
    
    # All pings are taken from the core_distances index
    all_pings = core_distances.index
    label_map = pd.Series(0, index=all_pings, name='cluster_id', dtype=int)

    # Log initial labels
    df0 = label_map.to_frame()
    df0['dendogram_scale'] = np.nan
    label_history.append(df0.reset_index())

    current_label_id = 1

    # Iteratively process edges grouped by weight (scale)
    for scale, edges_to_remove in edges_sorted_df.groupby(edges_sorted_df, sort=False):

        border_map = _build_border_map(scale, core_distances, d_graph) # can be computed without thinking of edges
        idx_from = edges_to_remove.index.get_level_values('from')
        affected_clusters = set(label_map.loc[idx_from].unique()) 

        for cluster_id in affected_clusters:
            if cluster_id == -1:
                continue
 
            members = label_map.index[label_map == cluster_id]
            remaining_members = set(members)
            
            G = _build_graph_pd(members, edges_sorted_df, edges_to_remove)
            components = _connected_components(G)
            
            non_spurious = []

            for comp in components:
                # Look up this component's borders in the global map
                comp_borders = set().union(*(border_map.get(ts, set()) for ts in comp))
                full_cluster = set(comp).union(comp_borders)
                if ((max(full_cluster) - min(full_cluster)) >= dur_min * 60) and (len(comp) >= min_cluster_size):
                    non_spurious.append(comp)
                    
            if not non_spurious:
                pass # cluster has disappeared

            # cluster has just shrunk
            elif len(non_spurious) == 1:
                component = non_spurious[0]
                label_map.loc[list(component)] = cluster_id
                remaining_members = remaining_members.difference(component)

            # true cluster split: multiple valid subclusters
            else:
                new_ids = []
                for component in non_spurious:
                    label_map.loc[list(component)] = current_label_id
                    remaining_members = remaining_members.difference(component)
                        
                    new_ids.append(current_label_id)
                    current_label_id += 1

                hierarchy.append((scale, cluster_id, new_ids))

            label_map.loc[list(remaining_members)] = -1

        # log label map after this scale
        df = label_map.to_frame()
        df['dendogram_scale'] = scale
        label_history.append(df.reset_index())
        
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

def _build_graph_pd(nodes, edges_sorted_df, removed_edges):
    graph = defaultdict(set)
    
    mask1 = edges_sorted_df.index.get_level_values('from').isin(nodes) & edges_sorted_df.index.get_level_values('to').isin(nodes)
    sub = edges_sorted_df.loc[mask1]
    
    mask2 = sub.index.get_level_values('from') != sub.index.get_level_values('to')
    mask3 = ~sub.index.isin(removed_edges.index)

    for u, v in sub[mask2&mask3].index:
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

def _piecewise_linear_cdf(eps):
    """
    Example of a custom, piecewise CDF for stability calculations.
    """
    x = np.asarray(eps)
    y = np.zeros_like(x, dtype=float)

    m = (x >= 5)  & (x <= 20)
    y[m] = 2 * (x[m] - 5)

    m = (x > 20) & (x <= 80)
    y[m] = 30 + 1.5 * (x[m] - 20)

    m = (x > 80) & (x <= 200)
    y[m] = 120 + (x[m] - 80)

    m = x > 200
    y[x > 200] = 240

    return y/240

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
    
    # 5. An exit occurs where the cluster_id changes.
    # The 'eps_min' for that point-cluster pair is the scale at which it was last seen.
    # When a point drops out of all clusters, its next_cluster_id is NaN.
    exit_events = df[df['cluster_id'] != df['next_cluster_id']].copy()
    exit_events.rename(columns={'dendogram_scale': 'eps_min'}, inplace=True)
    
    # 6. For points that never exit a cluster (i.e., stay in it until the MST is one component),
    # their eps_min is effectively infinite. They contribute to stability until the end.
    # We find these by identifying the last known state for each point.
    last_state = df.drop_duplicates(subset='time', keep='last')
    # Filter for those that were not already marked as an exit event (i.e., next_cluster_id was NaN)
    never_exited = last_state[last_state['next_cluster_id'].isna()]
    
    # Combine the two types of stability events
    stability_points = pd.concat([
        exit_events[['time', 'cluster_id', 'eps_min', 'eps_max']],
        never_exited[['time', 'cluster_id', 'dendogram_scale', 'eps_max']].assign(eps_min=np.inf)
    ])
    
    # 7. Apply the provided CDF to calculate the stability contribution of each point.
    stability_points['stability_term'] = (
        cdf_function(stability_points['eps_max']) - cdf_function(stability_points['eps_min'])
    )

    # 8. Sum the contributions for each cluster to get the final stability score.
    final_stability = stability_points.groupby('cluster_id')['stability_term'].sum()

    return final_stability.reset_index().rename(columns={'stability_term': 'cluster_stability'})

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

def _build_hdbscan_graphs(coords, ts_idx, neighbors, core_dist, use_lon_lat):
    """
    Computes all graphs required for the HDBSCAN algorithm in one pass.

    Returns
    -------
    edges_sorted : np.recarray
        [from, to, weight] sorted descending by weight.
    d_graph : pd.Series
        Symmetric graph of raw distances, MultiIndex (from, to).
    """
    mrd_graph = {}
    u_list, v_list, d_list = [], [], []

    for u, u_neighbors in neighbors.items():
        i = ts_idx[u]
        for v in u_neighbors:
            if u >= v:
                continue
            
            j = ts_idx[v]
            dist = (utils._haversine_distance(coords[i], coords[j])
                    if use_lon_lat else np.linalg.norm(coords[i] - coords[j]))
            dist = np.round(dist * 4) / 4

            mrd_graph[(u, v)] = max(core_dist[u], core_dist[v], dist)
            u_list.append(u)
            v_list.append(v)
            d_list.append(dist)


    idx = pd.MultiIndex.from_arrays([u_list, v_list], names=["from", "to"])
    d_graph_part = pd.Series(d_list, index=idx)
    
    rev = d_graph_part.copy()
    rev.index = rev.index.swaplevel(0, 1)
    d_graph = pd.concat([d_graph_part, rev])


    # --- Build MST from MRD graph ---
    mst_arr = _mst(mrd_graph)

    # --- Extend and sort MST with self-loops ---
    self_loops_items = list(core_dist.items())
    if not self_loops_items:
        self_loops_full = np.empty(0, dtype=mst_arr.dtype)
    else:
        self_loops = np.array(
            self_loops_items,
            dtype=[('from', 'int64'), ('weight', 'float64')]
        )
        self_loops_full = np.empty(len(self_loops), dtype=mst_arr.dtype)
        self_loops_full['from'] = self_loops['from']
        self_loops_full['to'] = self_loops['from']
        self_loops_full['weight'] = self_loops['weight']
    
    all_edges = np.concatenate([mst_arr, self_loops_full])
    
    order = np.argsort(all_edges["weight"])[::-1]
    sorted_edges = all_edges[order]
    
    edges_sorted_df = pd.Series(
        sorted_edges['weight'],
        index=pd.MultiIndex.from_arrays(
            [sorted_edges['from'], sorted_edges['to']],
            names=['from', 'to']
        ),
        name='weight'
    )
    return edges_sorted_df, d_graph

def hdbscan_labels(data, time_thresh, min_pts = 2, min_cluster_size = 1, dur_min=5, traj_cols=None, **kwargs):
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
    passthrough_cols : list, optional
        Columns to propagate for later stop summarization.
    traj_cols : dict, optional
        Mapping for key columns.
    **kwargs
        Passed to internal helpers.

    Returns
    -------
    pd.Series
        Cluster label for each row; –1 for noise.
    """
    # Check if user wants long and lat and datetime
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)
    # Load default col names
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    
    if traj_cols['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        first = arr[0]
        if any(x != first for x in arr[1:]):
            raise ValueError("Multi-user data? Groupby or use hdbscan_per_user instead.")
    
    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    time_pairs, times = _find_temp_neighbors(data[traj_cols[t_key]], time_thresh, use_datetime)

    neighbors = _build_neighbor_graph(time_pairs, times)
    ts_idx = {ts: i for i, ts in enumerate(times)}

    core_distances, coords = _compute_core_distance(data, time_pairs, times, use_lon_lat, traj_cols, min_pts)

    edges_sorted, d_graph = _build_hdbscan_graphs(coords, ts_idx, neighbors, core_distances, use_lon_lat)
    
    core_distances = pd.Series(core_distances).sort_index()
    core_distances.index.name = 'time'

    label_history_df, hierarchy_df = hdbscan(
        edges_sorted_df=edges_sorted,
        core_distances=core_distances,
        d_graph=d_graph,
        min_cluster_size=min_cluster_size,
        dur_min=dur_min)
    
    cluster_stability_df = compute_cluster_stability(label_history_df) # default old func
    # cluster_stability_df = compute_cluster_stability(label_history_df, cdf_function=_piecewise_linear_cdf)
    selected_clusters = select_most_stable_clusters(hierarchy_df, cluster_stability_df)

    final_labels = pd.Series(-1, index=core_distances.index, name='cluster', dtype=int)
    
    if not selected_clusters: # Handle case with no stable clusters
        final_labels.index = data.index
        return final_labels
        
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

        include_border_points = True
        if include_border_points:
            # 2. Find border points for these unclaimed cores at this scale
            border_map = _build_border_map(scale, core_distances, d_graph)
            potential_borders = set().union(*(border_map.get(ts, set()) for ts in unclaimed_cores))
            
            # Assign only unclaimed border points
            unclaimed_borders = potential_borders - claimed_points
            
            # 3. Assign labels and update claimed set
            all_new_members = unclaimed_cores.union(unclaimed_borders)
        else:
            # just the core points
            all_new_members = unclaimed_cores
        
        if all_new_members:
            final_labels.loc[list(all_new_members)] = cid
            claimed_points.update(all_new_members)

    # Align index with original dataframe before returning
    final_labels.index = data.index
    
    return final_labels

def st_hdbscan(
    data,
    time_thresh,
    min_pts=2,
    min_cluster_size=1,
    dur_min=5,
    complete_output=False,
    passthrough_cols=[],
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
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        first = arr[0]
        if any(x != first for x in arr[1:]):
            raise ValueError("Multi-user data? Use lachesis_per_user instead.")
        passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]
    else:
        uid_col = None
        
    labels_hdbscan = hdbscan_labels(
        data=data,
        time_thresh=time_thresh,
        min_pts=min_pts,
        min_cluster_size=min_cluster_size,
        dur_min=dur_min,
        passthrough_cols=passthrough_cols,
        traj_cols=traj_cols,
        **kwargs
    )

    merged = data.join(labels_hdbscan)
    merged = merged[merged.cluster != -1]

    stop_table = merged.groupby('cluster', as_index=False, sort=False).apply(
        lambda grouped_data: utils.summarize_stop(
            grouped_data,
            complete_output=complete_output,
            traj_cols=traj_cols,
            keep_col_names=True,
            passthrough_cols=passthrough_cols,
            **kwargs
        ),
        include_groups=False
    )

    return stop_table

def st_hdbscan_per_user(
    data,
    time_thresh,
    min_pts=2,
    min_cluster_size=1,
    dur_min=5,
    complete_output=False,
    passthrough_cols=[],
    traj_cols=None,
    **kwargs
):
    """
    Run HDBSCAN-based stop detection on each user separately, then concatenate results.
    Raises if 'user_id' not in traj_cols or missing from data.
    """

    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("st_hdbscan_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    pt_cols = passthrough_cols + [uid]

    results = [
        st_hdbscan(
            data=group,
            time_thresh=time_thresh,
            min_pts=min_pts,
            min_cluster_size=min_cluster_size,
            dur_min=dur_min,
            complete_output=complete_output,
            passthrough_cols=pt_cols,
            traj_cols=traj_cols,
            **kwargs
        )
        for _, group in data.groupby(uid, sort=False)
    ]
    return pd.concat(results, ignore_index=True)