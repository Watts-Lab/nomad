import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import warnings
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.filters import to_timestamp
from nomad.stop_detection.preprocessing import _find_neighbors

def _compute_core_distance(G, min_pts):
    result = {}
    
    for node in G.nodes():
        edges = sorted(G.edges(node, data='weight'), key=lambda e: e[2])
        result[node] = edges[min_pts - 1][2] if len(edges) >= min_pts else np.inf
    
    core_distances = pd.Series(result)
    core_distances.index.name = 'time'
    return core_distances

def _build_border_map(scale, core_distances, G):
    """
    For a given threshold `scale`, assign each non-core point to its nearest
    core neighbor (preceding or succeeding in time) within distance <= scale.

    Parameters
    ----------
    scale : float
        Distance threshold at this dendogram level.
    core_distances : pd.Series
        Indexed by timestamp; contains each point's "core distance."
        Must be sorted by index (timestamp).
    G : nx.Graph
        Weighted graph of distances between temporally-close points.

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

    rows = [
        (b, c, G.edges[b, c]["weight"])
        for b, c in pairs
        if G.has_edge(b, c) and np.round(G.edges[b, c]["weight"] * 4) / 4 <= scale
    ]

    if not rows:
        return defaultdict(set)

    dists = pd.Series(
        [w for _, _, w in rows],
        index=pd.MultiIndex.from_tuples([(b, c) for b, c, _ in rows], names=["border", "core"]),
        name="weight",
    )
    
    # For each border, pick the core with minimal distance (ties break naturally)
    best = dists.groupby(level="border").idxmin().values
    core_to_border = defaultdict(set)
    for border, core in best:
        core_to_border[core].add(border)
    
    return core_to_border # return this but instead using something with input as G

def cluster_hierarchy(edges_sorted, core_distances, G, min_cluster_size, dur_min=5):
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
    for scale, edges_to_remove in edges_sorted.groupby(edges_sorted, sort=False):

        border_map = _build_border_map(scale, core_distances, G)
        idx_from = edges_to_remove.index.get_level_values('from')
        affected_clusters = set(label_map.loc[idx_from].unique())

        for cluster_id in affected_clusters:
            if cluster_id == -1:
                continue

            members = label_map.index[label_map == cluster_id]
            remaining_members = set(members)

            # TODO: try to keep the original graph (mst + self loops), instead of building a new one and then getting the connected components
            # use the networkx filter edges method (all edges <)
            # benchmark prior to this change
            adj = _build_graph_pd(members, edges_sorted, edges_to_remove)
            components = _connected_components(adj)
            
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

# def compute_cluster_duration(cluster):
#     max_time = max(cluster)
#     min_time = min(cluster)
#     return (max_time - min_time) * 60

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
    edges_sorted_df : pd.Series
        MST + self-loops sorted descending by weight, MultiIndex (from, to).
    """
    H = G.copy()
    for u, v, data in H.edges(data=True):
        d = np.round(data["weight"] * 4) / 4
        data["weight"] = max(core_dist.at[u], core_dist.at[v], d)

    mst = nx.minimum_spanning_tree(H)

    mst.add_edges_from((node, node, {'weight': weight}) for node, weight in core_dist.items())

    all_edges = nx.to_pandas_edgelist(mst, source='from', target='to')
    all_edges.sort_values('weight', ascending=False, inplace=True)

    all_edges.set_index(['from', 'to'], inplace=True)
    return all_edges['weight']

def hdbscan_labels(data,
                   time_thresh,
                   min_pts = 2,
                   min_cluster_size = 1,
                   dur_min=5,
                   delta_roam=None,
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
    
    # Handle empty data
    if data.empty:
        return pd.Series([], dtype=int, name='cluster')
    
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

    G = _find_neighbors(data, time_thresh, traj_cols, dist_thresh=None,
                    weighted=True, use_datetime=use_datetime, use_lon_lat=use_lon_lat,
                    return_trees=False, relabel_nodes=True)
    
    # neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}
    # time_pairs, times = _find_temp_neighbors(data[traj_cols[t_key]], time_thresh, use_datetime)
    # neighbors = _build_neighbor_graph(time_pairs, times)
    # ts_idx = {ts: i for i, ts in enumerate(times)}
    # core_distances, coords = _compute_core_distance(data, time_pairs, times, use_lon_lat, traj_cols, min_pts)

    core_distances = _compute_core_distance(G, min_pts)

    edges_sorted = _build_hdbscan_graphs(G, core_distances)

    label_history_df, hierarchy_df = cluster_hierarchy(
        edges_sorted=edges_sorted,
        core_distances=core_distances,
        G=G,
        min_cluster_size=min_cluster_size,
        dur_min=dur_min,
    )

    if delta_roam is None:
        cluster_stability_df = compute_cluster_stability(label_history_df)
        selected_clusters = select_most_stable_clusters(hierarchy_df, cluster_stability_df)
    else:
        selected_clusters = select_clusters_by_epsilon(hierarchy_df, label_history_df, epsilon=delta_roam)

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
            border_map = _build_border_map(scale, core_distances, G)
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
        if len(arr) > 0:
            first = arr[0]
            if any(x != first for x in arr[1:]):
                raise ValueError("Multi-user data? Use hdbscan_per_user instead.")
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
    
    # Filter out noise points
    merged = merged[merged.cluster != -1]

    if merged.empty:
        # Get column names by calling summarize function on dummy data
        cols = utils._get_empty_stop_columns(
            data.columns, complete_output, passthrough_cols, traj_cols, 
            keep_col_names=True, is_grid_based=False, **kwargs
        )
        return pd.DataFrame(columns=cols, dtype=object)

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
