import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import warnings
from nomad import data
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

def _borders_from_cores(scale, core_set, core_distances, G):
    """
    For each core in `core_set`, walk its G-neighbors and assign any non-core node
    reachable within `scale` to its nearest core by edge weight.
    Returns {core_ts: set(border_ts)}.
    """
    border_to_best = {}  # border_ts -> (core_ts, edge_weight)
    for core_ts in core_set:
        for nb, edge_data in G[core_ts].items():
            if nb == core_ts:
                continue
            if core_distances.at[nb] > scale:  # nb is non-core at this scale
                w = np.round(edge_data['weight'] * 4) / 4
                if w <= scale:
                    if nb not in border_to_best or w < border_to_best[nb][1]:
                        border_to_best[nb] = (core_ts, w)
    result = defaultdict(set)
    for border_ts, (core_ts, _) in border_to_best.items():
        result[core_ts].add(border_ts)
    return result

def cluster_hierarchy(edges_sorted, core_distances, G, H, min_cluster_size,
                      data, traj_cols, s_tree, node_times, dist_thresh,
                      dur_min=5, min_pts=2):
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
    H : nx.Graph
        Precomputed hierarchy graph (MST plus self-loops).
    min_cluster_size : int
        Minimum number of core points for a cluster to be considered valid.
    dur_min : int
        Minimum duration in minutes for a cluster to be considered valid.

    Returns
    -------
    tuple
        (label_history_df, hierarchy_df)
    """
    _, coord_key1, coord_key2, _, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, {})

    hierarchy = []
    label_history = []

    # Build full ping index once and reuse it for label snapshots.
    all_pings = pd.Index(G.nodes(), name='time')
    nx.set_node_attributes(H, 0, 'cluster_id')
    nx.set_node_attributes(H, -1, 'temp_cluster_id')

    # Initial state is known: all nodes belong to cluster 0.
    label_history.append(pd.DataFrame({
        'time': all_pings,
        'cluster_id': 0,
        'dendogram_scale': np.nan,
    }))

    current_label_id = 1

    # Per-cluster border state. Three plain dicts replace the old flat border_map:
    #   _cluster_border_map   : cluster_id -> {core_ts: set(border_ts)}  (computed lazily, cached here)
    #   _cluster_border_pool  : cluster_id -> set(border_ts)             (inherited pool; None = unrestricted)
    #   _cluster_birth_scale  : cluster_id -> float                      (scale at which cluster was born)
    #
    # Invariant: a cluster's border pool is always a subset of its parent's border pool.
    # When a cluster splits, each child inherits only the borders whose assigned core
    # fell inside that child's component.
    _cluster_border_map   = {}           # populated lazily via _get_border_map()
    _cluster_border_pool  = {0: None}    # root is unrestricted
    _cluster_birth_scale  = {0: np.inf}  # root has no meaningful birth scale
 
    def _get_border_map(cluster_id):
        """Return the pre-computed border map for *cluster_id*.
        Non-root clusters are always populated eagerly at split time; the root (0)
        has no parent borders and falls back to an empty map."""
        if cluster_id not in _cluster_border_map:
            _cluster_border_map[cluster_id] = defaultdict(set)
        return _cluster_border_map[cluster_id]

    # Iteratively process pruning events grouped by weight (scale)
    for scale, edges_to_remove in edges_sorted.groupby(edges_sorted, sort=False):
        edges_batch = list(edges_to_remove.index)
        idx_from = edges_to_remove.index.get_level_values('from')
        idx_to = edges_to_remove.index.get_level_values('to')
        event_nodes = idx_from.union(idx_to)

        # Remove both regular edges and self-loops at this scale.
        H.remove_edges_from(edges_batch)
        # Remove edges, everything has temp_cluster_id of -1
        nx.set_node_attributes(H, -1, 'temp_cluster_id')

        _split_entries = {}
        _parent_comp_count = defaultdict(int)

        # Assigns a temp_id from 0 to k for each component, where k is the number of children of a single parent_id
        # Drops non-cores
        for seed in event_nodes:
            if not H.has_node(seed):
                continue

            if H.nodes[seed]['temp_cluster_id'] != -1:
                continue

            parent_id = H.nodes[seed]['cluster_id']
            component_nodes = nx.node_connected_component(H, seed)

            if len(component_nodes) == 1:
                node = next(iter(component_nodes))
                if not H.has_edge(node, node):
                    H.remove_node(node)
                    continue

            temp_id = _parent_comp_count[parent_id]
            # default dict guards against missing keys, so this is safe even if parent_id is new (defaults to 0).
            _parent_comp_count[parent_id] += 1
            for node in component_nodes:
                H.nodes[node]['temp_cluster_id'] = temp_id
                _split_entries[node] = (parent_id, temp_id)

        # split_df: DataFrame indexed by sorted timestamp, columns = (parent_id, temp_id).
        # parent_id identifies the cluster being split; temp_id identifies which sub-component the node belongs to.
        split_df = pd.DataFrame.from_dict(
            _split_entries, orient='index', columns=['parent_id', 'temp_id']
        ).sort_index()

        for parent_id in split_df['parent_id'].unique():
            parent_df = split_df[split_df['parent_id'] == parent_id]
            components = [set(grp.index) for _, grp in parent_df.groupby('temp_id')]

            border_map = _get_border_map(parent_id)
            # union of core timestamps and border timestamps for this parent cluster
            all_ts = sorted(set(parent_df.index) | set().union(*border_map.values()))
            cluster_df = pd.Series(-1, index=all_ts, name='cluster')
            cluster_df.loc[parent_df.index] = parent_id

            if len(components) >= 2:
                # Iterate chronologically; 'active_temp_id' is the current main thread.
                # The check fires on the first step away from the active component
                # (prev_temp_id == active but curr_temp_id != active).
                # Separating "decide" (top if) from "advance window" (bottom if) avoids
                # duplicating the coordinate update in both the normal and switch branches.
                active_temp_id = parent_df.iloc[0]['temp_id']
                prev_core = None; prev_coords = None
                prev_prev_core = None; prev_prev_coords = None
                prev_temp_id = None

                for curr_time in parent_df.index:
                    curr_temp_id = parent_df.at[curr_time, 'temp_id']

                    if curr_temp_id != active_temp_id and prev_temp_id == active_temp_id:
                        # First step away from active: evaluate the transition.
                        check_time  = prev_core
                        future_core = curr_time

                        core_time_range = sorted(abs(nb - check_time) for nb in G[check_time])[min_pts - 1]

                        anchor        = prev_prev_core   if prev_prev_core   is not None else check_time
                        anchor_coords = prev_prev_coords if prev_prev_coords is not None else prev_coords

                        future_pos    = np.searchsorted(node_times, future_core)
                        future_coords = data.iloc[future_pos][[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy()

                        denom = future_core - anchor
                        alpha = (check_time - anchor) / denom if denom != 0 else 0.0
                        counterfactual_coords = anchor_coords + alpha * (future_coords - anchor_coords)

                        if use_lon_lat:
                            spatial_nb_idx = s_tree.query_radius(
                                # _find_neighbors builds BallTree in [lat, lon] radians.
                                np.radians(counterfactual_coords[[1, 0]]).reshape(1, -1),
                                r=dist_thresh / 6_371_000,
                            )[0]
                        else:
                            spatial_nb_idx = s_tree.query_radius(
                                np.asarray(counterfactual_coords).reshape(1, -1),
                                r=dist_thresh,
                            )[0]

                        if len(spatial_nb_idx) >= min_pts:
                            counterfactual_time_range = np.sort(np.abs(node_times[spatial_nb_idx] - check_time))[min_pts - 1]
                            new_active_cluster = (core_time_range <= counterfactual_time_range)
                        else:
                            new_active_cluster = False

                        if not new_active_cluster:
                            split_df.at[check_time, 'parent_id'] = -1
                            cluster_df.at[check_time] = -1
                        else:
                            active_temp_id = curr_temp_id

                    if curr_temp_id == active_temp_id:
                        prev_prev_core, prev_prev_coords = prev_core, prev_coords
                        prev_core = curr_time
                        _pos = np.searchsorted(node_times, curr_time)
                        prev_coords = data.iloc[_pos][[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy()

                    prev_temp_id = curr_temp_id

            non_spurious = []
            nodes_to_drop = set()

            for component_nodes in components:
                border_nodes = set()
                for ts in component_nodes:
                    border_nodes.update(border_map.get(ts, ()))

                comp_min = min(component_nodes)
                comp_max = max(component_nodes)
                if border_nodes:
                    comp_min = min(comp_min, min(border_nodes))
                    comp_max = max(comp_max, max(border_nodes))

                if ((comp_max - comp_min) >= dur_min * 60) and (len(component_nodes) >= min_cluster_size):
                    non_spurious.append(component_nodes)
                else:
                    nodes_to_drop.update(component_nodes)

            if nodes_to_drop:
                H.remove_nodes_from(nodes_to_drop)

            if len(non_spurious) == 0:
                continue

            if len(non_spurious) == 1:
                # Remaining child already has parent_id.
                continue

            new_ids = []
            for component in non_spurious:
                for node in component:
                    if H.has_node(node):
                        H.nodes[node]['cluster_id'] = current_label_id

                new_ids.append(current_label_id)
                current_label_id += 1
            
            # Partition the parent's border pool among the newly minted children.
            # Each child inherits only the borders whose assigned core fell in its component.
            parent_map = _get_border_map(parent_id)
            all_cores_at_scale = core_distances.index[core_distances <= scale]
            raw_at_scale = _borders_from_cores(scale, all_cores_at_scale, core_distances, G)
            for component, child_id in zip(non_spurious, new_ids):
                core_set = set(component)
                child_pool = set()
                for core_ts, borders in parent_map.items():
                    if core_ts in core_set:
                        child_pool.update(borders)
                _cluster_birth_scale[child_id] = scale
                _cluster_border_pool[child_id] = child_pool
                child_map = defaultdict(set)
                for core_ts, borders in raw_at_scale.items():
                    if core_ts in core_set:
                        kept = borders & child_pool if child_pool else borders
                        if kept:
                            child_map[core_ts] = kept
                _cluster_border_map[child_id] = child_map

            hierarchy.append((scale, parent_id, new_ids))

        # O(N) per scale: get_node_attributes returns {node: cluster_id}.
        cluster_ids = pd.Series(nx.get_node_attributes(H, 'cluster_id'))
        cluster_ids = cluster_ids.reindex(all_pings, fill_value=-1)

        label_history.append(pd.DataFrame({
            'time': cluster_ids.index,
            'cluster_id': cluster_ids.values,
            'dendogram_scale': scale,
        }))

    # combine label history into one DataFrame
    label_history_df = pd.concat(label_history, ignore_index=True)
    # build cluster lineage for all clusters
    hierarchy_df = _build_cluster_lineage(hierarchy)
    return label_history_df, hierarchy_df


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
    H : nx.Graph
        Hierarchy graph with mutual-reachability MST edges and core-distance
        self-loops.
    edges_sorted_df : pd.Series
        H sorted descending by weight, MultiIndex (from, to).
    """
    G_copy = G.copy()
    for u, v, data in G_copy.edges(data=True):
        d = np.round(data["weight"] * 4) / 4
        data["weight"] = max(core_dist.at[u], core_dist.at[v], d)

    H = nx.minimum_spanning_tree(G_copy)

    H.add_edges_from((node, node, {'weight': weight}) for node, weight in core_dist.items())

    all_edges = nx.to_pandas_edgelist(H, source='from', target='to')
    all_edges.sort_values('weight', ascending=False, inplace=True)

    all_edges.set_index(['from', 'to'], inplace=True)
    return H, all_edges['weight']

def hdbscan_labels(data,
                   time_thresh,
                   min_pts = 2,
                   min_cluster_size = 1,
                   dur_min=5,
                   delta_roam=None,
                   dist_thresh=None,
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

    G, t_tree, s_tree = _find_neighbors(data, time_thresh, traj_cols, dist_thresh,
                    weighted=True, use_datetime=use_datetime, use_lon_lat=use_lon_lat,
                    return_trees=True, relabel_nodes=True)
    node_times = np.asarray(list(G), dtype=np.float64)

    core_distances = _compute_core_distance(G, min_pts)

    H, edges_sorted = _build_hdbscan_graphs(G, core_distances)

    label_history_df, hierarchy_df = cluster_hierarchy(
        edges_sorted=edges_sorted,
        core_distances=core_distances,
        G=G,
        H=H,
        min_cluster_size=min_cluster_size,
        data=data,
        traj_cols=traj_cols,
        s_tree=s_tree,
        node_times=node_times,
        dist_thresh=dist_thresh,
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
            all_cores_at_scale = core_distances.index[core_distances <= scale]
            border_map = _borders_from_cores(scale, all_cores_at_scale, core_distances, G)
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
