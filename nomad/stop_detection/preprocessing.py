import pandas as pd
import numpy as np
from collections import defaultdict
from nomad.stop_detection import utils
from nomad.filters import to_timestamp
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree # for haverside distance case
import networkx as nx

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree


def _find_temp_neighbors(times, time_thresh, return_tree=False, relabel_nodes=True):
    """Return the time-neighbor graph, and optionally its KDTree."""
    t_tree = KDTree(times[:, None])
    pairs = t_tree.query_pairs(r=time_thresh * 60, output_type="ndarray")

    G = nx.Graph()
    G.add_nodes_from(range(len(times)))
    G.add_edges_from(pairs)

    if relabel_nodes:
        G = nx.relabel_nodes(G, dict(enumerate(times)))
    
    return (G, t_tree) if return_tree else G


def _find_spatial_neighbors(coords, dist_thresh=None, weighted=False,
                            use_lon_lat=False, return_tree=False, times=None, relabel_nodes=False):
    """Return the spatial neighbor graph, and optionally its KDTree or BallTree."""
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))

    if use_lon_lat:
        earth_radius = 6_371_000
        s_tree = BallTree(coords, metric="haversine")
        
        radius = np.pi if dist_thresh is None else dist_thresh / earth_radius
        if weighted:
            indices, distances = s_tree.query_radius(coords, r=radius, return_distance=True)

            counts = np.array([len(x) for x in indices])
            row = np.repeat(np.arange(len(coords)), counts)
            col = np.concatenate(indices)
            dist = np.concatenate(distances)
            dist = dist * earth_radius

            mask = row < col
            G.add_weighted_edges_from(
                np.column_stack((row[mask], col[mask], dist[mask]))
            )

        elif dist_thresh is not None:
            indices = s_tree.query_radius(coords, r=radius, return_distance=False)

            counts = np.array([len(x) for x in indices])
            row = np.repeat(np.arange(len(coords)), counts)
            col = np.concatenate(indices)

            mask = row < col
            G.add_edges_from(np.column_stack((row[mask], col[mask])))

    else:
        s_tree = KDTree(coords)

        if weighted:
            radius = np.inf if dist_thresh is None else dist_thresh
            sdm = s_tree.sparse_distance_matrix(
                s_tree,
                max_distance=radius,
                output_type="ndarray"
            )
            mask = sdm["i"] < sdm["j"]
            G.add_weighted_edges_from(
                np.column_stack((sdm["i"][mask], sdm["j"][mask], sdm["v"][mask]))
            )

        elif dist_thresh is not None:
            pairs = s_tree.query_pairs(r=dist_thresh, output_type="ndarray")
            G.add_edges_from(pairs)

    if relabel_nodes and times is not None:
        G = nx.relabel_nodes(G, dict(enumerate(times)))
        
    return (G, s_tree) if return_tree else G


def _find_neighbors(data, time_thresh, traj_cols, dist_thresh=None,
                    weighted=False, use_datetime=False, use_lon_lat=False,
                    return_trees=False, relabel_nodes=True):
    """Combine time and spatial neighbors into the final graph."""
    if use_lon_lat:
        coords = np.radians(
            data[[traj_cols["latitude"], traj_cols["longitude"]]].values
        )
    else:
        coords = data[[traj_cols["x"], traj_cols["y"]]].values

    if use_datetime:
        times = to_timestamp(data[traj_cols["datetime"]]).values
    else:
        times = data[traj_cols["timestamp"]].values

    temp_result = _find_temp_neighbors(times, time_thresh, return_tree=return_trees)
    time_graph, t_tree = temp_result if return_trees else (temp_result, None)

    spatial_graph = None
    s_tree = None

    if dist_thresh is not None or weighted or return_trees:
        spatial_result = _find_spatial_neighbors(
            coords,
            dist_thresh=dist_thresh,
            weighted=weighted,
            use_lon_lat=use_lon_lat,
            return_tree=return_trees,
        )
        spatial_graph, s_tree = spatial_result if return_trees else (spatial_result, None)

    if spatial_graph is None:
        G = time_graph
    elif dist_thresh is None:
        G = spatial_graph.edge_subgraph(time_graph.edges).copy() if weighted else time_graph
    else:
        G = spatial_graph.edge_subgraph(time_graph.edges).copy() if weighted else nx.intersection(time_graph, spatial_graph)

    G.add_nodes_from(time_graph.nodes)

    if relabel_nodes:
        G = nx.relabel_nodes(G, dict(enumerate(times)))

    return (G, t_tree, s_tree) if return_trees else G