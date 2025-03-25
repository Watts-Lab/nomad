import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import datetime as dt
from datetime import timedelta
import itertools
import os
import nomad.constants as constants

def _diameter(coords, metric='euclidean'):
    """
    Calculate the diameter of a set of coordinates, defined as the maximum pairwise distance.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates, where each row represents a point in space.
    metric : str, optional
        Distance metric to use. Supported metrics include 'euclidean' (default) 
        and 'haversine'. If 'haversine' is used, coordinates should be in degrees.
    
    Returns
    -------
    float
        The diameter of the coordinate set, i.e., the maximum pairwise distance.
        Returns 0 if there are fewer than two coordinates.
    """
    if len(coords) < 2:
        return 0
        
    if metric == 'haversine':
        coords = np.radians(coords)
        pairwise_dists = pdist(coords,
                               metric=lambda u, v: _haversine_distance(u, v))
        return np.max(pairwise_dists)
    return np.max(pdist(coords, metric=metric))


def _medoid(coords, metric='euclidean'):
    """
    Calculate the medoid of a set of coordinates, defined as the point with the minimal 
    sum of distances to all other points.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates, where each row represents a point in space.
    metric : str, optional
        Distance metric to use. Supported metrics include 'euclidean' (default) 
        and 'haversine'. If 'haversine' is used, coordinates should be in degrees.
    
    Returns
    -------
    numpy.ndarray
        The medoid of the coordinate set, represented as a single point. If there
        is only one point, returns that point.
    """
    if len(coords) < 2:
        return coords[0]
    
    if metric == 'haversine':
        coords = np.radians(coords)
        distances = _pairwise_haversine(coords)
    else:
        distances = cdist(coords, coords, metric=metric)
    
    sum_distances = np.sum(distances, axis=1)
    medoid_index = np.argmin(sum_distances)
    return coords[medoid_index, :]


def _haversine_distance(coord1, coord2):
    """
    Compute the haversine distance between two points on Earth.

    Parameters:
        coord1: [lat1, lon1] in radians
        coord2: [lat2, lon2] in radians

    Returns:
        Distance in meters.
    """
    earth_radius_meters = 6371000  # Earth's radius in meters
    delta_lat = coord2[0] - coord1[0]
    delta_lon = coord2[1] - coord1[1]
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(coord1[0]) * np.cos(
        coord2[0]) * np.sin(delta_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_meters * c  # Distance in meters


def _pairwise_haversine(coords):
    """
    Compute the pairwise Haversine distances between a set of coordinates.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates, where each row represents a point in [latitude, longitude] 
        in radians.
    
    Returns
    -------
    numpy.ndarray
        A symmetric 2D array where the element at [i, j] represents the Haversine 
        distance between the i-th and j-th points.
    """
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = _haversine_distance(coords[i], coords[j])
            distances[j, i] = distances[i, j]
    return distances

    
def _update_diameter(c_j, coords_prev, D_prev, metric='euclidean'):
    """
    Update the diameter of a set of coordinates when a new point is added.
    
    Parameters
    ----------
    c_j : numpy.ndarray
        The new point being added to the coordinate set.
    coords_prev : numpy.ndarray
        Array of existing coordinates, where each row represents a point.
    D_prev : float
        The previous diameter of the coordinate set, before adding the new point.
    metric : str, optional
        Distance metric to use. Supported metrics are 'euclidean' (default) and 'haversine'.
        If 'haversine' is used, coordinates should be in degrees.
    
    Returns
    -------
    float
        The updated diameter of the coordinate set, considering the new point.
    """
    if metric == 'euclidean':
        X_prev = coords_prev[:, 0]
        Y_prev = coords_prev[:, 1]
        x_j, y_j = c_j[0], c_j[1]
        new_dists = np.sqrt((X_prev - x_j) ** 2 + (Y_prev - y_j) ** 2)

    elif metric == 'haversine':
        coords_prev = np.radians(coords_prev)
        c_j = np.radians(c_j)

        lat_j, lon_j = c_j[0], c_j[1]
        new_dists = np.array([
            _haversine_distance([lat_j, lon_j], [lat_i, lon_i])
            for lat_i, lon_i in coords_prev
        ])

    else:
        raise ValueError("metric must be 'euclidean' or 'haversine'")

    D_i_jp1 = np.max([D_prev, np.max(new_dists)])

    return D_i_jp1