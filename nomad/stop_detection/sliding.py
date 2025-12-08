import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, MultiPoint
import warnings
import nomad.io.base as loader


def _get_location_center(coords, metric='euclidean'):
    """
    Calculate the center of a location cluster.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates (n_points, 2)
    metric : str
        'euclidean' for projected coordinates, 'haversine' for lat/lon

    Returns
    -------
    tuple
        (x, y) coordinates of the center
    """
    if metric == 'haversine':
        # For geographic coordinates, use angle-based mean
        # Convert to radians for calculation
        coords_rad = np.radians(coords)
        x = coords_rad[:, 0]
        y = coords_rad[:, 1]

        # Convert to 3D Cartesian coordinates
        cos_y = np.cos(y)
        cart_x = cos_y * np.cos(x)
        cart_y = cos_y * np.sin(x)
        cart_z = np.sin(y)

        # Average and convert back
        avg_x = np.mean(cart_x)
        avg_y = np.mean(cart_y)
        avg_z = np.mean(cart_z)

        lon = np.arctan2(avg_y, avg_x)
        hyp = np.sqrt(avg_x**2 + avg_y**2)
        lat = np.arctan2(avg_z, hyp)

        return np.degrees(lon), np.degrees(lat)
    else:
        # For projected coordinates, use simple mean
        return np.mean(coords[:, 0]), np.mean(coords[:, 1])


def _get_location_extent(points, epsilon, crs=None):
    """
    Calculate the spatial extent of a location.

    Parameters
    ----------
    points : list of shapely Points
        Points in the location cluster
    epsilon : float
        Buffer distance in meters (or degrees for unprojected)
    crs : str, optional
        Coordinate reference system

    Returns
    -------
    shapely.Polygon
        Convex hull buffered by epsilon
    """
    if len(points) == 1:
        return points[0].buffer(epsilon)

    multipoint = MultiPoint(points)
    convex_hull = multipoint.convex_hull
    return convex_hull.buffer(epsilon)


def cluster_locations_dbscan(
    stops,
    epsilon=100,
    num_samples=1,
    distance_metric='euclidean',
    agg_level='user',
    traj_cols=None,
    **kwargs
):
    """
    Cluster stops into locations using DBSCAN.

    Parameters
    ----------
    stops : pd.DataFrame or gpd.GeoDataFrame
        Stop/staypoint data with spatial columns
    epsilon : float, default 100
        Maximum distance between stops in the same location (meters for haversine/euclidean)
    num_samples : int, default 1
        Minimum number of stops required to form a location
    distance_metric : str, default 'euclidean'
        Distance metric: 'euclidean' for projected coords, 'haversine' for lat/lon
    agg_level : str, default 'user'
        'user' = separate locations per user, 'dataset' = shared locations across users
    traj_cols : dict, optional
        Column name mappings for coordinates and user_id
    **kwargs
        Additional arguments passed to column detection

    Returns
    -------
    tuple of (pd.DataFrame, gpd.GeoDataFrame)
        - stops with added 'location_id' column (NaN for unclustered stops)
        - locations GeoDataFrame with cluster centers and extents

    Examples
    --------
    >>> # Cluster stops within 50 meters into locations
    >>> stops_labeled, locations = cluster_locations_dbscan(
    ...     stops, epsilon=50, num_samples=2
    ... )

    >>> # Use dataset-level clustering (locations shared across users)
    >>> stops_labeled, locations = cluster_locations_dbscan(
    ...     stops, epsilon=100, agg_level='dataset'
    ... )
    """
    if not isinstance(stops, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Input 'stops' must be a pandas DataFrame or GeoDataFrame")

    if stops.empty:
        # Return empty results with proper schema
        stops_out = stops.copy()
        stops_out['location_id'] = pd.Series(dtype='Int64')
        locations = gpd.GeoDataFrame(
            columns=['center', 'extent'],
            geometry='center',
            crs=stops.crs if isinstance(stops, gpd.GeoDataFrame) else None
        )
        return stops_out, locations

    # Parse column names
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)
    loader._has_spatial_cols(stops.columns, traj_cols)

    # Determine coordinate columns
    if 'longitude' in traj_cols and traj_cols['longitude'] in stops.columns:
        coord_key1, coord_key2 = 'longitude', 'latitude'
        use_lon_lat = True
    elif 'x' in traj_cols and traj_cols['x'] in stops.columns:
        coord_key1, coord_key2 = 'x', 'y'
        use_lon_lat = False
    else:
        raise ValueError("Could not find spatial columns in stops data")

    # Override distance metric based on coordinate type if not specified
    if distance_metric == 'euclidean' and use_lon_lat:
        warnings.warn(
            "Using haversine metric for lat/lon coordinates instead of euclidean",
            UserWarning
        )
        distance_metric = 'haversine'

    # Get user_id column if present
    user_col = None
    if 'user_id' in traj_cols and traj_cols['user_id'] in stops.columns:
        user_col = traj_cols['user_id']

    # Check aggregation level
    if agg_level == 'user' and user_col is None:
        warnings.warn(
            "agg_level='user' requires user_id column; falling back to 'dataset'",
            UserWarning
        )
        agg_level = 'dataset'

    stops_out = stops.copy()
    stops_out['location_id'] = pd.Series(dtype='Int64')

    location_list = []
    location_id_counter = 0

    # Group by user if needed
    if agg_level == 'user':
        groups = stops_out.groupby(user_col, sort=False)
    else:
        groups = [(None, stops_out)]

    for group_key, group_data in groups:
        if group_data.empty:
            continue

        # Extract coordinates
        coords = group_data[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy(dtype='float64')

        # For haversine, convert to radians
        if distance_metric == 'haversine':
            # Convert epsilon from meters to radians (approximate)
            # Earth radius in meters
            epsilon_rad = epsilon / 6371000.0
            coords_for_clustering = np.radians(coords)
        else:
            epsilon_rad = epsilon
            coords_for_clustering = coords

        # Run DBSCAN
        clusterer = DBSCAN(
            eps=epsilon_rad,
            min_samples=num_samples,
            metric=distance_metric,
            algorithm='ball_tree'
        )

        labels = clusterer.fit_predict(coords_for_clustering)

        # Assign location IDs (offset by counter for multi-group)
        group_loc_ids = labels.copy()
        valid_mask = labels >= 0
        group_loc_ids[valid_mask] += location_id_counter
        group_loc_ids[~valid_mask] = -1

        # Update stops with location IDs
        stops_out.loc[group_data.index, 'location_id'] = group_loc_ids

        # Create location entries for each cluster
        unique_labels = labels[labels >= 0]
        if len(unique_labels) > 0:
            for label in np.unique(unique_labels):
                cluster_mask = labels == label
                cluster_coords = coords[cluster_mask]
                cluster_indices = group_data.index[cluster_mask]

                # Calculate center
                center_x, center_y = _get_location_center(
                    cluster_coords,
                    metric=distance_metric
                )
                center_point = Point(center_x, center_y)

                # Calculate extent (convex hull + buffer)
                cluster_points = [Point(x, y) for x, y in cluster_coords]
                extent = _get_location_extent(
                    cluster_points,
                    epsilon,
                    crs=stops.crs if isinstance(stops, gpd.GeoDataFrame) else None
                )

                location_entry = {
                    'location_id': location_id_counter + label,
                    'center': center_point,
                    'extent': extent,
                    'n_stops': cluster_mask.sum()
                }

                # Add user_id if available
                if user_col is not None and agg_level == 'user':
                    location_entry[user_col] = group_key

                location_list.append(location_entry)

            # Update counter for next group
            location_id_counter += len(np.unique(unique_labels))

    # Convert location_id to nullable integer (NaN for unclustered)
    stops_out['location_id'] = stops_out['location_id'].replace(-1, pd.NA).astype('Int64')

    # Create locations GeoDataFrame
    if location_list:
        locations = gpd.GeoDataFrame(
            location_list,
            geometry='center',
            crs=stops.crs if isinstance(stops, gpd.GeoDataFrame) else None
        )
        # Set extent as additional geometry column
        locations['extent'] = gpd.GeoSeries(
            [loc['extent'] for loc in location_list],
            crs=stops.crs if isinstance(stops, gpd.GeoDataFrame) else None
        )
    else:
        locations = gpd.GeoDataFrame(
            columns=['location_id', 'center', 'extent', 'n_stops'],
            geometry='center',
            crs=stops.crs if isinstance(stops, gpd.GeoDataFrame) else None
        )

    return stops_out, locations


def cluster_locations_per_user(
    stops,
    epsilon=100,
    num_samples=1,
    distance_metric='euclidean',
    traj_cols=None,
    **kwargs
):
    """
    Convenience function to cluster locations per user.

    This is equivalent to calling cluster_locations_dbscan with agg_level='user'.

    Parameters
    ----------
    stops : pd.DataFrame or gpd.GeoDataFrame
        Stop data with user_id column
    epsilon : float, default 100
        Maximum distance between stops in same location (meters)
    num_samples : int, default 1
        Minimum stops required to form a location
    distance_metric : str, default 'euclidean'
        'euclidean' or 'haversine'
    traj_cols : dict, optional
        Column name mappings
    **kwargs
        Additional arguments

    Returns
    -------
    tuple of (pd.DataFrame, gpd.GeoDataFrame)
        Stops with location_id and locations table

    Raises
    ------
    ValueError
        If user_id column is not found
    """
    traj_cols_temp = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in stops.columns:
        raise ValueError(
            "cluster_locations_per_user requires a 'user_id' column "
            "specified in traj_cols or kwargs"
        )

    return cluster_locations_dbscan(
        stops,
        epsilon=epsilon,
        num_samples=num_samples,
        distance_metric=distance_metric,
        agg_level='user',
        traj_cols=traj_cols,
        **kwargs
    )


# Alias for convenience
generate_locations = cluster_locations_dbscan
