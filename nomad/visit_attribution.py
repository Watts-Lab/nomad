
import geopandas as gpd
import nomad.io.base as loader
import nomad.constants as constants
from shapely.geometry import Point
import warnings
import pandas as pd
import nomad.io.base as loader

def point_in_polygon(traj, labels, stop_table, poi_table, traj_cols, is_datetime, is_long_lat):
    # If either labels or stop_table is empty, there's nothing to do
    if labels.empty or stop_table.empty:
        stop_table['location'] = pd.Series(dtype='object')
        return stop_table

    # merge labels with data
    traj_with_labels = traj.copy()
    time_col = traj_cols['datetime'] if is_datetime else traj_cols['timestamp']
    traj_with_labels = traj_with_labels.merge(labels, left_on=time_col, right_index=True, how='left')

    # compute the location for each cluster
    space_cols = [traj_cols['longitude'], traj_cols['latitude']] if is_long_lat else [traj_cols['x'], traj_cols['y']]
    pings_df = traj_with_labels.groupby('cluster')[space_cols].mean()
    
    locations = poi_map(traj=pings_df,
                        poi_table=poi_table,
                        traj_cols=traj_cols)

    # Map the mode location back to the stop_table
    stop_table['location'] = locations

    return stop_table
    
def majority_poi(traj, labels, poi_table, traj_cols, is_datetime, is_long_lat):
    if labels.empty:
        return pd.Series(dtype='object')
    
    # merge labels with data
    traj_with_labels = traj.copy()
    time_col = traj_cols['datetime'] if is_datetime else traj_cols['timestamp']
    traj_with_labels = traj_with_labels.merge(labels, left_on=time_col, right_index=True, how='left')

    # compute the location for each cluster
    space_cols = [traj_cols['longitude'], traj_cols['latitude']] if is_long_lat else [traj_cols['x'], traj_cols['y']]
    pings_df = traj_with_labels[space_cols].copy()
    
    traj_with_labels["building_id"] = poi_map(traj=pings_df,
                                              poi_table=poi_table,
                                              traj_cols=traj_cols)

    locations = traj_with_labels.groupby('cluster')['building_id'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    return locations


def poi_map(traj, poi_table, traj_cols=None, max_distance=4, traj_crs=None, **kwargs):
    """
    Map elements in traj to closest polygon in poi_table with an allowed distance buffer.

    Parameters
    ----------
    traj : pd.DataFrame
        The trajectory DataFrame containing x and y or longitude and latitude coordinates.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
    traj_cols : list
        The columns in the trajectory DataFrame to be used for mapping.
    max_distance : float
        The maximum distance to search for a nearest POI.
    traj_crs : str, optional
        The Coordinate Reference System of the trajectory coordinates.
        If None, it will be inferred.
    **kwargs : dict
        Additional keyword arguments, potentially including 'latitude' and 'longitude' keys.

    Returns
    -------
    pd.Series
        A Series, indexed like `traj`, containing the building IDs for each point.
    """
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)
    loader._has_spatial_cols(traj.columns, traj_cols)

    use_long_lat = 'latitude' in kwargs and 'longitude' in kwargs and \
                   kwargs['latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    

def poi_map(data, poi_table, max_distance=0, data_crs=None, traj_cols=None, **kwargs):
    """
    Map elements in traj to closest polygon in poi_table with an allowed distance buffer.

    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        The trajectory (or stop) data. Can be a DataFrame with coordinate columns or a
        GeoDataFrame with point geometries.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
    traj_cols : list, optional
        Columns in the trajectory DataFrame to be used for mapping.
    max_distance : float
        The maximum distance to search for a nearest POI.
    data_crs : str or pyproj.CRS, optional
        The Coordinate Reference System of the trajectory coordinates. Ignored if
        `traj` is a GeoDataFrame.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pd.Series
        A Series, indexed like `traj`, containing the building IDs for each point.
    """
    # Column name handling
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    loader._has_spatial_cols(data.columns, traj_cols)

    # use_lat_lon and CRS handling
    if not (traj_crs is None or isinstance(traj_crs, (str, pyproj.CRS))):
        raise TypeError(f"CRS {data_crs} must be a string, a pyproj.CRS object, or None.")
        
    if isinstance(data, gpd.GeoDataFrame):
        pings_gdf = data
        if traj_crs and not pyproj.CRS(pings_gdf.crs).equals(pyproj.CRS(data_crs)):
            raise ValueError(f"Provided CRS {data_crs} conflicts with traj CRS {data.crs}.")

    elif isinstance(data, pd.DataFrame):
        use_lon_lat = (traj_cols['latitude'] in data.columns and traj_cols['longitude'] in data.columns)
        if traj_crs and not pyproj.CRS(data_crs).equals(pyproj.CRS("EPSG:4326")):
            use_lon_lat = False
            if not (traj_cols['x'] in data.columns and traj_cols['y'] in data.columns):
                raise ValueError(f"Provided CRS {data_crs} is incompatible with spherical coordinates: 'longitude' and 'latitude'"
                                "if using alternate coordinates, set arguments 'x' and 'y' to coordinate names."
                                )
        if use_lon_lat:
            x_col, y_col = traj_cols['longitude'], traj_cols['latitude']
            if traj_crs is None:
                warnings.warn("Argument `data_crs` not provided, assuming EPSG:4326")
                assigned_crs = pyproj.CRS("EPSG:4326")
            else:
                assigned_crs = pyproj.CRS(data_crs)
        else:
            x_col, y_col = traj_cols['x'], traj_cols['y']}
            if data_crs is None:
                warnings.warn(f"Argument `data_crs` not provided, assuming CRS {poi_table.crs} from `poi_table`")
                assigned_crs = poi_table.crs
            else:
                assigned_crs = pyproj.CRS(data_crs)

        if assigned_crs is None:
            warnings.warn(f"No CRS provided for `data` and `poi_table`, spatial operations may be innacurate.")
        elif assigned_crs != poi_table.crs:
            raise ValueError("CRS for `data` does not match CRS for `poi_table`.")

        pings_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(data[x_col], data[y_col]),
            crs=assigned_crs,
            index=data.index)

    else:
        raise TypeError("`traj` must be a pandas DataFrame or a GeoDataFrame.")

    # wrap sjoin_nearest
    joined = gpd.sjoin_nearest(
        pings_gdf,
        poi_table[['building_id', 'geometry']],
        how="left",
        max_distance=max_distance
    )
    building_ids = joined[~joined.index.duplicated(keep='first')]['building_id']
    return building_ids.reindex(data.index)

def oracle_map(traj, true_visits, traj_cols, **kwargs):
    """
    Map elements in traj to ground truth location based solely on the record's time.

    Parameters
    ----------
    traj : pd.DataFrame
        The trajectory DataFrame containing x and y coordinates.
    true_visits : pd.DataFrame
        A visitation table containing location IDs, start times, and durations/end times.
    traj_cols : list
        The columns in the trajectory DataFrame to be used for mapping.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pd.Series
        A Series containing the location IDs corresponding to the pings in the trajectory.
    """ 
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)

    loader._has_time_cols(traj.columns, traj_cols)
    loader._has_time_cols(true_visits.columns, traj_cols)

    end_col_present = _has_end_cols(true_visits.columns, traj_cols)
    duration_col_present = _has_duration_cols(true_visits.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        print("Missing required (end or duration) temporal columns for true_visits dataframe.")
        return False
    
    use_datetime = False
    if traj_cols['timestamp'] in traj.columns:
        time_col_in = traj_cols['timestamp']
        time_key = 'timestamp'
    elif traj_cols['start_timestamp'] in traj.columns:
        time_col_in = traj_cols['start_timestamp']
        time_key = 'start_timestamp'
    else:
        use_datetime = True 

    # TO DO: Check the same thing for true_visits start, and duration/end
    # TO DO: Conversion of everything to UTC
    # Loop through ground truth and numpy timestamps diff
    
    # Check whether to use timestamp or datetime columns
    is_long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in traj.columns and kwargs['longitude'] in traj.columns
    
    return location_ids