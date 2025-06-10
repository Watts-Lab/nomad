
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

    
def poi_map(traj, poi_table, traj_cols=None, max_distance=4, **kwargs):
    """
    Map elements in traj to closest polygon in poi_table with an allowed distance buffer.

    Parameters
    ----------
    traj : pd.DataFrame
        The trajectory DataFrame containing x and y coordinates.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
    traj_cols : list
        The columns in the trajectory DataFrame to be used for mapping.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pd.Series
        A Series containing the building IDs corresponding to the pings in the trajectory.
    """
    # TO DO: warning if CRS is None. If is_long_lat inform that EPSG:4326 will be used.
    # TO DO: if not is_long_lat and CRS is None, Raise Error and inform of CRS of poi_table. 
    # TO DO: if POI_table has no CRS Raise Error. If poi_table has different CRS? ValueError suggest reprojection

    # Check if user wants long and lat
    is_long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    # Set initial schema
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)
    loader._has_spatial_cols(traj.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns and not is_long_lat:
        is_long_lat = False
        pings_df = traj[[traj_cols['x'], traj_cols['y']]].copy()
    else:
        is_long_lat = True
        pings_df = traj[[traj_cols['longitude'],  traj_cols['latitide']]].copy()

    # Build pings GeoDataFrame
    # use gpd.points_from_xy
    pings_df["pings_geometry"] = pings_df.apply(lambda row: Point(row[traj_cols['longitude']], row[traj_cols['latitude']]) if is_long_lat else Point(row[traj_cols['x']], row[traj_cols['y']]), axis=1)
    pings_df = gpd.GeoDataFrame(pings_df, geometry="pings_geometry", crs=poi_table.crs)
    
    # First spatial join (within)
    pings_df = gpd.sjoin(pings_df, poi_table, how="left", predicate="within")
   
    # Identify unmatched pings
    unmatched_mask = pings_df["building_id"].isna()
    unmatched_pings = pings_df[unmatched_mask].drop(columns=["building_id", "index_right"])

    if not unmatched_pings.empty:
        # Nearest spatial join for unmatched pings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
            nearest = gpd.sjoin_nearest(unmatched_pings, poi_table, how="left", max_distance=max_distance)

        # Keep only the first match for each original ping
        nearest = nearest.groupby(nearest.index).first()

        # Update original DataFrame with nearest matches
        pings_df.loc[unmatched_mask, "building_id"] = nearest["building_id"].values

    return pings_df["building_id"]

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