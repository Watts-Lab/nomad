import geopandas as gpd
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection.ta_dbscan import _stop_metrics
from shapely.geometry import Point
import warnings
import pandas as pd
import nomad.io.base as loader
import pyproj
import pdb

# TO DO: change to stops_to_poi
def point_in_polygon(data, poi_table, method='centroid', data_crs = None, max_distance = 0,
                     cluster_label = None, location_id=None, traj_cols = None, **kwargs):
    ''' 
        cluster label can also be passed on traj_cols or on kwargs
    '''
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, defaults={})
        
    # check if it is stop table
    end_col_present = loader._has_end_cols(data.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(data.columns, traj_cols)
    is_stop_table = (end_col_present or duration_col_present)

    if is_stop_table:
        # is stop table
        if method=='majority':
            raise TypeError("Method `majority' requires ping data with cluster labels,\
                            but a stop table was provided")
        elif method=='centroid':
            stop_table = data.copy()
            location = poi_map(
                data=stop_table,
                poi_table=poi_table,
                max_distance=max_distance,
                data_crs=data_crs,
                location_id=location_id,
                traj_cols=traj_cols,
                **kwargs)
            return location
            
        else:
            raise ValueError(f"Method {method} not among implemented methods: `centroid' and `majority'")

    else:
        # is labeled pings
        if not cluster_label: #try defaults and raise
            if 'cluster_label' in data.columns:
                cluster_label = 'cluster_label'
            elif 'cluster' in data.columns:
                cluster_label = 'cluster'
            else:
                raise ValueError(f"Argument `cluster_label` is required for visit attribution of labeled pings.")

        clustered_pings = data.loc[data[cluster_label] != -1].copy()
        if method=='majority': 
            location = poi_map(
                data=clustered_pings,
                poi_table=poi_table,
                max_distance=max_distance,
                data_crs=data_crs,
                location_id=location_id,
                traj_cols=traj_cols,
                **kwargs                
            )
            loc_col = location.name
            clustered_pings = clustered_pings.join(location)
            
            location = clustered_pings.groupby(cluster_label)[loc_col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            
            return data[[cluster_label]].join(location, on=cluster_label)[loc_col]
            
        elif method=='centroid': # should be medoid?
            loader._has_spatial_cols(data.columns, traj_cols, exclusive=True)
            use_lon_lat = ('latitude' in traj_cols and 'longitude' in traj_cols)
            if use_lon_lat:
                warnings.warn("Spherical ('longitude', 'latitude') coordinates were passed. Centroids will not agree with geodetic distances")                
                centr_data = clustered_pings.groupby(cluster_label).agg({traj_cols['longitude']:'mean', traj_cols['latitude']:'mean'})
            else:
                centr_data = clustered_pings.groupby(cluster_label).agg({traj_cols['x']:'mean', traj_cols['y']:'mean'})

            location = poi_map(
                data=centr_data,
                poi_table=poi_table,
                max_distance=max_distance,
                data_crs=data_crs,
                location_id=location_id,
                traj_cols=traj_cols,
                **kwargs)
            loc_col = location.name
            
            return data[[cluster_label]].join(location, on=cluster_label)[loc_col]

        else:
            raise ValueError(f"Method {method} not among implemented methods: `centroid' and `majority'")

    return None
    
# change to point_in_polygon, move to filters.py
def poi_map(data, poi_table, max_distance=0, data_crs=None, location_id=None, traj_cols=None, **kwargs):
    """
    Map elements in traj to closest polygon in poi_table with an allowed distance buffer.

    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        The spatial data. Can be a DataFrame with coordinate columns or a
        GeoDataFrame with point geometries.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
    traj_cols : list, optional
        Columns in the trajectory DataFrame to be used for mapping.
    max_distance : float
        The maximum distance to search for a nearest POI.
    data_crs : str or pyproj.CRS, optional
        The Coordinate Reference System of the trajectory coordinates. Ignored if
        `data` is a GeoDataFrame.
    location : str
        The column name of the location_id in poi_table, poi_table.index is used if None
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pd.Series
        A Series, indexed like `data`, containing the building IDs for each point.
    """
    # column name handling
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, defaults={})
        
    if poi_table.crs is None:
        raise ValueError(f"poi_table must have crs attribute for spatial join.")
   
    # Determine which geometry to use
    if isinstance(data, gpd.GeoDataFrame):
        pings_gdf = data.geometry
        # if geodataframe, data_crs is ignored but we Raise if conflicting crs because it is suspect
        if data_crs and not pyproj.CRS(pings_gdf.crs).equals(pyproj.CRS(data_crs)):
            raise ValueError(f"Provided CRS {data_crs} conflicts with traj CRS {data.crs}.")

    if isinstance(data, pd.DataFrame):
        # check that user specified x,y or lat, lon but not both
        loader._has_spatial_cols(data.columns, traj_cols, exclusive=True)
        use_lon_lat = ('latitude' in traj_cols and 'longitude' in traj_cols)

        if use_lon_lat:
            if data_crs:
                data_crs = pyproj.CRS(data_crs)
                if data_crs.is_projected:
                    warnings.warn(f"Provided CRS {data_crs.name} is a projected coordinate system, but "
                                 "spherical ('longitude', 'latitude') coordinates were passed. Did you mean to pass data_crs='EPSG:4326'?"
                                 )
            else: # we assume EPSG:4326
                warnings.warn("Argument `data_crs` not provided, assuming EPSG:4326 for ('longitude', 'latitude') coordinates")
                data_crs = pyproj.CRS("EPSG:4326")
            
            pings_gdf= gpd.points_from_xy(
                data[traj_cols['longitude']],
                data[traj_cols['latitude']],
                crs=data_crs) # order matters: lon first
        else:
            if not data_crs:
                raise ValueError(f"data_crs must be provided when using projected coordinates.")
            data_crs = pyproj.CRS(data_crs)
            if data_crs.is_geographic:
                warnings.warn(f"Provided CRS {data_crs.name} is a geographic coordinate system. "
                             "This will lead to errors if passed coordinates ('x', 'y') are projected."
                             f"Did you mean to use {poi_table.crs}?"
                             )
            pings_gdf= gpd.points_from_xy(
                data[traj_cols['x']],
                data[traj_cols['y']],
                crs=data_crs)
    else:
        raise TypeError("`data` must be a pandas DataFrame or a GeoDataFrame.")

    if data_crs != pyproj.CRS(poi_table.crs):
        raise ValueError("CRS for `data` does not match CRS for `poi_table`.")

    use_poi_idx = True
    if location_id is not None:
        loc_col = location_id
        if location_id in poi_table:
            use_poi_idx=False
        else:
            warnings.warn(f"{location_id} column not found in {poi_table.columns}, defaulting to poi_table.index for spatial join.")
    else:
        loc_col = 'location_id'
        warnings.warn(f"location_id column not provided, defaulting to poi_table.index for spatial join.")

        
    if max_distance>0:
        if data_crs.is_geographic:
            warnings.warn(f"Provided CRS {data_crs.name} is a geographic coordinate system. "
                             "This will lead to errors when computing euclidean distances."
                             f"Did you mean to use `max_distance=0'?"
                         )        
        
        p_idx, idx = poi_table.sindex.nearest(pings_gdf, max_distance=max_distance, return_all=False)
        if use_poi_idx:
            s = pd.Series(poi_table.iloc[idx].index, index=data.index[p_idx])
            s.name = loc_col
        else:
            s = pd.Series(poi_table.iloc[idx][loc_col], index=data.index[p_idx])
            s.name = loc_col
            
        return s.reindex(data.index)

    else: # default max_distance = 0
        p_idx, idx = poi_table.sindex.query(pings_gdf, predicate="within") # boundary counts; use "contains" to exclude it
        if use_poi_idx:
            s = pd.Series(poi_table.iloc[idx].index, index=data.index[p_idx]) # might have duplicates
            s = s.loc[~s.index.duplicated()]
            s.name = loc_col
        else:
            s = pd.Series(poi_table.iloc[idx][loc_col], index=data.index[p_idx])
            s = s.loc[~s.index.duplicated()]
            s.name = loc_col        
        return s.reindex(data.index)


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