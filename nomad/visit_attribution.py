import geopandas as gpd
import nomad.io.base as loader
import nomad.constants as constants
from shapely.geometry import Point

def point_in_polygon():
    gdf = gpd.read_file('../garden_city.geojson')
    poi_labels = poi_map(plot_df,
                     poi_table=gdf)
    

def poi_map(traj, poi_table, traj_cols=None, crs=None, **kwargs):
    """
    Map pings in the trajectory to the POI table.

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

    pings_df["pings_geometry"] = pings_df.apply(lambda row: Point(row[traj_cols['longitude']], row[traj_cols['latitude']]) if is_long_lat else Point(row[traj_cols['x']], row[traj_cols['y']]), axis=1)
    pings_df = gpd.GeoDataFrame(pings_df, geometry="pings_geometry", crs=poi_table.crs)
    
    pings_df = gpd.sjoin(pings_df, poi_table, how="left", predicate="within")

    # Is this necessary at this stage? we lose information silently...
    pings_df["building_id"] = pings_df["building_id"].where(pings_df["building_id"].notna(), None)

    return pings_df["building_id"]


def identify_stop(alg_out, traj, stop_table, poi_table, method='mode'):
    """
    Given the output of a stop detection algorithm, maps each cluster to a location
    by the method specified.
   
    Parameters
    ----------
    alg_out : pd.DataFrame
        DataFrame containing cluster assignments (one row per ping), indexed by ping ID.
        Must have a column 'cluster' indicating each ping's cluster.
    traj : pd.DataFrame
        DataFrame containing ping coordinates (x, y) by ping ID.
    stop_table : pd.DataFrame
        DataFrame containing stop clusters (one row per cluster), with a 'cluster_id' column.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
    method : str, optional
        The method to use for mapping clusters to locations. Options are:
        - 'mode': Assigns the most frequent location ID associated with each cluster.
        - 'centroid': Assigns the centroid of the cluster to the location.
    
    Returns
    -------
    pd.DataFrame
        Updated stop_table with a 'location' column indicating location associated with each cluster.
    """
    # If either alg_out or stop_table is empty, there's nothing to do
    if alg_out.empty or stop_table.empty:
        stop_table['location'] = pd.Series(dtype='object')
        return stop_table

    merged_df = traj.copy()
    merged_df['cluster'] = alg_out

    # Compute the location for each cluster
    if method == 'centroid':
        pings_df = merged_df.groupby('cluster')[['x', 'y']].mean()
        locations = poi_map(pings_df, poi_table)

    elif method == 'mode':
        pings_df = merged_df[['x', 'y']].copy() #must we copy the data?
        merged_df["building_id"] = poi_map(pings_df, poi_table)
        locations = merged_df.groupby('cluster')['building_id'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mode' or 'centroid'.")

    # Map the mode location back to the stop_table
    stop_table['location'] = locations

    # Let's return locations instead?
    return stop_table
