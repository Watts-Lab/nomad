import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon


def to_projection(df,
                  latitude,
                  longitude,
                  from_crs="EPSG:4326",
                  to_crs="EPSG:3857",
                  spark_session=None):
    """
    Projects latitude and longitude columns from one CRS to another.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing latitude and longitude columns.
    latitude : str
        Name of the latitude column.
    longitude : str
        Name of the longitude column.
    from_crs : str, optional
        EPSG code for the original CRS (default is "EPSG:4326").
    to_crs : str, optional
        EPSG code for the target CRS (default is "EPSG:3857").

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'x' and 'y' columns representing projected coordinates.
    """
    if latitude not in df.columns or longitude not in df.columns:
        raise ValueError(f"Latitude or longitude columns '{latitude}', '{longitude}' not found in DataFrame.")

    proj_cols = _to_projection(df[latitude],
                               df[longitude],
                               from_crs,
                               to_crs)

    df['x'] = proj_cols.x
    df['y'] = proj_cols.y

    return df


def _to_projection(lat_col,
                   long_col,
                   from_crs,
                   to_crs):
    """
    Helper function to project latitude/longitude columns to a new CRS.
    """
    gdf = gpd.GeoSeries(gpd.points_from_xy(long_col, lat_col),
                        crs=from_crs)
    projected = gdf.to_crs(to_crs)
    output = pd.DataFrame({'x': projected.x, 'y': projected.y})

    return output


def filter_to_box(df, polygon, latitude, longitude):
    '''
    Filters DataFrame to keep points within a specified polygon's bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latitude and longitude columns.
    polygon : shapely.geometry.Polygon
        Polygon defining the area to retain points within.
    latitude : str
        Name of the latitude column.
    longitude : str
        Name of the longitude column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with points inside the polygon's bounds.
    '''
    if not isinstance(polygon, Polygon):
        raise TypeError("Polygon parameter must be a Shapely Polygon object.")

    if latitude not in df.columns or longitude not in df.columns:
        raise ValueError(f"Latitude or longitude columns '{latitude}', '{longitude}' not found in DataFrame.")

    min_x, min_y, max_x, max_y = polygon.bounds

    # TO DO: handle different column names and/or defaults as in daphmeIO. i.e. traj_cols as parameter

    return df[(df[longitude].between(min_y, max_y)) & (df[latitude].between(min_x, max_x))]


def coarse_filter(df: DataFrame, 
                  bounding_wkt: str, 
                  spark: SparkSession,
                  longitude_col: str = LONGITUDE, 
                  latitude_col: str = LATITUDE, 
                  id_col: str = UID) -> DataFrame:
    """Filters a DataFrame based on whether geographical points (defined by longitude and latitude) fall within a specified geometry.

    Parameters
    ----------
    df : DataFrame
        The Spark DataFrame to be filtered. It should contain columns corresponding to longitude and latitude values, as well as an id column.
    
    bounding_wkt : str
        The Well-Known Text (WKT) string representing the bounding geometry within which points are tested for inclusion. The WKT should define a polygon in the EPSG:4326 coordinate reference system.
    
    spark : SparkSession
        The active SparkSession instance used to execute Spark operations.
    
    longitude_col : str, default "longitude"
        The name of the column in 'df' containing longitude values. Longitude values should be in the EPSG:4326 coordinate reference system.
    
    latitude_col : str, default "latitude"
        The name of the column in 'df' containing latitude values. Latitude values should be in the EPSG:4326 coordinate reference system.
    
    id_col : str, default "id"
        The name of the column in 'df' containing user IDs.
    
    Returns
    ----------
    DataFrame
        A new Spark DataFrame filtered to include only rows where the point (longitude, latitude) falls within the specified geometric boundary defined by 'bounding_wkt'. This DataFrame includes all original columns from 'df' and an additional column 'in_geo' that is true if the point falls within the specified geometric boundary and false otherwise.

    Example
    ----------
    >>> # Assuming a SparkSession `spark` and a DataFrame `df` are predefined
    >>> bounding_wkt = "POLYGON((...))"  # Replace with actual WKT
    >>> filtered_df = coarse_filter(df, bounding_wkt, spark)
    >>> filtered_df.show()
    """
    
    df = df.withColumn("coordinate", F.expr(f"ST_MakePoint({longitude_col}, {latitude_col})"))
    df.createOrReplaceTempView("temp_df")
    
    query = f"""
        WITH temp_df AS (
            SELECT *,
                   ST_Contains(ST_GeomFromWKT('{bounding_wkt}'), coordinate) AS in_geo
            FROM temp_df
        ),

        UniqueIDs AS (
            SELECT DISTINCT {id_col} 
            FROM temp_df
            WHERE in_geo
        )

        SELECT t.*
        FROM temp_df t
        WHERE t.{id_col} IN (SELECT {id_col} FROM UniqueIDs)
        """
    
    return spark.sql(query)