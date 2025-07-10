import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

import geopandas as gpd
import pyproj
from shapely.geometry import Polygon, Point
from shapely import wkt
import warnings

import nomad.io.base as loader
from nomad.constants import DEFAULT_SCHEMA, SEC_PER_UNIT

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

def to_projection(
    traj: pd.DataFrame,
    input_crs = None,
    output_crs = None,
    spark_session: SparkSession = None,
    **kwargs
):
    """
    Projects coordinate columns from one Coordinate Reference System (CRS) to another.

    This function takes a DataFrame containing coordinate columns and projects them from one CRS to another specified CRS. 
    It supports both local and distributed computation using Spark. (TODO: SPARK)

    If columns names are not specified in `kwargs`, the function will attempt to use default column names.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory DataFrame containing coordinate columns.
    input_crs : str, optional
        EPSG code for the original CRS.
    output_crs : str, optional
        EPSG code for the target CRS.
    spark_session : SparkSession, optional.
        Spark session for distributed computation, if needed.
    **kwargs :
        Additional parameters to specify names of spatial columns to project from.
        E.g., 'latitude', 'longitude' or 'x', 'y'.

    Returns
    -------
    (pd.Series, pd.Series)
        A pair of Series with the new projected coordinates.

    Raises
    ------
    ValueError
        If expected coordinate columns are missing.
    """
    # Check whether traj contains spatial columns
    # Uncomment exclusive=True when the function is committed.
    loader._has_spatial_cols(traj.columns, kwargs)#, exclusive=True)

    # Check if user wants to project from x and y
    if ('x' in kwargs and 'y' in kwargs):
        input_x_col, input_y_col = kwargs['x'], kwargs['y']
        if input_crs is None:
            raise ValueError("input_crs not found in arguments.")
        if output_crs is None:
            output_crs = "EPSG:4326"
            warnings.warn("output_crs not provided. Defaulting to EPSG:4326.")
    else:
        input_x_col, input_y_col = kwargs['longitude'], kwargs['latitude']
        if input_crs is None:
            input_crs = "EPSG:4326"
            warnings.warn("input_crs not provided. Defaulting to EPSG:4326.")
        if output_crs is None:
            raise ValueError("output_crs not found in arguments.")

    if spark_session:
        return _to_projection_spark(traj, input_crs, output_crs, input_x_col, input_y_col, spark_session)
    else:
        return _to_projection(traj, input_crs, output_crs, input_x_col, input_y_col)


def _to_projection_spark(
    traj, 
    input_crs, 
    output_crs, 
    input_x_col, 
    input_y_col, 
    spark_session
):
    """
    Helper function to project latitude/longitude columns to a new CRS using Spark.
    """
    from sedona.register import SedonaRegistrator
    SedonaRegistrator.registerAll(spark_session)

    spark_df = spark_session.createDataFrame(traj)
    spark_df.createOrReplaceTempView("temp_view")

    query = f"""
        SELECT
            ST_X(ST_Transform(ST_Point({input_x_col}, {input_y_col}), '{input_crs}', '{output_crs}')) AS x,
            ST_Y(ST_Transform(ST_Point({input_x_col}, {input_y_col}), '{input_crs}', '{output_crs}')) AS y
        FROM temp_view
    """
    
    result_df = spark_session.sql(query)
    pandas_df = result_df.toPandas()
    return pandas_df['x'], pandas_df['y']

def filter_users(
    traj: pd.DataFrame,
    start_time,
    end_time,
    timezone = None,
    polygon = None,
    min_active_days = 1,
    min_pings_per_day = 1,
    traj_cols = None,
    crs = "EPSG:3857",
    spark_session = None,
    **kwargs):
    '''
    Subsets to users who have at least min_pings_per_day pings on min_active_days distinct days
    in the polygon within the timeframe start_time to start_time.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory DataFrame with latitude and longitude columns.
    start_time
        Start of the timeframe for filtering.
    end_time
        End of the timeframe for filtering.
    polygon : shapely.geometry.Polygon
        Polygon defining the area to retain points within.
        If None, no spatial filtering is applied.
    min_active_days, min_pings_per_day: int
        User is retained if they have at least min_pings_per_day pings on min_active_days distinct days.
        Defaults to 1.
    traj_cols : dict, optional
        A dictionary defining column mappings for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
        If not provided, the function will attempt to use default column names or those provided in `kwargs`.
    crs : str, optional
        Coordinate Reference System (CRS) for the polygon.
        Defaults to "EPSG:3857".
    spark_session : SparkSession, optional
        Spark session for distributed computation, if needed.
    **kwargs :
        Additional parameters like 'user_id', 'latitude', 'longitude', or 'datetime' column names.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with points inside the polygon's bounds.
    '''

    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)
    loader._has_spatial_cols(traj.columns, traj_cols)#, exclusive=True)
    loader._has_time_cols(traj.columns, traj_cols)

    # Check which spatial columns to project from
    if ('x' in kwargs and 'y' in kwargs):
        input_x_col, input_y_col = traj_cols['x'], traj_cols['y']
    else:
        input_x_col, input_y_col = traj_cols['longitude'], traj_cols['latitude']

    # Check which time column to use from
    if ('datetime' in kwargs):
        time_col = traj_cols['datetime']
        start_time = _timestamp_handling(start_time, "pd.timestamp", timezone)
        end_time = _timestamp_handling(end_time, "pd.timestamp", timezone)
    else:  # defaults to unix timestamps
        time_col = traj_cols['timestamp']
        start_time = _timestamp_handling(start_time, "unix", timezone)
        end_time = _timestamp_handling(end_time, "unix", timezone)

    # Check if polygon is a valid Shapely Polygon object
    if (polygon is not None) and (not isinstance(polygon, Polygon)):
        raise TypeError("Polygon parameter must be a Shapely Polygon object.")
    
    # Filter to the desired time range
    traj = traj[(traj[time_col] >= start_time) & (traj[time_col] <= end_time)].copy()

    if spark_session:
        return _filter_users_spark(
            traj, start_time, end_time, polygon.wkt, min_active_days, min_pings_per_day, traj_cols, input_x_col, input_y_col, time_col, spark_session
            )
    else:
        users = _filtered_users(
            traj, start_time, end_time, polygon, min_active_days, min_pings_per_day, traj_cols, input_x_col, input_y_col, time_col, crs
        )
        return traj[traj[traj_cols['user_id']].isin(users)]

def _filter_users_spark(
    traj,
    bounding_wkt,
    T0,
    T1,
    min_days,
    min_pings,
    traj_cols,
    input_x,
    input_y,
    spark,
):
    """
    Helper function that retains only users who have at least k distinct days 
    with pings inside the geometry between T0 and T1.

    TODO: I don't know if this works / if it could be more efficient
    """
    from sedona.register import SedonaRegistrator
    SedonaRegistrator.registerAll(spark)

    # Ensure timestamp column is proper type
    traj = traj.withColumn(
        traj_cols['timestamp'], 
        F.to_timestamp(F.col(traj_cols['timestamp']))
    )

    # Time filter
    traj_filtered = traj.filter(
        (F.col(traj_cols['timestamp']) >= F.to_timestamp(F.lit(T0))) & 
        (F.col(traj_cols['timestamp']) <= F.to_timestamp(F.lit(T1)))
    )

    # Add geometry point column
    traj_filtered = traj_filtered.withColumn(
        "coordinate", 
        F.expr(f"ST_Point(CAST({input_x} AS DECIMAL(24,20)), CAST({input_y} AS DECIMAL(24,20)))")
    )

    # Spatial filter + extract date
    traj_filtered.createOrReplaceTempView("temp_df")
    query = f"""
        SELECT *, DATE({traj_cols['timestamp']}) AS date
        FROM temp_df
        WHERE ST_Contains(ST_GeomFromWKT('{bounding_wkt}'), coordinate)
    """

    traj_inside = spark.sql(query)

    # Count pings per user per day
    daily_counts = traj_inside.groupBy(traj_cols['user_id'], "date").agg(
        F.count("*").alias("ping_count")
    )

    # Keep only user-day combos with at least `min_pings`
    filtered_days = daily_counts.filter(F.col("ping_count") >= min_pings)

    # Count how many qualifying days each user has
    qualifying_users = filtered_days.groupBy(traj_cols['user_id']).agg(
        F.countDistinct("date").alias("qualified_days")
    ).filter(
        F.col("qualified_days") >= min_days
    ).select(traj_cols['user_id'])

    result_traj = traj_inside.join(qualifying_users, on=traj_cols['user_id'], how='inner')

    return result_traj