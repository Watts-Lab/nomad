import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import warnings

from sedona.register import SedonaRegistrator
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

import nomad.io.base as loader
from nomad.constants import DEFAULT_SCHEMA
import pdb
import warnings
import numpy as np

def to_timestamp(
    datetime: pd.Series,
    tz_offset: pd.Series = None
) -> pd.Series:
    """
    Convert a datetime series into UNIX timestamps (seconds).
    
    Parameters
    ----------
    datetime : pd.Series
    tz_offset : pd.Series, optional

    Returns
    -------
    pd.Series
        UNIX timestamps as int64 values (seconds since epoch).
    """
    # Validate input type
    if not (
        
        pd.api.types.is_datetime64_any_dtype(datetime) or
        pd.api.types.is_string_dtype(datetime) or
        (pd.api.types.is_object_dtype(datetime) and _is_series_of_timestamps(datetime))
    ):
        raise TypeError(
            f"Input must be of type datetime64, string, or an array of Timestamp objects, "
            f"but it is of type {datetime.dtype}."
        )
    
    if tz_offset is not None:
        if not pd.api.types.is_integer_dtype(tz_offset):
            tz_offset = tz_offset.astype('int64')

    # datetime with timezone
    if isinstance(datetime.dtype, pd.DatetimeTZDtype):
        return datetime.astype("int64") // 10**9
    
    # datetime without timezone
    elif pd.api.types.is_datetime64_dtype(datetime):
        if tz_offset is not None:
            return datetime.astype("int64") // 10**9 - tz_offset
        warnings.warn(
                f"The input is timezone-naive. UTC will be assumed."
                "Consider localizing to a timezone or passing a timezone offset column.")
        return datetime.astype("int64") // 10**9
    
    # datetime as string
    elif pd.api.types.is_string_dtype(datetime):
        result = pd.to_datetime(datetime, errors="coerce", utc=True)

        # contains timezone e.g. '2024-01-01 12:29:00-02:00'
        if datetime.str.contains(r'(?:Z|[+\-]\d{2}:\d{2})$', regex=True, na=False).any():
            return result.astype('int64') // 10**9
        else:
            # naive e.g. "2024-01-01 12:29:00"
            if tz_offset is not None and not tz_offset.empty:
                return result.astype('int64') // 10**9 - tz_offset
            else:
                warnings.warn(
                    f"The input is timezone-naive. UTC will be assumed."
                    "Consider localizing to a timezone or passing a timezone offset column.")
                return result.astype('int64') // 10**9
                
    # datetime is pandas.Timestamp object
    else:
        if tz_offset is not None and not tz_offset.empty:
            f = np.frompyfunc(lambda x: x.timestamp(), 1, 1)
            return pd.Series(f(datetime).astype("float64"), index=datetime.index)
        else:
            return datetime.astype('int64') // 10**9

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


def _to_projection(
    traj,
    input_crs,
    output_crs,
    input_x_col,
    input_y_col
):
    """
    Helper function to project latitude/longitude columns to a new CRS.
    """
    gdf = gpd.GeoSeries(gpd.points_from_xy(traj[input_x_col], traj[input_y_col]), crs=input_crs)
    projected = gdf.to_crs(output_crs)
    return projected.x, projected.y


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
            ST_X(ST_Transform(ST_Point({input_x_col}, {input_y_col}), '{input_crs}', '{output_crs}')),
            ST_Y(ST_Transform(ST_Point({input_x_col}, {input_y_col}), '{input_crs}', '{output_crs}'))
        FROM temp_view
    """
    result_df = spark_session.sql(query)
    return result_df.toPandas()  # TODO: make it return as pd.Series


def filter_users(
    traj: pd.DataFrame,
    start_time: str,
    end_time: str,
    polygon: Polygon = None,
    min_active_days: int = 1,
    min_pings_per_day: int = 1,
    traj_cols: dict = None,
    crs: str = "EPSG:3857",
    spark_session: SparkSession = None,
    **kwargs
) -> pd.DataFrame:
    '''
    Subsets to users who have at least min_pings_per_day pings on min_active_days distinct days
    in the polygon within the timeframe start_time to start_time.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory DataFrame with latitude and longitude columns.
    start_time : str
        Start of the timeframe for filtering (as a string, or datetime).
    end_time : str
        End of the timeframe for filtering (as a string, or datetime).
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
        input_x_col, input_y_col = kwargs['x'], kwargs['y']
    else:
        input_x_col, input_y_col = kwargs['longitude'], kwargs['latitude']

    # Check if polygon is a valid Shapely Polygon object
    if (polygon is not None) and (not isinstance(polygon, Polygon)):
        raise TypeError("Polygon parameter must be a Shapely Polygon object.")
    
    # Filter to the desired time range
    traj = traj[(traj[traj_cols['datetime']] >= start_time) & (traj[traj_cols['datetime']] <= end_time)].copy()

    if spark_session:
        return _filter_users_spark(
            traj, start_time, end_time, polygon.wkt, min_active_days, min_pings_per_day, traj_cols, input_x_col, input_y_col, spark_session
            )
    else:
        users = _filtered_users(
            traj, start_time, end_time, polygon, min_active_days, min_pings_per_day, traj_cols, input_x_col, input_y_col, crs
        )
        return traj[traj[traj_cols['user_id']].isin(users)]


def _filtered_users(
    traj,
    start_time,
    end_time,
    polygon,
    min_active_days,
    min_pings_per_day,
    traj_cols,
    input_x,
    input_y,
    crs
):
    """
    Helper function that returns a series containing users who have at least 
    k distinct days with at least m pings in the polygon within the timeframe T0 to T1.
    """
    # Filter by time range
    traj_filtered = traj[(traj[traj_cols['datetime']] >= start_time) & (traj[traj_cols['datetime']] <= end_time)].copy()
    traj_filtered[traj_cols['datetime']] = pd.to_datetime(traj_filtered[traj_cols['datetime']])

    # Filter points inside the polygon
    if polygon is not None:
        traj_filtered = _in_geo(traj_filtered, input_x, input_y, polygon, crs)
    else:
        traj_filtered['in_geo'] = True
    traj_filtered['date'] = pd.to_datetime(traj_filtered[traj_cols['datetime']].dt.date)

    # Count pings per user per date inside the polygon
    daily_ping_counts = (
        traj_filtered[traj_filtered['in_geo']]
        .groupby([traj_cols['user_id'], 'date'])
        .size()
        .reset_index(name='ping_count')
    )

    # Filter users who have at least `m` pings on a given day
    users_with_m_pings = daily_ping_counts[daily_ping_counts['ping_count'] >= min_pings_per_day]

    # Count distinct days per user that satisfy the `m` pings condition
    users_with_k_days = (
        users_with_m_pings
        .groupby(traj_cols['user_id'])['date']
        .nunique()
        .reset_index(name='days_in_polygon')
    )

    # Select users who have at least `k` such days
    filtered_users = users_with_k_days[users_with_k_days['days_in_polygon'] >= min_active_days][traj_cols['user_id']]

    return filtered_users


def _in_geo(
    traj,
    input_x,
    input_y,
    polygon,
    crs
):
    """
    Helper function that adds a new column to the DataFrame indicating 
    whether points are inside the polygon or not.
    """
    points = gpd.GeoSeries(gpd.points_from_xy(traj[input_x], traj[input_y]), crs=crs)
    traj = traj.reset_index(drop=True)
    traj['in_geo'] = points.within(polygon)

    return traj


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
    TODO: IMPLEMENT min_pings
    Helper function that retains only users who have at least k distinct days 
    with pings inside the geometry between T0 and T1.
    """
    from sedona.register import SedonaRegistrator
    
    SedonaRegistrator.registerAll(spark)

    traj = traj.withColumn(traj_cols['timestamp'], F.to_timestamp(F.col(traj_cols['timestamp'])))
    traj_filtered = traj.filter(
        (F.col(traj_cols['timestamp']) >= F.to_timestamp(F.lit(T0))) & 
        (F.col(traj_cols['timestamp']) <= F.to_timestamp(F.lit(T1)))
    )
    traj_filtered = traj_filtered.withColumn(
        "coordinate", 
        F.expr(f"ST_Point(CAST({input_x} AS DECIMAL(24,20)), CAST({input_y} AS DECIMAL(24,20)))")
    )

    traj_filtered.createOrReplaceTempView("temp_df")
    query = f"""
        SELECT *, DATE({traj_cols['timestamp']}) AS date
        FROM temp_df
        WHERE ST_Contains(ST_GeomFromWKT('{bounding_wkt}'), coordinate)
    """

    traj_inside = spark.sql(query)

    user_day_counts = traj_inside.groupBy(traj_cols['user_id']).agg(
        F.countDistinct("date").alias("distinct_days")
    )
    users_with_k_days = user_day_counts.filter(
        F.col("distinct_days") >= min_days
    ).select(traj_cols['user_id'])

    result_traj = traj_inside.join(users_with_k_days, on=traj_cols['user_id'], how='inner')

    return result_traj


def q_filter(df: pd.DataFrame,
             qbar: float,
             traj_cols: dict = None,
             user_id: str = DEFAULT_SCHEMA["user_id"],
             timestamp: str = DEFAULT_SCHEMA["timestamp"]):
    """
    Computes the q statistic for each user as the proportion of unique hours with pings 
    over the total observed hours (last hour - first hour) and filters users where q > qbar.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with user_id and timestamp columns.
    qbar : float
        The threshold q value; users with q > qbar will be retained.
    traj_cols : dict, optional
        Dictionary containing column mappings, 
        e.g., {"user_id": "user_id", "timestamp": "timestamp"}.
    user_id : str, optional
        Name of the user_id column (default is "user_id").
    timestamp : str, optional
        Name of the timestamp column (default is "timestamp").

    Returns
    -------
    pd.Series
        A Series containing the user IDs for users whose q_stat > qbar.
    """
    user_col = traj_cols.get("user_id", user_id) if traj_cols else user_id
    timestamp_col = traj_cols.get("timestamp", timestamp) if traj_cols else timestamp

    user_q_stats = df.groupby(user_col).apply(
        lambda group: _compute_q_stat(group, timestamp_col)
    ).reset_index(name='q_stat')

    # Filter users where q > qbar
    filtered_users = user_q_stats[user_q_stats['q_stat'] > qbar][user_col]

    return filtered_users


# the user can pass **kwargs with timestamp or datetime, then if you absolutely need datetime then 
# create a variable, not a column in the dataframe
def q_stats(df: pd.DataFrame, user_id: str, timestamp: str):
    
    """
    Computes the q statistic for each user as the proportion of unique hours with pings 
    over the total observed hours (last hour - first hour).

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing user IDs and timestamps.
    user_id : str
        The name of the column containing user IDs.
    timestamp_col : str
        The name of the column containing timestamps.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing each user and their respective q_stat.
    """
    # only create a DATETIME column if it doesn't already exist (but what if the user knows datetime is wrong?)
    df[timestamp] = pd.to_datetime(df[timestamp], unit='s')

    q_stats = df.groupby(user_id).apply(
        lambda group: _compute_q_stat(group, timestamp)
    ).reset_index(name='q_stat')

    return q_stats


def _compute_q_stat(user, timestamp_col):
    user['hour_period'] = user[timestamp_col].dt.to_period('h')
    unique_hours = user['hour_period'].nunique()

    # Calculate total observed hours (difference between last and first hour)
    first_hour = user[timestamp_col].min()
    last_hour = user[timestamp_col].max()
    total_hours = (last_hour - first_hour).total_seconds() / 3600

    # Compute q as the proportion of unique hours to total hours
    q_stat = unique_hours / total_hours if total_hours > 0 else 0
    return q_stat