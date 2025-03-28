import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

from sedona.register import SedonaRegistrator
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

import nomad.daphmeIO as loader
from nomad.constants import DEFAULT_SCHEMA
import pdb

def to_projection(
    traj: pd.DataFrame,
    input_crs: str = "EPSG:4326",
    output_crs: str = "EPSG:3857",
    traj_cols: dict = None,
    output_x_col: str = "x",
    output_y_col: str = "y",
    spark_session: SparkSession = None,
    **kwargs
) -> pd.DataFrame:
    """
    Projects coordinate columns from one Coordinate Reference System (CRS) to another.

    This function takes a DataFrame containing coordinate columns and projects them from one CRS to another specified CRS. 
    It supports both local and distributed computation using Spark. (TODO: SPARK)

    If `traj_cols` is not provided, the function will attempt to use default column names or those provided in `kwargs`.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory DataFrame containing coordinate columns.
    input_crs : str, optional
        EPSG code for the original CRS.
        Defaults to "EPSG:4326".
    output_crs : str, optional
        EPSG code for the target CRS.
        Defaults to "EPSG:3857".
    traj_cols : dict, optional
        A dictionary defining column mappings for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
        If not provided, the function will attempt to use default column names or those provided in `kwargs`.
    output_x_col : str, optional
        Name of the projected x column.
        Defaults to 'x'.
    output_y_col : str, optional
        Name of the projected y column.
        Defaults to 'y'.
    spark_session : SparkSession, optional.
        Spark session for distributed computation, if needed.
    **kwargs :
        Additional parameters like 'latitude' or 'longitude' column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'x' and 'y' columns representing projected coordinates.

    Raises
    ------
    ValueError
        If expected coordinate columns are missing.
    """
    traj = traj.copy()

    # Check if user wants to project from x and y
    spatial_cols_provided = (
        'x' in kwargs and 'y' in kwargs 
        and kwargs['x'] in traj.columns 
        and kwargs['y'] in traj.columns
    )

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(DEFAULT_SCHEMA, traj_cols)

    # Test to check for spatial columns
    loader._has_spatial_cols(traj.columns, traj_cols)

    # Setting long and lat as defaults if not specified by user in either traj_cols or kwargs
    if not spatial_cols_provided and traj_cols['longitude'] in traj.columns and traj_cols['latitude'] in traj.columns:
        input_x_col, input_y_col = traj_cols['longitude'], traj_cols['latitude']
    else:
        input_x_col, input_y_col = traj_cols['x'], traj_cols['y']

    if input_x_col not in traj.columns or input_y_col not in traj.columns:
        raise ValueError(f"Coordinate columns '{input_x_col}' and/or '{input_y_col}' not found in DataFrame.")

    if spark_session:
        return _to_projection_spark(traj, input_crs, output_crs, input_x_col, input_y_col, output_x_col, output_y_col, spark_session)
    else:
        return _to_projection(traj, input_crs, output_crs, input_x_col, input_y_col, output_x_col, output_y_col)


def _to_projection(
    traj,
    input_crs,
    output_crs,
    input_x_col,
    input_y_col,
    output_x_col,
    output_y_col
):
    """
    Helper function to project latitude/longitude columns to a new CRS.
    """
    gdf = gpd.GeoSeries(gpd.points_from_xy(traj[input_x_col], traj[input_y_col]), crs=input_crs)
    projected = gdf.to_crs(output_crs)
    traj[output_x_col] = projected.x
    traj[output_y_col] = projected.y

    return traj


def _to_projection_spark(
    traj, 
    input_crs, 
    output_crs, 
    input_x_col, 
    input_y_col, 
    output_x_col,
    output_y_col,
    spark_session
):
    """
    Helper function to project latitude/longitude columns to a new CRS using Spark.
    """
    SedonaRegistrator.registerAll(spark_session)
    spark_df = spark_session.createDataFrame(traj)
    spark_df.createOrReplaceTempView("temp_view")
    query = f"""
        SELECT *, 
            ST_X(ST_Transform(ST_Point({input_x_col}, {input_y_col}), '{input_crs}', '{output_crs}')) AS {output_x_col},
            ST_Y(ST_Transform(ST_Point({input_x_col}, {input_y_col}), '{input_crs}', '{output_crs}')) AS {output_y_col}
        FROM temp_view
    """
    result_df = spark_session.sql(query)
    return result_df.toPandas()


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

    # Check if user wants to use long and lat
    long_lat = (
        'latitude' in kwargs and 'longitude' in kwargs
        and kwargs['latitude'] in traj.columns
        and kwargs['longitude'] in traj.columns
    )

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if not long_lat and traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns:
        input_x, input_y = traj_cols['x'], traj_cols['y']
    else:
        input_x, input_y = traj_cols['longitude'], traj_cols['latitude']

    # Check if polygon is a valid Shapely Polygon object
    if (polygon is not None) and (not isinstance(polygon, Polygon)):
        raise TypeError("Polygon parameter must be a Shapely Polygon object.")
    
    # Filter to the desired time range
    traj = traj[(traj[traj_cols['datetime']] >= start_time) & (traj[traj_cols['datetime']] <= end_time)].copy()

    if spark_session:
        return _filter_users_spark(
            traj, start_time, end_time, polygon.wkt, min_active_days, min_pings_per_day, traj_cols, input_x, input_y, spark_session
            )
    else:
        users = _filtered_users(
            traj, start_time, end_time, polygon, min_active_days, min_pings_per_day, traj_cols, input_x, input_y, crs
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


def compute_persistence(df: pd.DataFrame, 
                        max_gap: int,
                        user_col: str, 
                        timestamp_col: str,) -> pd.DataFrame:
    """
    TODO: FIX CODE 

    Computes the persistence for each user, defined as:
    P[X[t, t+max_gap] is not empty | X[t] is not empty], where X[t] = 1 indicates a ping
    occurred at time t, and X[t, t+max_gap] indicates any ping in the interval [t, t+max_gap].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'uid' (user ID) and 'local_timestamp' (datetime of pings).
    max_gap : int
        The maximum number of hours to check for the presence of pings after time t.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the persistence for each user ('uid', 'persistence').
    """
    df['hour'] = pd.to_datetime(df[timestamp_col]).dt.floor('h')
    df_hourly = df.groupby([user_col, 'hour']).size().clip(upper=1).reset_index(name='ping')

    # Generate a complete range of hours for each user within the target time window
    uids = df[user_col].unique()
    all_hours = pd.MultiIndex.from_product(
        [uids, pd.date_range(start=pd.Timestamp('2024-01-01 08:00:00'),
                             end=pd.Timestamp('2024-01-15 07:00:00'), freq='h')],
        names=[user_col, 'hour']
    )

    # Create a DataFrame with all hours for all users
    all_hours_df = pd.DataFrame(index=all_hours).reset_index()

    # Merge with the hourly pings data
    df_complete = pd.merge(all_hours_df, df_hourly, on=[user_col, 'hour'], how='left').fillna(0)
    df_complete['ping'] = df_complete['ping'].astype(int)

    # Initialize a column to store whether there is a ping within the next max_gap hours
    df_complete['has_future_ping'] = 0

    # Compute the condition: Check for pings within [t, t+max_gap] for each user
    for uid, group in df_complete.groupby(user_col):
        for idx, row in group.iterrows():
            current_hour = row['hour']
            future_pings = group[(group['hour'] > current_hour) & 
                                 (group['hour'] <= current_hour + pd.Timedelta(hours=max_gap))]
            if future_pings['ping'].sum() > 0:
                df_complete.at[idx, 'has_future_ping'] = 1

    # Compute persistence for each user
    user_persistence_list = []
    for uid, group in df_complete.groupby(user_col):
        numerator = ((group['ping'] == 1) & (group['has_future_ping'] == 1)).sum()
        denominator = (group['ping'] == 1).sum()
        persistence = numerator / denominator if denominator > 0 else 0
        user_persistence_list.append({user_col: uid, 'persistence': persistence})

    # Convert to DataFrame
    persistence_df = pd.DataFrame(user_persistence_list)

    return persistence_df


def filter_users_by_persistence_new(df: pd.DataFrame, max_gap: int, epsilon: float) -> pd.Series:
    """
    Filters users based on their persistence value, keeping only users with persistence above epsilon.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'uid' (user ID) and 'local_timestamp' (datetime of pings).
    max_gap : int
        The maximum number of hours to check for the presence of pings after time t.
    epsilon : float
        The persistence threshold; only users with persistence > epsilon will be retained.

    Returns
    -------
    pd.Series
        A Series containing the user IDs of users whose persistence > epsilon.
    """
    # Compute persistence for each user
    persistence_df = user_persistence_new(df, max_gap)

    # Filter users with persistence > epsilon
    filtered_users = persistence_df[persistence_df['persistence'] > epsilon]['uid']

    return filtered_users