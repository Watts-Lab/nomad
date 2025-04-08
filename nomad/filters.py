import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

from nomad.io.base import _is_series_of_timestamps
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
    df: pd.DataFrame,
    traj_cols: dict = None,
    longitude: str = None,
    latitude: str = None,
    x: str = None,
    y: str = None,
    from_crs: str = "EPSG:4326",
    to_crs: str = "EPSG:3857",
    spark_session: SparkSession = None
):
    """
    Projects coordinate columns from one CRS to another.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing coordinate columns.
    traj_cols : dict, optional
        Dictionary containing column mappings,
        e.g., {"latitude": "lat_col", "longitude": "lon_col"}.
    longitude : str, optional
        Name of the longitude column.
    latitude : str, optional
        Name of the latitude column.
    x : str, optional
        Name of the x coordinate column.
    y : str, optional
        Name of the y coordinate column.
    from_crs : str, optional
        EPSG code for the original CRS (default is "EPSG:4326", spherical).
    to_crs : str, optional
        EPSG code for the target CRS (default is "EPSG:3857", web mercator).
    spark_session : SparkSession, optional
        Spark session for distributed computation, if needed.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'x' and 'y' columns representing projected coordinates.
    """

    # User can pass latitude and longitude as kwargs,
    # or x and y, OR traj_cols (prioritizing latitude and longitude).
    if longitude is not None and latitude is not None:
        lon_col = longitude
        lat_col = latitude
    elif x is not None and y is not None:
        lon_col = x
        lat_col = y
    elif traj_cols is not None:
        lon_col = traj_cols.get("longitude", DEFAULT_SCHEMA["longitude"])
        lat_col = traj_cols.get("latitude", DEFAULT_SCHEMA["latitude"])
    else:
        lon_col = DEFAULT_SCHEMA["longitude"]
        lat_col = DEFAULT_SCHEMA["latitude"]

    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"Longitude or latitude columns '{lon_col}' and '{lat_col}' not found in DataFrame.")

    if spark_session:
        return _to_projection_spark(df, lon_col, lat_col, from_crs, to_crs, spark_session)
    else:
        proj_cols = _to_projection(df[lon_col], df[lat_col], from_crs, to_crs)
        result_df = df.copy()
        result_df['x'] = list(proj_cols['x'])
        result_df['y'] = list(proj_cols['y'])
        return result_df

def _to_projection(
    long_col,
    lat_col,
    from_crs: str,
    to_crs: str
):
    """
    Helper function to project latitude/longitude columns to a new CRS.
    """
    gdf = gpd.GeoSeries(gpd.points_from_xy(long_col, lat_col),
                        crs=from_crs)
    projected = gdf.to_crs(to_crs)
    output = pd.DataFrame({'x': projected.x, 'y': projected.y})

    return output


def _to_projection_spark(
    df, 
    longitude_col, 
    latitude_col, 
    source_crs, 
    target_crs, 
    spark_session
):
    """
    Helper function to project latitude/longitude columns to a new CRS using Spark.
    """
    from sedona.register import SedonaRegistrator
    
    SedonaRegistrator.registerAll(spark_session)
    spark_df = spark_session.createDataFrame(df)
    spark_df.createOrReplaceTempView("temp_view")
    query = f"""
        SELECT *, 
            ST_X(ST_Transform(ST_Point({longitude_col}, {latitude_col}), '{source_crs}', '{target_crs}')) AS x,
            ST_Y(ST_Transform(ST_Point({longitude_col}, {latitude_col}), '{source_crs}', '{target_crs}')) AS y
        FROM temp_view
    """
    result_df = spark_session.sql(query)
    return result_df.toPandas()


def filter_to_polygon(
    df: pd.DataFrame,
    polygon: Polygon,
    k: int,
    T0: str,
    T1: str,
    traj_cols: dict = None,
    user_col: str = DEFAULT_SCHEMA["user_id"],
    timestamp_col: str = DEFAULT_SCHEMA["timestamp"],
    longitude_col: str = DEFAULT_SCHEMA["longitude"],
    latitude_col: str = DEFAULT_SCHEMA["latitude"],
    spark_session: SparkSession = None
):
    '''
    Filters DataFrame to keep points within a specified polygon's bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latitude and longitude columns.
    polygon : shapely.geometry.Polygon
        Polygon defining the area to retain points within.
    k : int
        Minimum number of distinct days with pings inside the polygon for the user to be retained.
    T0 : str
        Start of the timeframe for filtering (as a string, or datetime).
    T1 : str
        End of the timeframe for filtering (as a string, or datetime).
    traj_cols : dict, optional
        Dictionary containing column mappings, 
        e.g., {"user_id": "user_id", "timestamp": "timestamp"}.
    user_col : str, optional
        Name of the user column (default is "user_id").
    timestamp_col : str, optional
        Name of the timestamp column (default is "timestamp").
    longitude_col : str, optional
        Name of the longitude column (default is "longitude").
    latitude_col : str, optional
        Name of the latitude column (default is "latitude").
    spark_session : SparkSession, optional
        Spark session for distributed computation, if needed.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with points inside the polygon's bounds.
    '''
    if traj_cols:
        user_col = traj_cols.get("user_id", user_col)
        timestamp_col = traj_cols.get("timestamp", timestamp_col)
        longitude_col = traj_cols.get("longitude", longitude_col)
        latitude_col = traj_cols.get("latitude", latitude_col)
    
    if longitude_col not in df.columns or latitude_col not in df.columns:
        raise ValueError(f"Longitude or latitude columns '{longitude_col}', '{latitude_col}' not found in DataFrame.")

    if not isinstance(polygon, Polygon):
        raise TypeError("Polygon parameter must be a Shapely Polygon object.")

    if spark_session:
        return _filter_to_polygon_spark(
            df, polygon.wkt, k, T0, T1, spark_session, user_col, timestamp_col, longitude_col, latitude_col
        )

    else:
        users = _filtered_users(
            df, k, T0, T1, polygon, user_col, timestamp_col, latitude_col, longitude_col
        )
        return df[df[user_col].isin(users)]


def _filtered_users(
    df: pd.DataFrame,
    k: int,
    T0: str,
    T1: str,
    polygon: Polygon,
    user_col: str,
    timestamp_col: str,
    latitude_col: str,
    longitude_col: str
) -> pd.DataFrame:
    """
    Subsets to users who have at least k distinct days with pings in the polygon 
    within the timeframe T0 to T1.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing user data with latitude, longitude, and timestamp.
    k : int
        Minimum number of distinct days with pings inside the polygon for the user to be retained.
    T0 : str
        Start of the timeframe (as a string, or datetime).
    T1 : str
        End of the timeframe (as a string, or datetime).
    polygon : Polygon
        The polygon to check whether pings are inside.
    user_col : str
        Name of the column containing user identifiers.
    timestamp_col : str
        Name of the column containing timestamps (as strings or datetime).
    latitude_col : str
        Name of the column containing latitude values.
    longitude_col : str
        Name of the column containing longitude values.

    Returns
    -------
    pd.DataFrame
        A Series containing the user IDs for users who have at 
        least k distinct days with pings inside the polygon.
    """
    df_filtered = df[(df[timestamp_col] >= T0) & (df[timestamp_col] <= T1)].copy()
    df_filtered[timestamp_col] = pd.to_datetime(df_filtered[timestamp_col])
    df_filtered = _in_geo(df_filtered, longitude_col, latitude_col, polygon)
    df_filtered['date'] = df_filtered[timestamp_col].dt.date

    filtered_users = (
        df_filtered[df_filtered['in_geo']]
        .groupby(user_col)['date']
        .nunique()
        .reset_index()
    )

    filtered_users = filtered_users[filtered_users['date'] >= k][user_col]

    return filtered_users


def _in_geo(
    df: pd.DataFrame,
    longitude_col: str,
    latitude_col: str,
    polygon: Polygon
) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame indicating whether points are 
    inside the polygon (1) or not (0).
    """

    points = gpd.GeoSeries(gpd.points_from_xy(df[longitude_col], df[latitude_col]), crs="EPSG:4326")
    df = df.reset_index(drop=True)
    df['in_geo'] = points.within(polygon)

    return df


def _filter_to_polygon_spark(
    df: pyspark.sql.DataFrame,
    bounding_wkt: str,
    k: int,
    T0: str,
    T1: str,
    spark: SparkSession,
    user_col: str,
    timestamp_col: str,
    longitude_col: str,
    latitude_col: str,
):
    """
    Filters a Spark DataFrame to include rows where geographical points fall within a specified geometry,
    and retains only users who have at least k distinct days with pings inside the geometry between T0 and T1.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input Spark DataFrame containing coordinate columns.
    bounding_wkt : str
        Well-Known Text (WKT) string representing the bounding geometry (e.g., a polygon) in EPSG:4326 CRS.
    k : int
        Minimum number of distinct days with pings inside the polygon for the user to be retained.
    T0 : str
        Start of the timeframe for filtering (as a string compatible with date parsing).
    T1 : str
        End of the timeframe for filtering (as a string compatible with date parsing).
    spark : SparkSession
        The active SparkSession instance for executing Spark operations.
    user_col : str
        Name of the column containing user IDs.
    timestamp_col : str
        Name of the column containing timestamps.
    longitude_col : str
        Name of the longitude column.
    latitude_col : str
        Name of the latitude column.

    Returns
    -------
    pyspark.sql.DataFrame
        Filtered DataFrame including only rows within the specified timeframe, inside the specified geometry,
        and belonging to users with at least k distinct days with pings inside the geometry.
    """
    from sedona.register import SedonaRegistrator
    
    SedonaRegistrator.registerAll(spark)

    df = df.withColumn(timestamp_col, F.to_timestamp(F.col(timestamp_col)))
    df_filtered = df.filter(
        (F.col(timestamp_col) >= F.to_timestamp(F.lit(T0))) & 
        (F.col(timestamp_col) <= F.to_timestamp(F.lit(T1)))
    )
    df_filtered = df_filtered.withColumn(
        "coordinate", 
        F.expr(f"ST_Point(CAST({longitude_col} AS DECIMAL(24,20)), CAST({latitude_col} AS DECIMAL(24,20)))")
    )

    df_filtered.createOrReplaceTempView("temp_df")
    query = f"""
        SELECT *, DATE({timestamp_col}) AS date
        FROM temp_df
        WHERE ST_Contains(ST_GeomFromWKT('{bounding_wkt}'), coordinate)
    """

    df_inside = spark.sql(query)

    user_day_counts = df_inside.groupBy(user_col).agg(
        F.countDistinct("date").alias("distinct_days")
    )
    users_with_k_days = user_day_counts.filter(
        F.col("distinct_days") >= k
    ).select(user_col)
    result_df = df_inside.join(users_with_k_days, on=user_col, how='inner')

    return result_df


def coarse_filter(df: pd.DataFrame):
    pass


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