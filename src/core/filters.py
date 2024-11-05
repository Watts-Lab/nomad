import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

from constants import DEFAULT_SCHEMA


# user can pass latitude and longitude as kwargs, user can pass x and y, OR traj_cols (prioritizing latitude, longitude). 
def to_projection(df: pd.DataFrame,
                  traj_cols: dict = None,
                  latitude: str = DEFAULT_SCHEMA["latitude"],
                  longitude: str = DEFAULT_SCHEMA["longitude"],
                  from_crs: str = "EPSG:4326",
                  to_crs: str = "EPSG:3857",
                  spark_session: SparkSession = None):
    """
    Projects latitude and longitude columns from one CRS to another.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing latitude and longitude columns.
    traj_cols : dict, optional
        Dictionary containing column mappings, 
        e.g., {"latitude": "latitude", "longitude": "longitude"}.
    latitude : str, optional
        Name of the latitude column (default is "latitude").
    longitude : str, optional
        Name of the longitude column (default is "longitude").
    from_crs : str, optional
        EPSG code for the original CRS (default is "EPSG:4326").
    to_crs : str, optional
        EPSG code for the target CRS (default is "EPSG:3857").
    spark_session : SparkSession, optional
        Spark session for distributed computation, if needed.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'x' and 'y' columns representing projected coordinates.
    """

    lat_col = traj_cols.get("latitude", latitude) if traj_cols else latitude
    lon_col = traj_cols.get("longitude", longitude) if traj_cols else longitude

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Latitude or longitude columns '{lat_col}', '{lon_col}' not found in DataFrame.")

    if spark_session:
        pass  #TODO
    else:
        proj_cols = _to_projection(df[lat_col], df[lon_col], from_crs, to_crs)
        df['x'] = proj_cols['x']
        df['y'] = proj_cols['y']

    return df


def _to_projection(lat_col,
                   long_col,
                   from_crs: str,
                   to_crs: str):
    """
    Helper function to project latitude/longitude columns to a new CRS.
    """
    gdf = gpd.GeoSeries(gpd.points_from_xy(long_col, lat_col),
                        crs=from_crs)
    projected = gdf.to_crs(to_crs)
    output = pd.DataFrame({'x': projected.x, 'y': projected.y})

    return output


def filter_to_box(df: pd.DataFrame,
                  polygon: Polygon,
                  traj_cols: dict = None,
                  latitude: str = DEFAULT_SCHEMA["latitude"],
                  longitude: str = DEFAULT_SCHEMA["longitude"],
                  spark_session: SparkSession = None):
    '''
    Filters DataFrame to keep points within a specified polygon's bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latitude and longitude columns.
    polygon : shapely.geometry.Polygon
        Polygon defining the area to retain points within.
    traj_cols : dict, optional
        Dictionary containing column mappings, 
        e.g., {"latitude": "latitude", "longitude": "longitude"}.
    latitude : str, optional
        Name of the latitude column (default is "latitude").
    longitude : str, optional
        Name of the longitude column (default is "longitude").
    spark_session : SparkSession, optional
        Spark session for distributed computation, if needed.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with points inside the polygon's bounds.
    '''
    lat_col = traj_cols.get("latitude", latitude) if traj_cols else latitude
    lon_col = traj_cols.get("longitude", longitude) if traj_cols else longitude

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Latitude or longitude columns '{lat_col}', '{lon_col}' not found in DataFrame.")

    if spark_session:
        pass  # TODO

    else:
        if not isinstance(polygon, Polygon):
            raise TypeError("Polygon parameter must be a Shapely Polygon object.")

        min_x, min_y, max_x, max_y = polygon.bounds

        return df[(df[longitude].between(min_y, max_y)) & (df[latitude].between(min_x, max_x))]


def _filter_to_box_spark(df: pd.DataFrame,
                         bounding_wkt: str,
                         spark: SparkSession,
                         longitude_col: str,
                         latitude_col: str,
                         id_col: str):
    """Filters a DataFrame based on whether geographical points
    (defined by longitude and latitude) fall within a specified geometry.

    Parameters
    ----------
    df : DataFrame
        The Spark DataFrame to be filtered. It should contain columns
        corresponding to longitude and latitude values, as well as an id column.

    bounding_wkt : str
        The Well-Known Text (WKT) string representing the bounding geometry
        within which points are tested for inclusion. The WKT should define
        a polygon in the EPSG:4326 coordinate reference system.

    spark : SparkSession
        The active SparkSession instance used to execute Spark operations.

    longitude_col : str, default "longitude"
        The name of the column in 'df' containing longitude values. Longitude
        values should be in the EPSG:4326 coordinate reference system.

    latitude_col : str, default "latitude"
        The name of the column in 'df' containing latitude values. Latitude
        values should be in the EPSG:4326 coordinate reference system.

    id_col : str, default "id"
        The name of the column in 'df' containing user IDs.

    Returns
    ----------
    DataFrame
        A new Spark DataFrame filtered to include only rows where the point
        (longitude, latitude) falls within the specified geometric boundary
        defined by 'bounding_wkt'. This DataFrame includes all original columns
        from 'df' and an additional column 'in_geo' that is true if the point
        falls within the specified geometric boundary and false otherwise.
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


def coarse_filter(df: pd.DataFrame):
    pass


def _filtered_users(df: pd.DataFrame,
                    k: int,
                    T0: str,
                    T1: str,
                    polygon: Polygon,
                    user_col: str,
                    timestamp_col: str,
                    latitude_col: str,
                    longitude_col: str) -> pd.DataFrame:
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
    latitude : str
        Name of the column containing latitude values.
    longitude : str
        Name of the column containing longitude values.

    Returns
    -------
    pd.Series
        A Series containing the user IDs for users who have at 
        least k distinct days with pings inside the polygon.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df_filtered = df[(df[timestamp_col] >= T0) & (df[timestamp_col] <= T1)]
    df_filtered = _in_geo(df_filtered, latitude_col, longitude_col, polygon)
    df_filtered['date'] = df_filtered[timestamp_col].dt.date

    filtered_users = (
        df_filtered[df_filtered['in_geo'] == 1]
        .groupby(user_col)['date']
        .nunique()
        .reset_index()
    )

    filtered_users = filtered_users[filtered_users['date'] >= k][user_col]

    return filtered_users


def _in_geo(df: pd.DataFrame,
            latitude_col: str,
            longitude_col: str,
            polygon: Polygon) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame indicating whether points are 
    inside the polygon (1) or not (0).
    """

    def _point_in_polygon(lat, lon):
        point = Point(lat, lon)
        return 1 if polygon.contains(point) else 0

    df['in_geo'] = df.apply(lambda row: _point_in_polygon(row[latitude_col], row[longitude_col]), axis=1)

    return df


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