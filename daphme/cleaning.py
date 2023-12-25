from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sedona.spark import *

def to_local_time(df: DataFrame, 
                  timezone_to: str,
                  timestamp_col: str = "timestamp",
                  epoch_unit: str = "seconds") -> DataFrame:
    """Transforms a column of epoch times in a Spark DataFrame to a local time zone specified timezone_to. Additional columns for date, hour, and day of the week are also added to the DataFrame.

    Parameters
    ----------
    df : DataFrame
        A Spark DataFrame containing a column with epoch times to be converted.

    timezone_to : str
        A valid timezone identifier for the local timezone of the data (e.g., "America/New_York", "UTC").

    timestamp_col : str (default "timestamp")
        The name of the column in 'df' containing epoch times.

    epoch_unit : str (default "seconds")
        The unit of epoch time in the 'timestamp_col'. Acceptable values include "seconds", "milliseconds", 
        "microseconds", and "nanoseconds". Defaults to "seconds" if not specified.

    Returns
    ----------
    DataFrame
        A new Spark DataFrame with all original columns from 'df' and the following additional columns:
            - 'local_timestamp': The local timestamp derived from the original epoch time.
            - 'date': The date extracted from 'local_timestamp'.
            - 'date_hour': The date and hour extracted from 'local_timestamp'.
            - 'day_of_week': The day of the week (1 for Sunday, 2 for Monday, ..., 7 for Saturday) derived from 'local_timestamp'.

    Example
    ----------
    >>> # Assuming a SparkSession `spark` and a DataFrame `df` with a 'timestamp' column are predefined.
    >>> timezone_str = "America/New_York"
    >>> converted_df = to_local_time(df, timezone_str)
    >>> converted_df.show()
    """

    divisor = {
        "seconds": 1,
        "milliseconds": 1000,
        "microseconds": 1000000,
        "nanoseconds": 1000000000
    }.get(epoch_unit, 1)

    df = df.withColumn(
        "local_timestamp",
        F.from_utc_timestamp(
            F.to_timestamp(F.col(timestamp_col) / divisor),
            timezone_to
        ))

    df = df.withColumn(
        "date",
        F.to_date(F.col("local_timestamp"))
    ).withColumn(
        "date_hour",
        F.date_format(F.col("local_timestamp"), "yyyy-MM-dd HH")
    ).withColumn(
        "day_of_week",
        F.dayofweek(F.col("local_timestamp"))
    )

    return df

def to_mercator(df: DataFrame, 
                spark: SparkSession,
                longitude_col: str = "longitude", 
                latitude_col: str = "latitude") -> DataFrame:
    """Converts geographic coordinates from EPSG:4326 (WGS 84) to EPSG:3857 (Web Mercator projection).
    
    Parameters
    ----------
    df : DataFrame
        A Spark DataFrame containing columns corresponding to longitude and latitude values in EPSG:4326.

    spark : SparkSession
        The active SparkSession instance used to execute Spark SQL operations.

    Returns
    ----------
    DataFrame
        A new Spark DataFrame with all original columns from 'df' and additional columns:
        - 'x': The x coordinate (longitude) in EPSG:3857 coordinate system.
        - 'y': The y coordinate (latitude) in EPSG:3857 coordinate system.
        - 'mercator_coord': The point geometry (longitude, latitude) in EPSG:3857 coordinate system.

    Example
    ----------
    >>> # Assuming a SparkSession `spark` and a DataFrame `df` are predefined
    >>> mercator_df = to_mercator(df, spark)
    >>> mercator_df.show()
    """
    
    df.createOrReplaceTempView("df")
    
    query = f"""
    WITH mercator_df AS (
        SELECT *,
               ST_Transform(
                   ST_MakePoint({longitude_col}, {latitude_col}), 
                   'EPSG:4326', 'EPSG:3857'
               ) AS mercator_coord
        FROM df
    )
    SELECT *,
           ST_X(mercator_coord) AS x,
           ST_Y(mercator_coord) AS y
    FROM mercator_df
    """
    
    return spark.sql(query)

def coarse_filter(df: DataFrame, 
                  bounding_wkt: str, 
                  spark: SparkSession,
                  longitude_col: str = "longitude", 
                  latitude_col: str = "latitude", 
                  id_col: str = "id") -> DataFrame:
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