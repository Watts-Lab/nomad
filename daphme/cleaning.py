from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sedona.spark import *

def to_local_time(df: DataFrame, 
                  timezone_to: str,
                  timestamp_col: str = "timestamp") -> DataFrame:
    """
    Parameters
    ----------
    df: Spark dataframe with a column named timestamp_col, which contains timestamps
            
    timezone: a valid timezone identifier for the local timezone of the data. E.g.,"America/New_York", "UTC" 
    
    timestamp_col: (optional) name of the column containing timestamps
    
    Returns
    ----------
    Spark dataframe with additional columns
        'local_timestamp'
        'date'
        'date_hour'
        'day_of_week' (1 for Sunday, 2 for Monday, ..., 7 for Saturday)
    """
    
    # Convert timestamp to local timestamp
    df = df.withColumn(
        "local_timestamp",
        F.from_utc_timestamp(
            F.to_timestamp(F.col(timestamp_col) / 1000),  # divide by 1000 for milliseconds
            timezone_to  # convert from UTC to timezone specified by timezone_to
        ))

    # Add date, date_hour, and day_of_week columns
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

def to_mercator(df: DataFrame, spark: SparkSession):
    """
    Parameters
    ----------
    df: Spark dataframe with columns
        'latitude', containing latitudes of type Double() in EPSG:4326
        'longitude', containing longitudes of type Double() in EPSG:4326
        
    spark: Spark Session
    
    Returns
    ----------
    Spark dataframe with a new column 'mercator_coord' containing point geometries (lat, long) in EPSG:3857
    """
    
    df.createOrReplaceTempView("df")
    df = spark.sql("""
        SELECT *,
               ST_FlipCoordinates(
                   ST_Transform(
                       ST_MakePoint(longitude, latitude), 
                       'EPSG:4326', 'EPSG:3857'
                   )
               ) AS mercator_coord
        FROM df
        """)
    return df