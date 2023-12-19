from sedona.spark import *

def clean_coords(df, spark):
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