from sedona.spark import *

def clean_coords(df):
    """
    Takes in a spark dataframe with columns
        'latitude', containing latitudes in EPSG:4326
        'longitude', containing longitudes in EPSG:4326
    
    Returns a spark dataframe with a new column 'mercator_coord' containing point geometries (lat, long) in EPSG:3857
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