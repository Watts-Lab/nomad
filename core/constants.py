from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

UID = 'uid'
TIMESTAMP = 'timestamp'
MERCATOR_COORD = 'mercator_coord'
MERCATOR_X = 'x'
MERCATOR_Y = 'y'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
DATE = 'date'
DATE_HOUR = 'date_hour'
DAY_OF_WEEK = 'day_of_week'

default_schema = StructType([ 
    StructField(UID, StringType(), True), 
    StructField(TIMESTAMP, LongType(), True), 
    StructField(LATITUDE, DoubleType(), True), 
    StructField(LONGITUDE, DoubleType(), True), 
]) 