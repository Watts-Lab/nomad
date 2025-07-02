import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
from functools import partial
import re
from pyspark.sql import SparkSession
import sys
import os
# --------- Delete? ---------------
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.types as pat
import pyarrow.csv as pc_csv
# ----------------------------------
from nomad.constants import DEFAULT_SCHEMA
import warnings
import inspect
from nomad.constants import FILTER_OPERATORS
from nomad.io.base import _fallback_spatial_cols, _parse_traj_cols

def _is_traj_df_spark(df, traj_cols=None, **kwargs):
    if not isinstance(df, psp.sql.dataframe.DataFrame):
        return False

    traj_cols = _parse_traj_cols(df.columns, traj_cols, kwargs)

    if not _has_spatial_cols(df.columns, traj_cols) or not _has_time_cols(df.columns, traj_cols):
        return False

    if 'datetime' in traj_cols and traj_cols['datetime'] in df.columns:
        if not isinstance(df.schema[traj_cols['datetime']].dataType, TimestampType):
            return False

    if 'timestamp' in traj_cols and traj_cols['timestamp'] in df.columns:
        if not isinstance(df.schema[traj_cols['timestamp']].dataType, (IntegerType, LongType, TimestampType)):
            return False

    for col in ['latitude', 'longitude', 'x', 'y']:
        if col in traj_cols and traj_cols[col] in df.columns:
            if not isinstance(df.schema[traj_cols[col]].dataType, (FloatType, DoubleType)):
                return False

    for col in ['user_id', 'geohash']:
        if col in traj_cols and traj_cols[col] in df.columns:
            if not isinstance(df.schema[traj_cols[col]].dataType, StringType):
                return False

    return True

# Cast data types and address datetime issues

def _cast_traj_cols_spark(df, traj_cols):
    """
    Casts specified trajectory columns in a loaded Spark DataFrame to their expected data types, 
    with warnings for timestamp precision and timezone-naive datetimes.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame containing the data to be cast.
    traj_cols : dict
        Dictionary mapping expected trajectory column names 
        (e.g., 'latitude', 'longitude', 'timestamp', 'datetime', 'user_id', 'geohash')
        to the actual column names in the DataFrame.

    Returns
    -------
    pyspark.sql.DataFrame
        The DataFrame with specified columns cast to their expected types.

    Notes
    -----
    - The 'datetime' column is cast to Datetime64 or Object series (with Timestamps) if not already of that type.
      A warning is issued if the first row appears timezone-naive.
    - The 'timestamp' column is cast to IntegerType, with a warning for possible millisecond or nanosecond precision.
    - Spatial columns ('latitude', 'longitude', 'x', 'y') are cast to FloatType if necessary.
    - User identifier and geohash columns are cast to StringType.
    """

    # Cast 'datetime' column appropriately
    if 'datetime' in traj_cols and traj_cols['datetime'] in df.columns:
        datetime_col = traj_cols['datetime']
        if not isinstance(df.schema[datetime_col].dataType, TimestampType):
            df = df.withColumn(datetime_col, col(datetime_col).cast(TimestampType()))
        
        # Check if the first row is timezone-naive (PySpark doesn't store tz info)
        first_row = df.select(datetime_col).first()
        if first_row and first_row[0] is not None:
            if first_row[0].tzinfo is None:
                warnings.warn(
                    f"The '{datetime_col}' column appears to be timezone-naive. "
                    "Consider localizing to a timezone to avoid inconsistencies."
                )

    # Cast 'timestamp' column to IntegerType and check precision
    if 'timestamp' in traj_cols and traj_cols['timestamp'] in df.columns:
        timestamp_col = traj_cols['timestamp']
        if not isinstance(df.schema[timestamp_col].dataType, (IntegerType, LongType)):
            df = df.withColumn(timestamp_col, col(timestamp_col).cast(IntegerType()))

        # Check for millisecond/nanosecond values by inspecting first row
        first_timestamp = df.select(timestamp_col).first()
        if first_timestamp and first_timestamp[0] is not None: # different index?
            timestamp_length = len(str(first_timestamp[0]))
            
            if timestamp_length == 13:
                warnings.warn(
                    f"The '{timestamp_col}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies, converting to seconds is recommended."
                )
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{timestamp_col}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies, converting to seconds is recommended."
                )

    # Cast spatial columns to FloatType or DoubleType
    float_cols = ['latitude', 'longitude', 'x', 'y']
    for col_name in float_cols:
        if col_name in traj_cols and traj_cols[col_name] in df.columns:
            actual_col = traj_cols[col_name]
            if not isinstance(df.schema[actual_col].dataType, (FloatType, DoubleType)):
                df = df.withColumn(actual_col, col(actual_col).cast(FloatType()))

    # Cast identifier and geohash columns to StringType
    string_cols = ['user_id', 'geohash']
    for col_name in string_cols:
        if col_name in traj_cols and traj_cols[col_name] in df.columns:
            actual_col = traj_cols[col_name]
            if not isinstance(df.schema[actual_col].dataType, StringType):
                df = df.withColumn(actual_col, col(actual_col).cast(StringType()))

    return df
