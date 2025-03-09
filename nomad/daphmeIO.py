import pandas as pd
import geopandas as gpd
import pyspark as psp
from functools import partial
import multiprocessing
from multiprocessing import Pool
import warnings
import re
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType, IntegerType, LongType, FloatType, DoubleType, StringType

import sys
import os
import pyarrow.compute as pc
import pyarrow.dataset as ds
# from . import constants
from nomad import constants

# utils
def _update_schema(original, new_labels):
    
    updated_schema = dict(original)
    for label in new_labels:
        if label in constants.DEFAULT_SCHEMA:
            updated_schema[label] = new_labels[label]
    return updated_schema

def _is_traj_df(df, traj_cols = None, **kwargs):
    
    if not (isinstance(df, pd.DataFrame) or isinstance(df, gpd.GeoDataFrame)):
        return False
    
    if not traj_cols:
        traj_cols = _update_schema({}, kwargs) #kwargs ignored if traj_cols is passed
        
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    if not _has_spatial_cols(df.columns, traj_cols) or not _has_time_cols(df.columns, traj_cols):
        return False

    # check datetime type
    if 'datetime' in traj_cols and traj_cols['datetime'] in df.columns:
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[traj_cols['datetime']].dtype):
            return False

    # check timestamp is integer type
    if 'timestamp' in traj_cols and traj_cols['timestamp'] in df.columns:
        if not pd.core.dtypes.common.is_integer_dtype(df[traj_cols['timestamp']].dtype):
            return False

    float_cols = ['latitude', 'longitude', 'x', 'y']
    for col in float_cols:
        if col in traj_cols and traj_cols[col] in df.columns:
            if not pd.core.dtypes.common.is_float_dtype(df[traj_cols[col]].dtype):
                return False

    string_cols = ['user_id', 'geohash']
    for col in string_cols:
        if col in traj_cols and traj_cols[col] in df.columns:
            if not pd.core.dtypes.common.is_string_dtype(df[traj_cols[col]].dtype):
                return False

    return True

def _has_time_cols(col_names, traj_cols):
    
    temporal_exists = (
        ('datetime' in traj_cols and traj_cols['datetime'] in col_names) or
        ('timestamp' in traj_cols and traj_cols['timestamp'] in col_names)
    )

    if not temporal_exists:
        raise ValueError(
            "Could not find required temporal columns in {}. The dataset must contain or map to either 'datetime' or 'timestamp'.".format(col_names)
        )
    
    return temporal_exists

def _has_spatial_cols(col_names, traj_cols):
    
    spatial_exists = (
        ('latitude' in traj_cols and 'longitude' in traj_cols and 
         traj_cols['latitude'] in col_names and traj_cols['longitude'] in col_names) or
        ('x' in traj_cols and 'y' in traj_cols and 
         traj_cols['x'] in col_names and traj_cols['y'] in col_names) or
        ('geohash' in traj_cols and traj_cols['geohash'] in col_names)
    )

    if not spatial_exists:
        raise ValueError(
            "Could not find required spatial columns in {}. The dataset must contain or map to at least one of the following sets: "
            "('latitude', 'longitude'), ('x', 'y'), or 'geohash'.".format(col_names)
        )
    
    return spatial_exists

def _has_user_cols(col_names, traj_cols):
    
    user_exists = 'user_id' in traj_cols and traj_cols['user_id'] in col_names

    if not user_exists:
        raise ValueError(
            "Could not find required user identifier column in {}. The dataset must contain or map to 'user_id'.".format(col_names)
        )
    
    return user_exists

# SPARK check is traj_dataframe

def _is_traj_df_spark(df, traj_cols = None, **kwargs):
    
    if not (isinstance(df, psp.sql.dataframe.DataFrame)):
        return False
    
    if not traj_cols:
        traj_cols = _update_schema({}, kwargs) #kwargs ignored if traj_cols is passed
        
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    
    if not _has_spatial_cols(df.columns, traj_cols) or not _has_time_cols(df.columns, traj_cols):
        return False

    # check datetime type
    if 'datetime' in traj_cols and traj_cols['datetime'] in df.columns:
            if not isinstance(df.schema[traj_cols['datetime']].dataType, TimestampType):
                return False

    # check timestamp is integer type
    if 'timestamp' in traj_cols and traj_cols['timestamp'] in df.columns:
        if not isinstance(df.schema[traj_cols['timestamp']].dataType, (IntegerType, LongType, TimestampType)):
            return False

    # Check float columns
    float_cols = ['latitude', 'longitude', 'x', 'y']
    for col in float_cols:
        if col in traj_cols and traj_cols[col] in df.columns:
            if not isinstance(df.schema[traj_cols[col]].dataType, (FloatType, DoubleType)):
                return False

    # Check string columns
    string_cols = ['user_id', 'geohash']
    for col in string_cols:
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
    - The 'datetime' column is cast to TimestampType if not already of that type.
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
        if first_timestamp and first_timestamp[0] is not None:
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
    

def _cast_traj_cols(df, traj_cols):
    """
    Casts specified trajectory columns in a loaded DataFrame to their expected data types, 
    with warnings for timestamp precision and timezone-naive datetimes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be cast.
    traj_cols : dict
        Dictionary mapping expected trajectory column names 
        (e.g., 'latitude', 'longitude', 'timestamp', 'datetime', 'user_id', 'geohash')
        to the actual column names in the DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with specified columns cast to their expected types.

    Notes
    -----
    - The 'datetime' column is cast to datetime.datetime if not already of a datetime type.
      A warning is issued if it is timezone-naive.
    - The 'timestamp' column is expected to contain Unix timestamps in seconds.
      If values appear to be in milliseconds or nanoseconds, a warning will recommend conversion.
    - Spatial columns ('latitude', 'longitude', 'x', 'y') are cast to floats if necessary.
    - User identifier and geohash columns are cast to strings.
    """

    # cast 'datetime' column to datetime, warn if timezone-naive
    if 'datetime' in traj_cols and traj_cols['datetime'] in df:
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[traj_cols['datetime']].dtype):
            df[traj_cols['datetime']] = pd.to_datetime(df[traj_cols['datetime']])
            
            # Warn if datetime is timezone-naive
            if df[traj_cols['datetime']].dt.tz is None:
                warnings.warn(
                    f"The '{traj_cols['datetime']}' column is timezone-naive. "
                    "Consider localizing to a timezone to avoid inconsistencies."
                )

    # cast 'timestamp' column to integer, check precision and recommend seconds
    if 'timestamp' in traj_cols and traj_cols['timestamp'] in df:
        timestamp_col_name = traj_cols['timestamp']
        if not pd.core.dtypes.common.is_integer_dtype(df[timestamp_col_name].dtype):
            df[timestamp_col_name] = df[timestamp_col_name].astype(int)

        # Check for possible millisecond or nanosecond values and issue a warning
        first_timestamp = df[timestamp_col_name].iloc[0]
        timestamp_length = len(str(first_timestamp))
        
        if timestamp_length > 10:
            if timestamp_length == 13:
                warnings.warn(
                    f"The '{timestamp_col_name}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies, converting to seconds is recommended."
                )
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{timestamp_col_name}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies, converting to seconds is recommended."
                )

    # Cast spatial columns to float
    float_cols = ['latitude', 'longitude', 'x', 'y']
    for col in float_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_float_dtype(df[traj_cols[col]].dtype):
                df[traj_cols[col]] = df[traj_cols[col]].astype("float")

    # Cast identifier and geohash columns to string
    string_cols = ['user_id', 'geohash']
    for col in string_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_string_dtype(df[traj_cols[col]].dtype):
                df[traj_cols[col]] = df[traj_cols[col]].astype("str")

    return df

# spark casting functions (only on schema)    

def from_pandas(df, traj_cols=None, spark=None, **kwargs):
    """
    Parameters
    ----------
    spark : pyspark spark session
    """
    if not (isinstance(df, pd.DataFrame) or isinstance(df, gpd.GeoDataFrame)):
        raise TypeError(
            "Expected the data argument to be either a pandas DataFrame or a GeoPandas GeoDataFrame."
        )

    if not traj_cols:
        traj_cols = _update_schema({}, kwargs) #kwargs ignored if traj_cols is passed
 
    # Warn if any specified trajectory columns are not found in df
    for key, value in traj_cols.items():
        if value not in df:
            warnings.warn(f"Trajectory column '{value}' specified for '{key}' not found in df.")

    # Update traj_cols with default schema values when missing
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Perform spatial and temporal column checks
    _has_spatial_cols(df.columns, traj_cols)
    _has_time_cols(df.columns, traj_cols)

    # Cast trajectory columns as necessary
    return _cast_traj_cols(df, traj_cols)


def from_file(filepath, format="csv", traj_cols=None, **kwargs):
    """
    Load and cast trajectory data from a specified file path or list of paths.

    Parameters
    ----------
    filepath : str or list of str
        Path or list of paths to the file(s) or directories containing the data.
    format : str, optional
        The format of the data files, either 'csv' or 'parquet'.
    traj_cols : dict, optional
        Mapping of trajectory column names (e.g., 'latitude', 'timestamp').
    **kwargs :
        Additional arguments for reading CSV files, passed to pandas read_csv.

    Returns
    -------
    pd.DataFrame
        The DataFrame with specified trajectory columns cast to expected types.
    
    Notes
    -----
    - Checks before loading that files have required trajectory columns for analysis. 
    """
    
    assert format in ["csv", "parquet"]

    if not traj_cols:
        traj_cols = _update_schema({}, kwargs) #kwargs ignored if traj_cols is passed
        
    if isinstance(filepath, list):
        # Pyarrow does not support lists of directories
        datasets = [ds.dataset(path, format=format, partitioning="hive") for path in filepath]
        dataset = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        column_names = dataset.schema.names
        
    elif format == 'parquet' or os.path.isdir(filepath):
        dataset = ds.dataset(filepath, format=format, partitioning="hive")
        column_names = dataset.schema.names
    else:
        column_names = pd.read_csv(filepath, nrows=0).columns

    for key, value in traj_cols.items():
        if value not in column_names:
            warnings.warn(f"Trajectory column '{value}' specified for '{key}' not found in the data source.")

    # add default column names to traj_cols when missing
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    _has_spatial_cols(column_names, traj_cols)
    _has_time_cols(column_names, traj_cols)

    if format == "csv" and not os.path.isdir(filepath):
        # pass kwargs to read_csv without unexpected keyword argument errors
        read_csv_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(pd.read_csv).parameters}
        df = pd.read_csv(filepath, **read_csv_kwargs)
    else:
        df = dataset.to_table().to_pandas()

    return _cast_traj_cols(df, traj_cols)
        

def sample_users(filepath, format="csv", frac_users=1.0, traj_cols=None, **kwargs):
    """
    Samples unique user IDs from a specified file path or list of paths.

    Parameters
    ----------
    filepath : str or list of str
        Path or list of paths to the file(s) or directories containing the data.
    format : str, optional
        The format of the data files, either 'csv' or 'parquet'.
    frac_users : float, optional
        Fraction of users to sample, by default 1.0 (all users).
    traj_cols : dict, optional
        Mapping of trajectory column names, including 'user_id'.
    **kwargs :
        Additional arguments for reading CSV files, passed to pandas read_csv.

    Returns
    -------
    pd.Series
        A Series of sampled user IDs.
    """
    assert format in ["csv", "parquet"]
   
    if not traj_cols:
        traj_cols = _update_schema({}, kwargs) #kwargs ignored if traj_cols is passed
        
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    uid_col = traj_cols['user_id']

    if isinstance(filepath, list):
        datasets = [ds.dataset(path, format=format, partitioning="hive") for path in filepath]
        dataset = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        column_names = dataset.schema.names
    elif format == 'parquet' or os.path.isdir(filepath):
        dataset = ds.dataset(filepath, format=format, partitioning="hive")
        column_names = dataset.schema.names
    else:
        column_names = pd.read_csv(filepath, nrows=0).columns

    _has_user_cols(column_names, traj_cols)

    if format == "csv" and not os.path.isdir(filepath):
        df = pd.read_csv(filepath, usecols=[uid_col])
        user_ids = df[uid_col].unique()
    else:
        user_ids = pc.unique(dataset.to_table(columns=[uid_col])[uid_col]).to_pandas()

    return user_ids.sample(frac=frac_users) if frac_users < 1.0 else user_ids

def sample_from_file(filepath, users, format="csv", traj_cols=None, **kwargs):
    """
    Loads data for specified users from a file path or list of paths.

    Parameters
    ----------
    filepath : str or list of str
        Path or list of paths to the file(s) or directories containing the data.
    users : list
        List of user IDs to filter for in the dataset.
    format : str, optional
        The format of the data files, either 'csv' or 'parquet'.
    traj_cols : dict, optional
        Mapping of trajectory column names, including 'user_id'.
    **kwargs :
        Additional arguments for reading CSV files, passed to pandas read_csv.

    Returns
    -------
    pd.DataFrame
        A DataFrame with data filtered to specified users and cast trajectory columns.
    """
    assert format in ["csv", "parquet"]

    if not traj_cols:
        traj_cols = _update_schema({}, kwargs) #kwargs ignored if traj_cols is passed

    # Load schema for PyArrow datasets or CSV header to perform column checks
    if isinstance(filepath, list):
        datasets = [ds.dataset(path, format=format, partitioning="hive") for path in filepath]
        dataset = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        column_names = dataset.schema.names
    elif format == 'parquet' or os.path.isdir(filepath):
        dataset = ds.dataset(filepath, format=format, partitioning="hive")
        column_names = dataset.schema.names
    else:
        column_names = pd.read_csv(filepath, nrows=0).columns

    for key, value in traj_cols.items():
        if value not in column_names:
            warnings.warn(f"Trajectory column '{value}' specified for '{key}' not found in the data source.")

    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    uid_col = traj_cols['user_id']

    # Perform required column checks
    _has_user_cols(column_names, traj_cols)
    _has_spatial_cols(column_names, traj_cols)
    _has_time_cols(column_names, traj_cols)

    # Load data and filter by specified users
    if format == "csv" and not os.path.isdir(filepath):
        df = pd.read_csv(filepath)
        df = df[df[uid_col].isin(users)]
    else:
        df = dataset.to_table(
            filter=ds.field(uid_col).isin(users),
            columns=None  # to include partition columns
        ).to_pandas()

    return _cast_traj_cols(df, traj_cols)