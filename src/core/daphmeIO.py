import pyarrow.parquet as pq
import pandas as pd
from functools import partial
import multiprocessing
from multiprocessing import Pool
import re
from pyspark.sql import SparkSession
import sys
import os
import pyarrow.compute as pc
import pyarrow.dataset as ds
# from . import constants
import core.constants

def _update_schema(original, new_labels):
    
    updated_schema = dict(original)
    for label in new_labels:
        if label in constants.DEFAULT_SCHEMA:
            updated_schema[label] = new_labels[label]
    return updated_schema

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


def _cast_traj_cols(df, traj_cols):
    """
    Casts specified trajectory columns in a DataFrame to their expected data types, 
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
    - The 'datetime' column is converted to datetime if not already of a datetime type.
      A warning is issued if it is timezone-naive.
    - The 'timestamp' column is expected to contain Unix timestamps in seconds.
      If values appear to be in milliseconds or nanoseconds, a warning will recommend conversion.
    - Spatial columns ('latitude', 'longitude', 'x', 'y') are cast to floats if necessary.
    - User identifier and geohash columns are cast to strings.
    """

    # Convert 'datetime' column to datetime, warn if timezone-naive
    if 'datetime' in traj_cols and traj_cols['datetime'] in df:
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[traj_cols['datetime']].dtype):
            df[traj_cols['datetime']] = pd.to_datetime(df[traj_cols['datetime']])
            
            # Warn if datetime is timezone-naive
            if df[traj_cols['datetime']].dt.tz is None:
                warnings.warn(
                    f"The '{traj_cols['datetime']}' column is timezone-naive. "
                    "Consider localizing to a timezone to avoid inconsistencies."
                )

    # Convert 'timestamp' column to integer, check precision and recommend seconds
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

def _is_traj_df(df, traj_cols = None, **kwargs):
    
    if not (isinstance(df, pd.DataFrame) or isinstance(df, gpd.GeoDataFrame)):
        return False
    
    if not traj_cols:
        traj_cols = _update_schema({}, kwargs) #kwargs ignored if traj_cols is passed
        
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    
    if not _has_traj_cols(df, traj_cols):
        return False
    
    if 'datetime' in traj_cols and traj_cols['datetime'] in df:
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[traj_cols['datetime']].dtype):
            return False
    elif 'timestamp' in traj_cols and traj_cols['timestamp'] in df:
        if not pd.core.dtypes.common.is_integer_dtype(df[traj_cols['timestamp']].dtype):
            return False

    float_cols = ['latitude', 'longitude', 'x', 'y']
    for col in float_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_float_dtype(df[traj_cols[col]].dtype):
                return False

    string_cols = ['user_id', 'geohash']
    for col in string_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_string_dtype(df[traj_cols[col]].dtype):
                return False

    return True


def from_pandas(df, traj_cols=None, spark_enabled=False, **kwargs):
    
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
            filter=ds.field(uid_col).isin(users)
        ).to_pandas()

    return _cast_traj_cols(df, traj_cols)