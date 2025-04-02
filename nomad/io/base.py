import pyarrow.parquet as pq
import pandas as pd
from functools import partial
import multiprocessing
from multiprocessing import Pool
import re
import pdb
from pyspark.sql import SparkSession
import sys
import os
import pyarrow.compute as pc
import pyarrow.dataset as ds
from nomad.constants import DEFAULT_SCHEMA
import numpy as np
import geopandas as gpd
import warnings

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_string_dtype,
    is_float_dtype,
    is_integer_dtype,
)


# utils
def _update_schema(original, new_labels):
    updated_schema = dict(original)
    for label in new_labels:
        if label in DEFAULT_SCHEMA:
            updated_schema[label] = new_labels[label]
    return updated_schema

def _parse_traj_cols(columns, traj_cols, optional_args):
    """
    Internal helper to finalize trajectory column names using user input and defaults.
    """
    if traj_cols:
        for k in optional_args:
            if k in traj_cols and optional_args[k] != traj_cols[k]:
                raise ValueError(
                    f"Conflicting column name for '{k}': '{traj_cols[k]}' (from traj_cols) vs '{optional_args[k]}' (from keyword arguments)."
                )
        traj_cols = _update_schema(traj_cols, optional_args)
    else:
        traj_cols = _update_schema({}, optional_args)

    for key, value in traj_cols.items():
        if value not in columns:
            warnings.warn(f"Trajectory column '{value}' specified for '{key}' not found in DataFrame.")

    return _update_schema(DEFAULT_SCHEMA, traj_cols)
    
def _offset_seconds_from_ts(ts):
    """
    Given a Timestamp ts, return its UTC offset in seconds.
    Naive timestamps return 0.
    Missing values return None.
    """
    if pd.isnull(ts):
        return None
    if ts.tz is None:
        return 0
    offset_td = ts.utcoffset()
    return int(offset_td.total_seconds()) if offset_td is not None else 0

def _offset_string_hrs(offset_seconds):
    mapping = {}
    for offset in offset_seconds.unique():
        abs_off = abs(offset)
        hours = abs_off // 3600
        minutes = (abs_off % 3600) // 60
        hours_signed = hours if offset >= 0 else -hours
        mapping[offset] = f"{hours_signed:+03d}:{minutes:02d}"
    return offset_seconds.map(mapping)

def naive_datetime_from_unix_and_offset(utc_timestamps, timezone_offset):
    return pd.to_datetime(utc_timestamps + timezone_offset, unit='s')

# this should change in Spark, since parsing only allows naive datetimes
def _naive_to_localized_str(naive_dt, timezone_offset):
    if not pd.core.dtypes.common.is_datetime64_any_dtype(naive_dt.dtype):
        raise ValueError(
            "dtype {} is not supported, only dtype datetime64[ns] is supported.".format(naive_dt.dtype)
        )
    else:
        dt_str = naive_dt.astype(str) 
        offset_str = _offset_string_hrs(timezone_offset)
        return dt_str + offset_str

def _unix_offset_to_str(utc_timestamps, timezone_offset):
    if not pd.core.dtypes.common.is_integer_dtype(utc_timestamps.dtype):
        raise ValueError(
            "dtype {} is not supported, only dtype int is supported.".format(utc_timestamps.dtype)
        )
    else:
        dt_str = naive_datetime_from_unix_and_offset(utc_timestamps, timezone_offset).astype(str) 
        offset_str = _offset_string_hrs(timezone_offset)
        return dt_str + offset_str

def _is_series_of_timestamps(series):
    """Check if all elements in a pandas Series are of type pd.Timestamp."""
    is_timestamp_vectorized = np.frompyfunc(lambda x: isinstance(x, pd.Timestamp), 1, 1)
    return is_timestamp_vectorized(series.values).all()

def _has_mixed_timezones(series):
    """Check if there are elements in a pandas Series of pd.Timestamp objects with different tz."""
    if np.issubdtype(series.values.dtype, np.datetime64):
        return False
    series = series.astype("object")
    get_tz = np.frompyfunc(lambda x: x.tz, 1, 1)
    return len(set(get_tz(series.values))) > 1

def localize_from_offset(naive_dt, timezone_offset):
    localized_dt = naive_dt.copy()
    for offset in timezone_offset.unique():
        tz = timezone(timedelta(seconds=int(offset)))
        mask = timezone_offset == offset
        localized_dt.loc[mask] = naive_dt.loc[mask].dt.tz_localize(tz)
    return localized_dt

def zoned_datetime_from_ts_and_offset(utc_timestamps, timezone_offset):
    # get naive datetimes
    naive_dt = naive_datetime_from_unix_and_offset(utc_timestamps, timezone_offset)
    return localize_from_offset(naive_dt, timezone_offset)

def _extract_naive_and_offset(dt_str):
    offset_pattern = re.compile(r'(.*?)([+-]\d{2}:\d{2}|Z)?$')

    offset_match = dt_str.str.extract(offset_pattern)
    naive_str = offset_match[0]
    offset_part = offset_match[1]

    offset_part = offset_part.replace('Z', '+00:00')

    has_offset = offset_part.notna()
    offset_part = offset_part.fillna('+00:00')

    sign = np.where(offset_part.str[0] == '-', -1, 1)
    hours = offset_part.str[1:3].astype(int)
    minutes = offset_part.str[4:6].astype(int)
    offset_seconds = np.where(
        has_offset,
        sign * (hours * 3600 + minutes * 60),
        np.nan
    )

    return naive_str, offset_seconds

def _custom_parse_date(series, parse_dates, mixed_timezone_behavior, fixed_format, check_valid_datetime_str=True):
    """
    Convert pandas Series of datetime strings into datetime Series with efficient mixed timezone handling.

    Parameters
    ----------
    series : pd.Series
        Series of datetime strings or timestamps. All must have the same datetime format.
        
    mixed_timezone_behavior : {'utc', 'naive', 'object'}, default 'naive'
        - 'utc': Force datetime64 outputs to be UTC.
        - 'naive': Strip timezone information and return offsets separately.
        - 'object': Return as object dtype (boxed Timestamps).

    parse_dates : bool, default True
        If False, the original Series is returned unchanged, but its type is validated.

    check_valid_datetime_str : bool, default True
        If True and parse_dates=False, ensures that string values in the Series are valid datetime representations.

    fixed_format : str, optional
        Provide a fixed datetime format for improved performance.
        
    Returns
    -------
    tuple
        (datetime_series, offset_series or None)
        If behavior='naive', returns offsets as pandas Series of int seconds.
        Otherwise, offset_series is None.
    """

    col_name = series.name  # The actual column name in df

    # Validate input type
    if not (
        pd.core.dtypes.common.is_datetime64_any_dtype(series) or
        pd.core.dtypes.common.is_string_dtype(series) or
        (pd.core.dtypes.common.is_object_dtype(series) and _is_series_of_timestamps(series))
    ):
        raise TypeError(
            f"Column '{col_name}' (mapped as 'datetime' in traj_cols) must be of type datetime64, string, or an array of Timestamp objects, "
            f"but it is of type {series.dtype}."
        )

    if not parse_dates:
        # If check_valid_datetime_str=True and the dtype is string, validate the string format
        if check_valid_datetime_str and pd.core.dtypes.common.is_string_dtype(series):
            invalid_dates = pd.to_datetime(series, errors="coerce", utc=True).isna()
            if invalid_dates.any():
                raise ValueError(
                    f"Column '{col_name}' (mapped as 'datetime' in traj_cols) contains invalid datetime strings."
                    "Either fix the data or set check_valid_datetime_str=False to skip validation."
                )
        return series, None

    if mixed_timezone_behavior == "utc":
        result = pd.to_datetime(series, utc=True, format=fixed_format, errors='raise')
        return result, None

    elif mixed_timezone_behavior == "object":
        result = pd.to_datetime(series, utc=False, format=fixed_format, errors='raise')
        return result, None

    elif mixed_timezone_behavior == "naive":
        naive_str, offset_seconds = _extract_naive_and_offset(series)
        naive_times = pd.to_datetime(naive_str, errors='raise')
        offset_series = pd.Series(offset_seconds, index=series.index)
        return naive_times, offset_series

    else:
        raise ValueError("mixed_timezone_behavior must be one of 'utc', 'naive', or 'object'")


# for testing only
def _is_traj_df(df, traj_cols=None, parse_dates=True, check_valid_datetime_str=True, **kwargs):
    """For unit testing: checks that all present trajectory columns are of correct type."""
    
    if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        print("Failure: Input is not a DataFrame or GeoDataFrame.")
        return False

    traj_cols = _parse_traj_cols(df.columns, traj_cols, kwargs)

    if not _has_spatial_cols(df.columns, traj_cols):
        print("Failure: Missing required spatial columns.")
        return False

    if not _has_time_cols(df.columns, traj_cols):
        print("Failure: Missing required time columns.")
        return False

    datetime_col = traj_cols.get('datetime')
    if datetime_col and datetime_col in df.columns:

        col = df[datetime_col]
        if pd.core.dtypes.common.is_string_dtype(col):
            if not parse_dates and check_valid_datetime_str:
                invalid_dates = pd.to_datetime(df[datetime_col], errors="coerce", utc=True).isna()
                if invalid_dates.any():
                    print(f"Failure: Column '{datetime_col}' contains invalid datetime strings.")
                    return False

        elif pd.core.dtypes.common.is_object_dtype(col):
            if not _is_series_of_timestamps(df[datetime_col]):
                print(f"Failure: Column '{datetime_col}' is object type but not an array of Timestamp objects.")
                return False

        elif not pd.core.dtypes.common.is_datetime64_any_dtype(col):
            print(f"Failure: Column '{datetime_col}' is not a valid datetime type. Found dtype: {col.dtype}")
            return False

    timestamp_col = traj_cols.get('timestamp')
    if timestamp_col and timestamp_col in df.columns:
        if not pd.core.dtypes.common.is_integer_dtype(df[timestamp_col]):
            print(f"Failure: Column '{timestamp_col}' is not an integer type.")
            return False

    tz_offset_col = traj_cols.get('tz_offset')
    if tz_offset_col and tz_offset_col in df.columns:
        if not pd.core.dtypes.common.is_integer_dtype(df[tz_offset_col]):
            print(f"Failure: Column '{tz_offset_col}' is not an integer type.")
            return False

    for col_key in ['latitude', 'longitude', 'x', 'y']:
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
            if not pd.core.dtypes.common.is_float_dtype(df[col_name]):
                print(f"Failure: Column '{col_name}' (mapped as '{col_key}') is not a float type.")
                return False

    for col_key in ['user_id', 'geohash']:
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
            if not pd.core.dtypes.common.is_string_dtype(df[col_name]):
                print(f"Failure: Column '{col_name}' (mapped as '{col_key}') is not a string type.")
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
    
def _process_datetime_column(df, col, parse_dates, mixed_timezone_behavior, fixed_format, tz_offset_key, traj_cols):
    dtype = df[col].dtype

    if is_datetime64_any_dtype(df[col]):
        if df[col].dt.tz is None:
            warnings.warn(
                f"The '{col}' column is timezone-naive. "
                "Consider localizing to a timezone or using a unix timestamp to avoid inconsistencies."
            )

    elif is_string_dtype(df[col]):
        parsed, offset = _custom_parse_date(
            df[col], parse_dates=parse_dates,
            mixed_timezone_behavior=mixed_timezone_behavior,
            fixed_format=fixed_format)
        df[col] = parsed

        if parse_dates and tz_offset_key and tz_offset_key not in df:
            df[tz_offset_key] = offset

        if _is_series_of_timestamps(df[col]) and _has_mixed_timezones(df[col]):
            warnings.warn(
                f"The '{col}' column has mixed timezones. "
                "Consider localizing to a single timezone or using a unix timestamp to avoid inconsistencies."
            )
        elif is_datetime64_any_dtype(df[col]) and df[col].dt.tz is None and offset is None:
            warnings.warn(
                f"The '{col}' column is timezone-naive. "
                "Consider localizing to a timezone or using a unix timestamp to avoid inconsistencies."
            )

    elif _is_series_of_timestamps(df[col]):
        if _has_mixed_timezones(df[col]):
            warnings.warn(
                f"The '{col}' column has mixed timezones. "
                "Consider localizing to a single timezone or using a unix timestamp to avoid inconsistencies."
            )
        else:
            df[col] = pd.to_datetime(df[col], format=fixed_format)
            if df[col].dt.tz is None:
                warnings.warn(
                    f"The '{col}' column is timezone-naive. "
                    "Consider localizing to a timezone or using a unix timestamp to avoid inconsistencies."
                )
    else:
        raise ValueError(f"dtype {df[col].dtype} for '{col}' is not supported for datetime handling.")


def _cast_traj_cols(df, traj_cols, parse_dates, mixed_timezone_behavior, fixed_format=None):
    # Standardize datetime columns
    for key in ['datetime', 'start_datetime', 'end_datetime']:
        if key in traj_cols and traj_cols[key] in df:
            _process_datetime_column(
                df,
                traj_cols[key],
                parse_dates,
                mixed_timezone_behavior,
                fixed_format,
                traj_cols.get('tz_offset'),
                traj_cols
            )

    # Handle integer columns
    for key in ['tz_offset', 'duration', 'timestamp']:
        if key in traj_cols and traj_cols[key] in df:
            col = traj_cols[key]
            if not is_integer_dtype(df[col].dtype):
                df[col] = df[col].astype("Int64")

            if key == 'timestamp':
                ts_len = len(str(df[col].iloc[0]))
                if ts_len == 13:
                    warnings.warn(
                        f"The '{col}' column appears to be in milliseconds. "
                        "This may lead to inconsistencies, converting to seconds is recommended."
                    )
                elif ts_len == 19:
                    warnings.warn(
                        f"The '{col}' column appears to be in nanoseconds. "
                        "This may lead to inconsistencies, converting to seconds is recommended."
                    )

    # Handle float columns
    for key in ['latitude', 'longitude', 'x', 'y']:
        if key in traj_cols and traj_cols[key] in df:
            col = traj_cols[key]
            if not is_float_dtype(df[col].dtype):
                df[col] = df[col].astype("float")

    # Handle string columns
    for key in ['user_id', 'geohash']:
        if key in traj_cols and traj_cols[key] in df:
            col = traj_cols[key]
            if not is_string_dtype(df[col].dtype):
                df[col] = df[col].astype("str")

    return df

def from_df(df, traj_cols=None, parse_dates=True, mixed_timezone_behavior="naive", fixed_format=None, **kwargs):
    """
    Converts a DataFrame into a standardized trajectory format by validating and casting 
    specified spatial and temporal columns.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        The input DataFrame containing trajectory data.
    traj_cols : dict, optional
        Mapping of expected trajectory column names (e.g., 'latitude', 'longitude', 'datetime', 
        'user_id', etc.) to actual column names in `df`. If None, `kwargs` is used for inference.
    parse_dates : bool, default=True
        Whether to parse datetime columns as pandas datetime objects.
    mixed_timezone_behavior : {'utc', 'naive', 'object'}, default='naive'
        Controls how datetime columns with mixed time zones are handled:
        - `'utc'`: Convert all datetimes to UTC.
        - `'naive'`: Strip time zone information and store offsets separately.
        - `'object'`: Keep timestamps as `pd.Timestamp` objects with mixed time zones.
    fixed_format : str, optional
        Format string for faster parsing of datetime columns if known.
    **kwargs : dict
        Additional parameters for column inference when `traj_cols` is not provided.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with validated and correctly typed trajectory columns.

    Notes
    -----
    - Any specified `traj_cols` that do not exist in `df` will trigger a warning.
    - If `traj_cols` is not provided, missing trajectory columns are inferred from `kwargs` or filled with default schema values when possible.
    - Spatial columns are validated, and datetime columns are processed based on `parse_dates` and `mixed_timezone_behavior`.
    - If `mixed_timezone_behavior='naive'`, a separate column storing UTC offsets (in seconds) is added.
    """
    if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Expected the data argument to be either a pandas DataFrame or a GeoPandas GeoDataFrame.")

    traj_cols = _parse_traj_cols(df.columns, traj_cols, kwargs)

    _has_spatial_cols(df.columns, traj_cols)
    _has_time_cols(df.columns, traj_cols)

    return _cast_traj_cols(df, traj_cols, parse_dates=parse_dates,
                           mixed_timezone_behavior=mixed_timezone_behavior,
                           fixed_format=fixed_format)


def from_file(filepath, format="csv", traj_cols=None, parse_dates=True,
              mixed_timezone_behavior="naive", fixed_format=None, **kwargs):
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

    if isinstance(filepath, list):
        datasets = [ds.dataset(path, format=format, partitioning="hive") for path in filepath]
        dataset = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        column_names = dataset.schema.names
    elif format == 'parquet' or os.path.isdir(filepath):
        dataset = ds.dataset(filepath, format=format, partitioning="hive")
        column_names = dataset.schema.names
    else:
        column_names = pd.read_csv(filepath, nrows=0).columns

    traj_cols = _parse_traj_cols(column_names, traj_cols, kwargs)

    _has_spatial_cols(column_names, traj_cols)
    _has_time_cols(column_names, traj_cols)

    if format == "csv" and not os.path.isdir(filepath):
        read_csv_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(pd.read_csv).parameters}
        df = pd.read_csv(filepath, **read_csv_kwargs)
    else:
        df = dataset.to_table().to_pandas()

    return _cast_traj_cols(df, traj_cols, parse_dates=parse_dates,
                           mixed_timezone_behavior=mixed_timezone_behavior,
                           fixed_format=fixed_format)

        

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

    # Resolve trajectory column mapping
    traj_cols = _parse_traj_cols(_get_columns_from_source(filepath, format), traj_cols, kwargs)
    uid_col = traj_cols['user_id']

    if isinstance(filepath, list):
        datasets = [ds.dataset(path, format=format, partitioning="hive") for path in filepath]
        dataset = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
    elif format == 'parquet' or os.path.isdir(filepath):
        dataset = ds.dataset(filepath, format=format, partitioning="hive")
    else:
        df = pd.read_csv(filepath, usecols=[uid_col])
        user_ids = df[uid_col].unique()
        return user_ids.sample(frac=frac_users) if frac_users < 1.0 else user_ids

    _has_user_cols(dataset.schema.names, traj_cols)

    user_ids = pc.unique(dataset.to_table(columns=[uid_col])[uid_col]).to_pandas()
    return user_ids.sample(frac=frac_users) if frac_users < 1.0 else user_ids

def sample_from_file(filepath, users, format="csv", traj_cols=None,
                     parse_dates=True, mixed_timezone_behavior="naive",
                     fixed_format=None, **kwargs):
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

    if isinstance(filepath, list):
        datasets = [ds.dataset(path, format=format, partitioning="hive") for path in filepath]
        dataset = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        column_names = dataset.schema.names
    elif format == 'parquet' or os.path.isdir(filepath):
        dataset = ds.dataset(filepath, format=format, partitioning="hive")
        column_names = dataset.schema.names
    else:
        column_names = pd.read_csv(filepath, nrows=0).columns

    traj_cols = _parse_traj_cols(column_names, traj_cols, kwargs)
    uid_col = traj_cols['user_id']

    _has_user_cols(column_names, traj_cols)
    _has_spatial_cols(column_names, traj_cols)
    _has_time_cols(column_names, traj_cols)

    if format == "csv" and not os.path.isdir(filepath):
        df = pd.read_csv(filepath)
        df = df[df[uid_col].isin(users)]
    else:
        df = dataset.to_table(
            filter=ds.field(uid_col).isin(users),
            columns=None
        ).to_pandas()

    return _cast_traj_cols(df, traj_cols=traj_cols,
                           parse_dates=parse_dates,
                           mixed_timezone_behavior=mixed_timezone_behavior,
                           fixed_format=fixed_format)
