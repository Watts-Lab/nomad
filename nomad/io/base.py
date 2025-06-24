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
import pyarrow as pa
import pyarrow.csv as pc_csv
from nomad.constants import DEFAULT_SCHEMA
import numpy as np
import geopandas as gpd
import warnings
import inspect
from nomad.constants import FILTER_OPERATORS
import pdb

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_string_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype, 
    is_extension_array_dtype 
)
from pandas import Int64Dtype, Float64Dtype, StringDtype # For isinstance if needed later

# utils
def _update_schema(original, new_labels):
    updated_schema = dict(original)
    for label in new_labels:
        if label in DEFAULT_SCHEMA:
            updated_schema[label] = new_labels[label]
    return updated_schema

def _parse_traj_cols(columns, traj_cols, kwargs, warn=True, defaults=DEFAULT_SCHEMA):
    """
    Internal helper to finalize trajectory column names using user input and defaults.
    """
    if traj_cols:
        for k in kwargs:
            if k in traj_cols and kwargs[k] != traj_cols[k]:
                raise ValueError(
                    f"Conflicting column name for '{k}': '{traj_cols[k]}' (from traj_cols) vs '{kwargs[k]}' (from keyword arguments)."
                )
        traj_cols = _update_schema(traj_cols, kwargs)
    else:
        traj_cols = _update_schema({}, kwargs)

    if warn:
        for key, value in traj_cols.items():
            if value not in columns:
                warnings.warn(f"Trajectory column '{value}' specified for '{key}' not found in DataFrame.")

    return _update_schema(defaults, traj_cols)
    
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
    unique_offsets = offset_seconds.dropna().unique()
    for offset in unique_offsets:
        offset = int(offset) # Convert after dropna
        abs_off = abs(offset)
        hours = abs_off // 3600
        minutes = (abs_off % 3600) // 60
        hours_signed = hours if offset >= 0 else -hours
        mapping[offset] = f"{hours_signed:+03d}:{minutes:02d}"
    # Add NA mapping if necessary
    if offset_seconds.hasnans:
        mapping[pd.NA] = pd.NA 
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
        dt_str = naive_dt.astype(str).replace('NaT', pd.NA)
        offset_str = _offset_string_hrs(timezone_offset)
        return dt_str + offset_str

def _unix_offset_to_str(utc_timestamps, timezone_offset):
    # Check if integer-like (standard numpy int or nullable Int64)
    if not (is_integer_dtype(utc_timestamps.dtype) or isinstance(utc_timestamps.dtype, Int64Dtype)):
         try: 
              utc_timestamps = utc_timestamps.astype('Int64')
         except (ValueError, TypeError):
              raise ValueError(
                  f"dtype {utc_timestamps.dtype} is not supported for utc_timestamps, only integer types are supported."
              )

    dt = naive_datetime_from_unix_and_offset(utc_timestamps, timezone_offset)
    dt_str = dt.astype(str).replace('NaT', pd.NA).astype('string') 
    offset_str = _offset_string_hrs(timezone_offset).astype('string')

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

def _is_stop_df(df, traj_cols=None, parse_dates=True, check_valid_datetime_str=True, **kwargs):
    """Checks stop DataFrame structure and column types."""
    # Check DataFrame type
    if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        return False

    traj_cols = _parse_traj_cols(df.columns, traj_cols, kwargs)

    try:
        _has_spatial_cols(df.columns, traj_cols)
    except (ValueError, TypeError, Exception):
        print("Failure: Missing required spatial columns.")
        return False

    try:
        _has_time_cols(df.columns, traj_cols)
    except (ValueError, TypeError, Exception):
        print("Failure: Missing required (start) temporal columns.")
        return False

    # Check stop-specific requirement: end time OR duration
    end_col_present = _has_end_cols(df.columns, traj_cols)
    duration_col_present = _has_duration_cols(df.columns, traj_cols)

    if not (end_col_present or duration_col_present):
        print("Failure: Missing required (end or duration) temporal columns.")
        return False


    # Datetime types
    for col_key in ['datetime', 'start_datetime', 'end_datetime']:
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
            col = df[col_name]
            is_parsed_dt_col = parse_dates and is_datetime64_any_dtype(col)
            if is_parsed_dt_col: pass
            elif is_string_dtype(col) or isinstance(col.dtype, StringDtype):
                if not parse_dates and check_valid_datetime_str:
                     try: pd.to_datetime(col.dropna(), errors="raise", utc=True) #fastest check
                     except (ValueError, TypeError): return False
            elif is_object_dtype(col) and _is_series_of_timestamps(col): pass
            elif not is_datetime64_any_dtype(col):
                 if parse_dates: return False

    # Integer types
    for col_key in ['timestamp', 'start_timestamp', 'end_timestamp', 'tz_offset', 'duration']:
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
            col = df[col_name]
            # Check if standard numpy int OR pandas Int64Dtype
            if not ( is_integer_dtype(col.dtype) or isinstance(col.dtype, Int64Dtype) ):
                print(f"Failure: column {col_name} is not of integer type.")
                return False

    # Float types
    for col_key in ['latitude', 'longitude', 'x', 'y']:
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
             col = df[col_name]
             # Check if standard numpy float OR pandas Float64Dtype
             if not ( is_float_dtype(col.dtype) or isinstance(col.dtype, Float64Dtype) ):
                 print(f"Failure: column {col_name} is not of float type.")
                 return False

    # String types
    for col_key in ['user_id', 'geohash']:
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
             col_dtype = df[col_name].dtype
             # Allow standard string, nullable string, or object
             if not (is_string_dtype(col_dtype) or isinstance(col_dtype, StringDtype) or is_object_dtype(col_dtype)):
                 print(f"Failure: column {col_name} is not of string type.")
                 return False

    return True
        

# for testing only
def _is_traj_df(df, traj_cols=None, parse_dates=True, check_valid_datetime_str=True, **kwargs):
    """Checks trajectory DataFrame structure and column types."""
    if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        print("Failure: Input is not a DataFrame or GeoDataFrame.")
        return False
    traj_cols = _parse_traj_cols(df.columns, traj_cols, kwargs)

    try:
        _has_spatial_cols(df.columns, traj_cols)
    except (ValueError, TypeError, Exception):
        print("Failure: Missing required spatial columns.")
        return False

    try:
        _has_time_cols(df.columns, traj_cols)
    except (ValueError, TypeError, Exception):
        print("Failure: Missing required temporal columns.")
        return False

    for col_key in ['datetime', 'start_datetime']: # Trajectory relevant keys
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
            col = df[col_name]
            is_parsed_dt_col = parse_dates and is_datetime64_any_dtype(col)
            if is_parsed_dt_col: pass
            elif is_string_dtype(col) or isinstance(col.dtype, StringDtype):
                if not parse_dates and check_valid_datetime_str:
                     try: pd.to_datetime(col.dropna(), errors="raise", utc=True)
                     except (ValueError, TypeError):
                          print(f"Failure: Column '{col_name}' contains invalid datetime strings.")
                          return False
            elif is_object_dtype(col) and _is_series_of_timestamps(df[col_name]): pass # Use df[col_name]
            elif not is_datetime64_any_dtype(col):
                 if parse_dates:
                     print(f"Failure: Column '{col_name}' is not a valid datetime type after parsing. Found dtype: {col.dtype}")
                     return False

    # Integer types (Modified check)
    for col_key in ['timestamp', 'start_timestamp', 'tz_offset']: # Trajectory relevant keys
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
            col = df[col_name]
            # Check if standard numpy int  OR pandas Int64Dtype
            if not ( is_integer_dtype(col.dtype) or isinstance(col.dtype, Int64Dtype) ):
                 print(f"Failure: Column '{col_name}' (mapped as '{col_key}') is not an integer type. Found dtype: {col.dtype}")
                 return False

    # Float types (Modified check)
    for col_key in ['latitude', 'longitude', 'x', 'y']: # Spatial keys
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
             col = df[col_name]
             # Check if standard numpy float OR pandas Float64Dtype
             if not ( is_float_dtype(col.dtype) or isinstance(col.dtype, Float64Dtype) ):
                  print(f"Failure: Column '{col_name}' (mapped as '{col_key}') is not a float type. Found dtype: {col.dtype}")
                  return False

    # String types (Modified check)
    for col_key in ['user_id', 'geohash']: # Trajectory relevant keys
        col_name = traj_cols.get(col_key)
        if col_name and col_name in df.columns:
             col_dtype = df[col_name].dtype
             # Allow standard string, nullable string, or object
             if not (is_string_dtype(col_dtype) or isinstance(col_dtype, StringDtype) or is_object_dtype(col_dtype)):
                  print(f"Failure: Column '{col_name}' (mapped as '{col_key}') is not a string type. Found dtype: {col.dtype}")
                  return False
                 
    return True

def _has_time_cols(col_names, traj_cols):
    """Checks for at least one primary or start time column."""
    dt_col = traj_cols.get('datetime')
    ts_col = traj_cols.get('timestamp')
    start_dt_col = traj_cols.get('start_datetime')
    start_ts_col = traj_cols.get('start_timestamp')

    temporal_exists = (
        (dt_col and dt_col in col_names) or
        (ts_col and ts_col in col_names) or
        (start_dt_col and start_dt_col in col_names) or
        (start_ts_col and start_ts_col in col_names)
    )

    if not temporal_exists:
        raise ValueError(
            "Could not find required temporal columns in {}. The dataset must contain or map to "
            "at least one of 'datetime', 'timestamp', 'start_datetime', or 'start_timestamp'.".format(list(col_names))
        )

    return temporal_exists # Returns the boolean result

def _has_end_cols(col_names, traj_cols):
    """Checks if at least one end time column exists (returns bool)."""
    end_dt_col = traj_cols.get('end_datetime')
    end_ts_col = traj_cols.get('end_timestamp')
    # Corrected boolean logic
    end_dt_exists = (end_dt_col is not None and end_dt_col in col_names)
    end_ts_exists = (end_ts_col is not None and end_ts_col in col_names)

    end_exists = end_dt_exists or end_ts_exists
    return end_exists


def _has_duration_cols(col_names, traj_cols):
    """Checks if a duration column exists (returns bool)."""
    duration_col = traj_cols.get('duration')

    # Corrected boolean logic
    duration_exists = (duration_col is not None and duration_col in col_names)
    return duration_exists

def _has_spatial_cols(col_names, traj_cols, exclusive=False):
    
    if exclusive:
        single_spatial = (
            ('latitude' in traj_cols and 'longitude' in traj_cols and 
             traj_cols['latitude'] in col_names and traj_cols['longitude'] in col_names) ^
            ('x' in traj_cols and 'y' in traj_cols and 
             traj_cols['x'] in col_names and traj_cols['y'] in col_names) 
        )
        if not single_spatial:
            raise ValueError(
                f"Too many user provided spatial columns in arguments {traj_cols}, only one pair of spatial coordinates is required."
            )

    # load defaults here
    traj_cols = _update_schema(DEFAULT_SCHEMA, traj_cols)
    
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
    
def _process_datetime_column(df, col, parse_dates, mixed_timezone_behavior, fixed_format, traj_cols):
    """Processes a datetime column: parses strings, handles timezones, adds offset."""
    dtype = df[col].dtype

    if is_datetime64_any_dtype(df[col]):
        if df[col].dt.tz is None:
            warnings.warn(f"The '{col}' column is timezone-naive. Consider localizing or using unix timestamps.")

    elif is_string_dtype(df[col]) or isinstance(df[col], StringDtype):
        parsed, offset = _custom_parse_date(
            df[col], parse_dates=parse_dates,
            mixed_timezone_behavior=mixed_timezone_behavior,
            fixed_format=fixed_format,
            check_valid_datetime_str=False
        )

        df[col] = parsed
        # do not compute offset column if already exists
        has_tz = ('tz_offset' in traj_cols) and (traj_cols['tz_offset'] in df.columns)
       
        if parse_dates and mixed_timezone_behavior == 'naive' and not has_tz:
            if offset is not None and not offset.isna().all():
                df[traj_cols['tz_offset']] = offset.astype("Int64") #overwrite offset?

        if parse_dates and is_object_dtype(df[col].dtype) and _has_mixed_timezones(df[col]):
             warnings.warn(f"The '{col}' column has mixed timezones after processing.")
        elif is_datetime64_any_dtype(df[col].dtype) and df[col].dt.tz is None:
             if offset is None or offset.isna().any():
                 warnings.warn(f"The '{col}' column has timezone-naive records consider localizing or using unix timestamps.")

    elif is_object_dtype(dtype) and _is_series_of_timestamps(df[col]):
        if _has_mixed_timezones(df[col]):
            warnings.warn(f"The '{col}' column (object of Timestamps) has mixed timezones.")
        else:
            try:
                 converted = pd.to_datetime(df[col], errors='coerce', format=fixed_format)
                 if is_datetime64_any_dtype(converted.dtype):
                      df[col] = converted
                      if df[col].dt.tz is None: warnings.warn(f"The '{col}' column (object of Timestamps) is timezone-naive.")
            except Exception: pass


def _cast_traj_cols(df, traj_cols, parse_dates, mixed_timezone_behavior, fixed_format=None):
    df = df.copy() 

    # Datetime processing
    for key in ['datetime', 'start_datetime', 'end_datetime']:
        if key in traj_cols and traj_cols[key] in df:
            _process_datetime_column(
                df,
                traj_cols[key],
                parse_dates,
                mixed_timezone_behavior,
                fixed_format,
                traj_cols
            )

    # Handle integer columns
    for key in ['tz_offset', 'duration', 'timestamp']:
        if key in traj_cols and traj_cols[key] in df:
            col = traj_cols[key]
            if df[col].dtype != "Int64":
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
                df[col] = df[col].astype("float64")

    # Handle string columns
    for key in ['user_id', 'geohash']:
        if key in traj_cols and traj_cols[key] in df:
            col = traj_cols[key]
            if not is_string_dtype(df[col].dtype):
                df[col] = df[col].astype("str")
    return df

def _process_filters(filters, col_names):
    """
    Normalise *filters* into one pyarrow.dataset.Expression.

    Parameters
    ----------
    filters : None | ds.Expression | (str, str, Any) | list[ds.Expression|tuple]
        The user-supplied filter spec(s).  Multiple specs are AND-combined.
    col_names : Iterable[str]
        The set of valid column names in the dataset.

    Returns
    -------
    ds.Expression | None
        A single PyArrow filter expression, ready for Dataset.to_table(..., filter=expr).

    Raises
    ------
    TypeError  : if *filters* is not recognised.
    ValueError : if a column name is invalid or an operator unsupported.
    """
    if filters is None:
        return None

    # Always treat *filters* as an iterable of specs
    specs = filters if isinstance(filters, (list, tuple)) else [filters]

    exprs = []
    for spec in specs:
        # already a PyArrow expression → use as is
        if isinstance(spec, ds.Expression):
            exprs.append(spec)
            continue

        # shorthand (“col”, “op”, value) tuple
        if isinstance(spec, tuple) and len(spec) == 3:
            col, op, val = spec
            if col not in col_names:
                raise ValueError(f"Filter refers to unknown column {col!r}.")
            if op not in FILTER_OPERATORS:
                raise ValueError(f"Unsupported operator {op!r}.")
            exprs.append(FILTER_OPERATORS[op](ds.field(col), val))
            continue

        raise TypeError(
            "filters must be a PyArrow Expression, a (column, op, value) tuple, "
            "or a list/tuple containing those."
        )

    # Combine all expressions with logical AND
    flt = exprs[0]
    for e in exprs[1:]:
        flt = flt & e
    return flt

def table_columns(filepath, format="csv", include_schema=False, sep=","):
    """
    Return column names or the full schema of a data source.

    The 'sep' argument specifies the delimiter and is only used for 'csv' format;
    it is ignored when reading 'parquet' files.
    """
    assert format in {"csv", "parquet"}, "format must be 'csv' or 'parquet'"

    use_pyarrow_dataset = (
        format == "parquet" or
        isinstance(filepath, (list, tuple)) or
        os.path.isdir(filepath)
    )
    if use_pyarrow_dataset:
        file_format_obj = "parquet"
        if format == "csv":
            parse_options = pc_csv.ParseOptions(delimiter=sep)
            file_format_obj = ds.CsvFileFormat(parse_options=parse_options)
        
        if isinstance(filepath, list):
            if not filepath:
                raise ValueError("Input filepath list cannot be empty.")
            datasets = [ds.dataset(p, format=file_format_obj, partitioning="hive") for p in filepath]
            schema = ds.UnionDataset(schema=datasets[0].schema, children=datasets).schema
        else:
            schema = ds.dataset(filepath, format=file_format_obj, partitioning="hive").schema
        
        return schema if include_schema else pd.Index(schema.names)
    
    else:
        header = pd.read_csv(filepath, nrows=0, sep=sep)
        return header.dtypes if include_schema else header.columns

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
              mixed_timezone_behavior="naive", fixed_format=None, sep=",", **kwargs):
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

    column_names = table_columns(filepath, format=format, sep=sep)
    traj_cols = _parse_traj_cols(column_names, traj_cols, kwargs)

    _has_spatial_cols(column_names, traj_cols)
    _has_time_cols(column_names, traj_cols)

    use_pyarrow_dataset = (
        format == "parquet" or
        isinstance(filepath, (list, tuple)) or
        os.path.isdir(filepath)
    )

    if use_pyarrow_dataset:
        file_format_obj = "parquet"
        if format == "csv":
            parse_options = pc_csv.ParseOptions(delimiter=sep)
            file_format_obj = ds.CsvFileFormat(parse_options=parse_options)
        
        if isinstance(filepath, list):
            if not filepath:
                raise ValueError("Input filepath list cannot be empty.")
            datasets = [ds.dataset(p, format=file_format_obj, partitioning="hive") for p in filepath]
            dataset_obj = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        else:
            dataset_obj = ds.dataset(filepath, format=file_format_obj, partitioning="hive")

        arrow_flt = _process_filters(filters, column_names)
        df = (
            dataset_obj
            .to_table(
                filter=arrow_flt,
                columns=list(column_names))
            .to_pandas()
        )
    else:
        read_csv_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(pd.read_csv).parameters
        }
        df = pd.read_csv(filepath, sep=sep, **read_csv_kwargs)

    return _cast_traj_cols(
        df,
        traj_cols=traj_cols,
        parse_dates=parse_dates,
        mixed_timezone_behavior=mixed_timezone_behavior,
        fixed_format=fixed_format,
    )

def sample_users(filepath, format="csv", size=1.0, traj_cols=None, seed=None, sep=",", **kwargs):

    assert format in {"csv", "parquet"}

    column_names = table_columns(filepath, format=format, sep=sep)
    traj_cols = _parse_traj_cols(column_names, traj_cols, kwargs)
    _has_user_cols(column_names, traj_cols)
    uid_col = traj_cols["user_id"]

    use_pyarrow_dataset = (
        format == "parquet" or
        isinstance(filepath, (list, tuple)) or
        os.path.isdir(filepath)
    )

    if use_pyarrow_dataset:
        file_format_obj = "parquet"
        if format == "csv":
            parse_options = pc_csv.ParseOptions(delimiter=sep)
            file_format_obj = ds.CsvFileFormat(parse_options=parse_options)
        
        if isinstance(filepath, list):
            if not filepath:
                raise ValueError("Input filepath list cannot be empty.")
            datasets = [ds.dataset(p, format=file_format_obj, partitioning="hive") for p in filepath]
            dataset_obj = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        else:
            dataset_obj = ds.dataset(filepath, format=file_format_obj, partitioning="hive")

        user_ids = pc.unique(dataset_obj.to_table(columns=[uid_col])[uid_col]).to_pandas()
    else:
        read_csv_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(pd.read_csv).parameters
        }
        user_ids = (
            pd.read_csv(filepath, usecols=[uid_col], sep=sep, **read_csv_kwargs)[uid_col]
            .drop_duplicates()
        )

    if isinstance(size, int):
        return user_ids.sample(n=min(size, len(user_ids)), random_state=seed, replace=False)
    if 0.0 < size <= 1.0:
        return user_ids.sample(frac=size, random_state=seed, replace=False)
    raise ValueError("size must be an int ≥ 1 or a float in (0, 1].")


def sample_from_file(
    filepath,
    users=None,
    format="csv",
    traj_cols=None,
    frac_users=1.0,
    frac_records=1.0,
    seed=None,
    parse_dates=True,
    mixed_timezone_behavior="naive",
    fixed_format=None,
    sep=",",
    **kwargs
):
    """
    Read and sample trajectory data from a file.

    Parameters
    ----------
    filepath : str or Path
        Path to the input file.
    users : list of hashable or None, default None
        If provided, only include these user IDs; if None, include all users.
    format : str, default "csv"
        Data format, e.g. "csv" or "parquet".
    traj_cols : dict or None, default None
        Mapping of trajectory column names (e.g. {"uid": "user_id"}), or None to use defaults.
    frac_users : float, default 1.0
        Fraction of users to sample (0 < frac_users ≤ 1).
    frac_records : float, default 1.0
        Fraction of each user’s records to sample (0 < frac_records ≤ 1).
    seed : int or None, default None
        Random seed for reproducibility.
    parse_dates : bool, default True
        Whether to parse date/time columns as datetime objects.
    mixed_timezone_behavior : str, default "naive"
        How to handle mixed‐timezone timestamps; options might include "naive", "utc", etc.
    fixed_format : str or None, default None
        If specified, enforce this input format (overrides autodetection).
    **kwargs
        Passed through to the underlying reader (e.g. `pandas.read_csv`).

    Returns
    -------
    pandas.DataFrame
        Sampled trajectory data.
    """
    assert format in {"csv", "parquet"}

    column_names = table_columns(filepath, format=format, sep=sep)
    traj_cols_ = _parse_traj_cols(column_names, traj_cols, kwargs)

    _has_spatial_cols(column_names, traj_cols_)
    _has_time_cols(column_names, traj_cols_)

    if users is None and frac_users < 1.0:
        users = sample_users(
            filepath,
            format=format,
            size=frac_users,
            traj_cols=traj_cols,
            seed=seed,
            sep=sep,
            **kwargs,
        )

    use_pyarrow_dataset = (
        format == "parquet" or
        isinstance(filepath, (list, tuple)) or
        os.path.isdir(filepath)
    )

    if use_pyarrow_dataset:
        file_format_obj = "parquet"
        if format == "csv":
            parse_options = pc_csv.ParseOptions(delimiter=sep)
            file_format_obj = ds.CsvFileFormat(parse_options=parse_options)
        
        if isinstance(filepath, list):
            if not filepath:
                raise ValueError("Input filepath list cannot be empty.")
            datasets = [ds.dataset(p, format=file_format_obj, partitioning="hive") for p in filepath]
            dataset_obj = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        else:
            dataset_obj = ds.dataset(filepath, format=file_format_obj, partitioning="hive")

        if users is None:
            df = dataset_obj.to_table(columns=list(column_names)).to_pandas()
        else:
            df = dataset_obj.to_table(
                filter=ds.field(traj_cols_['user_id']).isin(users),
                columns=list(column_names)
            ).to_pandas()
    else:
        df = pd.read_csv(filepath, sep=sep)
        if users is not None:
            df = df[df[traj_cols_['user_id']].isin(users)]

    if 0.0 < frac_records < 1.0:
        df = df.sample(frac=frac_records, random_state=seed)

    return _cast_traj_cols(
        df,
        traj_cols=traj_cols_,
        parse_dates=parse_dates,
        mixed_timezone_behavior=mixed_timezone_behavior,
        fixed_format=fixed_format,
    )

def to_file(df, path, format="csv",
            traj_cols=None, output_traj_cols=None,
            partition_by=None, filesystem=None,
            use_offset=False,
            **kwargs):
    df = df.copy()
    assert format in {"csv", "parquet"}

    traj_cols = _parse_traj_cols(df.columns, traj_cols, kwargs)
    _has_spatial_cols(df.columns, traj_cols)
    _has_time_cols(df.columns, traj_cols)

    if output_traj_cols == "default":
        output_traj_cols = DEFAULT_SCHEMA
    if output_traj_cols is None:
        output_traj_cols = traj_cols

    if use_offset and traj_cols["tz_offset"] not in df.columns:
        raise ValueError(f"use_offset=True but tz_offset column '{traj_cols['tz_offset']}' not found in df")

    for k in ["datetime", "start_datetime", "end_datetime"]:
        if k in traj_cols and traj_cols[k] in df.columns:
            col = df[traj_cols[k]]
            if is_string_dtype(col) or isinstance(col, StringDtype):
                continue
            if is_datetime64_any_dtype(col):
                if use_offset and col.dt.tz is None:
                    df[traj_cols[k]] = _naive_to_localized_str(col, df[traj_cols["tz_offset"]])
                else:
                    df[traj_cols[k]] = col.astype(str)
            elif is_object_dtype(col) and _is_series_of_timestamps(col):
                if use_offset and all(ts.tz is None for ts in col.dropna()):
                    df[traj_cols[k]] = _naive_to_localized_str(col, df[traj_cols["tz_offset"]])
                else:
                    df[traj_cols[k]] = col.astype(str)

    df = df.rename(columns={traj_cols[k]: output_traj_cols[k]
                            for k in traj_cols
                            if k in output_traj_cols and traj_cols[k] in df.columns})

    other_kwargs = {k: v for k, v in kwargs.items() if k not in traj_cols}
    if format == "csv":
        df.to_csv(path, index=False, **other_kwargs)
    else:
        table = pa.Table.from_pandas(df, preserve_index=False)
        ds.write_dataset(table, base_dir=str(path),
                         format="parquet",
                         partitioning=partition_by,
                         partitioning_flavor='hive',
                         filesystem=filesystem,
                         **other_kwargs)
