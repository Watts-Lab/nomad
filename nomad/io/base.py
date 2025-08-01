import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
import pyproj
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
import pyarrow.fs as pafs
import pyarrow.types as pat
import pathlib
import pyarrow.csv as pc_csv
from nomad.constants import DEFAULT_SCHEMA, FILTER_OPERATORS
import numpy as np
import warnings
import inspect
import pdb

from shapely import wkt
import shapely.geometry as sh_geom

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
def _fallback_spatial_cols(col_names, traj_cols, kwargs):
    '''
    Helper function to decide whether to use latitude and longitude or x,y
    for processing algorithms
    '''
    traj_cols = _parse_traj_cols(col_names, traj_cols, kwargs, defaults={}, warn=False)
    
    # check for sufficient spatial coords
    _has_spatial_cols(col_names, traj_cols, exclusive=True)

    use_lon_lat = ('latitude' in traj_cols and 'longitude' in traj_cols)
    if use_lon_lat:
        coord_key1, coord_key2 = 'longitude', 'latitude'
    else:
        coord_key1, coord_key2 = 'x', 'y'
            
    return coord_key1, coord_key2, use_lon_lat

def _fallback_time_cols_dt(col_names, traj_cols, kwargs):
    '''
    Helper to decide whether to use datetime vs timestamp in cases of ambiguity
    '''
    traj_cols = _parse_traj_cols(col_names, traj_cols, kwargs, defaults={}, warn=False)
    # check for explicit datetime usage
    t_keys = ['datetime', 'start_datetime', 'timestamp', 'start_timestamp']
    
    if 'timestamp' in kwargs or 'start_timestamp' in kwargs: # prioritize timestamp 
        t_keys = t_keys[-2:] + t_keys[:2]
    
    if 'datetime' in kwargs or 'start_datetime' in kwargs: # prioritize datetime 
        t_keys = t_keys[-2:] + t_keys[:2]

    # load defaults and check for time columns
    traj_cols = _update_schema(DEFAULT_SCHEMA, traj_cols)
    _has_time_cols(col_names, traj_cols) # error if no columns
    
    for t_key in t_keys:
        if traj_cols[t_key] in col_names:
            use_datetime = (t_key in ['datetime', 'start_datetime']) ## necessary?
            break
            
    return t_key, use_datetime

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
    """Return True if lon/lat, x/y or geohash columns are present; else raise."""
    if exclusive:
        has_lon_lat = (
            'latitude' in traj_cols and 'longitude' in traj_cols and
            traj_cols['latitude'] in col_names and traj_cols['longitude'] in col_names
        )
        has_x_y = (
            'x' in traj_cols and 'y' in traj_cols and
            traj_cols['x'] in col_names and traj_cols['y'] in col_names
        )

        if has_lon_lat and has_x_y:
            raise ValueError("Too many spatial columns; provide only one pair.")
        if not has_lon_lat and not has_x_y:
            raise ValueError(
                f"No spatial columns provided for spatial ops; please provide "
                "explicit column names to be used for either ('latitude','longitude') or ('x','y')."
            )
        return True

    else:
        traj_cols = _update_schema(DEFAULT_SCHEMA, traj_cols)

        has_lon_lat = (
            'latitude' in traj_cols and 'longitude' in traj_cols and
            traj_cols['latitude'] in col_names and traj_cols['longitude'] in col_names
        )
        has_x_y = (
            'x' in traj_cols and 'y' in traj_cols and
            traj_cols['x'] in col_names and traj_cols['y'] in col_names
        )
        has_geohash = (
            'geohash' in traj_cols and traj_cols['geohash'] in col_names
        )

        if not (has_lon_lat or has_x_y or has_geohash):
            raise ValueError(
                f"I didn't find spatial columns in {col_names}; please provide "
                "column names for ('latitude','longitude') or ('x','y') or 'geohash'."
            )
        return True

def _has_user_cols(col_names, traj_cols):
    
    user_exists = 'user_id' in traj_cols and traj_cols['user_id'] in col_names

    if not user_exists:
        raise ValueError(
            "Could not find required user identifier column in {}. The dataset must contain or map to 'user_id'.".format(col_names)
        )
    
    return user_exists

# SPARK check is traj_dataframe
    
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

    for key in ['date', 'utc_date']:
        if key in traj_cols and traj_cols[key] in df:
            if parse_dates:
                df[traj_cols[key]] = pd.to_datetime(df[traj_cols[key]]).dt.date
                
    # Handle integer columns
    for key in ['tz_offset', 'duration', 'timestamp', 'start_timestamp', 'end_timestamp']:
        if key in traj_cols and traj_cols[key] in df:
            col = traj_cols[key]
            if df[col].dtype != "Int64":
                df[col] = df[col].astype("Int64")

            if key == 'timestamp':
                if len(df)>0:
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

def _process_filters(filters, col_names, use_pyarrow_dataset, traj_cols=None, schema=None):
    """
    Build one pyarrow.Expression from filters, resolving aliases and
    coercing literals for timestamp vs string columns.

    Parameters
    ----------
    filters   : None | ds.Expression | (col, op, val) | list of same
    col_names : iterable of actual column names
    use_pyarrow_dataset : bool, whether to use pyarrow expressions
        or a pandas series as a mask
    traj_cols : dict, optional, maps logical names → actual names
    schema    : pyarrow.Schema, optional, for type lookups

    Returns
    -------
    ds.Expression or callable function that generates a mask
    """
    if filters is None:
        return None
        
    traj_cols = traj_cols or {}
    specs = [filters] if isinstance(filters, (ds.Expression, tuple)) else list(filters)

    if use_pyarrow_dataset:
        exprs = []
        for spec in specs:
            if isinstance(spec, ds.Expression):
                exprs.append(spec)
            elif (isinstance(spec, tuple) and len(spec) == 3):
                col, op, val = spec
                if col in col_names:
                    pass
                elif col in traj_cols and traj_cols[col] in col_names:
                    col = traj_cols[col]
                else:
                    raise KeyError(f"Filter column {col!r} not found in {col_names}")

                if op not in FILTER_OPERATORS:
                    raise ValueError(f"Unsupported operator {op!r}")

                if schema is not None:
                    pa_type = schema.field(col).type
                    if pat.is_timestamp(pa_type) and isinstance(val, str):
                        warnings.warn(f"Coercing filter value {val!r} to pandas.Timestamp for column {col!r}")
                        val = pd.Timestamp(val)
                    elif pat.is_string(pa_type) and isinstance(val, (pd.Timestamp, np.datetime64)):
                        val = pd.Timestamp(val).isoformat()
                        warnings.warn(f"Coercing filter datetime {val!r} to ISO string {val} for column {col!r}")
                        
                exprs.append(FILTER_OPERATORS[op](ds.field(col), val))

            else:
                raise TypeError(
                    "filters must be a ds.Expression or a (column, op, value) tuple, "
                    "or a list of those."
                )

        out = exprs[0]
        for e in exprs[1:]:
            out &= e
        return out

    else:
        def mask_func(df):
            mask = pd.Series(True, index=df.index)
            for col, op, val in specs:
                col = col if col in col_names else traj_cols.get(col, col)
                if col not in df.columns:
                    raise ValueError(f"Unknown filter column {col!r}")
                if op not in FILTER_OPERATORS:
                    raise ValueError(f"Unsupported operator {op!r}")

                if schema is not None:
                    dtype = schema[col]
                    if dtype.name.startswith("datetime") and isinstance(val, str):
                        warnings.warn(f"Coercing {val!r} → pandas.Timestamp for {col!r}")
                        val = pd.Timestamp(val)
                    elif dtype.name.startswith("object") and isinstance(val, (pd.Timestamp, np.datetime64)):
                        new_val = pd.Timestamp(val).isoformat()
                        warnings.warn(f"Coercing {val!r} → {new_val!r} for {col!r}")
                        val = new_val

                mask &= FILTER_OPERATORS[op](df[col], val)
            return mask

        return mask_func


def _is_directory(path):
    """
    True if *path* points to a directory, locally or on a remote/URI FS.

    Accepts str, pathlib.Path, or os.PathLike.
    """
    # pathlib.Path → str (Arrow wants str)
    if isinstance(path, (pathlib.Path, os.PathLike)):
        return pathlib.Path(path).is_dir()

    # Try Arrow’s universal resolver
    try:
        fs, rel = pafs.FileSystem.from_uri(path)
        # ``rel`` is "" for the bucket root ("s3://bucket") → treat as dir
        rel = rel.rstrip("/") or rel
        info = fs.get_file_info(rel)
        return info.type == pafs.FileType.Directory
    except (pa.ArrowInvalid, ValueError):
        # Arrow could not parse → assume local path string
        return os.path.isdir(path)
    
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
        _is_directory(filepath)
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

def from_df(df, traj_cols=None, parse_dates=True, mixed_timezone_behavior="naive", fixed_format=None, filters=None, **kwargs):
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
    
    return _cast_traj_cols(df.copy(), traj_cols, parse_dates=parse_dates,
                           mixed_timezone_behavior=mixed_timezone_behavior,
                           fixed_format=fixed_format)
    


def from_file(filepath,
              format="csv",
              parse_dates=True,
              mixed_timezone_behavior="naive",
              fixed_format=None,
              sep=",",
              filters=None,
              sort_times=False,
              traj_cols=None,
              **kwargs):
    """
    Load and cast trajectory data from a specified file path or list of paths.

    Parameters
    ----------
    filepath : str or list of str
        Path or list of paths to the file(s) or directories containing the data.
    format : str, optional
        The format of the data files, either 'csv' or 'parquet'.
    traj_cols : dict, optional
        Mapping from trajectory fields (e.g. 'latitude', 'timestamp') to
        column names in the input.
    parse_dates : bool, default True
        Whether to parse timestamp columns as datetime.
    mixed_timezone_behavior : {'utc', 'naive', 'object'}, default='naive'
        Controls how datetime columns with mixed time zones are handled:
        - `'utc'`: Convert all datetimes to UTC.
        - `'naive'`: Strip time zone information and store offsets separately.
        - `'object'`: Keep timestamps as `pd.Timestamp` objects with mixed time zones.
    fixed_format : str, optional
        strftime format string for datetime parsing.
    sep : str, default ','
        Field delimiter for CSV input.
    filters : pyarrow.dataset.Expression or tuple or list of tuples, optional
        Read‐time filter for Parquet. Accepts a PyArrow Expression,
        a (column, operator, value) tuple, or a list of such tuples
        (AND‐chained).
    **kwargs : dict
        Additional parameters for column inference when `traj_cols` is not provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with trajectory columns cast and dates parsed.
    """
    assert format in ["csv", "parquet"]

    column_names = table_columns(filepath, format=format, sep=sep)
    col_schema = None
    if filters is not None:
        col_schema = table_columns(filepath, format=format,
                                   include_schema=True, sep=sep)
        
    traj_cols = _parse_traj_cols(column_names, traj_cols, kwargs)

    _has_spatial_cols(column_names, traj_cols)
    _has_time_cols(column_names, traj_cols)

    use_pyarrow_dataset = (
        format == "parquet" or
        isinstance(filepath, (list, tuple)) or
        _is_directory(filepath)
    )

    if use_pyarrow_dataset:
        file_format_obj = "parquet"
        if format == "csv":
            parse_options = pc_csv.ParseOptions(delimiter=sep)
            conv_options = pc_csv.ConvertOptions(
                column_types={traj_cols['datetime']: pa.string()} if traj_cols['datetime'] in column_names else None
            )
            file_format_obj = ds.CsvFileFormat(parse_options=parse_options, convert_options=conv_options)
        
        if isinstance(filepath, list):
            if not filepath:
                raise ValueError("Input filepath list cannot be empty.")
            datasets = [ds.dataset(p, format=file_format_obj, partitioning="hive") for p in filepath]
            dataset_obj = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        else:
            dataset_obj = ds.dataset(filepath, format=file_format_obj, partitioning="hive")

        arrow_flt = _process_filters(filters,
                             col_names=column_names,
                             traj_cols=traj_cols,
                             schema=col_schema,
                             use_pyarrow_dataset=use_pyarrow_dataset)
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
        read_csv_kwargs['parse_dates'] = False
        df = pd.read_csv(filepath, sep=sep, **read_csv_kwargs)
                # build a boolean mask from tuple filters
        if filters is not None:
            mask_func = _process_filters(
                filters,
                col_names=df.columns,
                traj_cols=traj_cols,
                schema=schema,
                use_pyarrow_dataset=False
            )
            df = df[mask_func(df)]

    df = _cast_traj_cols(
        df,
        traj_cols=traj_cols,
        parse_dates=parse_dates,
        mixed_timezone_behavior=mixed_timezone_behavior,
        fixed_format=fixed_format,
    )

    #sorting
    t_keys = ['timestamp', 'start_timestamp', 'datetime', 'start_datetime']
    ts_col = next((traj_cols[k] for k in t_keys if traj_cols[k] in df.columns), None)
    uid_col = traj_cols['user_id']
    
    if uid_col in df.columns:
        # Multi-user case: always safe to sort.
        return df.sort_values(by=[uid_col, ts_col], ignore_index=True)

    if sort_times:
        warnings.warn(
            f"Sorting by timestamp only, as user ID column '{uid_col}' was not found. If this is a multi-user "
            f"dataset, map the correct user ID column to avoid mixing trajectories.",
            UserWarning
        )
        return df.sort_values(by=[ts_col], ignore_index=True)
    
    return df

    
def sample_users(
    filepath,
    format="csv",
    size=1.0,
    seed=None,
    sep=",",
    filters=None,
    within=None,
    poly_crs=None,
    data_crs=None,
    traj_cols=None,
    **kwargs
):
    """
    Sample users from a dataset, with optional read-time filtering for Parquet.

    Parameters
    ----------
    filepath : str or Path
        Path to the data file or directory.
    format : {'csv','parquet'}, default 'csv'
        Input format.
    size : float or int, default 1.0
        Fraction (0–1] or absolute number of users to sample.
    seed : int, optional
        Random seed for reproducibility.
    sep : str, default ','
        CSV delimiter.
    filters : pyarrow.dataset.Expression or tuple or list of tuples, optional
        Read-time filter(s) for Parquet inputs. Ignored for single CSV files.
    within : shapely Polygon/MultiPolygon or WKT str, default None
        If supplied, keep only points whose coordinates fall inside this polygon.
    data_crs : str or pyproj.CRS, optional
        CRS for `data` when it is a plain DataFrame; ignored if `data` is a GeoDataFrame.
    traj_cols : dict, optional
        Mapping of logical names ('user_id', etc.) to actual column names.
    **kwargs
        Passed through to the underlying reader.

    Returns
    -------
    pd.Series
        Sampled user IDs.
    """
    assert format in {"csv", "parquet"}

    column_names = table_columns(filepath, format=format, sep=sep)
    schema = None
    if filters is not None:
        schema = table_columns(filepath, format=format,
                               include_schema=True, sep=sep)

    if within is not None:
        coord_key1, coord_key2, use_lat_lon = _fallback_spatial_cols(column_names, traj_cols, kwargs)
        
        # normalise *poly* to a shapely geometry
        if isinstance(within, str):
            poly = wkt.loads(within)
        elif isinstance(within, sh_geom.base.BaseGeometry):
            poly = within
        elif isinstance(within, gpd.GeoSeries):
            poly = within.unary_union
        elif isinstance(within, gpd.GeoDataFrame):
            poly = within.geometry.unary_union
        else:
            raise TypeError("within must be WKT, shapely, GeoSeries, or GeoDataFrame")

        if data_crs is None:
            if use_lat_lon:
                data_crs = "EPSG:4326"
                warnings.warn("data_crs not provided; assuming EPSG:4326 for "
                              "longitude/latitude coordinates.")
            else:
                raise ValueError(
                    "data_crs must be supplied when using projected x/y columns, "
                    "or provide latitude/longitude columns instead."
                )
                
        data_crs = pyproj.CRS(data_crs)

        # CRS check / reprojection
        poly_crs_final = getattr(within, "crs", None) or poly_crs
        if poly_crs_final is None:
            warnings.warn("Polygon CRS unspecified; assuming it matches data_crs.")
        else:
            src_crs = pyproj.CRS(poly_crs_final)
            if not src_crs.equals(data_crs):
                poly = gpd.GeoSeries([poly], crs=src_crs).to_crs(data_crs).iloc[0]
        
        minx, miny, maxx, maxy = poly.bounds
        bbox_specs = [
            (coord_key1, ">=", minx), (coord_key1, "<=", maxx),
            (coord_key2, ">=", miny), (coord_key2, "<=", maxy),
        ]
        if filters is None:
            filters = bbox_specs
        elif isinstance(filters, tuple):
            filters = [filters] + bbox_specs
        elif isinstance(filters, list):
            filters = filters + bbox_specs
        else:  # raw ds.Expression – box filter can’t be merged, keep filters unchanged
            pass
    
    # Resolve trajectory column names
    traj_cols = _parse_traj_cols(column_names, traj_cols, kwargs)
    _has_user_cols(column_names, traj_cols)
    uid_col = traj_cols["user_id"]

    use_pyarrow_dataset = (
        format == "parquet" or
        isinstance(filepath, (list, tuple)) or
        _is_directory(filepath)
    )

    if use_pyarrow_dataset:
        file_format_obj = "parquet"
        if format == "csv":
            parse_options = pc_csv.ParseOptions(delimiter=sep)
            conv_options = pc_csv.ConvertOptions(
                column_types={traj_cols['datetime']: pa.string()} if traj_cols['datetime'] in column_names else None
            )
            file_format_obj = ds.CsvFileFormat(parse_options=parse_options, convert_options=conv_options)
        
        if isinstance(filepath, list):
            if not filepath:
                raise ValueError("Input filepath list cannot be empty.")
            datasets = [ds.dataset(p, format=file_format_obj, partitioning="hive") for p in filepath]
            dataset_obj = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        else:
            dataset_obj = ds.dataset(filepath, format=file_format_obj, partitioning="hive")

        # Apply filters at scan time
        arrow_flt = _process_filters(filters,
                                     col_names=column_names,
                                     traj_cols=traj_cols,
                                     schema=schema,
                                     use_pyarrow_dataset=use_pyarrow_dataset) # What happens with timezones??
        if within is not None:
            table = dataset_obj.to_table(columns=[uid_col, coord_key1, coord_key2], filter=arrow_flt)
            df = table.to_pandas()
            pts = gpd.GeoSeries(gpd.points_from_xy(df[coord_key1], df[coord_key2]),
                                 crs=data_crs)
            user_ids = df.loc[pts.within(poly), uid_col].drop_duplicates()
        else:
            table = dataset_obj.to_table(columns=[uid_col], filter=arrow_flt)
            user_ids = pc.unique(table[uid_col]).to_pandas()

    else:
        read_csv_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(pd.read_csv).parameters
        }
        if filters is None:
            df = pd.read_csv(filepath, usecols=[uid_col], sep=sep, **read_csv_kwargs)
        else:                                    # need all columns for filtering
            df = pd.read_csv(filepath, sep=sep, **read_csv_kwargs)
    
            # build a boolean mask from tuple filters
            mask_func = _process_filters(
                filters,
                col_names=df.columns,
                traj_cols=traj_cols,
                schema=schema,
                use_pyarrow_dataset=False
            )
            df = df[mask_func(df)]
    
        if within is not None:
            pts = gpd.GeoSeries(gpd.points_from_xy(df[coord_key1], df[coord_key2]),
                                     crs=data_crs)
            user_ids = df.loc[pts.within(poly), uid_col].drop_duplicates()
        else:
            user_ids = df[uid_col].drop_duplicates()
    # Sample as int count or fraction
    if isinstance(size, int):
        return user_ids.sample(n=min(size, len(user_ids)), random_state=seed, replace=False)
    if 0.0 < size <= 1.0:
        return user_ids.sample(frac=size, random_state=seed, replace=False)
    raise ValueError("size must be an int ≥ 1 or a float in (0, 1].")

def sample_from_file(
    filepath,
    users=None,
    format="csv",
    frac_users=1.0,
    frac_records=1.0,
    seed=None,
    parse_dates=True,
    mixed_timezone_behavior="naive",
    fixed_format=None,
    sep=",",
    filters=None,
    within=None,
    poly_crs=None,
    data_crs=None,
    sort_times=True,
    traj_cols=None,
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
    sep : str
        Separator character for reader. Defaults to ",".
    filters : pyarrow.dataset.Expression or tuple or list of tuples, optional
        Read-time filter(s) for Parquet inputs. Ignored for single CSV files.
    **kwargs
        Passed through to the underlying reader (e.g. `pandas.read_csv`).

    Returns
    -------
    pandas.DataFrame
        Sampled trajectory data.
    """
    assert format in {"csv", "parquet"}

    column_names = table_columns(filepath, format=format, sep=sep)
    schema = None
    if filters is not None:
        schema = table_columns(filepath, format=format, include_schema=True, sep=sep)

    poly = None
    coord_key1 = coord_key2 = None
    if within is not None:
        # decide which coordinate pair to use
        coord_key1, coord_key2, use_lat_lon = _fallback_spatial_cols(
            column_names, traj_cols, kwargs
        )

        # normalise the polygon
        if isinstance(within, str):
            poly = wkt.loads(within)
        elif isinstance(within, sh_geom.base.BaseGeometry):
            poly = within
        elif isinstance(within, gpd.GeoSeries):
            poly = within.unary_union
        elif isinstance(within, gpd.GeoDataFrame):
            poly = within.geometry.unary_union
        else:
            raise TypeError(
                "within must be WKT, shapely geometry, GeoSeries or GeoDataFrame."
            )

        # CRS handling
        if data_crs is None:
            if use_lat_lon:
                data_crs = "EPSG:4326"
                warnings.warn(
                    "data_crs not provided; assuming EPSG:4326 for longitude/latitude."
                )
            else:
                raise ValueError(
                    "data_crs must be supplied when using projected x/y columns, "
                    "or provide latitude/longitude columns instead."
                )

        data_crs = pyproj.CRS(data_crs)
        src_crs = getattr(within, "crs", None) or poly_crs
        if src_crs is not None and not pyproj.CRS(src_crs).equals(data_crs):
            poly = gpd.GeoSeries([poly], crs=src_crs).to_crs(data_crs).iloc[0]

        minx, miny, maxx, maxy = poly.bounds
        bbox_specs = [
            (coord_key1, ">=", minx),
            (coord_key1, "<=", maxx),
            (coord_key2, ">=", miny),
            (coord_key2, "<=", maxy),
        ]
        if filters is None:
            filters = bbox_specs
        elif isinstance(filters, tuple):
            filters = [filters] + bbox_specs
        elif isinstance(filters, list):
            filters = filters + bbox_specs
    
    traj_cols = _parse_traj_cols(column_names, traj_cols, kwargs)

    _has_spatial_cols(column_names, traj_cols)
    _has_time_cols(column_names, traj_cols)

    use_pyarrow_dataset = (
        format == "parquet" or
        isinstance(filepath, (list, tuple)) or
        _is_directory(filepath)
    )

    if use_pyarrow_dataset:
        file_format_obj = "parquet"
        if format == "csv":
            parse_options = pc_csv.ParseOptions(delimiter=sep)
            conv_options = pc_csv.ConvertOptions(
                column_types={traj_cols['datetime']: pa.string()} if traj_cols['datetime'] in column_names else None
            )
            file_format_obj = ds.CsvFileFormat(parse_options=parse_options, convert_options=conv_options)
        
        if isinstance(filepath, list):
            if not filepath:
                raise ValueError("Input filepath list cannot be empty.")
            datasets = [ds.dataset(p, format=file_format_obj, partitioning="hive") for p in filepath]
            dataset_obj = ds.UnionDataset(schema=datasets[0].schema, children=datasets)
        else:
            dataset_obj = ds.dataset(filepath, format=file_format_obj, partitioning="hive")
            
        arrow_flt = _process_filters(
            filters,
            col_names=column_names,
            traj_cols=traj_cols,
            schema=schema,
            use_pyarrow_dataset=True
        )

        if users is not None:
            arrow_ids = pa.array(users)
            user_expr = ds.field(traj_cols['user_id']).isin(arrow_ids)
            arrow_flt = user_expr if arrow_flt is None else (arrow_flt & user_expr)
        
        df = dataset_obj.to_table(filter=arrow_flt,
                                  columns=list(column_names)).to_pandas()

    else:
        df = pd.read_csv(filepath, sep=sep)
        if filters is not None:
            mask = _process_filters(
                filters,
                col_names=df.columns,
                traj_cols=traj_cols,
                schema=schema,
                use_pyarrow_dataset=False
            )(df)
            df = df[mask]

        if users is not None:
            df = df[df[traj_cols['user_id']].isin(users)]

    if poly is not None and not df.empty:
        pts = gpd.GeoSeries(
            gpd.points_from_xy(df[coord_key1], df[coord_key2]), crs=data_crs
        )
        df = df[pts.within(poly)]
    
    if (users is None) and frac_users:
        # build the user‐ID index
        user_ids = df[traj_cols['user_id']].drop_duplicates()
        # integer count
        if isinstance(frac_users, int):
            n = min(frac_users, len(user_ids))
            chosen = user_ids.sample(n=n, random_state=seed, replace=False)
        # fractional
        elif 0.0 < frac_users <= 1.0:
            chosen = user_ids.sample(frac=frac_users, random_state=seed, replace=False)
        else:
            raise ValueError("frac_users must be an int ≥ 1 or a float in (0, 1].")
        # restrict df to those users
        df = df[df[traj_cols['user_id']].isin(chosen)]

    if (frac_records) and (0.0 < frac_records < 1.0):
        df = df.sample(frac=frac_records, random_state=seed)
        
    df = _cast_traj_cols(
        df,
        traj_cols=traj_cols,
        parse_dates=parse_dates,
        mixed_timezone_behavior=mixed_timezone_behavior,
        fixed_format=fixed_format,
    )

    #sorting
    t_keys = ['timestamp', 'start_timestamp', 'datetime', 'start_datetime']
    ts_col = next((traj_cols[k] for k in t_keys if traj_cols[k] in df.columns), None)
    uid_col = traj_cols['user_id']
    
    if uid_col in df.columns:
        # Multi-user case: always safe to sort.
        return df.sort_values(by=[uid_col, ts_col], ignore_index=True)

    if sort_times:
        warnings.warn(
            f"Sorting by timestamp only, as user ID column '{uid_col}' was not found. If this is a multi-user "
            f"dataset, map the correct user ID column to avoid mixing trajectories.",
            UserWarning
        )
        return df.sort_values(by=[ts_col], ignore_index=True)
    
    return df

def to_file(df, path, format="csv",
            traj_cols=None, output_traj_cols=None,
            partition_by=None, filesystem=None,
            use_offset=False,
            **kwargs):
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

    # create directory if not exists    
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        warnings.warn(f"Parent directory '{parent_dir}' does not exist and will be created.")
        os.makedirs(parent_dir)
        
    if format=="parquet" or _is_directory(path):
        other_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(ds.write_dataset).parameters
        }
        table = pa.Table.from_pandas(df, preserve_index=False)

        ds.write_dataset(table, base_dir=str(path),
                         format=format,
                         partitioning=partition_by,
                         partitioning_flavor='hive',
                         filesystem=filesystem,
                         **other_kwargs)        
    
    else:
        other_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(df.to_csv).parameters
        }
        df.to_csv(path, index=False, **other_kwargs)
