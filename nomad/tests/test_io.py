import pytest
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import pygeohash as gh
# import pdb # Removed unused import
from nomad.io import base as loader
# Assuming constants.DEFAULT_SCHEMA is available via import
from nomad import constants

# Define the keys explicitly for parametrization
_col_variation_keys = [
    "default-basic", "alt-names-basic", "alt-names-dt-xy",
    "alt-names-ts-gh", "default-dt-xy"
]

_stop_col_variation_keys = [
    "default_dt_start_end", "default_dt_start_duration",
    "default_ts_start_end", "default_ts_start_duration",
    "alt_dt_start_end", "alt_dt_start_duration",
    "alt_ts_start_end", "alt_ts_start_duration"
]


@pytest.fixture
def base_df():
    """Provides trajectory data from gc_sample.csv with added columns."""
    test_dir = Path(__file__).resolve().parent
    data_path = test_dir.parent / "data" / "gc_sample.csv"
    if not data_path.exists():
        pytest.skip(f"Test data file not found at {data_path}")
    df = pd.read_csv(data_path)

    df['tz_offset'] = 0
    df.loc[df.index[:5000],'tz_offset'] = -7200
    df.loc[df.index[-5000:], 'tz_offset'] = 3600
    df['tz_offset'] = df['tz_offset'].astype('Int64')

    df['local_datetime'] = loader._unix_offset_to_str(df.timestamp, df.tz_offset)

    gdf = gpd.GeoSeries(gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    projected = gdf.to_crs("EPSG:3857")
    df['x'] = projected.x.astype('Float64')
    df['y'] = projected.y.astype('Float64')
    df['latitude'] = df['latitude'].astype('Float64')
    df['longitude'] = df['longitude'].astype('Float64')
    df['timestamp'] = df['timestamp'].astype('Int64')

    df['geohash'] = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=7), axis=1).astype('string')
    df['uid'] = df['uid'].astype('string')
    df['local_datetime'] = df['local_datetime'].astype('string')
    return df

@pytest.fixture
def col_variations():
    """Provides variations for trajectory column name tests.
       Format: (indices_to_select, names_for_selected_cols, default_schema_keys)"""
    # Base col names for indices: ['uid', 'timestamp', 'latitude', 'longitude', 'tz_offset', 'local_datetime', 'x', 'y', 'geohash']
    variations = {
        "default-basic": ([0, 1, 2, 3], ["uid", "timestamp", "latitude", "longitude"], ["user_id", "timestamp", "latitude", "longitude"]),
        "alt-names-basic": ([0, 1, 2, 3], ["user", "unix_time", "lat", "lon"], ["user_id", "timestamp", "latitude", "longitude"]),
        "alt-names-dt-xy": ([0, 5, 6, 7, 4], ["user", "event_zoned_datetime", "device_x", "device_y", "offset"], ["user_id", "datetime", "x", "y", "tz_offset"]),
        "alt-names-ts-gh": ([0, 1, 8], ["user", "unix_ts", "geohash_7"], ["user_id", "timestamp", "geohash"]),
        "default-dt-xy": ([0, 5, 6, 7, 4], ["uid", "local_datetime", "x", "y", "tz_offset"], ["user_id", "datetime", "x", "y", "tz_offset"])
    }
    # Ensure keys match the explicitly defined list
    assert set(variations.keys()) == set(_col_variation_keys)
    return variations


@pytest.fixture
def stop_df():
    """Provides stop data from gc_stops.csv with added timestamp columns."""
    test_dir = Path(__file__).resolve().parent
    data_path = test_dir.parent / "data" / "gc_stops.csv"
    if not data_path.exists():
         pytest.skip(f"Test data file not found at {data_path}")
    df = pd.read_csv(data_path)

    start_dt = pd.to_datetime(df['start_time'], errors='coerce', utc=True)
    end_dt = pd.to_datetime(df['end_time'], errors='coerce', utc=True)

    df['start_ts'] = (start_dt.astype('int64') // 10**9).astype('Int64')
    df['end_ts'] = (end_dt.astype('int64') // 10**9).astype('Int64')

    df['duration'] = pd.to_numeric(df['duration'], errors='coerce').astype('Float64')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').astype('Float64')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').astype('Float64')
    df['start_time'] = df['start_time'].astype('string')
    df['end_time'] = df['end_time'].astype('string')
    return df

@pytest.fixture
def stop_col_variations():
    """Provides common variations for testing stop data handling.
       Format: (indices_to_select, names_for_selected_cols, default_schema_keys)"""
    # Base col names for indices: ['start_time', 'end_time', 'longitude', 'latitude', 'diameter', 'n_pings', 'duration', 'start_ts', 'end_ts']
    variations = {
        "default_dt_start_end": ([0, 1, 3, 2], ["start_time", "end_time", "latitude", "longitude"], ["start_datetime", "end_datetime", "latitude", "longitude"]),
        "default_dt_start_duration": ([0, 6, 3, 2], ["start_time", "duration", "latitude", "longitude"], ["start_datetime", "duration", "latitude", "longitude"]),
        "default_ts_start_end": ([7, 8, 3, 2], ["start_ts", "end_ts", "latitude", "longitude"], ["start_timestamp", "end_timestamp", "latitude", "longitude"]),
        "default_ts_start_duration": ([7, 6, 3, 2], ["start_ts", "duration", "latitude", "longitude"], ["start_timestamp", "duration", "latitude", "longitude"]),
        "alt_dt_start_end": ([0, 1, 3, 2], ["begin_dt", "finish_dt", "lat", "lon"], ["start_datetime", "end_datetime", "latitude", "longitude"]),
        "alt_dt_start_duration": ([0, 6, 3, 2], ["start_when", "stop_secs", "latitude", "long"], ["start_datetime", "duration", "latitude", "longitude"]),
        "alt_ts_start_end": ([7, 8, 3, 2], ["unix_begin", "unix_finish", "latitude", "longitude"], ["start_timestamp", "end_timestamp", "latitude", "longitude"]),
        "alt_ts_start_duration": ([7, 6, 3, 2], ["t_start", "dur_sec", "stop_lat", "longitude"], ["start_timestamp", "duration", "latitude", "longitude"]),
    }
    # Ensure keys match the explicitly defined list
    assert set(variations.keys()) == set(_stop_col_variation_keys)
    return variations


@pytest.fixture
def value_na_input_df():
    """Small DataFrame with NAs for value/type testing."""
    data = {
        'user_id': ['A', 'A', 'B', 'B', 'C'],
        'timestamp': [1672560000.0, 1672578000.0, np.nan, 1672617600.0, 1672531200.0],
        'latitude': [40.7128, 34.0522, np.nan, 36.1699, 40.7128],
        'longitude': [-74.0060, -118.2437, np.nan, -115.1398, -74.0060],
        'datetime': ["2023-01-01T10:00:00Z", "2023-01-01T09:00:00Z", pd.NA, "2023-01-01T18:00:00Z", "2023-01-01T00:00:00Z"]
    }
    df = pd.DataFrame(data)
    df['user_id'] = df['user_id'].astype('string')
    df['timestamp'] = df['timestamp'].astype('Float64')
    df['latitude'] = df['latitude'].astype('Float64')
    df['longitude'] = df['longitude'].astype('Float64')
    df['datetime'] = df['datetime'].astype('string')
    return df

@pytest.fixture
def expected_value_na_output_df():
    """Expected result after processing value_na_input_df with from_df."""
    data = {
        'user_id': ['A', 'A', 'B', 'B', 'C'],
        'timestamp': [1672560000, 1672578000, pd.NA, 1672617600, 1672531200],
        'latitude': [40.7128, 34.0522, np.nan, 36.1699, 40.7128],
        'longitude': [-74.0060, -118.2437, np.nan, -115.1398, -74.0060],
        'datetime': [pd.Timestamp('2023-01-01 10:00:00'), pd.Timestamp('2023-01-01 09:00:00'), pd.NaT, pd.Timestamp('2023-01-01 18:00:00'), pd.Timestamp('2023-01-01 00:00:00')],
        'tz_offset': [0, 0, pd.NA, 0, 0],
    }
    dtypes = {
        'user_id': 'string', 'timestamp': 'Int64', 'latitude': 'Float64',
        'longitude': 'Float64', 'datetime': 'datetime64[ns]', 'tz_offset': 'Int64',
    }
    expected_df = pd.DataFrame(data)
    for col, dtype_str in dtypes.items():
         # Avoid recasting datetime64[ns] which pd.Timestamp handles correctly
        if not (isinstance(expected_df[col].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype) and dtype_str == 'datetime64[ns]'):
           expected_df[col] = expected_df[col].astype(dtype_str)
    return expected_df


# --- Tests ---

# Use the explicitly defined list of keys for parametrization
@pytest.mark.parametrize("variation", _col_variation_keys)
def test_from_df_name_handling(base_df, col_variations, variation):
    """Tests from_df with trajectory data using various column name mappings."""
    indices, names, keys = col_variations[variation] # Access fixture data using the key
    df_subset = base_df.iloc[:, indices].copy()
    df_subset.columns = names

    traj_cols = dict(zip(keys, names)) if keys else None

    parse_dates_flag = 'datetime' in keys if keys else 'local_datetime' in names or 'datetime' in names
    mixed_tz = 'naive'

    result = loader.from_df(df_subset, traj_cols=traj_cols, parse_dates=parse_dates_flag, mixed_timezone_behavior=mixed_tz)

    assert loader._is_traj_df(result, traj_cols=traj_cols, parse_dates=parse_dates_flag), \
        f"_is_traj_df validation failed for trajectory variation '{variation}'"

# Use the explicitly defined list of keys for parametrization
@pytest.mark.parametrize("variation_name", _stop_col_variation_keys)
def test_from_df_stop_table(stop_df, stop_col_variations, variation_name):
    """Tests from_df with stop data using various column name mappings."""
    indices, names, keys = stop_col_variations[variation_name] # Access fixture data using the key
    df_subset = stop_df.iloc[:, indices].copy()
    df_subset.columns = names

    traj_cols = dict(zip(keys, names))

    parse_dates_flag = any(k in traj_cols for k in ['start_datetime', 'end_datetime'])
    mixed_tz = 'naive'

    result = loader.from_df(df_subset, traj_cols=traj_cols, parse_dates=parse_dates_flag, mixed_timezone_behavior=mixed_tz)

    assert loader._is_stop_df(result, traj_cols=traj_cols, parse_dates=parse_dates_flag), \
        f"_is_stop_df validation failed for stop variation '{variation_name}'"

def test_from_df_values_types_na(value_na_input_df, expected_value_na_output_df):
    """Tests from_df for correct values, type casting, and NA handling."""
    traj_cols = {
        "user_id": "user_id", "timestamp": "timestamp", "latitude": "latitude",
        "longitude": "longitude", "datetime": "datetime",
    }
    mixed_tz = 'naive'

    result = loader.from_df(value_na_input_df.copy(), traj_cols=traj_cols,
                            parse_dates=True, mixed_timezone_behavior=mixed_tz)

    pd.testing.assert_frame_equal(result, expected_value_na_output_df, check_dtype=True)

    # Optional sanity check with internal validation
    final_traj_cols_for_check = traj_cols.copy()
    if 'tz_offset' in result.columns:
         final_traj_cols_for_check['tz_offset'] = 'tz_offset'
    assert loader._is_traj_df(result, traj_cols=final_traj_cols_for_check, parse_dates=True), \
        "Resulting DataFrame failed _is_traj_df validation in NA test"

    