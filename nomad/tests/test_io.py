import pytest
import warnings
from pathlib import Path
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np
import geopandas as gpd
import pygeohash as gh
import pyarrow.dataset as ds
import pdb
from nomad.io import base as loader
from nomad.filters import to_timestamp
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
def io_sources():
    root = Path(__file__).resolve().parent.parent / "data"
    return [
        (root / "gc_sample.csv",           "csv"),     # single CSV
        (root / "single_parquet",          "parquet"), # single‐file Parquet dir
        (root / "partitioned_csv",         "csv"),     # hive-partitioned CSV dir
        (root / "partitioned_parquet",     "parquet"), # hive-partitioned Parquet dir
    ]


@pytest.fixture
def base_df():
    test_dir = Path(__file__).resolve().parent
    data_path = test_dir.parent / "data" / "gc_sample.csv"
    df = pd.read_csv(data_path)

    df['tz_offset'] = 0
    df.loc[df.index[:5000],'tz_offset'] = -7200
    df.loc[df.index[-5000:], 'tz_offset'] = 3600
    df['tz_offset'] = df['tz_offset'].astype('Int64')

    # create string datetime column
    df['local_datetime'] = loader._unix_offset_to_str(df.timestamp, df.tz_offset)

    # create x, y columns in web mercator
    gdf = gpd.GeoSeries(gpd.points_from_xy(df.longitude, df.latitude),
	 		crs="EPSG:4326")
    projected = gdf.to_crs("EPSG:3857")
    df['x'] = projected.x
    df['y'] = projected.y
    
    df['geohash'] = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=7), axis=1)
    # col names:  ['uid', 'timestamp', 'tz_offset', 'longitude', 'latitude',  'local_datetime', 'x', 'y', 'geohash'
    # dtypes: [object, int64, int64, float64, float64, object, float64, float64, object]
    return df

@pytest.fixture
def dated_df(base_df):
    """
    Returns a DataFrame with default trajectory column names,
    and a derived 'date' column for partitioning.
    """
    df = base_df.rename(columns={
        "uid": "user_id",
        "local_datetime": "datetime"
    }).copy()
    df["date"] = df["datetime"].str[:10].astype(object)
    return df

@pytest.fixture
def col_variations():
    variations = {
        "default-basic": ([0, 1, 2, 3], ["uid", "timestamp", "latitude", "longitude"],
                          ["user_id", "timestamp", "latitude", "longitude"]),
        "alt-names-basic": ([0, 1, 2, 3], ["user", "unix_time", "lat", "lon"],
                            ["user_id", "timestamp", "latitude", "longitude"]),
        "alt-names-dt-xy": ([0, 5, 6, 7, 2], ["user", "event_zoned_datetime", "device_x", "device_y", "offset"],
                            ["user_id", "datetime", "x", "y", "tz_offset"]),
        "alt-names-ts-gh": ([0, 1, 8], ["user", "unix_ts", "geohash_7"],
                            ["user_id", "timestamp", "geohash"]),
        "default-dt-xy": ([0, 5, 6, 7, 2], ["uid", "local_datetime", "x", "y", "tz_offset"],
                          ["user_id", "datetime", "x", "y", "tz_offset"]),
    }
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

# Correctly handles names when importing trajectories
@pytest.mark.parametrize("variation", _col_variation_keys)
def test_from_df_name_handling(base_df, col_variations, variation):
    """Tests from_df with trajectory data using various column name mappings."""
    indices, names, keys = col_variations[variation]
    df_subset = base_df.iloc[:, indices].copy()
    df_subset.columns = names

    traj_cols = dict(zip(keys, names)) if keys else None

    result = loader.from_df(df_subset, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior='naive')

    assert loader._is_traj_df(result, traj_cols=traj_cols, parse_dates=True), "from_df() output is not a valid trajectory DataFrame"


# Correctly handles names when importing stops
@pytest.mark.parametrize("variation_name", _stop_col_variation_keys)
def test_from_df_stop_table(stop_df, stop_col_variations, variation_name):
    """Tests from_df with stop data using various column name mappings."""
    indices, names, keys = stop_col_variations[variation_name]
    df_subset = stop_df.iloc[:, indices].copy()
    df_subset.columns = names

    traj_cols = dict(zip(keys, names))

    result = loader.from_df(df_subset, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior='naive')

    assert loader._is_stop_df(result, traj_cols=traj_cols, parse_dates=True), \
        f"_is_stop_df validation failed for stop variation '{variation_name}'"

# date parsing works correctly
def test_date_parsing_from_df(base_df):
    df_subset = base_df.iloc[:, [0, 5, 6, 7]].copy()
    df_subset.columns = ["user_id", "datetime", "x", "y"]
    expected_tz_offset = base_df.tz_offset
    expected_ts = base_df.timestamp

    result = loader.from_df(df_subset, parse_dates=True, mixed_timezone_behavior="naive")
    result['timestamp'] = to_timestamp(result.datetime, result.tz_offset)

    assert (result.timestamp.values == expected_ts.values).all() and (result.tz_offset.values == expected_tz_offset.values).all()

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


@pytest.mark.parametrize("idx", range(4), ids=["csv-file", "parquet-file",
                                               "csv-hive", "parquet-hive"])
def test_table_columns(idx, io_sources):
    path, fmt = io_sources[idx]

    if fmt == "csv":
        csv_file = next(path.rglob("*.csv")) if path.is_dir() else path
        expected_cols = list(pd.read_csv(csv_file, nrows=0).columns)
    else:  # parquet
        expected_cols = list(ds.dataset(path, format="parquet",
                                        partitioning="hive").schema.names)

    cols = loader.table_columns(path, format=fmt)
    assert list(cols) == expected_cols


@pytest.mark.parametrize("bad_alias", [False, True])
def test_sampling_pipeline(io_sources, bad_alias):
    csv_path, _ = io_sources[0]

    if bad_alias:
        with pytest.raises(Exception):
            loader.sample_users(csv_path, format="csv", user_id="not_there")
        return

    ids = loader.sample_users(csv_path, format="csv",
                              user_id="uid", size=0.25, seed=123)

    df = loader.sample_from_file(csv_path,
                                 users=None,          # let it sample again
                                 format="csv",
                                 user_id="uid",
                                 frac_users=0.25,
                                 seed=123)

    assert loader._is_traj_df(df, traj_cols={"user_id": "uid"}, parse_dates=True)
    assert_series_equal(
        pd.Series(sorted(df["uid"].unique())),
        pd.Series(sorted(ids)),
        check_names=False,
    )


def test_from_file_alias_equivalence_csv(io_sources):
    path, fmt = io_sources[0]                 # gc_sample.csv

    # supply ONLY keyword aliases (no traj_cols)
    alias_kwargs = dict(
        user_id="uid",
        timestamp="timestamp",
        latitude="latitude",
        longitude="longitude",
        parse_dates=True,
    )

    df_via_file = loader.from_file(path, format=fmt, **alias_kwargs)

    raw = pd.read_csv(path)                   # plain pandas load
    df_via_df   = loader.from_df(raw, **alias_kwargs)

    assert_frame_equal(df_via_file, df_via_df, check_dtype=True)
    assert loader._is_traj_df(df_via_file,
                              traj_cols={k: v for k, v in alias_kwargs.items()
                                                   if k in {"user_id",
                                                            "timestamp",
                                                            "latitude",
                                                            "longitude"}},
                              parse_dates=True)


# Round-trip test for the writer/reader pipeline
@pytest.mark.parametrize("fmt", ["csv", "parquet"], ids=["writer-csv", "writer-parquet"])
def test_to_file_roundtrip(tmp_path, fmt, dated_df):
    """
    Tests to_file and from_file using standard columns and output_traj_cols remapping.
    """
    dated_df = loader.from_df(dated_df)
    output_traj_cols = {
        "user_id": "u",
        "timestamp": "ts",
        "latitude": "lat",
        "longitude": "lon",
        "tz_offset": "offset",
        "datetime": "event_time"
    }

    if fmt == "csv":
        out_path = tmp_path / "trip.csv"
        loader.to_file(dated_df, out_path, format="csv", output_traj_cols=output_traj_cols)
    else:
        out_path = tmp_path / "trip_parquet"
        loader.to_file(dated_df, out_path, format="parquet", output_traj_cols=output_traj_cols, partition_by=["date"])

    df_round = loader.from_file(out_path, format=fmt, traj_cols=output_traj_cols, parse_dates=True)
    assert loader._is_traj_df(df_round, traj_cols=output_traj_cols, parse_dates=True)

    df_out = df_round.rename(columns={v: k for k, v in output_traj_cols.items()})
    df_out = df_out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df_exp = dated_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    assert_frame_equal(df_out, df_exp, check_dtype=True)
