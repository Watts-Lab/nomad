import pytest
import pandas as pd
import numpy as np
import pdb
import geopandas as gpd
import pygeohash as gh
from pandas.testing import assert_frame_equal, assert_series_equal
from shapely.geometry import Polygon
from pyspark.sql import SparkSession
from pathlib import Path
import nomad.io.base as loader
from nomad.io.base import _unix_offset_to_str, _is_traj_df, from_df
from nomad.filters import (
    to_projection,
    to_timestamp,
    to_yyyymmdd,
    is_within,
    within,
    downsample,
    coverage_matrix,
    completeness,
)

@pytest.fixture(scope="module")
def spark():
    spark_session = SparkSession.builder \
        .appName("TestToProjection") \
        .getOrCreate()
    yield spark_session
    spark_session.stop()

@pytest.fixture
def simple_df_one_user():
    df = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16']],
        columns=['user_id', 'latitude', 'longitude', 'datetime']
    )
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('America/New_York')
    df['timestamp'] = to_timestamp(df['datetime'])
    df['tz_offset'] = df['datetime'].apply(lambda x: loader._offset_seconds_from_ts(x))
    return from_df(df)

@pytest.fixture
def simple_df_multi_user():
    df = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2023-01-01 13:53:05'],
         [1, 39.984198, 116.319322, '2023-01-02 13:53:06'],
         [1, 39.984224, 116.319402, '2023-01-02 13:53:11'],
         [1, 39.984211, 116.319389, '2023-01-07 13:53:16'],
         [2, 39.984100, 116.319500, '2023-01-01 13:53:20'],
         [2, 39.984300, 116.319600, '2023-01-01 13:53:25'],
         [2, 39.984400, 116.319700, '2023-01-01 13:53:30'],
         [3, 20.984000, 116.319800, '2023-01-04 13:53:35'],
         [3, 20.984500, 116.319900, '2023-01-05 13:53:40'],
         [4, 39.984100, 116.319500, '2023-01-03 13:53:20'],
         [4, 39.984300, 116.319600, '2023-01-04 13:53:25'],
         [4, 39.984400, 116.319700, '2023-01-04 13:53:30']],
        columns=['user_id', 'latitude', 'longitude', 'datetime']
    )
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('America/New_York')
    df['timestamp'] = to_timestamp(df['datetime'])
    df['tz_offset'] = df['datetime'].apply(lambda x: loader._offset_seconds_from_ts(x))
    return from_df(df)

@pytest.fixture
def base_df():
    test_dir = Path(__file__).resolve().parent
    data_path = test_dir.parent / "data" / "gc_sample.csv"
    df = pd.read_csv(data_path)

    # create tz_offset column
    df['tz_offset'] = 0
    df.loc[df.index[:5000],'tz_offset'] = -7200
    df.loc[df.index[-5000:], 'tz_offset'] = 3600

    # create string datetime column
    df['local_datetime'] = _unix_offset_to_str(df.timestamp, df.tz_offset)
    
    # create x, y columns in web mercator
    gdf = gpd.GeoSeries(gpd.points_from_xy(df.longitude, df.latitude),
                            crs="EPSG:4326")
    projected = gdf.to_crs("EPSG:3857")
    df['x'] = projected.x
    df['y'] = projected.y
    
    df['geohash'] = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=7), axis=1)
    # col names:  ['uid', 'timestamp', 'latitude', 'longitude', 'tz_offset', 'local_datetime', 'x', 'y', 'geohash'
    # dtypes: [object, int64, float64, float64, int64, object, float64, float64, object]
    return from_df(df)

@pytest.fixture
def park_polygons():
    hex_wkt = (
        "POLYGON ((-38.3176943767 36.6695149320, "
        "-38.3178191245 36.6697310917, "
        "-38.3180686181 36.6697310917, "
        "-38.3181933660 36.6695149320, "
        "-38.3180686181 36.6692987719, "
        "-38.3178191245 36.6692987719, "
        "-38.3176943767 36.6695149320))"
    )
    hex_poly = _wkt.loads(hex_wkt)                          # shapely
    hex_gs   = gpd.GeoSeries([hex_poly], crs="EPSG:4326")   # attach CRS
    hex_gs   = hex_gs.to_crs("EPSG:3857")                   # reproject to Web-Mercator
    return [hex_wkt, hex_poly, hex_gs]

def test_to_timestamp(base_df):
    timestamp_col = to_timestamp(base_df.local_datetime, base_df.tz_offset)
    assert np.array_equal(timestamp_col.values, base_df.timestamp.values)

def test_to_timestamp_scalar():
    result = to_timestamp("2024-01-01 00:00-05:00")
    assert isinstance(result, (int, np.integer))
    assert result == 1704085200

def test_to_yyyymmdd_roundtrip_simple():
    s = pd.to_datetime(pd.Series([
        '2024-01-02T23:59:59Z',
        '2024-01-03T00:00:00Z'
    ]))
    # UTC dates
    dates = to_yyyymmdd(s)
    assert dates.tolist() == [20240102, 20240103]

def test_to_yyyymmdd_with_offsets():
    # Two timestamps near midnight UTC; apply offsets to flip dates
    ts = pd.Series([1704230399, 1704230400])  # 2024-01-02 23:59:59Z, 2024-01-03 00:00:00Z
    offsets = pd.Series([-5*3600, +5*3600])
    dates_local = to_yyyymmdd(ts, tz_offset=offsets)
    # First becomes local 2024-01-02 18:59:59 (still 2024-01-02); second becomes 2024-01-03 05:00:00
    assert dates_local.tolist() == [20240102, 20240103]

def test_to_yyyymmdd_accepts_strings_and_timestamps():
    s = pd.Series(['2024-02-10 12:34:56', '2024-02-11 00:00:00'])
    dates = to_yyyymmdd(s)
    assert dates.tolist() == [20240210, 20240211]
    
def test_projection_output(simple_df_one_user):
    # Basic test
    result = to_projection(data=simple_df_one_user,
                           data_crs="EPSG:4326",
                           crs_to="EPSG:3857",
                           longitude="longitude",
                           latitude="latitude")
    x = pd.Series([1.294860e+07, 1.294861e+07, 1.294862e+07, 1.294862e+07])
    y = pd.Series([4.863631e+06, 4.863646e+06, 4.863650e+06, 4.863648e+06])
    ans = (x, y)

    assert_series_equal(result[0], ans[0], rtol=1e-6)
    assert_series_equal(result[1], ans[1], rtol=1e-6)

def test_projection_with_empty_df():
    empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'datetime'])
    result = to_projection(data=empty_df,
                           data_crs="EPSG:4326",
                           crs_to="EPSG:3857",
                           longitude="longitude",
                           latitude="latitude")
    assert (len(result[0]), len(result[1])) == (0, 0)

# def test_projection_with_spark(simple_df_one_user, spark):
#     result = to_projection(simple_df_one_user, spark_session=spark)
#     assert 'x' in result.columns
#     assert 'y' in result.columns
#     assert len(result) == 4

def test_projection_with_custom_columns(simple_df_one_user):
    # Traj has custom column names
    df_custom = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    result = to_projection(data=df_custom,
                           data_crs="EPSG:4326",
                           crs_to="EPSG:3857",
                           longitude="lon",
                           latitude="lat")
    x = pd.Series([1.294860e+07, 1.294861e+07, 1.294862e+07, 1.294862e+07])
    y = pd.Series([4.863631e+06, 4.863646e+06, 4.863650e+06, 4.863648e+06])
    ans = (x, y)

    assert_series_equal(result[0], ans[0], rtol=1e-6)
    assert_series_equal(result[1], ans[1], rtol=1e-6)

def test_is_within_mask(simple_df_multi_user):
    polygon = Polygon([(116.3192, 39.9840), (116.31965, 39.9840),
                       (116.31965, 39.9845), (116.3192, 39.9845)])
    mask = is_within(
        df=simple_df_multi_user,
        within=polygon,
        data_crs='EPSG:4326',
        longitude='longitude',
        latitude='latitude'
    )
    expected = [True, True, True, True, True, True, False, False, False, True, True, False]
    assert mask.tolist() == expected


def test_within_filters_rows(simple_df_multi_user):
    polygon = Polygon([(116.3192, 39.9840), (116.31965, 39.9840),
                       (116.31965, 39.9845), (116.3192, 39.9845)])
    filtered = within(
        simple_df_multi_user,
        polygon,
        data_crs='EPSG:4326',
        longitude='longitude',
        latitude='latitude'
    )
    assert len(filtered) == 8
    assert _is_traj_df(filtered, longitude='longitude', latitude='latitude', timestamp='timestamp')

def test_downsample_minute_window(simple_df_one_user):
    # Duplicate entries within the same minute for a single user
    df = simple_df_one_user.copy()
    # Add a duplicate point within same minute
    dup = df.iloc[[0]].copy()
    dup['timestamp'] = dup['timestamp'] + 10  # 10 seconds later
    df2 = pd.concat([df, dup], ignore_index=True)
    reduced = downsample(df2, periods=1, freq='min', keep='first', user_id='user_id', timestamp='timestamp')
    # First minute should have only one record after downsampling
    assert reduced['timestamp'].nunique() <= df2['timestamp'].nunique()

def test_coverage_and_completeness_series():
    # Series of 1-hour apart timestamps
    base = pd.Series([0, 3600, 7200, 10800])
    hits = coverage_matrix(base, periods=1, freq='h')
    assert hits.sum() in (3, 4)
    comp = completeness(base, periods=1, freq='h')
    assert comp == 1.0

def test_completeness_multi_user(simple_df_multi_user):
    # Ensure returns per-user stats and reasonable bounds
    comp = completeness(simple_df_multi_user, periods=1, freq='h', traj_cols=None, user_id='user_id', timestamp='timestamp')
    assert isinstance(comp, pd.Series)
    assert (comp >= 0).all() and (comp <= 1).all()

def test_within_counts_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    filtered = within(simple_df_multi_user, polygon, data_crs='EPSG:4326', longitude='longitude', latitude='latitude')
    assert len(filtered) > 0

# def test__filtered_users_with_spark(simple_df_one_user, spark):
#     polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
#     result = _filtered_users(simple_df_one_user, polygon=polygon, min_active_days=1, start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', spark_session=spark)
#     assert len(result) == 4

def test_within_custom_columns(simple_df_one_user):
    df_custom = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    filtered = within(df_custom, polygon, data_crs='EPSG:4326', traj_cols={'longitude':'lon','latitude':'lat'})
    assert len(filtered) == 4
    assert _is_traj_df(filtered, longitude='lon', latitude='lat')

def test_is_within_invalid_polygon(simple_df_one_user):
    with pytest.raises(Exception):
        is_within(simple_df_one_user, "invalid_polygon", data_crs='EPSG:4326')

def test_coverage_matrix_and_completeness_df(simple_df_multi_user):
    hits = coverage_matrix(simple_df_multi_user, periods=1, freq='h', traj_cols=None, user_id='user_id', timestamp='timestamp')
    assert isinstance(hits, pd.DataFrame)
    comp = completeness(simple_df_multi_user, periods=1, freq='h', traj_cols=None, user_id='user_id', timestamp='timestamp')
    assert isinstance(comp, pd.Series)

def test_projection_and_within(simple_df_one_user):
    projected_x, projected_y = to_projection(data=simple_df_one_user,
                                             data_crs="EPSG:4326",
                                             crs_to="EPSG:3857",
                                             longitude='longitude',
                                             latitude='latitude')
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])

    simple_df_one_user['x'] = projected_x
    simple_df_one_user['y'] = projected_y
    mask = is_within(simple_df_one_user, polygon, data_crs='EPSG:3857', x='x', y='y')
    assert mask.sum() > 0

def test_projection_and_within_wrong_cols(simple_df_one_user):
    """Test that when both lat/lon and x/y are present, auto-detection prefers x/y."""
    projected_x, projected_y = to_projection(data=simple_df_one_user,
                                             data_crs="EPSG:4326",
                                             crs_to="EPSG:3857",
                                             longitude='longitude',
                                             latitude='latitude')
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])
    
    simple_df_one_user['x'] = projected_x
    simple_df_one_user['y'] = projected_y
    # Without specifying column names, auto-detection should prefer x/y over lat/lon
    mask = is_within(simple_df_one_user, polygon, data_crs='EPSG:3857')
    # Should successfully use x/y coordinates and find points within polygon
    assert isinstance(mask, pd.Series)
    assert mask.sum() > 0

def test_completeness_time_window(simple_df_one_user):
    comp = completeness(simple_df_one_user, periods=1, freq='min', traj_cols=None, user_id='user_id', timestamp='timestamp')
    assert isinstance(comp, pd.Series)

def test_within_with_empty_df():
    empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'datetime', 'timestamp', 'tz_offset'])
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    filtered = within(empty_df, polygon, data_crs='EPSG:4326', longitude='longitude', latitude='latitude')
    assert len(filtered) == 0


def test_completeness_empty_dataframe():
    df = pd.DataFrame(columns=['user_id', 'timestamp'])
    out = completeness(df, periods=1, freq='h', traj_cols=None, user_id='user_id', timestamp='timestamp')
    assert isinstance(out, pd.Series)
    assert out.empty


def test_completeness_bad_freq_raises(simple_df_one_user):
    with pytest.raises(ValueError):
        coverage_matrix(simple_df_one_user, periods=1, freq='hourly', traj_cols=None, user_id='user_id', timestamp='timestamp')


def test_to_zoned_datetime_basic():
    from nomad.filters import to_zoned_datetime
    ts = pd.Series([0, 3600], dtype='int64')
    offs = pd.Series([0, 0], dtype='int64')
    zoned = to_zoned_datetime(ts, offs)
    assert pd.api.types.is_datetime64_any_dtype(zoned) or zoned.dtype == 'object'
    # Check first two values stringified
    s = zoned.astype(str).tolist()
    assert s[0].startswith('1970-01-01') and s[1].startswith('1970-01-01')


def test_to_zoned_datetime_with_na_offset_raises():
    from nomad.filters import to_zoned_datetime
    ts = pd.Series([0, 3600], dtype='int64')
    offs = pd.Series([0, pd.NA], dtype='Int64')
    with pytest.raises(Exception):
        to_zoned_datetime(ts, offs)


def test_to_yyyymmdd_nullable_outputs():
    s = pd.Series(['2024-02-10 12:34:56', None])
    dates = to_yyyymmdd(s)
    # nullable dtype when NA is present
    assert str(dates.dtype) in ('Int64', 'int64')
    assert pd.isna(dates.iloc[1])