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
from nomad.filters import to_projection, filter_users, _in_geo, to_timestamp

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
    
def test_projection_output(simple_df_one_user):
    # Basic test
    result = to_projection(traj=simple_df_one_user,
                           input_crs="EPSG:4326",
                           output_crs="EPSG:3857",
                           longitude="longitude",
                           latitude="latitude")
    x = pd.Series([1.294860e+07, 1.294861e+07, 1.294862e+07, 1.294862e+07])
    y = pd.Series([4.863631e+06, 4.863646e+06, 4.863650e+06, 4.863648e+06])
    ans = (x, y)

    assert_series_equal(result[0], ans[0], rtol=1e-6)
    assert_series_equal(result[1], ans[1], rtol=1e-6)

def test_projection_with_empty_df():
    empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'datetime'])
    result = to_projection(traj=empty_df,
                           input_crs="EPSG:4326",
                           output_crs="EPSG:3857",
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
    result = to_projection(traj=df_custom,
                           input_crs="EPSG:4326",
                           output_crs="EPSG:3857",
                           longitude="lon",
                           latitude="lat")
    x = pd.Series([1.294860e+07, 1.294861e+07, 1.294862e+07, 1.294862e+07])
    y = pd.Series([4.863631e+06, 4.863646e+06, 4.863650e+06, 4.863648e+06])
    ans = (x, y)

    assert_series_equal(result[0], ans[0], rtol=1e-6)
    assert_series_equal(result[1], ans[1], rtol=1e-6)

def test_projection_invalid_columns(simple_df_one_user):
    df_invalid = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    with pytest.raises(ValueError):
        to_projection(traj=df_invalid)

def test_in_geo(simple_df_multi_user):
    polygon = Polygon([(116.3192, 39.9840), (116.31965, 39.9840),
                       (116.31965, 39.9845), (116.3192, 39.9845)])
    df = _in_geo(traj=simple_df_multi_user, 
                 input_x='longitude', 
                 input_y='latitude', 
                 polygon=polygon, 
                 crs='EPSG:4326')
    assert 'in_geo' in df.columns
    expected_values = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    assert df['in_geo'].tolist() == expected_values


def test_filter_no_polygon(simple_df_multi_user):
    result = filter_users(simple_df_multi_user,
                          start_time=pd.Timestamp('2023-01-01 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2023-01-08 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326', 
                          longitude='longitude', 
                          latitude='latitude',
                          timestamp='timestamp')
    assert len(result) == 12
    assert _is_traj_df(result, longitude='longitude', latitude='latitude', timestamp='timestamp')

def test_filter_datetime(simple_df_multi_user):
    result = filter_users(simple_df_multi_user,
                          start_time=pd.Timestamp('2023-01-01 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2023-01-08 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326', 
                          longitude='longitude', 
                          latitude='latitude',
                          datetime='datetime')
    assert len(result) == 12
    assert _is_traj_df(result, longitude='longitude', latitude='latitude', datetime='datetime')

def test_filter_users_within_bounds(simple_df_one_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_one_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2008-10-23 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2008-10-24 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 4
    assert _is_traj_df(result)

def test_filter_users_outside_bounds(simple_df_one_user):
    polygon = Polygon([(116.3180, 39.9830), (116.3185, 39.9830), (116.3185, 39.9835), (116.3180, 39.9835)])
    result = filter_users(simple_df_one_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2008-10-23 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2008-10-24 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 0
    assert _is_traj_df(result)

def test_filter_users_within_bounds_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user, 
                          polygon=polygon,
                          start_time=pd.Timestamp('2023-01-01 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2023-01-08 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 10
    assert _is_traj_df(result)

def test_filter_users_outside_bounds_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3180, 39.9830), (116.3185, 39.9830), (116.3185, 39.9835), (116.3180, 39.9835)])
    result = filter_users(simple_df_multi_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2023-01-01 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2023-01-08 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 0
    assert _is_traj_df(result)

# def test_filter_users_with_spark(simple_df_one_user, spark):
#     polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
#     result = filter_users(simple_df_one_user, polygon=polygon, min_active_days=1, start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', spark_session=spark)
#     assert len(result) == 4

def test_filter_users_with_custom_columns(simple_df_one_user):
    df_custom = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(df_custom,
                          polygon=polygon,
                          start_time=pd.Timestamp('2008-10-23 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2008-10-24 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326',
                          longitude='lon',
                          latitude='lat')
    assert len(result) == 4
    assert _is_traj_df(result, longitude='lon', latitude='lat')

def test_filter_users_invalid_polygon(simple_df_one_user):
    with pytest.raises(TypeError):
        filter_users(simple_df_one_user,
                     "invalid_polygon",
                     start_time=pd.Timestamp('2008-10-23 00:00:00', tz='America/New_York'),
                     end_time=pd.Timestamp('2008-10-24 00:00:00', tz='America/New_York'),
                     min_active_days=1,
                     crs='EPSG:4326')

def test_filter_users_k2(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2023-01-01 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2023-01-08 00:00:00', tz='America/New_York'),
                          min_active_days=2,
                          crs='EPSG:4326')
    assert len(result) == 7  # Users 1, 2, and 4 have points within the polygon on at least 2 distinct days
    assert _is_traj_df(result)

def test_filter_users_k4(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2023-01-01 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2023-01-08 00:00:00', tz='America/New_York'),
                          min_active_days=4,
                          crs='EPSG:4326')
    assert len(result) == 0  # No users have points within the polygon on at least 3 distinct days
    assert _is_traj_df(result)

def test_projection_and_filter_users(simple_df_one_user):
    projected_x, projected_y = to_projection(traj=simple_df_one_user,
                                             input_crs="EPSG:4326",
                                             output_crs="EPSG:3857",
                                             longitude='longitude',
                                             latitude='latitude')
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])

    simple_df_one_user['x'] = projected_x
    simple_df_one_user['y'] = projected_y
    result = filter_users(simple_df_one_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2008-10-23 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2008-10-24 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          x='x',
                          y='y')
    assert len(result) == 4
    assert _is_traj_df(result)

def test_projection_and_filter_users_wrong_cols(simple_df_one_user):
    projected_x, projected_y = to_projection(traj=simple_df_one_user,
                                             input_crs="EPSG:4326",
                                             output_crs="EPSG:3857",
                                             longitude='longitude',
                                             latitude='latitude')
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])
    
    simple_df_one_user['x'] = projected_x
    simple_df_one_user['y'] = projected_y
    result = filter_users(simple_df_one_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2008-10-23 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2008-10-24 00:00:00', tz='America/New_York'),
                          min_active_days=1)
    assert len(result) == 0
    assert _is_traj_df(result)

def test_filter_users_within_time_frame(simple_df_one_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_one_user,
                          polygon=polygon,
                          start_time='2008-10-23 13:53:00',
                          end_time='2008-10-23 13:53:10',
                          timezone='America/New_York',
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 2
    assert _is_traj_df(result)

def test_filter_users_within_time_frame_datetime(simple_df_one_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_one_user,
                          polygon=polygon,
                          start_time='2008-10-23 13:53:00',
                          end_time='2008-10-23 13:53:10',
                          timezone='America/New_York',
                          min_active_days=1,
                          crs='EPSG:4326',
                          datetime='datetime')
    assert len(result) == 2
    assert _is_traj_df(result)

def test_filter_users_outside_time_frame(simple_df_one_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_one_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2008-10-23 13:53:20', tz='America/New_York'),
                          end_time=pd.Timestamp('2008-10-23 13:53:30', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 0
    assert _is_traj_df(result)

def test_filter_users_within_time_frame_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user,
                          polygon=polygon,
                          start_time='2023-01-01 05:00:00',
                          end_time='2023-01-03 05:00:00',
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 6
    assert _is_traj_df(result)

def test_filter_users_outside_time_frame_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user,
                          polygon=polygon,
                          start_time=pd.Timestamp('2023-01-05 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2023-01-08 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 1
    assert _is_traj_df(result)

def test_filter_users_with_empty_df():
    empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'datetime', 'timestamp', 'tz_offset'])
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(empty_df,
                          polygon=polygon,
                          start_time=pd.Timestamp('2008-10-23 00:00:00', tz='America/New_York'),
                          end_time=pd.Timestamp('2008-10-24 00:00:00', tz='America/New_York'),
                          min_active_days=1,
                          crs='EPSG:4326')
    assert len(result) == 0