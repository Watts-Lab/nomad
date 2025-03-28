import pytest
import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from shapely.geometry import Polygon
from pyspark.sql import SparkSession

from nomad.filters import to_projection, filter_users, _in_geo


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
    return df

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
    return df

def test_projection_output(simple_df_one_user):
    result = to_projection(traj=simple_df_one_user)
    ans = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05', 1.294860e+07, 4.863631e+06],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06', 1.294861e+07, 4.863646e+06],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11', 1.294862e+07, 4.863650e+06],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16', 1.294862e+07, 4.863648e+06]],
        columns=['user_id', 'latitude', 'longitude', 'datetime', 'x', 'y']
    )

    assert isinstance(result.iloc[0]['x'], float)
    assert isinstance(result.iloc[0]['y'], float)
    assert_frame_equal(result.reset_index(drop=True), ans.reset_index(drop=True), atol=1e-6)

# def test_projection_with_spark(simple_df_one_user, spark):
#     result = to_projection(simple_df_one_user, spark_session=spark)
#     assert 'x' in result.columns
#     assert 'y' in result.columns
#     assert len(result) == 4

def test_projection_with_custom_columns(simple_df_one_user):
    df_custom = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    result = to_projection(traj=df_custom, longitude='lon', latitude='lat')
    assert 'x' in result.columns
    assert 'y' in result.columns
    assert len(result) == 4

def test_projection_with_traj_cols(simple_df_one_user):
    df_custom = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    traj_cols = {'longitude': 'lon', 'latitude': 'lat'}
    result = to_projection(traj=df_custom, traj_cols=traj_cols, output_x_col='x_proj', output_y_col='y_proj')
    assert 'x_proj' in result.columns
    assert 'y_proj' in result.columns
    assert len(result) == 4

def test_projection_invalid_columns(simple_df_one_user):
    df_invalid = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    with pytest.raises(ValueError):
        to_projection(traj=df_invalid)

def test_in_geo(simple_df_multi_user):
    polygon = Polygon([(116.3192, 39.9840), (116.31965, 39.9840),
                       (116.31965, 39.9845), (116.3192, 39.9845)])
    df = _in_geo(traj=simple_df_multi_user, 
                 from_x='longitude', from_y='latitude', 
                 polygon=polygon, crs='EPSG:4326')
    assert 'in_geo' in df.columns
    expected_values = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    assert df['in_geo'].tolist() == expected_values

def test_filter_no_polygon(simple_df_multi_user):
    result = filter_users(simple_df_multi_user,
                          start_time='2023-01-01 00:00:00', end_time='2023-01-08 00:00:00', min_active_days=1,
                          crs='EPSG:4326', longitude='longitude', latitude='latitude')
    assert len(result) == 12

def test_filter_users_within_bounds(simple_df_one_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_one_user, polygon=polygon, 
                               start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', min_active_days=1,
                               crs='EPSG:4326', longitude='longitude', latitude='latitude')
    assert len(result) == 4

def test_filter_users_outside_bounds(simple_df_one_user):
    polygon = Polygon([(116.3180, 39.9830), (116.3185, 39.9830), (116.3185, 39.9835), (116.3180, 39.9835)])
    result = filter_users(simple_df_one_user, polygon=polygon, 
                               start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', min_active_days=1,
                               crs='EPSG:4326', longitude='longitude', latitude='latitude')
    assert len(result) == 0

def test_filter_users_within_bounds_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user, polygon=polygon,
                               start_time='2023-01-01 00:00:00', end_time='2023-01-08 00:00:00',  min_active_days=1,
                               crs='EPSG:4326')
    assert len(result) == 10

def test_filter_users_outside_bounds_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3180, 39.9830), (116.3185, 39.9830), (116.3185, 39.9835), (116.3180, 39.9835)])
    result = filter_users(simple_df_multi_user, polygon=polygon, 
                               start_time='2023-01-01 00:00:00', end_time='2023-01-08 00:00:00', min_active_days=1,
                               crs='EPSG:4326')
    assert len(result) == 0

# def test_filter_users_with_spark(simple_df_one_user, spark):
#     polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
#     result = filter_users(simple_df_one_user, polygon=polygon, min_active_days=1, start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', spark_session=spark)
#     assert len(result) == 4

def test_filter_users_with_custom_columns(simple_df_one_user):
    df_custom = simple_df_one_user.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(df_custom, polygon=polygon,
                               start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00',  min_active_days=1, 
                               crs='EPSG:4326', longitude='lon', latitude='lat')
    assert len(result) == 4

def test_filter_users_invalid_polygon(simple_df_one_user):
    with pytest.raises(TypeError):
        filter_users(simple_df_one_user, "invalid_polygon",
                          start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', min_active_days=1,
                          crs='EPSG:4326')

def test_filter_users_k2(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user, polygon=polygon,
                               start_time='2023-01-01 00:00:00', end_time='2023-01-08 00:00:00', min_active_days=2,
                               crs='EPSG:4326')
    assert len(result) == 7  # Users 1, 2, and 4 have points within the polygon on at least 2 distinct days

def test_filter_users_k4(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user, polygon=polygon, 
                               start_time='2023-01-01 00:00:00', end_time='2023-01-08 00:00:00', min_active_days=4,
                               crs='EPSG:4326')
    assert len(result) == 0  # No users have points within the polygon on at least 3 distinct days

def test_projection_and_filter_users(simple_df_one_user):
    projected_df = to_projection(traj=simple_df_one_user)
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])
    result = filter_users(projected_df, polygon=polygon,
                               start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', min_active_days=1)
    assert len(result) == 4

def test_projection_and_filter_users_wrong_cols(simple_df_one_user):
    projected_df = to_projection(traj=simple_df_one_user)
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])
    result = filter_users(projected_df, polygon=polygon,
                               start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', min_active_days=1,
                               longitude='longitude', latitude='latitude')
    assert len(result) == 0

def test_filter_users_within_time_frame(simple_df_one_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_one_user, polygon=polygon, 
                               start_time='2008-10-23 13:53:00', end_time='2008-10-23 13:53:10', min_active_days=1,
                               crs='EPSG:4326')
    assert len(result) == 4

def test_filter_users_outside_time_frame(simple_df_one_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_one_user, polygon=polygon, 
                               start_time='2008-10-23 13:53:20', end_time='2008-10-23 13:53:30', min_active_days=1,
                               crs='EPSG:4326')
    assert len(result) == 0 

def test_filter_users_within_time_frame_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user, polygon=polygon, 
                               start_time='2023-01-01 00:00:00', end_time='2023-01-03 00:00:00', min_active_days=1,
                               crs='EPSG:4326')
    assert len(result) == 7

def test_filter_users_outside_time_frame_multi_user(simple_df_multi_user):
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(simple_df_multi_user, polygon=polygon, 
                               start_time='2023-01-05 00:00:00', end_time='2023-01-08 00:00:00', min_active_days=1, 
                               crs='EPSG:4326')
    assert len(result) == 4

def test_projection_with_empty_df():
    empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'datetime'])
    result = to_projection(traj=empty_df)
    assert 'x' in result.columns
    assert 'y' in result.columns
    assert len(result) == 0

def test_filter_users_with_empty_df():
    empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'datetime'])
    polygon = Polygon([(116.3190, 39.9840), (116.3200, 39.9840), (116.3200, 39.9850), (116.3190, 39.9850)])
    result = filter_users(empty_df, polygon=polygon, 
                               start_time='2008-10-23 00:00:00', end_time='2008-10-24 00:00:00', min_active_days=1,
                               crs='EPSG:4326')
    assert len(result) == 0