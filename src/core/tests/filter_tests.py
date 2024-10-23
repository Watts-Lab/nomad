import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import pytest
import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from shapely.geometry import Polygon
import filters as F


@pytest.fixture
def simple_df_one_user():
    df = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16']],
        columns=['user_id', 'latitude', 'longitude', 'timestamp']
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
        columns=['user_id', 'latitude', 'longitude', 'timestamp']
    )
    return df


def test_filter_to_box_lat_lon(simple_df_one_user):
    """
    Test that filter_to_box filters points correctly within a polygon
    in lat/lon space. Polygon covers full space.
    """

    polygon = Polygon([(39.99, 116.31), (39.99, 116.32),
                       (39.98, 116.32), (39.98, 116.31)])

    result = F.filter_to_box(df=simple_df_one_user,
                             latitude='latitude',
                             longitude='longitude',
                             polygon=polygon)

    assert len(result) == len(simple_df_one_user)
    assert_frame_equal(result.reset_index(drop=True), simple_df_one_user.reset_index(drop=True))


def test_filter_to_box_lat_lon2(simple_df_one_user):
    """
    Test that filter_to_box filters points correctly within a polygon
    in lat/lon space. Polygon is smaller than full space.
    """

    polygon = Polygon([(39.98422, 116.3192), (39.98422, 116.31935),
                       (39.98400, 116.31935), (39.98400, 116.3192)])
    result = F.filter_to_box(df=simple_df_one_user,
                             latitude='latitude',
                             longitude='longitude',
                             polygon=polygon)

    ans = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06']],
        columns=['user_id', 'latitude', 'longitude', 'timestamp']
    )

    assert len(result) == 2
    assert_frame_equal(result.reset_index(drop=True), ans.reset_index(drop=True))


def test_to_projection(simple_df_one_user):
    """Test that to_projection correctly converts latitude/longitude to projected x/y."""
    result = F.to_projection(df=simple_df_one_user,
                             latitude='latitude',
                             longitude='longitude')
    ans = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05', 1.294860e+07, 4.863631e+06],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06', 1.294861e+07, 4.863646e+06],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11', 1.294862e+07, 4.863650e+06],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16', 1.294862e+07, 4.863648e+06]],
        columns=['user_id', 'latitude', 'longitude', 'timestamp', 'x', 'y']
    )

    assert 'x' in result.columns
    assert 'y' in result.columns
    assert isinstance(result.iloc[0]['x'], float)
    assert isinstance(result.iloc[0]['y'], float)
    assert_frame_equal(result.reset_index(drop=True), ans.reset_index(drop=True), atol=1e-6)


def test_to_projection_and_filter_by_xy(simple_df_one_user):
    """Test to_projection followed by filter_to_box in x/y space."""
    projected_df = F.to_projection(df=simple_df_one_user,
                             latitude='latitude',
                             longitude='longitude')
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])
    result = F.filter_to_box(df=projected_df, 
                             latitude='x',
                             longitude='y',
                             polygon=polygon)

    ans = pd.DataFrame(
        [[1, 39.984211, 116.319389, '2008-10-23 13:53:16', 1.294862e+07, 4.863648e+06]],
        columns=['user_id', 'latitude', 'longitude', 'timestamp', 'x', 'y']
    )

    assert len(result) == 1
    assert 'x' in result.columns
    assert 'y' in result.columns
    assert_frame_equal(result.reset_index(drop=True), ans.reset_index(drop=True), atol=1e-6)


def test_custom_column_names():
    """Test that projection works with custom column names."""
    df = pd.DataFrame({
        'custom_lat': [39.984094, 39.984198],
        'custom_lon': [116.319236, 116.319322]
    })

    result = F.to_projection(df=df,
                             latitude='custom_lat',
                             longitude='custom_lon')

    assert 'x' in result.columns
    assert 'y' in result.columns

    assert isinstance(result.iloc[0]['x'], float)
    assert isinstance(result.iloc[0]['y'], float)


def test_in_geo(simple_df_multi_user):
    """
    Test the _in_geo function for correctly tagging points inside the polygon.
    """
    polygon = Polygon([(39.9840, 116.3192), (39.9840, 116.31965),
                       (39.9845, 116.31965), (39.9845, 116.3192)])

    df = F._in_geo(df=simple_df_multi_user,
                   latitude_col='latitude',
                   longitude_col='longitude',
                   polygon=polygon)

    assert 'in_geo' in df.columns

    expected_values = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    assert df['in_geo'].tolist() == expected_values


def test_filtered_users_basic(simple_df_multi_user):
    """
    Test the filtered_users function for basic functionality: filtering users based on k days.
    """
    T0 = '2023-01-01'
    T1 = '2023-01-06'

    polygon = Polygon([(39.9840, 116.3192), (39.9840, 116.31965),
                       (39.9845, 116.31965), (39.9845, 116.3192)])

    filtered_users = F._filtered_users(
        df=simple_df_multi_user,
        k=2,
        T0=T0,
        T1=T1,
        polygon=polygon,
        user_col='user_id',
        timestamp_col='timestamp',
        latitude_col='latitude',
        longitude_col='longitude'
    )

    assert filtered_users.tolist() == [1, 4]


def test_filtered_users_timeframe(simple_df_multi_user):
    """
    Test that the function correctly filters based on the timeframe T0 to T1.
    """
    T0 = '2023-01-03'
    T1 = '2023-01-08'

    polygon = Polygon([(39.9840, 116.3192), (39.9840, 116.31965),
                       (39.9845, 116.31965), (39.9845, 116.3192)])

    filtered_users = F._filtered_users(
        df=simple_df_multi_user,
        k=2,
        T0=T0,
        T1=T1,
        polygon=polygon,
        user_col='user_id',
        timestamp_col='timestamp',
        latitude_col='latitude',
        longitude_col='longitude'
    )

    assert filtered_users.tolist() == [4]


def test_filtered_users_no_users(simple_df_multi_user):
    """
    Test the edge case where no users meet the condition of at least k distinct days.
    """
    T0 = '2023-01-01'
    T1 = '2023-01-06'

    polygon = Polygon([(39.9840, 116.3192), (39.9840, 116.31965),
                       (39.9845, 116.31965), (39.9845, 116.3192)])

    filtered_users = F._filtered_users(
        df=simple_df_multi_user,
        k=4,
        T0=T0,
        T1=T1,
        polygon=polygon,
        user_col='user_id',
        timestamp_col='timestamp',
        latitude_col='latitude',
        longitude_col='longitude'
    )

    assert filtered_users.empty


###### VERIFY BELOW


# def test_filtered_users_distinct_days(simple_df_multi_user, sample_polygon):
#     """
#     Test that the function correctly counts distinct days for users.
#     """
#     # Define the timeframe
#     T0 = '2023-01-01'
#     T1 = '2023-01-03'
    
#     # Test with k=1 distinct day
#     filtered_result = filtered_users(
#         simple_df_multi_user, 'user_id', 'timestamp', 1, T0, T1,
#         'latitude', 'longitude', sample_polygon
#     )

#     # User 1 has 2 distinct days with pings in the polygon, user 2 has 0 days
#     expected_users = [1]
#     assert filtered_result['user_id'].tolist() == expected_users


# def test_filtered_users_empty_input():
#     """
#     Test the edge case where the input DataFrame is empty.
#     """
#     # Empty DataFrame
#     empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'timestamp'])

#     # Define the timeframe and polygon
#     T0 = '2023-01-01'
#     T1 = '2023-01-03'
#     polygon = Polygon([(39.9840, 116.3192), (39.9840, 116.3197),
#                        (39.9845, 116.3197), (39.9845, 116.3192)])

#     # Test with k=1 (no users in empty input)
#     filtered_result = filtered_users(
#         empty_df, 'user_id', 'timestamp', 1, T0, T1,
#         'latitude', 'longitude', polygon
#     )

#     # The result should be an empty DataFrame
#     assert filtered_result.empty