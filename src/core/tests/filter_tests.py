import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import pytest
import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from shapely.geometry import Polygon
from filters import to_projection, filter_to_box


@pytest.fixture
def simple_df_one_user():
    df = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16']],
        columns=['uid', 'latitude', 'longitude', 'time']
    )
    return df


@pytest.fixture
def simple_df_multi_user():
    df = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16'],
         [2, 39.984100, 116.319500, '2008-10-23 13:53:20'],
         [2, 39.984300, 116.319600, '2008-10-23 13:53:25'],
         [2, 39.984400, 116.319700, '2008-10-23 13:53:30'],
         [3, 39.984000, 116.319800, '2008-10-23 13:53:35'],
         [3, 39.984500, 116.319900, '2008-10-23 13:53:40']],
        columns=['uid', 'latitude', 'longitude', 'time']
    )
    return df


@pytest.fixture
def sample_polygon():
    return Polygon([(39.9840, 116.3192), (39.9840, 116.3197),
                    (39.9845, 116.3197), (39.9845, 116.3192)])


def test_filter_to_box_lat_lon(simple_df_one_user):
    """
    Test that filter_to_box filters points correctly within a polygon
    in lat/lon space. Polygon covers full space.
    """

    polygon = Polygon([(39.99, 116.31), (39.99, 116.32),
                       (39.98, 116.32), (39.98, 116.31)])

    result = filter_to_box(simple_df_one_user, polygon, 'latitude', 'longitude')

    assert len(result) == len(simple_df_one_user)
    assert_frame_equal(result.reset_index(drop=True), simple_df_one_user.reset_index(drop=True))


def test_filter_to_box_lat_lon2(simple_df_one_user):
    """
    Test that filter_to_box filters points correctly within a polygon
    in lat/lon space. Polygon is smaller than full space.
    """

    polygon = Polygon([(39.98422, 116.3192), (39.98422, 116.31935),
                       (39.98400, 116.31935), (39.98400, 116.3192)])
    result = filter_to_box(simple_df_one_user, polygon, 'latitude', 'longitude')

    ans = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06']],
        columns=['uid', 'latitude', 'longitude', 'time']
    )

    assert len(result) == 2
    assert_frame_equal(result.reset_index(drop=True), ans.reset_index(drop=True))


def test_to_projection(simple_df_one_user):
    """Test that to_projection correctly converts latitude/longitude to projected x/y."""
    result = to_projection(simple_df_one_user, 'latitude', 'longitude')
    ans = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05', 1.294860e+07, 4.863631e+06],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06', 1.294861e+07, 4.863646e+06],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11', 1.294862e+07, 4.863650e+06],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16', 1.294862e+07, 4.863648e+06]],
        columns=['uid', 'latitude', 'longitude', 'time', 'x', 'y']
    )

    assert 'x' in result.columns
    assert 'y' in result.columns
    assert isinstance(result.iloc[0]['x'], float)
    assert isinstance(result.iloc[0]['y'], float)
    assert_frame_equal(result.reset_index(drop=True), ans.reset_index(drop=True), atol=1e-6)


def test_to_projection_and_filter_by_xy(simple_df_one_user):
    """Test to_projection followed by filter_to_box in x/y space."""
    projected_df = to_projection(simple_df_one_user, 'latitude', 'longitude')
    polygon = Polygon([(1.294861e+07, 4.863647e+06), (1.294861e+07, 4.863649e+06),
                       (1.294863e+07, 4.863649e+06), (1.294863e+07, 4.863647e+06)])
    result = filter_to_box(projected_df, polygon, 'x', 'y')

    ans = pd.DataFrame(
        [[1, 39.984211, 116.319389, '2008-10-23 13:53:16', 1.294862e+07, 4.863648e+06]],
        columns=['uid', 'latitude', 'longitude', 'time', 'x', 'y']
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

    result = to_projection(df, 'custom_lat', 'custom_lon')

    assert 'x' in result.columns
    assert 'y' in result.columns

    assert isinstance(result.iloc[0]['x'], float)
    assert isinstance(result.iloc[0]['y'], float)

###### VERIFY BELOW

def test_in_geo(simple_df_multi_user, sample_polygon):
    """
    Test the _in_geo function for correctly tagging points inside the polygon.
    """
    df = _in_geo(simple_df_multi_user, 'latitude', 'longitude', sample_polygon)
    
    assert 'in_geo' in df.columns

    expected_values = [1, 1, 1, 0, 0, 0, 0, 0]
    assert df_with_in_geo['in_geo'].tolist() == expected_values


# Test for filtered_users function
def test_filtered_users_basic(simple_df_multi_user, sample_polygon):
    """
    Test the filtered_users function for basic functionality: filtering users based on k days.
    """
    # Define the timeframe
    T0 = '2023-01-01'
    T1 = '2023-01-03'
    
    # Test with k=2 distinct days
    filtered_result = filtered_users(
        simple_df_multi_user, 'user_id', 'timestamp', 2, T0, T1,
        'latitude', 'longitude', sample_polygon
    )

    # Check if filtered_result is a DataFrame
    assert isinstance(filtered_result, pd.DataFrame)

    # Only user 1 should have at least 2 distinct days with pings in the polygon
    expected_users = [1]
    assert filtered_result['user_id'].tolist() == expected_users


def test_filtered_users_timeframe(simple_df_multi_user, sample_polygon):
    """
    Test that the function correctly filters based on the timeframe T0 to T1.
    """
    # Define a narrower timeframe, only including data from January 1st, 2023
    T0 = '2023-01-01'
    T1 = '2023-01-01'
    
    # Test with k=1 distinct day
    filtered_result = filtered_users(
        simple_df_multi_user, 'user_id', 'timestamp', 1, T0, T1,
        'latitude', 'longitude', sample_polygon
    )

    # Only user 1 should have pings within this timeframe and inside the polygon
    expected_users = [1]
    assert filtered_result['user_id'].tolist() == expected_users


def test_filtered_users_distinct_days(simple_df_multi_user, sample_polygon):
    """
    Test that the function correctly counts distinct days for users.
    """
    # Define the timeframe
    T0 = '2023-01-01'
    T1 = '2023-01-03'
    
    # Test with k=1 distinct day
    filtered_result = filtered_users(
        simple_df_multi_user, 'user_id', 'timestamp', 1, T0, T1,
        'latitude', 'longitude', sample_polygon
    )

    # User 1 has 2 distinct days with pings in the polygon, user 2 has 0 days
    expected_users = [1]
    assert filtered_result['user_id'].tolist() == expected_users


def test_filtered_users_no_users(simple_df_multi_user, sample_polygon):
    """
    Test the edge case where no users meet the condition of at least k distinct days.
    """
    # Define the timeframe
    T0 = '2023-01-01'
    T1 = '2023-01-03'
    
    # Test with k=3 (no users have 3 distinct days in the polygon)
    filtered_result = filtered_users(
        simple_df_multi_user, 'user_id', 'timestamp', 3, T0, T1,
        'latitude', 'longitude', sample_polygon
    )

    # The result should be an empty DataFrame
    assert filtered_result.empty


def test_filtered_users_empty_input():
    """
    Test the edge case where the input DataFrame is empty.
    """
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=['user_id', 'latitude', 'longitude', 'timestamp'])

    # Define the timeframe and polygon
    T0 = '2023-01-01'
    T1 = '2023-01-03'
    polygon = Polygon([(39.9840, 116.3192), (39.9840, 116.3197),
                       (39.9845, 116.3197), (39.9845, 116.3192)])

    # Test with k=1 (no users in empty input)
    filtered_result = filtered_users(
        empty_df, 'user_id', 'timestamp', 1, T0, T1,
        'latitude', 'longitude', polygon
    )

    # The result should be an empty DataFrame
    assert filtered_result.empty