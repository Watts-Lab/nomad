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
def simple_df():
    """Fixture to generate a simple DataFrame with mock latitude, longitude, and time data."""
    df = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
         [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
         [1, 39.984211, 116.319389, '2008-10-23 13:53:16']],
        columns=['uid', 'latitude', 'longitude', 'time']
    )
    return df


def test_filter_to_box_lat_lon(simple_df):
    """
    Test that filter_to_box filters points correctly within a polygon
    in lat/lon space. Polygon covers full space.
    """

    polygon = Polygon([(39.99, 116.31), (39.99, 116.32),
                       (39.98, 116.32), (39.98, 116.31)])

    result = filter_to_box(simple_df, polygon, 'latitude', 'longitude')

    assert len(result) == len(simple_df)
    assert_frame_equal(result.reset_index(drop=True), simple_df.reset_index(drop=True))


def test_filter_to_box_lat_lon2(simple_df):
    """
    Test that filter_to_box filters points correctly within a polygon
    in lat/lon space. Polygon is smaller than full space.
    """

    polygon = Polygon([(39.98422, 116.3192), (39.98422, 116.31935),
                       (39.98400, 116.31935), (39.98400, 116.3192)])
    result = filter_to_box(simple_df, polygon, 'latitude', 'longitude')

    ans = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
         [1, 39.984198, 116.319322, '2008-10-23 13:53:06']],
        columns=['uid', 'latitude', 'longitude', 'time']
    )

    assert len(result) == 2
    assert_frame_equal(result.reset_index(drop=True), ans.reset_index(drop=True))


def test_to_projection(simple_df):
    """Test that to_projection correctly converts latitude/longitude to projected x/y."""
    result = to_projection(simple_df, 'latitude', 'longitude')
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


def test_to_projection_and_filter_by_xy(simple_df):
    """Test to_projection followed by filter_to_box in x/y space."""
    projected_df = to_projection(simple_df, 'latitude', 'longitude')
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