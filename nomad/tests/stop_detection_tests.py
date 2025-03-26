import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import pdist, cdist
import pygeohash as gh
import itertools
from collections import defaultdict
import pytest
from pathlib import Path
from nomad import daphmeIO as loader
from nomad import constants
from nomad import filters
import nomad.stop_detection.ta_dbscan as DBSCAN


def extract_user(df, user_id):
    return df[df['user_id'] == user_id]

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
    df['local_datetime'] = loader._unix_offset_to_str(df.timestamp, df.tz_offset)

    # create x, y columns in web mercator
    gdf = gpd.GeoSeries(gpd.points_from_xy(df.longitude, df.latitude),
                            crs="EPSG:4326")
    projected = gdf.to_crs("EPSG:3857")
    df['x'] = projected.x
    df['y'] = projected.y
    
    df['geohash'] = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=7), axis=1)
    # col names:  ['uid', 'timestamp', 'latitude', 'longitude', 'tz_offset', 'local_datetime', 'x', 'y', 'geohash'
    # dtypes: [object, int64, float64, float64, int64, object, float64, float64, object]
    return df

# # Mock test (Test# 0) 
def test_print_sample_df(base_df):
    print(base_df.head())  
    assert not base_df.empty


##########################################
####          LACHESIS TESTS          #### 
##########################################



##########################################
####           DBSCAN TESTS           #### 
##########################################
