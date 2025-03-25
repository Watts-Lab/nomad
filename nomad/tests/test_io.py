import pytest
import warnings
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pygeohash as gh
import pdb
from nomad.io import base as loader
from nomad import constants

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

@pytest.fixture
def col_variations():
    col_vars = {
        "default-basic":([0, 1, 2, 3], ["user_id", "timestamp", "latitude", "longitude"], ["user_id", "timestamp", "latitude", "longitude"]),
        "alt-names-basic":([0, 1, 2, 3], ["uid", "unix_time", "lat", "lon"], ["user_id", "timestamp", "latitude", "longitude"]),
        "alt-names-dt-xy":([5,6,7], ["event_zoned_datetime", "device_x", "device_y"], ["datetime", "x", "y"]),
        "alt-names-ts-gh":([1,8], ["unix_ts", "geohash_7"], ["timestamp", "geohash"]),
        "default-dt-xy":([0, 4, 5, 6, 7], ["user_id", "tz_offset", "datetime", "x", "y"], [])
    }
    return col_vars

# # Mock test (Test# 0) 
# def test_print_sample_df(base_df):
#     print(base_df.head())  
#     assert not base_df.empty

# from_df simple and provided names (Test # 1)
@pytest.mark.parametrize("variation", ["default-basic", "alt-names-basic", "alt-names-dt-xy", "alt-names-ts-gh", "default-dt-xy"])
def test_from_df_name_handling(base_df, col_variations, variation):
    cols, col_names, keys = col_variations[variation]
    df = base_df.iloc[:, cols]
    df.columns = col_names

    traj_cols = None
    if len(keys)>0:
        traj_cols = dict(zip(keys, col_names))

    result = loader.from_df(df, traj_cols=traj_cols, parse_dates=False)

    assert loader._is_traj_df(result, traj_cols=traj_cols, parse_dates=False), "from_df() output is not a valid trajectory DataFrame"



# from_object has correct values in some entries

# from object fails if no spatial or temporal columns are provided (and no defaults)

# from_file works on data samples, i.e. _is_traj_df for pandas/pyarrow reads

# from_file works with pyspark/pyarrow (without instantiating) until _is_traj_df on data sample 3