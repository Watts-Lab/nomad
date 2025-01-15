import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import pdist, cdist
import numpy as np
import itertools
from collections import defaultdict
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import daphmeIO as loader
import constants as constants
import filters as filters
import stop_detection_modified as SD


def extract_user(df, user_id):
    return df[df['user_id'] == user_id]
    
@pytest.fixture
def sample_df_lon_lat_datetime():
    path = '../data/sample4/'
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = loader.from_file(path, traj_cols=traj_cols, format='csv')
    return df

def sample_df_lon_lat_timestamp():
    path = '../data/sample4/'
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = loader.from_file(path, traj_cols=traj_cols, format='csv')
    df['unix_time'] = df['local_datetime'].astype('int64') // 10**9
    return df

@pytest.fixture
def sample_df_x_y_datetime():
    path = '../data/sample4/'
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = loader.from_file(path, traj_cols=traj_cols, format='csv')
    df = filters.to_projection(df, latitude='dev_lat', longitude='dev_lon')
    return df

@pytest.fixture
def sample_df_x_y_timestamp():
    path = '../data/sample4/'
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = loader.from_file(path, traj_cols=traj_cols, format='csv')
    df = filters.to_projection(df, latitude='dev_lat', longitude='dev_lon')
    df['unix_time'] = df['local_datetime'].astype('int64') // 10**9
    return df


##########################################
####          LACHESIS TESTS          #### 
##########################################

DUR_MIN = 100
DT_MAX = 120
DELTA_ROAM = 10

def test_lon_lat_datetime_lachesis(sample_df_lon_lat_datetime):
    traj_cols = {'user_id':'user_id',
                 'latitude':'dev_lat',
                 'longitude':'dev_lon',
                 'datetime':'local_datetime'}

    df = sample_df_lon_lat_datetime
    user_df = extract_user(df, user_id = 'youthful_galileo')
    
    try:
        result = SD.lachesis(user_df, DUR_MIN, DT_MAX, DELTA_ROAM, traj_cols=traj_cols, complete_output=False)
    except Exception as e:
        pytest.fail(f"lachesis raised an error: {e}")

    
def test_lon_lat_timestamp_lachesis(sample_df_lon_lat_timestamp):
    traj_cols = {'user_id':'user_id',
                 'latitude':'dev_lat',
                 'longitude':'dev_lon',
                 'timestamp':'unix_time'}
    
    df = sample_df_lon_lat_timestamp
    user_df = extract_user(df, user_id = 'youthful_galileo')
    
    try:
        result = SD.lachesis(user_df, DUR_MIN, DT_MAX, DELTA_ROAM, traj_cols=traj_cols, complete_output=False)
    except Exception as e:
        pytest.fail(f"lachesis raised an error: {e}")

def test_x_y_datetime_lachesis(sample_df_x_y_datetime):
    traj_cols = {'user_id':'user_id',
                 'x':'x',
                 'y':'y',
                 'datetime':'local_datetime'}

    df = sample_df_x_y_datetime
    user_df = extract_user(df, user_id = 'youthful_galileo')
    
    try:
        result = SD.lachesis(user_df, DUR_MIN, DT_MAX, DELTA_ROAM, traj_cols=traj_cols, complete_output=False)
    except Exception as e:
        pytest.fail(f"lachesis raised an error: {e}")


def test_x_y_timestamp_lachesis(sample_df_x_y_timestamp):
    traj_cols = {'user_id':'user_id',
                 'x':'x',
                 'y':'y',
                 'timestamp':'unix_time'}

    df = sample_df_x_y_timestamp
    user_df = extract_user(sample_df_x_y_timestamp, user_id = 'youthful_galileo')
    
    try:
        result = SD.lachesis(user_df, DUR_MIN, DT_MAX, DELTA_ROAM, traj_cols=traj_cols, complete_output=False)
    except Exception as e:
        pytest.fail(f"lachesis raised an error: {e}")




