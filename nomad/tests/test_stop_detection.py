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

import nomad.io.base as loader
import nomad.constants as constants
import nomad.filters as filters
from stop_detection.ta_dbscan import temporal_dbscan, _temporal_dbscan_labels
from stop_detection.lachesis import lachesis, _lachesis_labels


def extract_user(df, user_id):
    return df[df['user_id'] == user_id]
    
@pytest.fixture
def sample_df_lon_lat_datetime():
    path = "../../data/sample4/"
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = loader.from_file(path, traj_cols=traj_cols, format='csv')
    return df

@pytest.fixture
def sample_df_x_y_timestamp():
    path = "../../data/sample4/"
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = loader.from_file(path, traj_cols=traj_cols, format='csv')
    df = filters.to_projection(df, longitude='dev_lon', latitude='dev_lat')
    df['unix_time'] = df['local_datetime'].astype('int64') // 10**9
    return df


##########################################
####          LACHESIS TESTS          #### 
##########################################

DUR_MIN = 60
DT_MAX = 120
DELTA_ROAM = 50

def test_lon_lat_datetime_lachesis(sample_df_lon_lat_datetime):
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = sample_df_lon_lat_datetime
    user_df = extract_user(df, user_id = 'wonderful_swirles')
    expected_durs = [68.0, 62.0, 60.0, 62.0, 121.0, 499.0]
    expected_labels = [-1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1,
                       -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, -1, -1, -1, 2, 2, 2, 2, 3, 3, 3, 
                       3, 3, 3, 4, 4, 4, 4, 4, 4, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       5, 5, 5, 5, 5, 5]

    actual_stops = sd.lachesis(traj=user_df, dur_min=DUR_MIN, dt_max=DT_MAX, delta_roam=DELTA_ROAM, traj_cols=traj_cols, complete_output=False)
    actual_labels = sd._lachesis_labels(traj=user_df, dur_min=DUR_MIN, dt_max=DT_MAX, delta_roam=DELTA_ROAM, traj_cols=traj_cols)

    assert expected_durs == list(actual_stops['duration'])
    assert expected_labels == list(actual_labels['cluster'])

def test_x_y_timestamp_lachesis(sample_df_x_y_timestamp):
    traj_cols = {'user_id':'user_id', 'x':'x', 'y':'y', 'timestamp':'unix_time'}
    df = sample_df_x_y_timestamp
    user_df = extract_user(df, user_id = 'wonderful_swirles')
    
    expected_durs = [62.0, 60.0, 62.0, 121.0, 499.0]
    expected_labels = [-1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, -1, -1, -1, -1, 0, 0, 0, 0, 0,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, -1, -1, -1, -1, -1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, -1, 4, 4, 4, 4, 4, 4]

    actual_stops = sd.lachesis(traj=user_df, dur_min=DUR_MIN, dt_max=DT_MAX, delta_roam=DELTA_ROAM, traj_cols=traj_cols, complete_output=False)
    actual_labels = sd._lachesis_labels(traj=user_df, dur_min=DUR_MIN, dt_max=DT_MAX, delta_roam=DELTA_ROAM, traj_cols=traj_cols)

    assert expected_durs == list(actual_stops['duration'])
    assert expected_labels == list(actual_labels['cluster'])

##########################################
####           DBSCAN TESTS           #### 
##########################################

TIME_THRESH = 10
DIST_THRESH = 10
MIN_PTS = 2

def test_lon_lat_datetime_dbscan(sample_df_lon_lat_datetime):
    traj_cols = {'user_id':'user_id', 'latitude':'dev_lat', 'longitude':'dev_lon', 'datetime':'local_datetime'}
    df = sample_df_lon_lat_datetime
    user_df = extract_user(df, user_id = 'wonderful_swirles')
    expected_durs = [5.0, 18.0, 12.0, 20.0, 15.0]
    expected_labels = [-1, -1, -1, -1, 4, 4, 4, -1, 
                       3, 3, 3, 3, 3, -1, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, -1, -1, 2, 2, 2, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, 1, 1, 1, 1, 1, -1, -1, 
                       -1, 0, 0, 0, -1, -1, -1]

    actual_stops = sd.temporal_dbscan(user_df, TIME_THRESH, DIST_THRESH, MIN_PTS, traj_cols=traj_cols, complete_output=False)
    actual_labels = sd._temporal_dbscan_labels(user_df, TIME_THRESH, DIST_THRESH, MIN_PTS, traj_cols=traj_cols)

    assert expected_durs == list(actual_stops['duration'])
    assert expected_labels == list(actual_labels['cluster'])

def test_x_y_timestamp_dbscan(sample_df_x_y_timestamp):
    traj_cols = {'user_id':'user_id', 'x':'x', 'y':'y', 'timestamp':'unix_time'}
    df = sample_df_x_y_timestamp
    user_df = extract_user(df, user_id = 'wonderful_swirles')
    
    expected_durs = [5.0, 18.0, 12.0, 20.0, 15.0]
    expected_labels = [-1, -1, -1, -1, 4, 4, 4, -1,
                       3, 3, 3, 3, 3, -1, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, -1, -1, 2, 2, 2, -1, -1,
                       -1, -1, -1, -1, -1, -1, -1, 
                       -1, 1, 1, 1, 1, 1, -1, -1, 
                       -1, 0, 0, 0, -1, -1, -1]

    actual_stops = sd.temporal_dbscan(user_df, TIME_THRESH, DIST_THRESH, MIN_PTS, traj_cols=traj_cols, complete_output=False)
    actual_labels = sd._temporal_dbscan_labels(user_df, TIME_THRESH, DIST_THRESH, MIN_PTS, traj_cols=traj_cols)

    assert expected_durs == list(actual_stops['duration'])
    assert expected_labels == list(actual_labels['cluster'])
