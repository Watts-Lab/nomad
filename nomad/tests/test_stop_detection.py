import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import pdist, cdist
import pygeohash as gh
from datetime import datetime, timedelta
import itertools
from collections import defaultdict
import pytest
from pathlib import Path
import nomad.io.base as loader
from nomad import constants
from nomad import filters
import nomad.stop_detection.ta_dbscan as DBSCAN
import nomad.stop_detection.lachesis as LACHESIS
import pdb

@pytest.fixture
def agent_traj_ground_truth():
    test_dir = Path(__file__).resolve().parent
    traj_path = test_dir.parent / "data" / "gc_3_stops.csv"
    df = loader.from_file(traj_path, timestamp='unix_timestamp', datetime='local_timestamp', user_id='identifier')
    return df

@pytest.fixture
def base_df():
    test_dir = Path(__file__).resolve().parent
    data_path = test_dir.parent / "data" / "gc_sample.csv"
    df = pd.read_csv(data_path)

    # create tz_offset column
    df['tz_offset'] = -3600
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
def single_user_df(base_df):
    uid = base_df.uid.iloc[0]
    return base_df[base_df.uid == uid].copy().drop(columns=['uid'])

@pytest.fixture
def column_variations():
    return {
        # 0->timestamp, 1->latitude, 2->longitude, 3->tz_offset, 4->local_datetime, 5->x, 6->y, 7->geohash
        "default-timestamp-xy": ([0, 5, 6], ["timestamp", "x", "y"], ["timestamp", "x", "y"]),
        "alt-timestamp-xy": ([0, 5, 6], ["unix_time", "merc_x", "merc_y"], ["timestamp", "x", "y"]),
        # note we want [longitude, latitude] for latlon
        "default-datetime-latlon": ([4, 2, 1], ["local_datetime", "longitude", "latitude"], ["datetime", "longitude", "latitude"]),
        "alt-datetime-latlon": ([4, 2, 1], ["event_time", "lon", "lat"], ["datetime", "longitude", "latitude"]),
        "explicit-start-datetime": ([4, 2, 1], ["start_dt_col", "lon", "lat"], ["start_datetime", "longitude", "latitude"])
    }

@pytest.mark.parametrize("mixed_tz_bhv", ['naive', 'utc', 'object'])
def test_to_timestamp(base_df, mixed_tz_bhv):
    df = base_df.iloc[:, [2, 3, 4, 5]].copy()
    traj_cols = {'latitude':'latitude', 'longitude':'longitude', 'datetime':'local_datetime'}

    df = loader.from_df(df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior=mixed_tz_bhv)

    if mixed_tz_bhv == 'naive':
        timestamp_col = filters.to_timestamp(df.local_datetime, df.tz_offset)
    else:
        timestamp_col = filters.to_timestamp(df.local_datetime, base_df.tz_offset)
    
    assert (timestamp_col.values==base_df.timestamp).all()

## ============================================= TESTS =========================================
def test_lachesis_output_is_valid_stop_df(base_df):
    """Tests if Lachesis concise output conforms to the stop DataFrame standard."""
    traj_cols = {
        "user_id": "uid", "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    df = loader.from_df(base_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")

    first_user = df[traj_cols["user_id"]].iloc[0]
    single_user_df = df[df[traj_cols["user_id"]] == first_user].copy()

    stops_df = LACHESIS.lachesis(
        traj=single_user_df,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        traj_cols=traj_cols,
        complete_output=False
    )
    del traj_cols['user_id']
    is_valid = loader._is_stop_df(stops_df, traj_cols=traj_cols, parse_dates=False)

    assert is_valid, "Lachesis concise output failed validation by _is_stop_df"


@pytest.mark.parametrize("variation_key", [
    "default-timestamp-xy",
    "alt-timestamp-xy",
    "default-datetime-latlon",
    "alt-datetime-latlon",
    "explicit-start-datetime",
])
@pytest.mark.parametrize("complete_output", [True, False])
@pytest.mark.parametrize("keep_col_names", [True, False])
def test_lachesis_name_handling_from_variants(single_user_df, column_variations, variation_key, complete_output, keep_col_names):
    indices, new_names, keys = column_variations[variation_key]
    df_subset = single_user_df.iloc[:, indices].copy()
    df_subset.columns = new_names

    # Build traj_cols from the keys => new_names
    traj_cols = dict(zip(keys, new_names))

    result = LACHESIS.lachesis(
        traj=df_subset,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        traj_cols=traj_cols,
        complete_output=complete_output,
        keep_col_names=keep_col_names
    )

    use_dt = keys[0] in ('datetime', 'start_datetime')
    uses_xy = 'x' in keys or 'y' in keys

    check_cols = {}
    if use_dt:
        check_cols['start_datetime'] = 'start_datetime'
    else:
        check_cols['start_timestamp'] = 'start_timestamp'

    if complete_output:
        if use_dt:
            check_cols['end_datetime'] = 'end_datetime'
        else:
            check_cols['end_timestamp'] = 'end_timestamp'

    check_cols['duration'] = 'duration'

    if uses_xy:
        check_cols['x'] = 'x'
        check_cols['y'] = 'y'
    else:
        check_cols['longitude'] = 'longitude'
        check_cols['latitude'] = 'latitude'

    final_traj_cols = traj_cols.copy() if keep_col_names else None

    assert loader._is_stop_df(
        result,
        traj_cols=final_traj_cols,
        parse_dates='datetime' in keys or 'start_datetime' in keys
    )

    start_key = 'start_datetime' if use_dt else 'start_timestamp'
    end_key = 'end_datetime' if use_dt else 'end_timestamp'

    expected_start = (
        traj_cols.get(keys[0]) if keep_col_names else check_cols[start_key]
    )

    if uses_xy:
        coord1_key, coord2_key = 'x', 'y'
    else:
        coord1_key, coord2_key = 'longitude', 'latitude'

    if keep_col_names:
        expected_coord1 = traj_cols.get(coord1_key)
        expected_coord2 = traj_cols.get(coord2_key)
    else:
        expected_coord1 = check_cols[coord1_key]
        expected_coord2 = check_cols[coord2_key]

    expected_columns = [expected_start, 'duration', expected_coord1, expected_coord2]
    if complete_output:
        end_val = check_cols[end_key]
        expected_columns = [expected_start, end_val, 'duration', expected_coord1, expected_coord2, 'diameter', 'n_pings', 'max_gap']

    actual_cols = list(result.columns)
    assert actual_cols == expected_columns, f"For {variation_key}, c={complete_output}, keep={keep_col_names}, got {actual_cols}, expected {expected_columns}"

    
##########################################
####          LACHESIS TESTS          #### 
##########################################
# Test to see if number of unique labels in lachesis_labels matches with number of stops from stop table
def test_lachesis_number_labels(single_user_df):
    """Tests if the output of Lachesis labels has same number of unique labels as the Lachesis stop table."""
    traj_cols = {
        "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    
    single_user_df = loader.from_df(single_user_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")

    stops_df = LACHESIS.lachesis(
        traj=single_user_df,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        traj_cols=traj_cols,
        complete_output=False,
        keep_col_names=False
    )
    
    labels = LACHESIS._lachesis_labels(
        traj=single_user_df,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        traj_cols=traj_cols
    )

    labels = labels[~(labels == -1)]

    assert len(stops_df) == labels.nunique()

# Test to see if they identify correct number of stops
def test_lachesis_ground_truth(agent_traj_ground_truth):
    lachesis_params = (45, 60, 3)
    traj_cols = {'user_id':'identifier',
             'x':'x',
             'y':'y',
             'timestamp':'unix_timestamp'}
    lachesis_out = LACHESIS._lachesis_labels(agent_traj_ground_truth,
                                             *lachesis_params,
                                             traj_cols)

    num_clusters = sum(lachesis_out.unique() > -1)
    assert num_clusters == 3

##########################################
####           DBSCAN TESTS           #### 
##########################################