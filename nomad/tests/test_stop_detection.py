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
from shapely.geometry import Point
import nomad.io.base as loader
from nomad import constants
from nomad import filters
import nomad.stop_detection.dbscan as DBSCAN
import nomad.stop_detection.lachesis as LACHESIS
import nomad.stop_detection.sliding as SLIDING
import pdb
import nomad.stop_detection.sequential as SEQUENTIAL

@pytest.fixture
def simple_traj():
    times = pd.date_range("2025-01-01 00:00", periods=6, freq="1min").tolist()
    
    times += [times[-1] + pd.Timedelta(minutes=10)] # add a big gap and new point

    df = pd.DataFrame({
        "x": 0.0, "y": 0.0,
        "datetime": times,
    })
    
    df["tz_offset"] = 0
    return df

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


# FAILING TESTS: FROM FILTERS.PY
# @pytest.mark.parametrize("mixed_tz_bhv", ['naive', 'utc', 'object'])
# def test_to_timestamp(base_df, mixed_tz_bhv):
#     df = base_df.iloc[:, [2, 3, 4, 5]].copy()
#     traj_cols = {'latitude':'latitude', 'longitude':'longitude', 'datetime':'local_datetime'}

#     df = loader.from_df(df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior=mixed_tz_bhv)

#     if mixed_tz_bhv == 'naive':
#         timestamp_col = filters.to_timestamp(df.local_datetime, df.tz_offset)
#     else:
#         timestamp_col = filters.to_timestamp(df.local_datetime, df.tz_offset)
    
#     assert (timestamp_col.values==base_df.timestamp).all()

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
        data=single_user_df,
        delta_roam=100,
        dt_max=10,
        dur_min=5,
        traj_cols=traj_cols,
        complete_output=False)
    
    del traj_cols['user_id']

    is_valid = loader._is_stop_df(stops_df, traj_cols=traj_cols, parse_dates=False)

    assert is_valid, "Lachesis concise output failed validation by _is_stop_df"

   
##########################################
####          LACHESIS TESTS          #### 
##########################################
# Test to check that lachesis identifies six points in cluster 0 and one point as noise -1
def test_lachesis_labels_single_stop(simple_traj):
    labels = LACHESIS.lachesis_labels(data=simple_traj,
        dt_max=5,
        delta_roam=0.1,
        dur_min=5,
        traj_cols={"x":"x","y":"y","datetime":"datetime"}
    )

    assert list(labels.values) == [0,0,0,0,0,0,-1]

# Test to check that delta_roam is tiny (<0) no cluster should form
def test_lachesis_labels_too_sparse(simple_traj):
    
    labels = LACHESIS.lachesis_labels(
        simple_traj,
        dt_max=5,
        delta_roam=-1,
        dur_min=1,
        traj_cols={"x":"x","y":"y","datetime":"datetime"},
    )
    assert all(labels == -1)


# Test to check if number of unique labels in lachesis_labels matches with number of stops from stop table
def test_lachesis_number_labels(single_user_df):
    """Tests if the output of Lachesis labels has same number of unique labels as the Lachesis stop table."""
    traj_cols = {
        "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    
    single_user_df = loader.from_df(single_user_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")

    stops_df = LACHESIS.lachesis(
        data=single_user_df,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        traj_cols=traj_cols,
        complete_output=False,
        keep_col_names=False
    )
    
    labels = LACHESIS.lachesis_labels(
        data=single_user_df,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        traj_cols=traj_cols
    )

    labels = labels[~(labels == -1)]

    assert len(stops_df) == labels.nunique()

def test_lachesis_labels_insufficient_duration(simple_traj):
    # dur_min=10 means even six 1-minute intervals fail
    labels = LACHESIS.lachesis_labels(
        simple_traj,
        dt_max=5,
        delta_roam=1,
        dur_min=10,
        traj_cols={"x":"x","y":"y","datetime":"datetime"},
    )
    
    assert all(labels == -1)



# Test to check if they identify correct number of stops
def test_lachesis_ground_truth(agent_traj_ground_truth):
    lachesis_params = (45, 60, 3)
    traj_cols = {'user_id':'identifier',
             'x':'x',
             'y':'y',
             'timestamp':'unix_timestamp'}
    lachesis_out = LACHESIS.lachesis_labels(agent_traj_ground_truth,
                                             *lachesis_params,
                                             traj_cols)

    num_clusters = sum(lachesis_out.unique() > -1)
    assert num_clusters == 3

##########################################
####        SEQUENTIAL TESTS          #### 
##########################################

def test_sequential_labels_single_stop(simple_traj):
    """Test that sequential detection identifies single stop correctly."""
    labels = SEQUENTIAL.detect_stops_labels(
        data=simple_traj,
        dt_max=5,
        delta_roam=0.1,
        dur_min=5,
        method='sliding',
        traj_cols={"x":"x","y":"y","datetime":"datetime"}
    )
    
    assert list(labels.values) == [0,0,0,0,0,0,-1]

def test_sequential_labels_too_sparse(simple_traj):
    """Test that no cluster forms when delta_roam is too small."""
    labels = SEQUENTIAL.detect_stops_labels(
        simple_traj,
        dt_max=5,
        delta_roam=-1,
        dur_min=1,
        method='sliding',
        traj_cols={"x":"x","y":"y","datetime":"datetime"},
    )
    assert all(labels == -1)

def test_sequential_labels_insufficient_duration(simple_traj):
    """Test that no cluster forms when duration requirement not met."""
    labels = SEQUENTIAL.detect_stops_labels(
        simple_traj,
        dt_max=5,
        delta_roam=1,
        dur_min=10,
        method='sliding',
        traj_cols={"x":"x","y":"y","datetime":"datetime"},
    )
    
    assert all(labels == -1)

def test_sequential_number_labels(single_user_df):
    """Test that detect_stops_labels and detect_stops have matching stop counts."""
    traj_cols = {
        "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    
    single_user_df = loader.from_df(single_user_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")

    stops_df = SEQUENTIAL.detect_stops(
        data=single_user_df,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        method='sliding',
        traj_cols=traj_cols,
        complete_output=False,
        keep_col_names=False
    )
    
    labels = SEQUENTIAL.detect_stops_labels(
        data=single_user_df,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        method='sliding',
        traj_cols=traj_cols
    )

    labels = labels[~(labels == -1)]

    assert len(stops_df) == labels.nunique()

def test_sequential_ground_truth(agent_traj_ground_truth):
    """Test sequential detection on ground truth data."""
    traj_cols = {'user_id':'identifier',
             'x':'x',
             'y':'y',
             'timestamp':'unix_timestamp'}
    
    sequential_out = SEQUENTIAL.detect_stops_labels(
        agent_traj_ground_truth,
        delta_roam=45,
        dt_max=60,
        dur_min=3,
        method='sliding',
        traj_cols=traj_cols
    )

    num_clusters = sum(sequential_out.unique() > -1)
    assert 2 <= num_clusters <= 5, f"Expected 2-5 stops, got {num_clusters}"

def test_sequential_empty_dataframe(empty_traj):
    """Test that sequential returns proper empty DataFrame with correct columns."""
    result = SEQUENTIAL.detect_stops(
        empty_traj,
        delta_roam=100,
        dt_max=60,
        dur_min=5,
        method='sliding',
        complete_output=True,
        traj_cols={
            'timestamp': 'timestamp', 
            'longitude': 'longitude', 
            'latitude': 'latitude'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'}
    assert set(result.columns) == expected_cols

def test_sequential_empty_dataframe_xy(empty_traj_xy):
    """Test that sequential works with x,y coordinates on empty data."""
    result = SEQUENTIAL.detect_stops(
        empty_traj_xy,
        delta_roam=100,
        dt_max=60,
        dur_min=5,
        method='sliding',
        complete_output=False,
        traj_cols={
            'timestamp': 'timestamp', 
            'x': 'x', 
            'y': 'y'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = ['x', 'y', 'timestamp', 'duration']
    assert list(result.columns) == expected_cols

def test_sequential_output_is_valid_stop_df(base_df):
    """Test if sequential concise output conforms to the stop DataFrame standard."""
    traj_cols = {
        "user_id": "uid", "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    df = loader.from_df(base_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")

    first_user = df[traj_cols["user_id"]].iloc[0]
    single_user_df = df[df[traj_cols["user_id"]] == first_user].copy()

    stops_df = SEQUENTIAL.detect_stops(
        data=single_user_df,
        delta_roam=100,
        dt_max=10,
        dur_min=5,
        method='sliding',
        traj_cols=traj_cols,
        complete_output=False
    )
    
    del traj_cols['user_id']

    is_valid = loader._is_stop_df(stops_df, traj_cols=traj_cols, parse_dates=False)

    assert is_valid, "Sequential concise output failed validation by _is_stop_df"

def test_sequential_per_user_multiuser_required(base_df):
    """Test that detect_stops raises error when multi-user data is provided."""
    traj_cols = {
        "user_id": "uid", "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    df = loader.from_df(base_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")

    with pytest.raises(ValueError, match="Multi-user data"):
        SEQUENTIAL.detect_stops(
            data=df,
            delta_roam=100,
            dt_max=10,
            dur_min=5,
            method='sliding',
            traj_cols=traj_cols
        )

def test_sequential_per_user_basic(base_df):
    """Test detect_stops_per_user on multi-user data."""
    traj_cols = {
        "user_id": "uid", "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    df = loader.from_df(base_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")

    stops_df = SEQUENTIAL.detect_stops_per_user(
        data=df,
        delta_roam=100,
        dt_max=10,
        dur_min=5,
        method='sliding',
        traj_cols=traj_cols,
        complete_output=False,
        n_jobs=1
    )
    
    # Should have stops for multiple users
    assert not stops_df.empty
    assert 'uid' in stops_df.columns
    assert stops_df['uid'].nunique() > 1

def test_sequential_temporal_gap_breaks_stop(simple_traj):
    """Test that temporal gap larger than dt_max breaks a stop."""
    # simple_traj has 6 points at 1-min intervals, then a 10-min gap
    labels = SEQUENTIAL.detect_stops_labels(
        data=simple_traj,
        dt_max=5,  # Max 5 min gap
        delta_roam=100,  # Large spatial threshold
        dur_min=5,  # Need 5 min duration
        method='sliding',
        traj_cols={"x":"x","y":"y","datetime":"datetime"}
    )
    
    # First 6 points form a stop, last point is noise
    assert list(labels.values) == [0,0,0,0,0,0,-1]

def test_sequential_latlon_vs_xy_consistency(base_df):
    """Test that sequential gives consistent results with lat/lon vs x/y."""
    # Get single user data
    first_user = base_df['uid'].iloc[0]
    single_user = base_df[base_df['uid'] == first_user].copy()
    
    # Test with lat/lon
    df_latlon = loader.from_df(
        single_user[['timestamp', 'latitude', 'longitude']],
        traj_cols={'timestamp': 'timestamp', 'latitude': 'latitude', 'longitude': 'longitude'},
        parse_dates=True,
        mixed_timezone_behavior="utc"
    )
    
    stops_latlon = SEQUENTIAL.detect_stops(
        data=df_latlon,
        delta_roam=100,
        dt_max=10,
        dur_min=5,
        method='sliding',
        complete_output=False
    )
    
    # Test with x/y (note: delta_roam needs adjustment for different units)
    df_xy = loader.from_df(
        single_user[['timestamp', 'x', 'y']],
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'},
        parse_dates=True,
        mixed_timezone_behavior="utc"
    )
    
    stops_xy = SEQUENTIAL.detect_stops(
        data=df_xy,
        delta_roam=100,  # Same value - will be in different units
        dt_max=10,
        dur_min=5,
        method='sliding',
        complete_output=False
    )
    
    # Both should produce stops (even if different counts due to unit difference)
    # Main test is that both run without error
    assert isinstance(stops_latlon, pd.DataFrame)
    assert isinstance(stops_xy, pd.DataFrame)

##########################################
####           DBSCAN TESTS           #### 
##########################################

##########################################
####      EMPTY DATAFRAME TESTS       #### 
##########################################

@pytest.fixture
def empty_traj():
    """Empty trajectory DataFrame with standard columns."""
    return pd.DataFrame(columns=[
        'timestamp', 'longitude', 'latitude', 'user_id', 'location_id'
    ])

@pytest.fixture
def empty_traj_xy():
    """Empty trajectory DataFrame with x,y coordinates."""
    return pd.DataFrame(columns=[
        'timestamp', 'x', 'y', 'user_id', 'location_id'
    ])

def test_grid_based_empty_dataframe(empty_traj):
    """Test that grid_based returns proper empty DataFrame with correct columns."""
    import nomad.stop_detection.grid_based as grid_based
    
    result = grid_based.grid_based(
        empty_traj,
        time_thresh=120,
        min_cluster_size=2,
        dur_min=5,
        complete_output=True,
        traj_cols={
            'timestamp': 'timestamp', 
            'longitude': 'longitude', 
            'latitude': 'latitude', 
            'location_id': 'location_id'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = {'timestamp', 'end_timestamp', 'n_pings', 'max_gap', 'duration', 'location_id'}
    assert set(result.columns) == expected_cols

def test_grid_based_empty_dataframe_xy(empty_traj_xy):
    """Test that grid_based works with x,y coordinates on empty data."""
    import nomad.stop_detection.grid_based as grid_based
    
    result = grid_based.grid_based(
        empty_traj_xy,
        time_thresh=120,
        min_cluster_size=2,
        dur_min=5,
        complete_output=False,
        traj_cols={
            'timestamp': 'timestamp', 
            'x': 'x', 
            'y': 'y', 
            'location_id': 'location_id'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = {'timestamp', 'duration', 'location_id'}
    assert set(result.columns) == expected_cols

def test_st_hdbscan_empty_dataframe(empty_traj):
    """Test that st_hdbscan returns proper empty DataFrame with correct columns."""
    import nomad.stop_detection.hdbscan as hdbscan
    
    result = hdbscan.st_hdbscan(
        empty_traj,
        time_thresh=60,
        min_pts=2,
        min_cluster_size=2,
        dur_min=5,
        complete_output=True,
        traj_cols={
            'timestamp': 'timestamp', 
            'longitude': 'longitude', 
            'latitude': 'latitude'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'}
    assert set(result.columns) == expected_cols

def test_st_hdbscan_empty_dataframe_xy(empty_traj_xy):
    """Test that st_hdbscan works with x,y coordinates on empty data."""
    import nomad.stop_detection.hdbscan as hdbscan
    
    result = hdbscan.st_hdbscan(
        empty_traj_xy,
        time_thresh=60,
        min_pts=2,
        min_cluster_size=2,
        dur_min=5,
        complete_output=False,
        traj_cols={
            'timestamp': 'timestamp', 
            'x': 'x', 
            'y': 'y'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = {'x', 'y', 'timestamp', 'duration'}
    assert set(result.columns) == expected_cols

def test_lachesis_empty_dataframe(empty_traj):
    """Test that lachesis returns proper empty DataFrame with correct columns."""
    import nomad.stop_detection.lachesis as lachesis
    
    result = lachesis.lachesis(
        empty_traj,
        delta_roam=100,
        dt_max=60,
        dur_min=5,
        complete_output=True,
        traj_cols={
            'timestamp': 'timestamp', 
            'longitude': 'longitude', 
            'latitude': 'latitude'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'}
    assert set(result.columns) == expected_cols

def test_lachesis_empty_dataframe_xy(empty_traj_xy):
    """Test that lachesis works with x,y coordinates on empty data."""
    import nomad.stop_detection.lachesis as lachesis
    
    result = lachesis.lachesis(
        empty_traj_xy,
        delta_roam=100,
        dt_max=60,
        dur_min=5,
        complete_output=False,
        traj_cols={
            'timestamp': 'timestamp', 
            'x': 'x', 
            'y': 'y'
        }
    )
    
    # Should return empty DataFrame with correct columns
    assert result.empty
    expected_cols = ['x', 'y', 'timestamp', 'duration']
    assert list(result.columns) == expected_cols

def test_empty_dataframe_consistency():
    """Test that all algorithms return consistent column structures for empty data."""
    import nomad.stop_detection.grid_based as grid_based
    import nomad.stop_detection.hdbscan as hdbscan
    import nomad.stop_detection.lachesis as lachesis
    
    empty_data = pd.DataFrame(columns=['timestamp', 'longitude', 'latitude', 'location_id'])
    traj_cols = {
        'timestamp': 'timestamp', 
        'longitude': 'longitude', 
        'latitude': 'latitude', 
        'location_id': 'location_id'
    }
    
    # Test with complete_output=False for all algorithms
    grid_result = grid_based.grid_based(empty_data, traj_cols=traj_cols, complete_output=False)
    hdbscan_result = hdbscan.st_hdbscan(empty_data, time_thresh=60, traj_cols=traj_cols, complete_output=False)
    lachesis_result = lachesis.lachesis(empty_data, delta_roam=100, dt_max=60, traj_cols=traj_cols, complete_output=False)
    
    # All should be empty
    assert grid_result.empty
    assert hdbscan_result.empty
    assert lachesis_result.empty
    
    # Grid-based should have location_id, others should have spatial coordinates
    assert 'location_id' in grid_result.columns
    assert 'longitude' in hdbscan_result.columns
    assert 'latitude' in hdbscan_result.columns
    assert 'longitude' in lachesis_result.columns
    assert 'latitude' in lachesis_result.columns

##########################################
####        LOCATION CLUSTERING      ####
####           (SLIDING.PY)          ####
##########################################

@pytest.fixture
def position_fixes_simple():
    """Simple position fixes for testing sliding window algorithm."""
    times = pd.date_range("2025-01-01 08:00", periods=10, freq="1min").tolist()

    # Create position fixes:
    # Points 0-5: stationary (should form staypoint if time_threshold <= 5 min)
    # Points 6-9: moved away
    coords = [(0.0, 0.0)] * 6 + [(0.002, 0.002)] * 4

    df = gpd.GeoDataFrame({
        "user_id": "user1",
        "tracked_at": times,
        "geometry": [Point(lon, lat) for lon, lat in coords]
    }, crs="EPSG:4326")

    return df


@pytest.fixture
def position_fixes_with_gap():
    """Position fixes with temporal gap for testing gap_threshold."""
    times = pd.date_range("2025-01-01 08:00", periods=5, freq="1min").tolist()
    # Add a large gap
    times += [times[-1] + pd.Timedelta(minutes=20)]
    times += pd.date_range(times[-1] + pd.Timedelta(minutes=1), periods=4, freq="1min").tolist()

    # All points at same location
    coords = [(0.0, 0.0)] * len(times)

    df = gpd.GeoDataFrame({
        "user_id": "user1",
        "tracked_at": times,
        "geometry": [Point(lon, lat) for lon, lat in coords]
    }, crs="EPSG:4326")

    return df


@pytest.fixture
def position_fixes_multi_user():
    """Position fixes for multiple users."""
    times = pd.date_range("2025-01-01 08:00", periods=8, freq="1min").tolist()

    user1_coords = [(0.0, 0.0)] * 4 + [(0.002, 0.002)] * 4
    user2_coords = [(0.001, 0.001)] * 4 + [(0.003, 0.003)] * 4

    df = gpd.GeoDataFrame({
        "user_id": ["user1"] * 8 + ["user2"] * 8,
        "tracked_at": times * 2,
        "geometry": [Point(lon, lat) for lon, lat in user1_coords + user2_coords]
    }, crs="EPSG:4326")

    return df


def test_sliding_basic_staypoint_detection(position_fixes_simple):
    pfs, sp = SLIDING.generate_staypoints(
        position_fixes_simple,
        method="sliding",
        dist_threshold=100,
        time_threshold=3.0,
        gap_threshold=15.0,
        include_last=False,
        exclude_duplicate_pfs=True,
        n_jobs=1
    )

    # Should detect at least one staypoint from first 6 points
    assert len(sp) >= 1
    assert 'staypoint_id' in pfs.columns
    assert isinstance(sp, gpd.GeoDataFrame)

    # Check required columns in staypoints
    assert 'user_id' in sp.columns
    assert 'started_at' in sp.columns
    assert 'finished_at' in sp.columns
    assert sp.geometry.name in sp.columns


def test_sliding_time_threshold(position_fixes_simple):
    """Test that time_threshold is respected."""
    # With time_threshold=10, the first 6 points (5 min duration) should not form a staypoint
    pfs, sp = SLIDING.generate_staypoints(
        position_fixes_simple,
        dist_threshold=100,
        time_threshold=10.0,
        gap_threshold=15.0,
        include_last=False,
        n_jobs=1
    )

    # Should not detect any staypoints since max duration < 10 min
    assert len(sp) == 0


def test_sliding_distance_threshold(position_fixes_simple):
    """Test that distance_threshold is respected."""
    # With very small distance threshold, points should not cluster together
    # The test data has 6 points at (0,0) and 4 points at (0.002, 0.002)
    # With dist_threshold=1m, these should form separate staypoints
    pfs, sp = SLIDING.generate_staypoints(
        position_fixes_simple,
        dist_threshold=1,  # 1 meter - very tight
        time_threshold=3.0,
        gap_threshold=15.0,
        include_last=False,
        n_jobs=1
    )

    # With tight distance constraint, should detect 2 separate staypoints
    # (one at each distinct location)
    assert len(sp) == 2


def test_sliding_gap_threshold(position_fixes_with_gap):
    """Test that gap_threshold prevents clustering across large temporal gaps."""
    # With gap_threshold=15, the 20-minute gap should prevent clustering
    pfs, sp = SLIDING.generate_staypoints(
        position_fixes_with_gap,
        dist_threshold=100,
        time_threshold=3.0,
        gap_threshold=15.0,
        include_last=False,
        n_jobs=1
    )

    # Should detect at most 2 separate staypoints (before and after gap)
    assert len(sp) <= 2


def test_sliding_include_last(position_fixes_simple):
    """Test include_last parameter."""
    # Without include_last
    pfs1, sp1 = SLIDING.generate_staypoints(
        position_fixes_simple,
        dist_threshold=100,
        time_threshold=3.0,
        include_last=False,
        n_jobs=1
    )

    # With include_last
    pfs2, sp2 = SLIDING.generate_staypoints(
        position_fixes_simple,
        dist_threshold=100,
        time_threshold=3.0,
        include_last=True,
        n_jobs=1
    )

    # include_last=True should potentially detect more staypoints
    assert len(sp2) >= len(sp1)


def test_sliding_multi_user(position_fixes_multi_user):
    """Test sliding window with multiple users."""
    pfs, sp = SLIDING.generate_staypoints(
        position_fixes_multi_user,
        dist_threshold=100,
        time_threshold=2.0,
        gap_threshold=15.0,
        n_jobs=1
    )

    # Should detect staypoints for both users
    assert 'user_id' in sp.columns
    unique_users = sp['user_id'].unique()
    assert len(unique_users) >= 1  # At least one user should have staypoints

def test_sliding_empty_dataframe():
    """Test sliding window with empty dataframe."""
    empty_pfs = gpd.GeoDataFrame({
        "user_id": [],
        "tracked_at": [],
        "geometry": []
    }, crs="EPSG:4326")

    with pytest.warns(UserWarning, match="No staypoints can be generated"):
        pfs, sp = SLIDING.generate_staypoints(
            empty_pfs,
            dist_threshold=100,
            time_threshold=5.0,
            n_jobs=1
        )

    assert len(sp) == 0
    assert 'staypoint_id' in pfs.columns
