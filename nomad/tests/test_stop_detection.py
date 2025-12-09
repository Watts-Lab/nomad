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
import nomad.stop_detection.dbscan as DBSCAN
import nomad.stop_detection.lachesis as LACHESIS
import nomad.stop_detection.sliding as SLIDING
import pdb

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
def stops_for_clustering():
    """Create sample stops for location clustering tests"""
    # Create stops at two distinct locations
    # Location 1: stops at (0, 0) with small variations
    # Location 2: stops at (1000, 1000) with small variations
    stops_data = {
        'x': [0, 5, 10, 1000, 1005, 1010, 2000],  # Third location with single stop
        'y': [0, 5, 10, 1000, 1005, 1010, 2000],
        'timestamp': [
            1704067200,  # 2024-01-01 00:00:00
            1704070800,  # 2024-01-01 01:00:00
            1704074400,  # 2024-01-01 02:00:00
            1704078000,  # 2024-01-01 03:00:00
            1704081600,  # 2024-01-01 04:00:00
            1704085200,  # 2024-01-01 05:00:00
            1704088800,  # 2024-01-01 06:00:00
        ],
        'duration': [10, 15, 20, 10, 15, 20, 10],  # minutes
        'user_id': ['user1', 'user1', 'user1', 'user1', 'user1', 'user1', 'user1']
    }
    return pd.DataFrame(stops_data)


@pytest.fixture
def multi_user_stops():
    """Create sample stops for multiple users"""
    stops_data = {
        'x': [0, 5, 10, 0, 5, 10],
        'y': [0, 5, 10, 0, 5, 10],
        'timestamp': [1704067200 + i*3600 for i in range(6)],
        'duration': [10, 15, 20, 10, 15, 20],
        'user_id': ['user1', 'user1', 'user1', 'user2', 'user2', 'user2']
    }
    return pd.DataFrame(stops_data)


def test_cluster_locations_basic(stops_for_clustering):
    """Test basic location clustering with default parameters"""
    traj_cols = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_dbscan(
        stops_for_clustering,
        epsilon=50,
        num_samples=1,
        distance_metric='euclidean',
        agg_level='dataset',
        traj_cols=traj_cols
    )
    
    # Should create 3 locations (2 clusters + 1 single stop at 2000,2000)
    assert len(locations) == 3
    
    # All stops should have location_id assigned
    assert 'location_id' in stops_labeled.columns
    
    # Check that first 3 stops belong to same location
    loc_ids = stops_labeled['location_id'].values
    assert loc_ids[0] == loc_ids[1] == loc_ids[2]
    
    # Check that stops 3,4,5 belong to same location (different from first)
    assert loc_ids[3] == loc_ids[4] == loc_ids[5]
    assert loc_ids[3] != loc_ids[0]
    
    # Check locations have proper columns
    assert 'center' in locations.columns
    assert 'extent' in locations.columns
    assert 'n_stops' in locations.columns


def test_cluster_locations_per_user(multi_user_stops):
    """Test per-user location clustering"""
    traj_cols = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_dbscan(
        multi_user_stops,
        epsilon=50,
        num_samples=1,
        distance_metric='euclidean',
        agg_level='user',
        traj_cols=traj_cols
    )
    
    # Should create 2 locations (one per user, both at same spatial location)
    assert len(locations) == 2
    
    # Check that each user has their own location
    assert 'user_id' in locations.columns
    user1_locs = locations[locations['user_id'] == 'user1']
    user2_locs = locations[locations['user_id'] == 'user2']
    assert len(user1_locs) == 1
    assert len(user2_locs) == 1
    
    # Check that user1 stops have different location_id than user2 stops
    user1_stops = stops_labeled[stops_labeled['user_id'] == 'user1']
    user2_stops = stops_labeled[stops_labeled['user_id'] == 'user2']
    assert user1_stops['location_id'].iloc[0] != user2_stops['location_id'].iloc[0]


def test_cluster_locations_dataset_level(multi_user_stops):
    """Test dataset-level location clustering (shared across users)"""
    traj_cols = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_dbscan(
        multi_user_stops,
        epsilon=50,
        num_samples=1,
        distance_metric='euclidean',
        agg_level='dataset',
        traj_cols=traj_cols
    )
    
    # Should create 1 shared location
    assert len(locations) == 1
    
    # All stops should have same location_id
    location_ids = stops_labeled['location_id'].dropna().unique()
    assert len(location_ids) == 1


def test_cluster_locations_empty_input():
    """Test handling of empty DataFrame"""
    empty_df = pd.DataFrame(columns=['x', 'y', 'timestamp', 'user_id'])
    traj_cols = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_dbscan(
        empty_df,
        epsilon=50,
        num_samples=1,
        traj_cols=traj_cols
    )
    
    assert len(stops_labeled) == 0
    assert len(locations) == 0
    assert 'location_id' in stops_labeled.columns


def test_cluster_locations_geographic_coords():
    """Test clustering with lat/lon coordinates using haversine metric"""
    # Create stops near SF (37.7749° N, 122.4194° W)
    stops_data = {
        'longitude': [-122.4194, -122.4195, -122.4196, -122.5000],  # ~100m apart, then far
        'latitude': [37.7749, 37.7750, 37.7751, 37.8000],
        'timestamp': [1704067200 + i*3600 for i in range(4)],
        'duration': [10, 15, 20, 10],
        'user_id': ['user1'] * 4
    }
    stops_df = pd.DataFrame(stops_data)
    
    traj_cols = {'longitude': 'longitude', 'latitude': 'latitude', 
                 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_dbscan(
        stops_df,
        epsilon=200,  # 200 meters
        num_samples=2,
        distance_metric='haversine',
        agg_level='dataset',
        traj_cols=traj_cols
    )
    
    # Should cluster first 3 stops, last one is noise
    assert len(locations) == 1
    assert stops_labeled.loc[0, 'location_id'] == stops_labeled.loc[1, 'location_id']
    assert pd.isna(stops_labeled.loc[3, 'location_id'])


def test_cluster_locations_per_user_convenience():
    """Test the cluster_locations_per_user convenience function"""
    stops_data = {
        'x': [0, 5, 10, 0, 5, 10],
        'y': [0, 5, 10, 0, 5, 10],
        'timestamp': [1704067200 + i*3600 for i in range(6)],
        'duration': [10, 15, 20, 10, 15, 20],
        'user_id': ['user1', 'user1', 'user1', 'user2', 'user2', 'user2']
    }
    stops_df = pd.DataFrame(stops_data)
    
    traj_cols = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_per_user(
        stops_df,
        epsilon=50,
        num_samples=1,
        traj_cols=traj_cols
    )
    
    # Should behave same as agg_level='user'
    assert len(locations) == 2
    assert 'user_id' in locations.columns


def test_location_center_calculation(stops_for_clustering):
    """Test that location centers are calculated correctly"""
    traj_cols = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_dbscan(
        stops_for_clustering,
        epsilon=50,
        num_samples=2,
        distance_metric='euclidean',
        agg_level='dataset',
        traj_cols=traj_cols
    )
    
    # First location should be centered around (5, 5) - mean of [0,5,10]
    # Second location should be centered around (1005, 1005)
    centers = [(loc.center.x, loc.center.y) for _, loc in locations.iterrows()]
    
    # Check first location center is near (5, 5)
    assert any(abs(cx - 5) < 10 and abs(cy - 5) < 10 for cx, cy in centers)
    
    # Check second location center is near (1005, 1005)
    assert any(abs(cx - 1005) < 10 and abs(cy - 1005) < 10 for cx, cy in centers)


def test_location_extent_calculation(stops_for_clustering):
    """Test that location extents (convex hull + buffer) are created"""
    traj_cols = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    
    stops_labeled, locations = SLIDING.cluster_locations_dbscan(
        stops_for_clustering,
        epsilon=50,
        num_samples=1,
        distance_metric='euclidean',
        agg_level='dataset',
        traj_cols=traj_cols
    )
    
    # All locations should have extent geometries
    assert all(locations['extent'].notna())
    
    # Extents should be polygons (or points buffered to polygons)
    from shapely.geometry import Polygon, Point
    assert all(isinstance(ext, Polygon) or isinstance(ext, Point) 
               for ext in locations['extent'])
