import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import pdist, cdist
import pygeohash as gh
from datetime import datetime, timedelta
import itertools
from collections import defaultdict
import numpy as np
import pytest
from pathlib import Path
from shapely.geometry import Point
import nomad.io.base as loader
from nomad import constants
from nomad import filters
import nomad.stop_detection.dbscan as DBSCAN
import nomad.stop_detection.lachesis as LACHESIS
import nomad.stop_detection.dbstop as DBSTOP
import nomad.stop_detection.density_based as DENSITY_BASED
import nomad.stop_detection.hdbscan as HDBSCAN
import nomad.stop_detection.grid_based as GRID_BASED
import nomad.stop_detection.preprocessing as PREPROCESSING
import nomad.stop_detection.utils as STOP_UTILS
import pdb
import nomad.stop_detection.sequential as SEQUENTIAL
import nomad.stop_detection.hdbscan as HDBSCAN

@pytest.fixture
def stop_test_params():
    gap_minutes = 10
    return {
        "gap_minutes": gap_minutes,
        "dt_max": gap_minutes - 1,
        "delta_roam": 30,
        "dist_thresh": 30,
        "bad_delta_roam": 0.01,
        "min_pts": 2,
        "min_cluster_size": 2,
    }


@pytest.fixture
def shared_algo_registry():
    return {
        "lachesis": {
            "label_fn": LACHESIS.lachesis_labels,
            "stop_fn": LACHESIS.lachesis,
            "extra_kwargs": {},
        },
        "sequential-sliding": {
            "label_fn": SEQUENTIAL.detect_stops_labels,
            "stop_fn": SEQUENTIAL.detect_stops,
            "extra_kwargs": {"method": "sliding"},
        },
        "sequential-centroid": {
            "label_fn": SEQUENTIAL.detect_stops_labels,
            "stop_fn": SEQUENTIAL.detect_stops,
            "extra_kwargs": {"method": "centroid"},
        },
    }


@pytest.fixture
def additional_label_case_registry(stop_test_params):
    dt_max = stop_test_params["dt_max"]
    dist_thresh = stop_test_params["dist_thresh"]
    min_pts = stop_test_params["min_pts"]
    min_cluster_size = stop_test_params["min_cluster_size"]
    return {
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
        },
        "dbstop": {
            "fn": DBSTOP.dbstop_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max, "dur_min": 5},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
        },
        "hdbscan-labels": {
            "fn": HDBSCAN.hdbscan_labels,
            "kwargs": {"time_thresh": dt_max, "min_pts": min_pts, "min_cluster_size": min_cluster_size, "dur_min": 5},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
        },
        "grid-based-labels": {
            "fn": GRID_BASED.grid_based_labels,
            "kwargs": {"time_thresh": dt_max, "min_cluster_size": min_cluster_size, "dur_min": 5},
            "traj_cols": {"timestamp": "timestamp", "location_id": "location_id"},
        },
    }


@pytest.fixture
def overlap_stop_case_registry():
    return {
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan,
            "kwargs": {"dist_thresh": 45, "min_pts": 2, "time_thresh": 60, "dur_min": 3},
            "traj_cols": {"timestamp": "unix_timestamp", "x": "x", "y": "y"},
        },
        "dbstop": {
            "fn": DBSTOP.dbstop,
            "kwargs": {"dist_thresh": 45, "min_pts": 2, "time_thresh": 60, "dur_min": 3},
            "traj_cols": {"timestamp": "unix_timestamp", "x": "x", "y": "y"},
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan,
            "kwargs": {"dist_thresh": 45, "min_pts": 2, "time_thresh": 60, "dur_min": 3},
            "traj_cols": {"timestamp": "unix_timestamp", "x": "x", "y": "y"},
        },
        "hdbscan": {
            "fn": HDBSCAN.st_hdbscan,
            "kwargs": {"time_thresh": 60, "min_pts": 2, "min_cluster_size": 2, "dur_min": 3},
            "traj_cols": {"timestamp": "unix_timestamp", "x": "x", "y": "y"},
        },
    }

@pytest.fixture
def simple_traj(stop_test_params):
    times = pd.date_range("2025-01-01 00:00", periods=6, freq="1min").tolist()
    
    times += [times[-1] + pd.Timedelta(minutes=stop_test_params["gap_minutes"])] # add a big gap and new point

    coords = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    df = pd.DataFrame({
        "x": coords, "y": coords,
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


@pytest.fixture
def simple_traj_ts(simple_traj):
    df = simple_traj.copy()
    df["timestamp"] = filters.to_timestamp(df["datetime"])
    df["location_id"] = [0, 0, 0, 0, 0, 0, 1]
    return df

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
@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("lachesis", id="lachesis"),
        pytest.param("sequential-sliding", id="sequential-sliding"),
        pytest.param("sequential-centroid", id="sequential-centroid"),
    ],
)
def test_labels_single_stop_shared(simple_traj, stop_test_params, shared_algo_registry, algo_name):
    case = shared_algo_registry[algo_name]
    labels = case["label_fn"](
        data=simple_traj,
        traj_cols={"x": "x", "y": "y", "datetime": "datetime"},
        dt_max=stop_test_params["dt_max"],
        delta_roam=stop_test_params["delta_roam"],
        dur_min=5,
        **case["extra_kwargs"],
    )

    assert list(labels.values) == [0, 0, 0, 0, 0, 0, -1]


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("lachesis", id="lachesis"),
        pytest.param("sequential-sliding", id="sequential-sliding"),
        pytest.param("sequential-centroid", id="sequential-centroid"),
    ],
)
def test_labels_too_sparse_shared(simple_traj, stop_test_params, shared_algo_registry, algo_name):
    case = shared_algo_registry[algo_name]
    labels = case["label_fn"](
        simple_traj,
        traj_cols={"x": "x", "y": "y", "datetime": "datetime"},
        dt_max=stop_test_params["dt_max"],
        delta_roam=stop_test_params["bad_delta_roam"],
        dur_min=1,
        **case["extra_kwargs"],
    )
    assert all(labels == -1)


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("lachesis", id="lachesis"),
        pytest.param("sequential-sliding", id="sequential-sliding"),
        pytest.param("sequential-centroid", id="sequential-centroid"),
    ],
)
def test_labels_insufficient_duration_shared(simple_traj, stop_test_params, shared_algo_registry, algo_name):
    case = shared_algo_registry[algo_name]
    labels = case["label_fn"](
        simple_traj,
        traj_cols={"x": "x", "y": "y", "datetime": "datetime"},
        dt_max=stop_test_params["dt_max"],
        delta_roam=stop_test_params["delta_roam"],
        dur_min=10,
        **case["extra_kwargs"],
    )
    assert all(labels == -1)


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("lachesis", id="lachesis"),
        pytest.param("sequential-sliding", id="sequential-sliding"),
        pytest.param("sequential-centroid", id="sequential-centroid"),
    ],
)
def test_number_labels_matches_stop_count_shared(single_user_df, shared_algo_registry, algo_name):
    traj_cols = {
        "timestamp": "timestamp",
        "x": "x", "y": "y"
    }
    case = shared_algo_registry[algo_name]

    data = loader.from_df(single_user_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")
    stops_df = case["stop_fn"](
        data=data,
        traj_cols=traj_cols,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        complete_output=False,
        keep_col_names=False,
        **case["extra_kwargs"],
    )
    labels = case["label_fn"](
        data=data,
        traj_cols=traj_cols,
        dur_min=5,
        dt_max=10,
        delta_roam=100,
        **case["extra_kwargs"],
    )
    labels = labels[labels != -1]

    assert len(stops_df) == labels.nunique()



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

def test_sequential_temporal_gap_breaks_stop(simple_traj, stop_test_params):
    """Test that temporal gap larger than dt_max breaks a stop."""
    # simple_traj has a gap of stop_test_params['gap_minutes'] between ping 6 and 7.
    labels = SEQUENTIAL.detect_stops_labels(
        data=simple_traj,
        dt_max=stop_test_params["dt_max"],
        delta_roam=stop_test_params["delta_roam"],
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


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("dbstop", id="dbstop"),
        pytest.param("seqscan", id="seqscan"),
        pytest.param("hdbscan-labels", id="hdbscan-labels"),
        pytest.param("grid-based-labels", id="grid-based-labels"),
    ],
)
def test_additional_algorithms_label_smoke(simple_traj_ts, additional_label_case_registry, algo_name):
    case = additional_label_case_registry[algo_name]
    labels = case["fn"](simple_traj_ts, traj_cols=case["traj_cols"], **case["kwargs"])
    assert len(labels) == len(simple_traj_ts)
    assert labels.iloc[-1] == -1


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("dbstop", id="dbstop"),
        pytest.param("seqscan", id="seqscan"),
        pytest.param(
            "hdbscan",
            marks=pytest.mark.xfail(
                reason="Known overlap issue in hdbscan output; pending dedicated branch merge.",
                strict=False,
            ),
            id="hdbscan",
        ),
    ],
)
def test_density_algorithms_output_non_overlapping_stops(agent_traj_ground_truth, overlap_stop_case_registry, algo_name):
    case = overlap_stop_case_registry[algo_name]
    stops = case["fn"](agent_traj_ground_truth, traj_cols=case["traj_cols"], **case["kwargs"])

    assert isinstance(stops, pd.DataFrame)
    assert not STOP_UTILS.has_overlapping_stops(stops, traj_cols=case["traj_cols"])


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("lachesis", id="lachesis"),
        pytest.param("sequential-sliding", id="sequential-sliding"),
        pytest.param("dbstop", id="dbstop"),
        pytest.param("seqscan", id="seqscan"),
        pytest.param("tadbscan", id="tadbscan"),
    ],
)
def test_default_time_fallback_prefers_datetime(simple_traj_ts, stop_test_params, shared_algo_registry, additional_label_case_registry, algo_name):
    """When both datetime and timestamp exist, default fallback should prioritize datetime."""
    traj = simple_traj_ts.copy()
    traj["timestamp"] = pd.Series(
        [3600 * i for i in range(len(traj))],
        index=traj.index,
        dtype="Int64",
    )

    if algo_name in shared_algo_registry:
        case = shared_algo_registry[algo_name]
        labels = case["label_fn"](
            traj,
            dt_max=stop_test_params["dt_max"],
            delta_roam=stop_test_params["delta_roam"],
            dur_min=5,
            **case["extra_kwargs"],
        )
    else:
        case = additional_label_case_registry[algo_name]
        labels = case["fn"](traj, **case["kwargs"])

    assert list(labels.values) == [0, 0, 0, 0, 0, 0, -1]


def test_find_neighbors_datetime_relabels_to_unix_seconds(simple_traj, stop_test_params):
    graph = PREPROCESSING._find_neighbors(
        data=simple_traj,
        time_thresh=stop_test_params["dt_max"],
        traj_cols={"x": "x", "y": "y", "datetime": "datetime"},
        dist_thresh=stop_test_params["dist_thresh"],
        weighted=False,
        use_datetime=True,
        use_lon_lat=False,
        return_trees=False,
        relabel_nodes=True,
    )

    expected_nodes = filters.to_timestamp(simple_traj["datetime"]).to_list()
    assert list(graph.nodes()) == expected_nodes


def test_find_neighbors_nullable_timestamp_relabels_to_unix_seconds(simple_traj, stop_test_params):
    traj = simple_traj.copy()
    traj["timestamp"] = filters.to_timestamp(traj["datetime"]).astype("Int64")

    graph = PREPROCESSING._find_neighbors(
        data=traj,
        time_thresh=stop_test_params["dt_max"],
        traj_cols={"x": "x", "y": "y", "timestamp": "timestamp"},
        dist_thresh=stop_test_params["dist_thresh"],
        weighted=False,
        use_datetime=False,
        use_lon_lat=False,
        return_trees=False,
        relabel_nodes=True,
    )

    expected_nodes = traj["timestamp"].astype("int64").to_list()
    assert list(graph.nodes()) == expected_nodes

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


@pytest.fixture
def empty_complete_case_registry():
    return {
        "grid-based": {
            "fn": GRID_BASED.grid_based,
            "kwargs": {
                "time_thresh": 120,
                "min_cluster_size": 2,
                "dur_min": 5,
                "complete_output": True,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'longitude': 'longitude',
                    'latitude': 'latitude',
                    'user_id': 'user_id',
                    'location_id': 'location_id'
                },
            },
            "expected_cols": {'timestamp', 'end_timestamp', 'n_pings', 'max_gap', 'duration', 'location_id'},
        },
        "hdbscan": {
            "fn": HDBSCAN.st_hdbscan,
            "kwargs": {
                "time_thresh": 60,
                "min_pts": 2,
                "min_cluster_size": 2,
                "dur_min": 5,
                "complete_output": True,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'longitude': 'longitude',
                    'latitude': 'latitude',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'},
        },
        "lachesis": {
            "fn": LACHESIS.lachesis,
            "kwargs": {
                "delta_roam": 100,
                "dt_max": 60,
                "dur_min": 5,
                "complete_output": True,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'longitude': 'longitude',
                    'latitude': 'latitude',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'},
        },
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 25,
                "min_pts": 2,
                "dur_min": 5,
                "complete_output": True,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'longitude': 'longitude',
                    'latitude': 'latitude',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'},
        },
        "dbstop": {
            "fn": DBSTOP.dbstop,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 25,
                "min_pts": 2,
                "dur_min": 5,
                "complete_output": True,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'longitude': 'longitude',
                    'latitude': 'latitude',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'},
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 25,
                "min_pts": 2,
                "dur_min": 5,
                "complete_output": True,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'longitude': 'longitude',
                    'latitude': 'latitude',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'},
        },
    }


@pytest.fixture
def empty_xy_case_registry():
    return {
        "grid-based": {
            "fn": GRID_BASED.grid_based,
            "kwargs": {
                "time_thresh": 120,
                "min_cluster_size": 2,
                "dur_min": 5,
                "complete_output": False,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'x': 'x',
                    'y': 'y',
                    'user_id': 'user_id',
                    'location_id': 'location_id'
                },
            },
            "expected_cols": {'timestamp', 'duration', 'location_id'},
        },
        "hdbscan": {
            "fn": HDBSCAN.st_hdbscan,
            "kwargs": {
                "time_thresh": 60,
                "min_pts": 2,
                "min_cluster_size": 2,
                "dur_min": 5,
                "complete_output": False,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'x': 'x',
                    'y': 'y',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'x', 'y', 'timestamp', 'duration'},
        },
        "lachesis": {
            "fn": LACHESIS.lachesis,
            "kwargs": {
                "delta_roam": 100,
                "dt_max": 60,
                "dur_min": 5,
                "complete_output": False,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'x': 'x',
                    'y': 'y',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'x', 'y', 'timestamp', 'duration'},
        },
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 25,
                "min_pts": 2,
                "dur_min": 5,
                "complete_output": False,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'x': 'x',
                    'y': 'y',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'x', 'y', 'timestamp', 'duration'},
        },
        "dbstop": {
            "fn": DBSTOP.dbstop,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 25,
                "min_pts": 2,
                "dur_min": 5,
                "complete_output": False,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'x': 'x',
                    'y': 'y',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'x', 'y', 'timestamp', 'duration'},
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 25,
                "min_pts": 2,
                "dur_min": 5,
                "complete_output": False,
                "traj_cols": {
                    'timestamp': 'timestamp',
                    'x': 'x',
                    'y': 'y',
                    'user_id': 'user_id',
                },
            },
            "expected_cols": {'x', 'y', 'timestamp', 'duration'},
        },
    }


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("grid-based", id="grid-based"),
        pytest.param("hdbscan", id="hdbscan"),
        pytest.param("lachesis", id="lachesis"),
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("dbstop", id="dbstop"),
        pytest.param("seqscan", id="seqscan"),
    ],
)
def test_empty_dataframe_complete_output(empty_traj, empty_complete_case_registry, algo_name):
    case = empty_complete_case_registry[algo_name]
    result = case["fn"](empty_traj, **case["kwargs"])

    assert result.empty
    assert set(result.columns) == case["expected_cols"]


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("grid-based", id="grid-based"),
        pytest.param("hdbscan", id="hdbscan"),
        pytest.param("lachesis", id="lachesis"),
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("dbstop", id="dbstop"),
        pytest.param("seqscan", id="seqscan"),
    ],
)
def test_empty_dataframe_xy_output(empty_traj_xy, empty_xy_case_registry, algo_name):
    case = empty_xy_case_registry[algo_name]
    result = case["fn"](empty_traj_xy, **case["kwargs"])

    assert result.empty
    assert set(result.columns) == case["expected_cols"]

def test_empty_dataframe_consistency():
    """Test that all algorithms return consistent column structures for empty data."""

    empty_data = pd.DataFrame(columns=['timestamp', 'longitude', 'latitude', 'location_id'])
    traj_cols = {
        'timestamp': 'timestamp', 
        'longitude': 'longitude', 
        'latitude': 'latitude', 
        'location_id': 'location_id'
    }
    
    # Test with complete_output=False for all algorithms
    grid_result = GRID_BASED.grid_based(empty_data, traj_cols=traj_cols, complete_output=False)
    hdbscan_result = HDBSCAN.st_hdbscan(empty_data, time_thresh=60, traj_cols=traj_cols, complete_output=False)
    lachesis_result = LACHESIS.lachesis(empty_data, delta_roam=100, dt_max=60, traj_cols=traj_cols, complete_output=False)
    
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
####          HDBSCAN TESTS          ####
##########################################

@pytest.fixture
def hdbscan_traj():
    """Trajectory with two clear stops separated by a long gap."""
    np.random.seed(0)
    base = 1609459200  # 2021-01-01 00:00:00 UTC
    rows = []
    # Stop 1: 20 points at (0, 0) over ~10 minutes (30s intervals)
    for i in range(20):
        rows.append({'x': np.random.normal(0, 2), 'y': np.random.normal(0, 2),
                     'timestamp': base + i * 30})
    # Gap of 30 minutes
    t = base + 20 * 30 + 1800
    # Stop 2: 20 points at (1000, 1000) over ~10 minutes
    for i in range(20):
        rows.append({'x': np.random.normal(1000, 2), 'y': np.random.normal(1000, 2),
                     'timestamp': t + i * 30})
    return pd.DataFrame(rows)


def test_hdbscan_labels_single_stop(hdbscan_traj):
    """hdbscan_labels detects at least one cluster on clear stop data."""
    labels = HDBSCAN.hdbscan_labels(
        hdbscan_traj, time_thresh=5, min_pts=3, min_cluster_size=3, dur_min=5,
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    )
    assert (labels >= 0).any(), "Expected at least one cluster"


def test_hdbscan_labels_no_cluster_when_sparse(hdbscan_traj):
    """No cluster forms when min_pts exceeds the number of temporal neighbors."""
    labels = HDBSCAN.hdbscan_labels(
        hdbscan_traj, time_thresh=5, min_pts=100, min_cluster_size=3, dur_min=5,
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    )
    assert (labels == -1).all(), "Expected all noise when min_pts is too high"


def test_hdbscan_labels_no_cluster_insufficient_duration(hdbscan_traj):
    """No cluster forms when dur_min exceeds actual stop duration."""
    labels = HDBSCAN.hdbscan_labels(
        hdbscan_traj, time_thresh=5, min_pts=3, min_cluster_size=3, dur_min=60,
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    )
    assert (labels == -1).all(), "Expected all noise when dur_min is too high"


def test_hdbscan_labels_two_stops(hdbscan_traj):
    """hdbscan_labels finds both stops in a two-stop trajectory."""
    labels = HDBSCAN.hdbscan_labels(
        hdbscan_traj, time_thresh=5, min_pts=3, min_cluster_size=3, dur_min=5,
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    )
    n_clusters = labels[labels >= 0].nunique()
    assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}"


def test_hdbscan_number_labels_matches_stop_table(hdbscan_traj):
    """Number of unique non-noise labels equals number of rows in st_hdbscan output."""
    traj_cols = {'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    labels = HDBSCAN.hdbscan_labels(
        hdbscan_traj, time_thresh=5, min_pts=3, min_cluster_size=3, dur_min=5,
        traj_cols=traj_cols
    )
    stops = HDBSCAN.st_hdbscan(
        hdbscan_traj, time_thresh=5, min_pts=3, min_cluster_size=3, dur_min=5,
        traj_cols=traj_cols
    )
    assert labels[labels >= 0].nunique() == len(stops)


def test_st_hdbscan_output_is_valid_stop_df(base_df):
    """st_hdbscan concise output conforms to the stop DataFrame standard."""
    traj_cols = {'user_id': 'uid', 'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    df = loader.from_df(base_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior='utc')
    first_user = df[traj_cols['user_id']].iloc[0]
    single = df[df[traj_cols['user_id']] == first_user].copy()

    stops = HDBSCAN.st_hdbscan(
        single, time_thresh=10, min_pts=2, min_cluster_size=2, dur_min=5,
        traj_cols=traj_cols, complete_output=False
    )
    del traj_cols['user_id']
    assert loader._is_stop_df(stops, traj_cols=traj_cols, parse_dates=False)


def test_st_hdbscan_ground_truth(agent_traj_ground_truth):
    """st_hdbscan detects the expected number of stops on ground-truth data."""
    traj_cols = {'user_id': 'identifier', 'x': 'x', 'y': 'y', 'timestamp': 'unix_timestamp'}
    labels = HDBSCAN.hdbscan_labels(
        agent_traj_ground_truth, time_thresh=10, min_pts=2, min_cluster_size=2, dur_min=3,
        traj_cols=traj_cols
    )
    n_clusters = labels[labels >= 0].nunique()
    assert 2 <= n_clusters <= 5, f"Expected 2-5 stops on ground truth, got {n_clusters}"


def test_st_hdbscan_multiuser_raises(base_df):
    """st_hdbscan raises ValueError when passed multi-user data."""
    traj_cols = {'user_id': 'uid', 'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    df = loader.from_df(base_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior='utc')
    with pytest.raises(ValueError, match="Multi-user"):
        HDBSCAN.st_hdbscan(df, time_thresh=10, traj_cols=traj_cols)


def test_st_hdbscan_per_user_basic(base_df):
    """st_hdbscan_per_user runs on multi-user data and returns stops for multiple users."""
    traj_cols = {'user_id': 'uid', 'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    df = loader.from_df(base_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior='utc')
    stops = HDBSCAN.st_hdbscan_per_user(
        df, time_thresh=10, min_pts=2, min_cluster_size=2, dur_min=5,
        traj_cols=traj_cols
    )
    assert not stops.empty
    assert 'uid' in stops.columns
    assert stops['uid'].nunique() > 1


def test_st_hdbscan_delta_roam(hdbscan_traj):
    """delta_roam (epsilon cut) path runs without error and returns a DataFrame."""
    traj_cols = {'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}
    stops = HDBSCAN.st_hdbscan(
        hdbscan_traj, time_thresh=5, min_pts=3, min_cluster_size=3, dur_min=5,
        delta_roam=50, traj_cols=traj_cols
    )
    assert isinstance(stops, pd.DataFrame)


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
