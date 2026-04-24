import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import pdist, cdist
import pygeohash as gh
import pytest
from pathlib import Path
import nomad.io.base as loader
from nomad import filters
import nomad.stop_detection.dbscan as DBSCAN
import nomad.stop_detection.lachesis as LACHESIS
import nomad.stop_detection.dbstop as DBSTOP
import nomad.stop_detection.density_based as DENSITY_BASED
import nomad.stop_detection.hdbscan as HDBSCAN
import nomad.stop_detection.grid_based as GRID_BASED
import nomad.stop_detection.preprocessing as PREPROCESSING
import nomad.stop_detection.utils as STOP_UTILS
import nomad.stop_detection.sequential as SEQUENTIAL
from pandas.api.types import is_integer_dtype

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
def fallback_label_case_registry(stop_test_params):
    dt_max = stop_test_params["dt_max"]
    dist_thresh = stop_test_params["dist_thresh"]
    min_pts = stop_test_params["min_pts"]
    return {
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
        },
        "dbstop": {
            "fn": DBSTOP.dbstop_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
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
def latlon_xy_consistency_case_registry():
    return {
        "sequential": {
            "fn": SEQUENTIAL.detect_stops,
            "kwargs": {},
        },
        "lachesis": {
            "fn": LACHESIS.lachesis,
            "kwargs": {
                # Required argument; other behavior uses implementation defaults.
                "delta_roam": 100,
            },
        },
        "dbstop": {
            "fn": DBSTOP.dbstop,
            "kwargs": {
                # Required args; keep notebook-like dbstop settings.
                "time_thresh": 60,
                "dist_thresh": 8,
                "min_pts": 3,
            },
        },
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 45,
                "min_pts": 2,
            },
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan,
            "kwargs": {
                "time_thresh": 60,
                "dist_thresh": 45,
                "min_pts": 2,
            },
        },
        "hdbscan": {
            "fn": HDBSCAN.st_hdbscan,
            "kwargs": {
                "time_thresh": 60,
            },
        },
    }


@pytest.fixture
def latlon_xy_label_consistency_case_registry():
    return {
        "tadbscan-labels": {
            "fn": DBSCAN.ta_dbscan_labels,
            "kwargs": {"dist_thresh": 45, "min_pts": 2, "time_thresh": 60},
        },
        "dbstop-labels": {
            "fn": DBSTOP.dbstop_labels,
            "kwargs": {"dist_thresh": 8, "min_pts": 3, "time_thresh": 60},
        },
        "seqscan-labels": {
            "fn": DENSITY_BASED.seqscan_labels,
            "kwargs": {"dist_thresh": 45},
        },
        "hdbscan-labels": {
            "fn": HDBSCAN.hdbscan_labels,
            "kwargs": {"time_thresh": 60},
        },
    }


@pytest.fixture
def label_concat_case_registry(stop_test_params):
    dt_max = stop_test_params["dt_max"]
    dist_thresh = stop_test_params["dist_thresh"]
    min_pts = stop_test_params["min_pts"]
    return {
        "dbstop": {
            "fn": DBSTOP.dbstop_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
            "supports_return_cores": True,
        },
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
            "supports_return_cores": True,
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan_labels,
            "kwargs": {"dist_thresh": dist_thresh, "min_pts": min_pts, "time_thresh": dt_max},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
            "supports_return_cores": True,
        },
        "lachesis": {
            "fn": LACHESIS.lachesis_labels,
            "kwargs": {"delta_roam": dist_thresh, "dt_max": dt_max, "dur_min": 5},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
            "supports_return_cores": False,
        },
        "sequential": {
            "fn": SEQUENTIAL.detect_stops_labels,
            "kwargs": {"delta_roam": dist_thresh, "dt_max": dt_max, "dur_min": 5, "method": "sliding"},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
            "supports_return_cores": False,
        },
        "hdbscan": {
            "fn": HDBSCAN.hdbscan_labels,
            "kwargs": {"time_thresh": dt_max, "min_pts": min_pts, "min_cluster_size": 2, "dur_min": 5},
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
            "supports_return_cores": False,
        },
        "grid-based": {
            "fn": GRID_BASED.grid_based_labels,
            "kwargs": {"time_thresh": dt_max, "min_cluster_size": 2, "dur_min": 5},
            "traj_cols": {"timestamp": "timestamp", "location_id": "location_id"},
            "supports_return_cores": False,
        },
    }


@pytest.fixture
def stop_df_schema_case_registry():
    return {
        "sequential": {
            "fn": SEQUENTIAL.detect_stops,
            "kwargs": {},
        },
        "lachesis": {
            "fn": LACHESIS.lachesis,
            "kwargs": {"delta_roam": 100},
        },
        "dbstop": {
            "fn": DBSTOP.dbstop,
            "kwargs": {"time_thresh": 60, "dist_thresh": 8, "min_pts": 3},
        },
        "tadbscan": {
            "fn": DBSCAN.ta_dbscan,
            "kwargs": {"time_thresh": 60, "dist_thresh": 45, "min_pts": 2},
        },
        "seqscan": {
            "fn": DENSITY_BASED.seqscan,
            "kwargs": {"time_thresh": 60, "dist_thresh": 45, "min_pts": 2},
        },
        "hdbscan": {
            "fn": HDBSCAN.st_hdbscan,
            "kwargs": {"time_thresh": 60},
        },
    }


@pytest.fixture
def per_user_wrapper_case_registry():
    return {
        "dbstop": {
            "stop_fn": DBSTOP.dbstop_per_user,
            "label_fn": DBSTOP.dbstop_labels_per_user,
            "single_user_label_fn": DBSTOP.dbstop_labels,
            "kwargs": {"dist_thresh": 100, "min_pts": 2, "time_thresh": 60},
        },
        "tadbscan": {
            "stop_fn": DBSCAN.ta_dbscan_per_user,
            "label_fn": DBSCAN.ta_dbscan_labels_per_user,
            "single_user_label_fn": DBSCAN.ta_dbscan_labels,
            "kwargs": {"dist_thresh": 100, "min_pts": 2, "time_thresh": 60},
        },
        "seqscan": {
            "stop_fn": DENSITY_BASED.seqscan_per_user,
            "label_fn": DENSITY_BASED.seqscan_labels_per_user,
            "single_user_label_fn": DENSITY_BASED.seqscan_labels,
            "kwargs": {"dist_thresh": 100, "min_pts": 2, "time_thresh": 60},
        },
        "hdbscan": {
            "stop_fn": HDBSCAN.st_hdbscan_per_user,
            "label_fn": HDBSCAN.hdbscan_labels_per_user,
            "single_user_label_fn": HDBSCAN.hdbscan_labels,
            "kwargs": {"time_thresh": 60, "min_pts": 2, "min_cluster_size": 2, "dur_min": 5},
        },
        "lachesis": {
            "stop_fn": LACHESIS.lachesis_per_user,
            "label_fn": LACHESIS.lachesis_labels_per_user,
            "single_user_label_fn": LACHESIS.lachesis_labels,
            "kwargs": {"dt_max": 60, "delta_roam": 100, "dur_min": 5},
        },
        "sequential": {
            "stop_fn": SEQUENTIAL.detect_stops_per_user,
            "label_fn": SEQUENTIAL.detect_stops_labels_per_user,
            "single_user_label_fn": SEQUENTIAL.detect_stops_labels,
            "kwargs": {"dt_max": 60, "delta_roam": 100, "dur_min": 5, "method": "sliding"},
        },
    }


@pytest.fixture
def stop_df_schema_input_case_registry():
    return {
        "xy-timestamp": {
            "cols": ["timestamp", "x", "y"],
            "traj_cols": {"timestamp": "timestamp", "x": "x", "y": "y"},
        },
        "latlon-timestamp": {
            "cols": ["timestamp", "longitude", "latitude"],
            "traj_cols": {"timestamp": "timestamp", "longitude": "longitude", "latitude": "latitude"},
        },
        "xy-datetime": {
            "cols": ["local_datetime", "x", "y"],
            "traj_cols": {"datetime": "local_datetime", "x": "x", "y": "y"},
        },
        "latlon-datetime": {
            "cols": ["local_datetime", "longitude", "latitude"],
            "traj_cols": {"datetime": "local_datetime", "longitude": "longitude", "latitude": "latitude"},
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
def per_user_test_data(base_df):
    traj_cols = {
        "user_id": "uid",
        "timestamp": "timestamp",
        "x": "x",
        "y": "y",
    }
    selected_users = base_df[traj_cols["user_id"]].drop_duplicates().head(4)
    sample_df = base_df[base_df[traj_cols["user_id"]].isin(selected_users)].copy()
    data = loader.from_df(sample_df, traj_cols=traj_cols, parse_dates=True, mixed_timezone_behavior="utc")
    return data, traj_cols

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
@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("sequential", id="sequential"),
        pytest.param("lachesis", id="lachesis"),
        pytest.param("dbstop", id="dbstop"),
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("seqscan", id="seqscan"),
        pytest.param("hdbscan", id="hdbscan"),
    ],
)
@pytest.mark.parametrize(
    "input_case",
    [
        pytest.param("xy-timestamp", id="xy-timestamp"),
        pytest.param("latlon-timestamp", id="latlon-timestamp"),
        pytest.param("xy-datetime", id="xy-datetime"),
        pytest.param("latlon-datetime", id="latlon-datetime"),
    ],
)
def test_stop_output_is_valid_stop_df(base_df, stop_df_schema_case_registry, stop_df_schema_input_case_registry, algo_name, input_case):
    case = stop_df_schema_case_registry[algo_name]
    input_cfg = stop_df_schema_input_case_registry[input_case]

    first_user = base_df["uid"].iloc[0]
    single_user = base_df[base_df["uid"] == first_user].head(1500).copy()
    input_df = single_user[input_cfg["cols"]].copy()

    if "datetime" in input_cfg["traj_cols"]:
        dt_col = input_cfg["traj_cols"]["datetime"]
        input_df[dt_col] = pd.to_datetime(input_df[dt_col], utc=True)

    stops_df = case["fn"](
        data=input_df,
        traj_cols=input_cfg["traj_cols"],
        complete_output=False,
        **case["kwargs"],
    )

    assert loader._is_stop_df(stops_df, traj_cols=input_cfg["traj_cols"], parse_dates=False)

   
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
        dt_max=60,
        delta_roam=100,
        complete_output=False,
        keep_col_names=False,
        **case["extra_kwargs"],
    )
    labels = case["label_fn"](
        data=data,
        traj_cols=traj_cols,
        dur_min=5,
        dt_max=60,
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


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("dbstop", id="dbstop"),
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("seqscan", id="seqscan"),
        pytest.param("hdbscan", id="hdbscan"),
        pytest.param("lachesis", id="lachesis"),
        pytest.param("sequential", id="sequential"),
    ],
)
@pytest.mark.parametrize(
    "n_jobs",
    [
        pytest.param(1, id="n_jobs_1"),
        pytest.param(2, id="n_jobs_2"),
    ],
)
def test_per_user_wrapper_outputs_valid_stop_df(per_user_test_data, per_user_wrapper_case_registry, algo_name, n_jobs):
    df, traj_cols = per_user_test_data
    case = per_user_wrapper_case_registry[algo_name]

    stops_df = case["stop_fn"](
        data=df,
        traj_cols=traj_cols,
        n_jobs=n_jobs,
        **case["kwargs"],
    )

    assert not stops_df.empty
    assert loader._is_stop_df(stops_df, traj_cols=traj_cols, parse_dates=False)
    assert traj_cols["user_id"] in stops_df.columns
    assert stops_df[traj_cols["user_id"]].nunique() > 1


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("dbstop", id="dbstop"),
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("seqscan", id="seqscan"),
        pytest.param("hdbscan", id="hdbscan"),
        pytest.param("lachesis", id="lachesis"),
        pytest.param("sequential", id="sequential"),
    ],
)
@pytest.mark.parametrize(
    "n_jobs",
    [
        pytest.param(1, id="n_jobs_1"),
        pytest.param(2, id="n_jobs_2"),
    ],
)
def test_per_user_label_wrapper_matches_single_user_reference(per_user_test_data, per_user_wrapper_case_registry, algo_name, n_jobs):
    df, traj_cols = per_user_test_data
    case = per_user_wrapper_case_registry[algo_name]

    labels = case["label_fn"](
        data=df,
        traj_cols=traj_cols,
        n_jobs=n_jobs,
        **case["kwargs"],
    )

    uid = traj_cols["user_id"]
    ts = traj_cols["timestamp"]

    expected_labels = pd.Series(index=df.index, dtype=labels.dtype)
    for _, group in df.groupby(uid, sort=False):
        expected_labels.loc[group.index] = case["single_user_label_fn"](
            data=group,
            traj_cols=traj_cols,
            **case["kwargs"],
        ).values

    computed = pd.DataFrame({
        uid: df[uid].values,
        ts: df[ts].values,
        "label": labels.values,
    })
    expected = pd.DataFrame({
        uid: df[uid].values,
        ts: df[ts].values,
        "label": expected_labels.values,
    })

    key_cols = [uid, ts]
    assert not computed.duplicated(key_cols).any()
    assert not expected.duplicated(key_cols).any()

    merged = computed.merge(
        expected,
        on=key_cols,
        how="inner",
        validate="one_to_one",
        suffixes=("_computed", "_expected"),
    )

    assert len(labels) == len(df)
    assert labels.index.equals(df.index)
    assert len(merged) == len(df)
    assert (merged["label_computed"] == merged["label_expected"]).all()

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

@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("sequential", id="sequential"),
        pytest.param("lachesis", id="lachesis"),
        pytest.param("dbstop", id="dbstop"),
        pytest.param("tadbscan", id="tadbscan"),
        pytest.param("seqscan", id="seqscan"),
        pytest.param("hdbscan", id="hdbscan"),
    ],
)
def test_stop_detection_latlon_vs_xy_consistency(base_df, latlon_xy_consistency_case_registry, algo_name):
    """Test that algorithms run with both lat/lon and x/y coordinates without errors."""
    # Get single user data
    first_user = base_df['uid'].iloc[0]
    single_user = base_df[base_df['uid'] == first_user].head(1500).copy()
    case = latlon_xy_consistency_case_registry[algo_name]
    
    # Test with lat/lon
    df_latlon = loader.from_df(
        single_user[['timestamp', 'latitude', 'longitude']],
        traj_cols={'timestamp': 'timestamp', 'latitude': 'latitude', 'longitude': 'longitude'},
        parse_dates=True,
        mixed_timezone_behavior="utc"
    )
    
    stops_latlon = case["fn"](
        data=df_latlon,
        traj_cols={'timestamp': 'timestamp', 'latitude': 'latitude', 'longitude': 'longitude'},
        **case["kwargs"],
    )
    
    # Test with x/y (note: delta_roam needs adjustment for different units)
    df_xy = loader.from_df(
        single_user[['timestamp', 'x', 'y']],
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'},
        parse_dates=True,
        mixed_timezone_behavior="utc"
    )
    
    stops_xy = case["fn"](
        data=df_xy,
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'},
        **case["kwargs"],
    )
    
    # Main test is that both run without error.
    assert isinstance(stops_latlon, pd.DataFrame)
    assert isinstance(stops_xy, pd.DataFrame)


@pytest.mark.parametrize(
    "algo_name",
    [
        pytest.param("tadbscan-labels", id="tadbscan-labels"),
        pytest.param("dbstop-labels", id="dbstop-labels"),
        pytest.param("seqscan-labels", id="seqscan-labels"),
        pytest.param("hdbscan-labels", id="hdbscan-labels"),
    ],
)
def test_density_label_algorithms_latlon_vs_xy_consistency(base_df, latlon_xy_label_consistency_case_registry, algo_name):
    first_user = base_df['uid'].iloc[0]
    single_user = base_df[base_df['uid'] == first_user].head(1500).copy()
    case = latlon_xy_label_consistency_case_registry[algo_name]

    labels_latlon = case["fn"](
        single_user[['timestamp', 'latitude', 'longitude']],
        traj_cols={'timestamp': 'timestamp', 'latitude': 'latitude', 'longitude': 'longitude'},
        **case["kwargs"],
    )
    labels_xy = case["fn"](
        single_user[['timestamp', 'x', 'y']],
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'},
        **case["kwargs"],
    )

    assert len(labels_latlon) == len(single_user)
    assert len(labels_xy) == len(single_user)


@pytest.mark.parametrize(
    "algo_name,return_cores",
    [
        pytest.param("dbstop", False, id="dbstop-series"),
        pytest.param("dbstop", True, id="dbstop-cores"),
        pytest.param("tadbscan", False, id="tadbscan-series"),
        pytest.param("tadbscan", True, id="tadbscan-cores"),
        pytest.param("seqscan", False, id="seqscan-series"),
        pytest.param("seqscan", True, id="seqscan-cores"),
        pytest.param("lachesis", False, id="lachesis-series"),
        pytest.param("sequential", False, id="sequential-series"),
        pytest.param("hdbscan", False, id="hdbscan-series"),
        pytest.param("grid-based", False, id="grid-based-series"),
    ],
)
def test_label_concat_with_empty_input_preserves_integer_schema(
    simple_traj_ts,
    label_concat_case_registry,
    algo_name,
    return_cores,
):
    case = label_concat_case_registry[algo_name]
    if return_cores and not case["supports_return_cores"]:
        pytest.skip("Algorithm does not expose return_cores.")

    non_empty = simple_traj_ts.copy()
    empty = non_empty.iloc[:0].copy()

    kwargs = dict(case["kwargs"])
    if case["supports_return_cores"]:
        kwargs["return_cores"] = return_cores

    labels_non_empty = case["fn"](non_empty, traj_cols=case["traj_cols"], **kwargs)
    labels_empty = case["fn"](empty, traj_cols=case["traj_cols"], **kwargs)

    concatenated = pd.concat([labels_empty, labels_non_empty])

    assert len(concatenated) == len(labels_non_empty)
    assert concatenated.index.equals(labels_non_empty.index)

    if return_cores:
        assert isinstance(labels_empty, pd.DataFrame)
        assert list(labels_empty.columns) == ["cluster", "core"]
        assert isinstance(concatenated, pd.DataFrame)
        assert list(concatenated.columns) == ["cluster", "core"]
        assert is_integer_dtype(concatenated["cluster"].dtype)
        assert is_integer_dtype(concatenated["core"].dtype)
    else:
        assert isinstance(labels_empty, pd.Series)
        assert labels_empty.name == "cluster"
        assert isinstance(concatenated, pd.Series)
        assert concatenated.name == "cluster"
        assert is_integer_dtype(concatenated.dtype)


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
def test_default_time_fallback_prefers_datetime(simple_traj_ts, stop_test_params, shared_algo_registry, fallback_label_case_registry, algo_name):
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
        case = fallback_label_case_registry[algo_name]
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
        "sequential": {
            "fn": SEQUENTIAL.detect_stops,
            "kwargs": {
                "delta_roam": 100,
                "dt_max": 60,
                "dur_min": 5,
                "method": "sliding",
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
        "sequential": {
            "fn": SEQUENTIAL.detect_stops,
            "kwargs": {
                "delta_roam": 100,
                "dt_max": 60,
                "dur_min": 5,
                "method": "sliding",
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
        pytest.param("sequential", id="sequential"),
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
        pytest.param("sequential", id="sequential"),
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
