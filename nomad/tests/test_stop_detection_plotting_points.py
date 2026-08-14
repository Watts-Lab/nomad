import inspect

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from nomad.stop_detection import density_algs, sequential_algs


ANCHOR_COLUMNS = ["cluster", "anchor_time", "anchor_x", "anchor_y"]


@pytest.fixture
def single_user_trajectory():
    data = pd.DataFrame(
        {
            "uid": ["user-1"] * 6,
            "timestamp": [0, 120, 240, 360, 480, 600],
            "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "y": [0.0] * 6,
        }
    )
    traj_cols = {
        "user_id": "uid",
        "timestamp": "timestamp",
        "x": "x",
        "y": "y",
    }
    return data, traj_cols


@pytest.mark.parametrize(
    ("method", "expected_times", "expected_x"),
    [
        ("sliding", [0] * 6, [0.0] * 6),
        ("centroid", [0, 120, 240, 360, 480, 600], [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]),
    ],
)
def test_sequential_return_anchors_returns_aligned_metadata(
    single_user_trajectory,
    method,
    expected_times,
    expected_x,
):
    data, traj_cols = single_user_trajectory

    anchor_points = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method=method,
        return_anchors=True,
        traj_cols=traj_cols,
    )

    assert list(anchor_points.columns) == ANCHOR_COLUMNS
    assert anchor_points.index.equals(data.index)
    assert len(anchor_points) == len(data)
    assert not anchor_points.empty
    assert anchor_points["anchor_time"].tolist() == expected_times
    assert anchor_points["anchor_x"].tolist() == pytest.approx(expected_x)
    assert anchor_points["anchor_y"].tolist() == [0.0] * 6

    labels_without_points = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method=method,
        traj_cols=traj_cols,
    )
    pd.testing.assert_series_equal(anchor_points["cluster"], labels_without_points)


def test_sequential_return_anchors_includes_noise(single_user_trajectory):
    data, traj_cols = single_user_trajectory

    anchor_points = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=20,
        return_anchors=True,
        traj_cols=traj_cols,
    )

    assert len(anchor_points) == len(data)
    assert set(anchor_points["cluster"]) == {-1}
    assert anchor_points[["anchor_time", "anchor_x", "anchor_y"]].isna().all().all()


def test_sequential_return_anchors_empty_schema(single_user_trajectory):
    data, traj_cols = single_user_trajectory

    anchor_points = sequential_algs.detect_stops_labels(
        data.iloc[:0],
        return_anchors=True,
        traj_cols=traj_cols,
    )

    assert anchor_points.empty
    assert list(anchor_points.columns) == ANCHOR_COLUMNS
    assert is_integer_dtype(anchor_points["cluster"])


def test_anchor_output_uses_canonical_metadata_names(single_user_trajectory):
    data, traj_cols = single_user_trajectory
    traj_cols = {
        **traj_cols,
        "start_timestamp": "anchor_timestamp",
        "label": "stop_label",
    }
    data = data.assign(anchor_timestamp=pd.NA, stop_label=pd.NA)

    anchor_points = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        return_anchors=True,
        traj_cols=traj_cols,
    )

    assert list(anchor_points.columns) == ANCHOR_COLUMNS


def test_anchor_time_preserves_datetime_values(single_user_trajectory):
    data, traj_cols = single_user_trajectory
    data = data.assign(datetime=pd.to_datetime(data["timestamp"], unit="s", utc=True))
    traj_cols = {**traj_cols, "datetime": "datetime"}
    traj_cols.pop("timestamp")

    anchor_points = sequential_algs.detect_stops_labels(
        data.drop(columns="timestamp"),
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method="sliding",
        return_anchors=True,
        traj_cols=traj_cols,
    )

    assert anchor_points["anchor_time"].tolist() == [data.loc[0, "datetime"]] * len(data)


def test_lachesis_does_not_advertise_anchor_or_core_output():
    parameters = inspect.signature(sequential_algs.lachesis_labels).parameters

    assert "return_anchors" not in parameters
    assert "return_cores" not in parameters


def test_grid_based_does_not_advertise_core_output():
    for function in (
        sequential_algs.grid_based_labels,
        sequential_algs.grid_based_labels_per_user,
    ):
        parameters = inspect.signature(function).parameters
        assert "return_cores" not in parameters
        assert "config_key" not in parameters


def test_point_output_algorithms_do_not_advertise_config_key():
    for function in (
        sequential_algs.detect_stops_labels,
        sequential_algs.detect_stops_labels_per_user,
        density_algs.ta_dbscan_labels,
        density_algs.ta_dbscan_labels_per_user,
        density_algs.dbstop_labels,
        density_algs.dbstop_labels_per_user,
        density_algs.seqscan_labels,
        density_algs.seqscan_labels_per_user,
        density_algs.hdbscan_labels,
        density_algs.hdbscan_labels_per_user,
    ):
        assert "config_key" not in inspect.signature(function).parameters


@pytest.mark.parametrize(
    ("label_function", "editable_parameters"),
    [
        (
            sequential_algs.detect_stops_labels,
            {"delta_roam", "dt_max", "dur_min", "method", "return_anchors"},
        ),
        (
            sequential_algs.lachesis_labels,
            {"dt_max", "delta_roam", "dur_min"},
        ),
        (
            sequential_algs.grid_based_labels,
            {"time_thresh", "min_cluster_size", "dur_min"},
        ),
        (
            density_algs.ta_dbscan_labels,
            {"dist_thresh", "min_pts", "time_thresh", "remove_overlaps", "return_cores"},
        ),
        (
            density_algs.dbstop_labels,
            {"dist_thresh", "min_pts", "time_thresh", "return_cores"},
        ),
        (
            density_algs.seqscan_labels,
            {"dist_thresh", "dur_min", "time_thresh", "min_pts", "back_merge", "return_cores"},
        ),
        (
            density_algs.hdbscan_labels,
            {
                "time_thresh",
                "min_pts",
                "min_cluster_size",
                "dur_min",
                "delta_roam",
                "dist_thresh",
                "return_cores",
            },
        ),
    ],
)
def test_algorithm_parameters_remain_editable(label_function, editable_parameters):
    assert editable_parameters <= set(inspect.signature(label_function).parameters)


@pytest.mark.parametrize(
    ("label_function", "kwargs"),
    [
        (
            density_algs.ta_dbscan_labels,
            {"dist_thresh": 10, "min_pts": 2, "time_thresh": 5},
        ),
        (
            density_algs.dbstop_labels,
            {"dist_thresh": 10, "min_pts": 2, "time_thresh": 5},
        ),
        (
            density_algs.seqscan_labels,
            {"dist_thresh": 10, "min_pts": 2, "time_thresh": 5},
        ),
        (
            density_algs.hdbscan_labels,
            {
                "time_thresh": 5,
                "min_pts": 2,
                "min_cluster_size": 2,
                "dur_min": 5,
            },
        ),
    ],
)
def test_density_return_cores_returns_label_metadata(
    single_user_trajectory,
    label_function,
    kwargs,
):
    data, traj_cols = single_user_trajectory

    core_points = label_function(
        data,
        return_cores=True,
        traj_cols=traj_cols,
        **kwargs,
    )

    assert len(core_points) == len(data)
    assert list(core_points.columns) == ["cluster", "core", "promotion_time"]
    assert is_integer_dtype(core_points["cluster"])
    assert is_integer_dtype(core_points["core"])
    assert core_points.loc[core_points["cluster"] < 0, "promotion_time"].isna().all()
    assert core_points.loc[core_points["core"] >= 0, "promotion_time"].notna().all()

    labels = label_function(data, traj_cols=traj_cols, **kwargs)
    pd.testing.assert_series_equal(
        core_points["cluster"],
        labels,
    )


@pytest.mark.parametrize(
    ("label_function", "kwargs", "expected_promotion_times"),
    [
        (density_algs.ta_dbscan_labels, {}, [120, 120, 120]),
        (density_algs.dbstop_labels, {}, [120, 120, 240]),
        (density_algs.seqscan_labels, {"dur_min": 0}, [240, 240, 240]),
    ],
)
def test_density_promotion_time_records_propagation(
    label_function,
    kwargs,
    expected_promotion_times,
):
    data = pd.DataFrame(
        {
            "uid": ["user-1"] * 3,
            "timestamp": [0, 120, 240],
            "x": [0.0, 0.4, 0.8],
            "y": [0.0] * 3,
        }
    )
    traj_cols = {
        "user_id": "uid",
        "timestamp": "timestamp",
        "x": "x",
        "y": "y",
    }

    output = label_function(
        data,
        dist_thresh=0.5,
        min_pts=2,
        time_thresh=5,
        return_cores=True,
        traj_cols=traj_cols,
        **kwargs,
    )

    assert output["cluster"].tolist() == [0, 0, 0]
    assert output.loc[0, "core"] < 0
    assert output.loc[1, "core"] == 0
    assert output.loc[2, "core"] < 0
    assert output["promotion_time"].tolist() == expected_promotion_times


@pytest.mark.parametrize("use_datetime", [False, True])
def test_hdbscan_promotion_time_identifies_claiming_core(monkeypatch, use_datetime):
    input_times = (
        pd.to_datetime([0, 120, 240], unit="s", utc=True)
        if use_datetime
        else [0, 120, 240]
    )
    time_column = "datetime" if use_datetime else "timestamp"
    data = pd.DataFrame(
        {
            "uid": ["user-1"] * 3,
            time_column: input_times,
            "x": [0.0, 0.1, 0.2],
            "y": [0.0] * 3,
        }
    )
    traj_cols = {
        "user_id": "uid",
        time_column: time_column,
        "x": "x",
        "y": "y",
    }
    core_distances = pd.Series([2.0, 1.0, 2.0], index=[0, 120, 240])
    label_history = pd.DataFrame(
        {"cluster_id": [7], "dendogram_scale": [1.0], "time": [120]}
    )

    monkeypatch.setattr(density_algs, "_compute_core_distance", lambda *_: core_distances)
    monkeypatch.setattr(
        density_algs,
        "_build_hdbscan_graphs",
        lambda *_: (None, pd.Series(dtype="float64")),
    )
    monkeypatch.setattr(
        density_algs,
        "cluster_hierarchy",
        lambda **_: (label_history, pd.DataFrame()),
    )
    monkeypatch.setattr(
        density_algs,
        "compute_cluster_stability",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        density_algs,
        "select_most_stable_clusters",
        lambda *_: [7],
    )
    monkeypatch.setattr(
        density_algs,
        "_borders_from_cores",
        lambda *_: {120: {0, 240}},
    )

    output = density_algs.hdbscan_labels(
        data,
        time_thresh=5,
        min_pts=2,
        min_cluster_size=2,
        dur_min=0,
        return_cores=True,
        traj_cols=traj_cols,
    )

    assert output["cluster"].tolist() == [7, 7, 7]
    assert output["core"].tolist() == [-1, 7, -1]
    assert output["promotion_time"].tolist() == [input_times[1]] * 3


def test_density_promotion_time_uses_original_datetime_values():
    data = pd.DataFrame(
        {
            "uid": ["user-1"] * 3,
            "datetime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:02", "2024-01-01 00:04"],
                utc=True,
            ),
            "x": [0.0, 0.4, 0.8],
            "y": [0.0] * 3,
        }
    )
    traj_cols = {
        "user_id": "uid",
        "datetime": "datetime",
        "x": "x",
        "y": "y",
    }

    output = density_algs.dbstop_labels(
        data,
        dist_thresh=0.5,
        min_pts=2,
        time_thresh=5,
        return_cores=True,
        traj_cols=traj_cols,
    )

    assert output["promotion_time"].tolist() == [
        data.loc[1, "datetime"],
        data.loc[1, "datetime"],
        data.loc[2, "datetime"],
    ]


def test_seqscan_retains_border_points_as_non_core():
    data = pd.DataFrame(
        {
            "uid": ["user-1"] * 5,
            "timestamp": [0, 120, 240, 360, 480],
            "x": [-0.9, 0.0, 0.3, 0.6, 0.9],
            "y": [0.0] * 5,
        }
    )
    traj_cols = {
        "user_id": "uid",
        "timestamp": "timestamp",
        "x": "x",
        "y": "y",
    }

    core_points = density_algs.seqscan_labels(
        data,
        dist_thresh=1,
        min_pts=2,
        time_thresh=5,
        dur_min=5,
        return_cores=True,
        traj_cols=traj_cols,
    )

    assert core_points.loc[0, "cluster"] >= 0
    assert core_points.loc[0, "core"] == -1
    assert pd.notna(core_points.loc[0, "promotion_time"])


def test_hdbscan_propagates_min_pts_to_cluster_hierarchy(
    single_user_trajectory,
    monkeypatch,
):
    data, traj_cols = single_user_trajectory
    original = density_algs.cluster_hierarchy
    received_min_pts = []

    def recording_cluster_hierarchy(*args, **kwargs):
        received_min_pts.append(kwargs.get("min_pts"))
        return original(*args, **kwargs)

    monkeypatch.setattr(density_algs, "cluster_hierarchy", recording_cluster_hierarchy)
    density_algs.hdbscan_labels(
        data,
        time_thresh=5,
        min_pts=3,
        min_cluster_size=2,
        dur_min=5,
        traj_cols=traj_cols,
    )

    assert received_min_pts == [3]


def test_sequential_per_user_return_anchors_is_aligned(single_user_trajectory):
    data, traj_cols = single_user_trajectory
    second_user = data.assign(
        uid="user-2",
        timestamp=data["timestamp"] + 10_000,
    )
    multi_user = pd.concat([data, second_user], ignore_index=True)

    anchor_points = sequential_algs.detect_stops_labels_per_user(
        multi_user,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method="sliding",
        return_anchors=True,
        traj_cols=traj_cols,
    )

    assert list(anchor_points.columns) == ANCHOR_COLUMNS
    assert anchor_points.index.equals(multi_user.index)
    assert is_integer_dtype(anchor_points["cluster"])
    assert anchor_points["anchor_time"].tolist() == [0] * 6 + [10_000] * 6


@pytest.mark.parametrize(
    ("per_user_function", "kwargs"),
    [
        (
            density_algs.ta_dbscan_labels_per_user,
            {"dist_thresh": 10, "min_pts": 2, "time_thresh": 5},
        ),
        (
            density_algs.dbstop_labels_per_user,
            {"dist_thresh": 10, "min_pts": 2, "time_thresh": 5},
        ),
        (
            density_algs.seqscan_labels_per_user,
            {"dist_thresh": 10, "min_pts": 2, "time_thresh": 5},
        ),
        (
            density_algs.hdbscan_labels_per_user,
            {
                "time_thresh": 5,
                "min_pts": 2,
                "min_cluster_size": 2,
                "dur_min": 5,
            },
        ),
    ],
)
def test_density_per_user_return_cores_has_label_metadata(
    single_user_trajectory,
    per_user_function,
    kwargs,
):
    data, traj_cols = single_user_trajectory
    second_user = data.assign(
        uid="user-2",
        timestamp=data["timestamp"] + 10_000,
    )
    multi_user = pd.concat([data, second_user], ignore_index=True)

    core_points = per_user_function(
        multi_user,
        return_cores=True,
        traj_cols=traj_cols,
        n_jobs=1,
        **kwargs,
    )

    assert len(core_points) == len(multi_user)
    assert list(core_points.columns) == ["cluster", "core", "promotion_time"]
    assert is_integer_dtype(core_points["cluster"])
    assert is_integer_dtype(core_points["core"])
