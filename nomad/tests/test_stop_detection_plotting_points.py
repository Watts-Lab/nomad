import inspect

import networkx as nx
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from nomad.stop_detection import density_algs, sequential_algs


POINT_COLUMNS = [
    "uid",
    "timestamp",
    "role",
    "start_timestamp",
    "label",
    "x",
    "y",
    "value",
    "value_name",
]


def _extract_core_points(timestamps, clusters, cores):
    data = pd.DataFrame(
        {
            "uid": ["user-1"] * len(timestamps),
            "timestamp": timestamps,
            "x": range(len(timestamps)),
            "y": [0] * len(timestamps),
        }
    )
    graph = nx.Graph()
    graph.add_nodes_from(timestamps)
    output = pd.DataFrame({"cluster": clusters, "core": cores})
    traj_cols = {
        "user_id": "uid",
        "timestamp": "timestamp",
        "start_timestamp": "start_timestamp",
        "label": "label",
        "x": "x",
        "y": "y",
    }
    return density_algs._format_density_points(
        data,
        graph,
        output,
        traj_cols,
        "timestamp",
        "x",
        "y",
        False,
    )


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


@pytest.mark.parametrize("method", ["sliding", "centroid"])
def test_sequential_return_anchors_returns_complete_points(
    single_user_trajectory,
    method,
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

    assert list(anchor_points.columns) == POINT_COLUMNS
    assert not anchor_points.empty
    assert is_integer_dtype(anchor_points["role"])
    assert set(anchor_points["role"]) == {-1, 1}

    labels_without_points = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method=method,
        traj_cols=traj_cols,
    )
    returned_labels = anchor_points.loc[
        anchor_points["role"] >= 0, "label"
    ].reset_index(drop=True).rename("cluster")
    pd.testing.assert_series_equal(returned_labels, labels_without_points)


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
    assert set(anchor_points["role"]) == {0}
    assert set(anchor_points["label"]) == {-1}


def test_sequential_return_anchors_empty_schema(single_user_trajectory):
    data, traj_cols = single_user_trajectory

    anchor_points = sequential_algs.detect_stops_labels(
        data.iloc[:0],
        return_anchors=True,
        traj_cols=traj_cols,
    )

    assert anchor_points.empty
    assert list(anchor_points.columns) == [
        "user_id",
        "timestamp",
        "role",
        "start_timestamp",
        "label",
        "x",
        "y",
        "value",
        "value_name",
    ]
    assert is_integer_dtype(anchor_points["role"])


def test_point_output_follows_traj_cols_names(single_user_trajectory):
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

    assert list(anchor_points.columns) == [
        "uid",
        "timestamp",
        "role",
        "anchor_timestamp",
        "stop_label",
        "x",
        "y",
        "value",
        "value_name",
    ]


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
def test_density_return_cores_returns_complete_schema(
    single_user_trajectory,
    label_function,
    kwargs,
):
    data, traj_cols = single_user_trajectory
    traj_cols = {
        **traj_cols,
        "start_timestamp": "core_timestamp",
        "label": "stop_label",
    }
    data = data.assign(core_timestamp=pd.NA, stop_label=pd.NA)

    core_points = label_function(
        data,
        return_cores=True,
        traj_cols=traj_cols,
        **kwargs,
    )

    assert len(core_points) == len(data)
    assert list(core_points.columns) == [
        "uid",
        "timestamp",
        "role",
        "core_timestamp",
        "stop_label",
        "x",
        "y",
        "value",
        "value_name",
        "cluster",
        "core",
    ]
    assert is_integer_dtype(core_points["role"])
    assert set(core_points["role"]) <= {-1, 0, 1}
    pd.testing.assert_series_equal(core_points["cluster"], core_points["stop_label"], check_names=False)

    labels = label_function(data, traj_cols=traj_cols, **kwargs)
    pd.testing.assert_series_equal(
        core_points["stop_label"].set_axis(data.index).rename("cluster"),
        labels,
    )


def test_density_core_points_retains_border_records_when_cluster_has_no_cores():
    core_points = _extract_core_points(
        timestamps=[0, 5],
        clusters=[0, 0],
        cores=[-1, -1],
    )

    assert core_points["role"].tolist() == [-1, -1]
    assert core_points["label"].tolist() == [0, 0]
    assert core_points["start_timestamp"].isna().all()


def test_density_core_points_uses_nearest_core_timestamp():
    core_points = _extract_core_points(
        timestamps=[0, 4, 6, 10],
        clusters=[0, 0, 0, 0],
        cores=[0, -1, -1, 1],
    )

    assert core_points["role"].tolist() == [1, -1, -1, 1]
    assert core_points["start_timestamp"].tolist() == [0, 0, 10, 10]


def test_density_core_points_handles_mixed_clusters_with_and_without_cores():
    core_points = _extract_core_points(
        timestamps=[0, 5, 10, 15],
        clusters=[0, 0, 1, 1],
        cores=[-1, -1, 0, -1],
    )

    assert core_points["role"].tolist() == [-1, -1, 1, -1]
    assert core_points.loc[:1, "start_timestamp"].isna().all()
    assert core_points.loc[2:, "start_timestamp"].tolist() == [10, 10]


def test_density_core_points_includes_noise():
    core_points = _extract_core_points(
        timestamps=[0, 5, 10],
        clusters=[-1, 0, 0],
        cores=[-1, 0, -1],
    )

    assert core_points["role"].tolist() == [0, 1, -1]
    assert core_points["label"].tolist() == [-1, 0, 0]


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

    assert core_points.loc[0, "label"] >= 0
    assert core_points.loc[0, "role"] == -1
    assert core_points.loc[0, "core"] == -1
    assert set(core_points["role"]) <= {-1, 0, 1}


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


def test_sequential_per_user_return_anchors_is_numeric(single_user_trajectory):
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

    assert set(anchor_points["uid"]) == {"user-1", "user-2"}
    assert is_integer_dtype(anchor_points["role"])
    assert set(anchor_points["role"]) == {-1, 1}


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
def test_density_per_user_return_cores_is_numeric(
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
    assert is_integer_dtype(core_points["role"])
    assert set(core_points["role"]) <= {-1, 0, 1}
