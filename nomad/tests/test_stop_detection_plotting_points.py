import inspect

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from nomad.stop_detection import density_algs, sequential_algs
from nomad.stop_detection.core_points import CORE_POINT_COLUMNS


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
def test_sequential_anchor_plotting_returns_anchor_points(
    single_user_trajectory,
    method,
):
    data, traj_cols = single_user_trajectory

    labels, anchor_points = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method=method,
        anchor_plotting=True,
        config_key=method,
        traj_cols=traj_cols,
    )

    assert len(labels) == len(data)
    assert list(anchor_points.columns) == CORE_POINT_COLUMNS
    assert not anchor_points.empty
    assert is_integer_dtype(anchor_points["role"])
    assert set(anchor_points["role"]) == {-1, 1}
    assert set(anchor_points["config_key"]) == {method}

    labels_without_plotting = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method=method,
        traj_cols=traj_cols,
    )
    pd.testing.assert_series_equal(labels, labels_without_plotting)


def test_sequential_anchor_plotting_path_writes_without_changing_return_type(
    single_user_trajectory,
    tmp_path,
):
    data, traj_cols = single_user_trajectory
    output_path = tmp_path / "anchor_points.parquet"

    labels = sequential_algs.detect_stops_labels(
        data,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        anchor_plotting_path=output_path,
        traj_cols=traj_cols,
    )

    assert isinstance(labels, pd.Series)
    assert output_path.exists()
    assert not pd.read_parquet(output_path).empty


def test_lachesis_does_not_advertise_anchor_or_core_plotting():
    parameters = inspect.signature(sequential_algs.lachesis_labels).parameters

    assert "anchor_plotting" not in parameters
    assert "anchor_plotting_path" not in parameters
    assert "core_plotting" not in parameters
    assert "core_plotting_path" not in parameters


def test_grid_based_does_not_advertise_or_return_core_points():
    for function in (
        sequential_algs.grid_based_labels,
        sequential_algs.grid_based_labels_per_user,
    ):
        parameters = inspect.signature(function).parameters
        assert "core_plotting" not in parameters
        assert "core_plotting_path" not in parameters
        assert "config_key" not in parameters


@pytest.mark.parametrize(
    ("label_function", "editable_parameters"),
    [
        (
            sequential_algs.detect_stops_labels,
            {"delta_roam", "dt_max", "dur_min", "method", "anchor_plotting"},
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
            {"dist_thresh", "min_pts", "time_thresh", "remove_overlaps", "core_plotting"},
        ),
        (
            density_algs.dbstop_labels,
            {"dist_thresh", "min_pts", "time_thresh", "core_plotting"},
        ),
        (
            density_algs.seqscan_labels,
            {"dist_thresh", "dur_min", "time_thresh", "min_pts", "back_merge", "core_plotting"},
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
                "core_plotting",
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
def test_density_core_plotting_returns_stable_schema(
    single_user_trajectory,
    label_function,
    kwargs,
):
    data, traj_cols = single_user_trajectory

    labels, core_points = label_function(
        data,
        core_plotting=True,
        config_key=label_function.__name__,
        traj_cols=traj_cols,
        **kwargs,
    )

    assert len(labels) == len(data)
    assert list(core_points.columns) == CORE_POINT_COLUMNS
    assert is_integer_dtype(core_points["role"])
    assert set(core_points["role"]) <= {-1, 1}

    labels_without_plotting = label_function(data, traj_cols=traj_cols, **kwargs)
    pd.testing.assert_series_equal(labels, labels_without_plotting)


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

    output, core_points = density_algs.seqscan_labels(
        data,
        dist_thresh=1,
        min_pts=2,
        time_thresh=5,
        dur_min=5,
        return_cores=True,
        core_plotting=True,
        traj_cols=traj_cols,
    )

    assert output.loc[0, "cluster"] >= 0
    assert output.loc[0, "core"] == -1
    assert set(core_points["role"]) == {-1, 1}


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


def test_sequential_per_user_anchor_plotting_is_numeric(single_user_trajectory):
    data, traj_cols = single_user_trajectory
    second_user = data.assign(
        uid="user-2",
        timestamp=data["timestamp"] + 10_000,
    )
    multi_user = pd.concat([data, second_user], ignore_index=True)

    labels, anchor_points = sequential_algs.detect_stops_labels_per_user(
        multi_user,
        delta_roam=10,
        dt_max=5,
        dur_min=5,
        method="sliding",
        anchor_plotting=True,
        traj_cols=traj_cols,
    )

    assert len(labels) == len(multi_user)
    assert set(anchor_points["user_id"]) == {"user-1", "user-2"}
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
def test_density_per_user_core_plotting_is_numeric(
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

    labels, core_points = per_user_function(
        multi_user,
        core_plotting=True,
        traj_cols=traj_cols,
        n_jobs=1,
        **kwargs,
    )

    assert len(labels) == len(multi_user)
    assert is_integer_dtype(core_points["role"])
    assert set(core_points["role"]) <= {-1, 1}
