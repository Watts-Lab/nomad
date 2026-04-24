import pandas as pd
import pytest

import nomad.contact_estimation as contact
from nomad.stop_detection.validation import compute_stop_detection_metrics, compute_visitation_errors


@pytest.fixture
def distinct_visit_tables():
    left = pd.DataFrame(
        {
            "pred_start": [
                pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:05:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:20:00", tz="UTC"),
            ],
            "pred_end": [
                pd.Timestamp("2024-01-01 00:05:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:10:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
            ],
            "pred_loc": ["home", "home", "cafe"],
            "pred_user": ["u1", "u1", "u1"],
        }
    )
    right = pd.DataFrame(
        {
            "truth_start": [
                int(pd.Timestamp("2024-01-01 00:00:00", tz="UTC").timestamp()),
                int(pd.Timestamp("2024-01-01 00:20:00", tz="UTC").timestamp()),
            ],
            "truth_end": [
                int(pd.Timestamp("2024-01-01 00:10:00", tz="UTC").timestamp()),
                int(pd.Timestamp("2024-01-01 00:30:00", tz="UTC").timestamp()),
            ],
            "truth_loc": ["home", "work"],
            "truth_user": ["u1", "u1"],
        }
    )
    left_kwargs = {
        "start_datetime": "pred_start",
        "end_datetime": "pred_end",
        "location_id": "pred_loc",
        "user_id": "pred_user",
    }
    right_traj_cols = {
        "start_timestamp": "truth_start",
        "end_timestamp": "truth_end",
        "location_id": "truth_loc",
        "user_id": "truth_user",
    }
    return left, right, left_kwargs, right_traj_cols


def test_overlapping_visits_supports_distinct_schemas(distinct_visit_tables):
    left, right, left_kwargs, right_traj_cols = distinct_visit_tables

    overlaps = contact.overlapping_visits(
        left,
        right,
        match_location=False,
        right_traj_cols=right_traj_cols,
        **left_kwargs,
    )

    assert overlaps["duration"].tolist() == [5, 5, 10]


def test_compute_visitation_errors_supports_distinct_schemas(distinct_visit_tables):
    left, right, left_kwargs, right_traj_cols = distinct_visit_tables

    overlaps = contact.overlapping_visits(
        left,
        right,
        match_location=False,
        right_traj_cols=right_traj_cols,
        **left_kwargs,
    )
    errors = compute_visitation_errors(
        overlaps,
        right,
        right_traj_cols=right_traj_cols,
        **left_kwargs,
    )

    assert errors["missed_fraction"] == 0.0
    assert errors["merged_fraction"] == pytest.approx(0.5)
    assert errors["split_fraction"] == pytest.approx(1.0)


def test_compute_stop_detection_metrics_supports_distinct_schemas():
    stops = pd.DataFrame(
        {
            "pred_start": [
                pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:20:00", tz="UTC"),
            ],
            "pred_end": [
                pd.Timestamp("2024-01-01 00:10:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
            ],
            "pred_loc": ["home", "work"],
            "pred_user": ["u1", "u1"],
            "pred_minutes": [10, 10],
        }
    )
    truth = pd.DataFrame(
        {
            "truth_start": [
                int(pd.Timestamp("2024-01-01 00:00:00", tz="UTC").timestamp()),
                int(pd.Timestamp("2024-01-01 00:20:00", tz="UTC").timestamp()),
            ],
            "truth_end": [
                int(pd.Timestamp("2024-01-01 00:10:00", tz="UTC").timestamp()),
                int(pd.Timestamp("2024-01-01 00:30:00", tz="UTC").timestamp()),
            ],
            "truth_loc": ["home", "work"],
            "truth_user": ["u1", "u1"],
            "truth_minutes": [10, 10],
        }
    )

    metrics = compute_stop_detection_metrics(
        stops,
        truth,
        prf_only=False,
        start_datetime="pred_start",
        end_datetime="pred_end",
        location_id="pred_loc",
        user_id="pred_user",
        duration="pred_minutes",
        right_traj_cols={
            "start_timestamp": "truth_start",
            "end_timestamp": "truth_end",
            "location_id": "truth_loc",
            "user_id": "truth_user",
            "duration": "truth_minutes",
        },
    )

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["missed_fraction"] == pytest.approx(0.0)
    assert metrics["merged_fraction"] == pytest.approx(0.0)
    assert metrics["split_fraction"] == pytest.approx(0.0)


def test_overlapping_visits_shared_schema_failure_mentions_right_traj_cols():
    left = pd.DataFrame({"pred_start": [0], "pred_end": [600], "pred_loc": ["home"]})
    right = pd.DataFrame({"truth_start": [300], "truth_end": [900], "truth_loc": ["home"]})

    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="right_traj_cols"):
            contact.overlapping_visits(
                left,
                right,
                start_timestamp="pred_start",
                end_timestamp="pred_end",
                location_id="pred_loc",
            )
