import pandas as pd
import pytest

import nomad.contact_estimation as contact
from nomad.stop_detection.validation import compute_stop_detection_metrics, compute_visitation_errors


def test_estimate_contacts_exact_location_and_duration_weight():
    stops = pd.DataFrame(
        {
            "user_id": ["a", "b", "c"],
            "start_timestamp": [0, 300, 0],
            "end_timestamp": [600, 900, 600],
            "location_id": ["cafe", "cafe", "park"],
        }
    )

    contacts = contact.estimate_contacts(stops)
    weighted = contact.compute_contact_weights(contacts)

    assert len(contacts) == 1
    assert contacts.loc[0, "user_id_1"] == "a"
    assert contacts.loc[0, "user_id_2"] == "b"
    assert contacts.loc[0, "location_id"] == "cafe"
    assert contacts.loc[0, "overlap_duration"] == 5
    assert "distance" not in contacts.columns
    assert "stop_id_1" not in contacts.columns
    assert "stop_id_2" not in contacts.columns
    assert weighted.name == "contact_weight"
    assert weighted.loc[0] == 5


def test_estimate_contacts_radius_and_linear_distance_weight():
    stops = pd.DataFrame(
        {
            "user_id": ["a", "b", "c"],
            "start_timestamp": [0, 0, 0],
            "end_timestamp": [600, 600, 600],
            "x": [0, 3, 30],
            "y": [0, 4, 0],
        }
    )

    contacts = contact.estimate_contacts(stops, distance_threshold=10)
    weighted = contact.compute_contact_weights(
        contacts, method="linear_distance", distance_threshold=10
    )

    assert len(contacts) == 1
    assert contacts.loc[0, "distance"] == pytest.approx(5)
    assert contacts.loc[0, "overlap_duration"] == 10
    assert weighted.name == "contact_weight"
    assert weighted.loc[0] == pytest.approx(5)


def test_estimate_contacts_reconstructs_timestamp_end_from_duration():
    stops = pd.DataFrame(
        {
            "user_id": ["a", "b"],
            "start_timestamp": [0, 600],
            "duration": [20, 30],
            "location_id": ["cafe", "cafe"],
        }
    )

    contacts = contact.estimate_contacts(stops)

    assert len(contacts) == 1
    assert contacts.loc[0, "contact_start"] == 600
    assert contacts.loc[0, "contact_end"] == 1200
    assert contacts.loc[0, "overlap_duration"] == 10


def test_estimate_contacts_uses_strict_temporal_overlap():
    stops = pd.DataFrame(
        {
            "user_id": ["a", "b"],
            "start_timestamp": [0, 600],
            "end_timestamp": [600, 1200],
            "location_id": ["cafe", "cafe"],
        }
    )

    contacts = contact.estimate_contacts(stops)

    assert contacts.empty


def test_estimate_contacts_empty_stops_has_expected_columns():
    stops = pd.DataFrame(columns=["user_id", "start_timestamp", "end_timestamp", "location_id"])

    contacts = contact.estimate_contacts(stops)

    assert contacts.empty
    assert contacts.columns.tolist() == [
        "user_id_1",
        "user_id_2",
        "contact_start",
        "contact_end",
        "overlap_duration",
        "location_id",
    ]


def test_estimate_contacts_excludes_same_user_overlap():
    stops = pd.DataFrame(
        {
            "user_id": ["a", "a"],
            "start_timestamp": [0, 300],
            "end_timestamp": [600, 900],
            "location_id": ["cafe", "cafe"],
        }
    )

    contacts = contact.estimate_contacts(stops)

    assert contacts.empty


def test_estimate_contacts_exact_location_requires_non_missing_location():
    stops = pd.DataFrame(
        {
            "user_id": ["a", "b"],
            "start_timestamp": [0, 300],
            "end_timestamp": [600, 900],
            "location_id": ["cafe", None],
        }
    )

    with pytest.raises(ValueError, match="non-missing location_id"):
        contact.estimate_contacts(stops)


def test_estimate_contacts_requires_user_id():
    stops = pd.DataFrame(
        {
            "start_timestamp": [0],
            "end_timestamp": [600],
            "location_id": ["cafe"],
        }
    )

    with pytest.raises(ValueError, match="user_id"):
        contact.estimate_contacts(stops)


def test_estimate_contacts_requires_end_or_duration():
    stops = pd.DataFrame(
        {
            "user_id": ["a"],
            "start_timestamp": [0],
            "location_id": ["cafe"],
        }
    )

    with pytest.raises(ValueError, match="end time or duration"):
        contact.estimate_contacts(stops)


def test_linear_distance_weight_clips_at_zero():
    contacts = pd.DataFrame(
        {
            "overlap_duration": [30, 30],
            "distance": [10, 15],
        }
    )

    weights = contact.compute_contact_weights(
        contacts,
        method="linear_distance",
        distance_threshold=10,
    )

    assert weights.tolist() == [0, 0]


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
            "truth_datetime": [
                pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:20:00", tz="UTC"),
            ],
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
