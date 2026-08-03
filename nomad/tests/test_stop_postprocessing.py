import pandas as pd
import pytest

from nomad.stop_detection.postprocessing import merge_stops


def test_merge_stops_merges_same_location_with_short_gap():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 35 * 60],
            'duration': [30, 20],
            'location_id': [4, 4],
        },
        index=[10, 20],
    )

    merged = merge_stops(stops)

    assert merged.index.tolist() == [10]
    assert merged.iloc[0].to_dict() == {
        'start_timestamp': 0,
        'location_id': 4,
        'duration': 55,
    }


def test_merge_stops_merges_gap_equal_to_threshold():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 40 * 60],
            'duration': [30, 20],
            'location_id': [4, 4],
        }
    )

    merged = merge_stops(stops, max_time_gap='10min')

    assert len(merged) == 1
    assert merged['duration'].tolist() == [60]


def test_merge_stops_keeps_gap_above_threshold_separate():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 41 * 60],
            'duration': [30, 20],
            'location_id': [4, 4],
        }
    )

    merged = merge_stops(stops, max_time_gap='10min')

    assert len(merged) == 2
    assert merged['duration'].tolist() == [30, 20]


def test_merge_stops_keeps_different_locations_separate():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 35 * 60],
            'duration': [30, 20],
            'location_id': [4, 8],
        }
    )

    merged = merge_stops(stops)

    assert merged['location_id'].tolist() == [4, 8]


def test_merge_stops_keeps_users_separate():
    stops = pd.DataFrame(
        {
            'user_id': ['a', 'b'],
            'start_timestamp': [0, 60],
            'duration': [30, 30],
            'location_id': [4, 4],
        }
    )

    merged = merge_stops(stops)

    assert len(merged) == 2
    assert merged['user_id'].tolist() == ['a', 'b']


def test_merge_stops_supports_missing_user_ids():
    stops = pd.DataFrame(
        {
            'user_id': [pd.NA, pd.NA],
            'start_timestamp': [0, 15 * 60],
            'duration': [10, 10],
            'location_id': [4, 4],
        }
    )

    merged = merge_stops(stops)

    assert len(merged) == 1
    assert merged['duration'].tolist() == [25]


def test_merge_stops_merges_multiple_consecutive_stops():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 35 * 60, 60 * 60],
            'duration': [30, 20, 15],
            'location_id': [4, 4, 4],
        }
    )

    merged = merge_stops(stops)

    assert len(merged) == 1
    assert merged['duration'].tolist() == [75]


def test_merge_stops_supports_datetime_and_explicit_end():
    stops = pd.DataFrame(
        {
            'start_datetime': pd.to_datetime(
                ['2026-01-01 09:00', '2026-01-01 09:35']
            ),
            'end_datetime': pd.to_datetime(
                ['2026-01-01 09:30', '2026-01-01 10:00']
            ),
            'location_id': [4, 4],
        }
    )

    merged = merge_stops(stops)

    assert len(merged) == 1
    assert merged['start_datetime'].iloc[0] == pd.Timestamp('2026-01-01 09:00')
    assert merged['end_datetime'].iloc[0] == pd.Timestamp('2026-01-01 10:00')


def test_merge_stops_supports_custom_column_mappings():
    stops = pd.DataFrame(
        {
            'person': ['a', 'a'],
            'started': [0, 35 * 60],
            'minutes': [30, 20],
            'destination': [4, 4],
        }
    )
    traj_cols = {
        'user_id': 'person',
        'start_timestamp': 'started',
        'duration': 'minutes',
        'location_id': 'destination',
    }

    merged = merge_stops(stops, traj_cols=traj_cols)

    assert merged.iloc[0].to_dict() == {
        'started': 0,
        'destination': 4,
        'person': 'a',
        'minutes': 55,
    }


def test_merge_stops_sorts_before_merging_and_preserves_first_stop_index():
    stops = pd.DataFrame(
        {
            'start_timestamp': [35 * 60, 0],
            'duration': [20, 30],
            'location_id': [4, 4],
        },
        index=[20, 10],
    )

    merged = merge_stops(stops)

    assert merged.index.tolist() == [10]
    assert merged['start_timestamp'].tolist() == [0]


def test_merge_stops_applies_requested_aggregations():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 35 * 60],
            'duration': [30, 20],
            'location_id': [4, 4],
            'n_pings': [5, 7],
            'geometry': ['first', 'second'],
        }
    )

    merged = merge_stops(
        stops,
        agg={'n_pings': 'sum', 'geometry': 'first'},
    )

    assert merged['n_pings'].tolist() == [12]
    assert merged['geometry'].tolist() == ['first']


def test_merge_stops_does_not_merge_missing_locations():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 60],
            'duration': [10, 10],
            'location_id': [pd.NA, pd.NA],
        }
    )

    merged = merge_stops(stops)

    assert len(merged) == 2


def test_merge_stops_returns_empty_input_unchanged():
    stops = pd.DataFrame(
        columns=['start_timestamp', 'duration', 'location_id']
    )

    merged = merge_stops(stops)

    pd.testing.assert_frame_equal(merged, stops)
    assert merged is not stops


def test_merge_stops_does_not_mutate_input():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 35 * 60],
            'duration': [30, 20],
            'location_id': [4, 4],
        }
    )
    original = stops.copy()

    merge_stops(stops)

    pd.testing.assert_frame_equal(stops, original)


def test_merge_stops_requires_end_or_duration():
    stops = pd.DataFrame({'start_timestamp': [0], 'location_id': [4]})

    with pytest.raises(ValueError, match='either end time or duration'):
        merge_stops(stops)


def test_merge_stops_requires_location_column():
    stops = pd.DataFrame({'start_timestamp': [0], 'duration': [10]})

    with pytest.raises(ValueError, match="Location column 'location_id'"):
        merge_stops(stops)


def test_merge_stops_rejects_invalid_time_gap_type():
    stops = pd.DataFrame(
        {'start_timestamp': [0], 'duration': [10], 'location_id': [4]}
    )

    with pytest.raises(TypeError, match='str or pd.Timedelta'):
        merge_stops(stops, max_time_gap=10)
