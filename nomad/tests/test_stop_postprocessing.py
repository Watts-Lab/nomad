import pandas as pd
import pytest

from nomad.stop_detection.postprocessing import fill_timestamp_gaps, merge_stops
from nomad.stop_detection.sequential_algs import grid_based


def test_fill_timestamp_gaps_adds_unassigned_intervals():
    stops = pd.DataFrame(
        {
            'cluster': [0, 1],
            'x': [1.0, 2.0],
            'y': [1.0, 2.0],
            'start_timestamp': [600, 1800],
            'duration': [10, 10],
            'building_id': ['home', 'work'],
        }
    )

    result = fill_timestamp_gaps(0, 3000, stops)

    assert result['start_timestamp'].tolist() == [0, 600, 1200, 1800, 2400]
    assert result['duration'].tolist() == [10, 10, 10, 10, 10]
    assert result['building_id'].tolist() == [
        'None', 'home', 'None', 'work', 'None'
    ]


def test_fill_timestamp_gaps_returns_empty_input_unchanged():
    stops = pd.DataFrame(
        columns=['start_timestamp', 'duration', 'building_id']
    )

    result = fill_timestamp_gaps(0, 600, stops)

    pd.testing.assert_frame_equal(result, stops)
    assert result is not stops


def test_fill_timestamp_gaps_does_not_add_rows_without_gaps():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 600],
            'duration': [10, 10],
            'building_id': ['home', 'work'],
        }
    )

    result = fill_timestamp_gaps(0, 1200, stops)

    pd.testing.assert_frame_equal(result, stops)


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
        'cluster': 0,
        'start_timestamp': 0,
        'location_id': 4,
        'duration': 55,
    }


def test_merge_stops_uses_grid_based_for_ping_table():
    pings = pd.DataFrame(
        {
            'timestamp': [0, 60, 20 * 60, 21 * 60],
            'location_id': [4, 4, 4, 4],
        }
    )

    merged = merge_stops(pings, time_thresh=10)

    assert merged.to_dict('records') == [
        {'cluster': 0, 'timestamp': 0, 'duration': 1, 'location_id': 4},
        {
            'cluster': 1,
            'timestamp': 20 * 60,
            'duration': 1,
            'location_id': 4,
        },
    ]


def test_merge_stops_ping_table_respects_location_changes():
    pings = pd.DataFrame(
        {
            'timestamp': [0, 60, 120],
            'location_id': [4, 8, 8],
        }
    )

    merged = merge_stops(pings)

    assert merged['location_id'].tolist() == [4, 8]
    assert merged['duration'].tolist() == [0, 1]


def test_merge_stops_ping_table_matches_grid_based():
    pings = pd.DataFrame(
        {
            'timestamp': [0, 60, 20 * 60, 21 * 60],
            'location_id': [4, 4, 4, 4],
        }
    )

    result = merge_stops(pings, time_thresh=10)
    expected = grid_based(
        pings,
        time_thresh=10,
        min_cluster_size=1,
        dur_min=0,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_merge_stops_custom_method_overrides_time_thresh():
    pings = pd.DataFrame(
        {'timestamp': [0, 60], 'location_id': [4, 4]}
    )

    def custom_algorithm(data, time_thresh, traj_cols):
        result = data.iloc[[0]].copy()
        result['time_thresh'] = time_thresh
        return result

    result = merge_stops(
        pings,
        time_thresh=60,
        method='custom',
        algorithm=custom_algorithm,
        algorithm_kwargs={'time_thresh': 5},
    )

    assert result['time_thresh'].tolist() == [5]


def test_merge_stops_requires_custom_method_for_algorithm():
    pings = pd.DataFrame(
        {'timestamp': [0, 60], 'location_id': [4, 4]}
    )

    with pytest.raises(ValueError, match="method='custom'"):
        merge_stops(pings, algorithm=lambda data: data)


def test_merge_stops_ping_table_keeps_single_ping():
    pings = pd.DataFrame({'timestamp': [0], 'location_id': [4]})

    merged = merge_stops(pings)

    assert merged.to_dict('records') == [
        {'cluster': 0, 'timestamp': 0, 'duration': 0, 'location_id': 4}
    ]


def test_merge_stops_rejects_multiple_users_in_ping_table():
    pings = pd.DataFrame(
        {
            'user_id': ['a', 'b'],
            'timestamp': [0, 60],
            'location_id': [4, 4],
        }
    )

    with pytest.raises(ValueError, match='one user per call'):
        merge_stops(pings)


def test_merge_stops_supports_datetime_pings():
    pings = pd.DataFrame(
        {
            'datetime': pd.to_datetime(
                ['2026-01-01 09:00', '2026-01-01 09:05', '2026-01-01 09:20']
            ),
            'location_id': [4, 4, 4],
        }
    )

    merged = merge_stops(pings, time_thresh=10)

    assert merged['datetime'].tolist() == [
        pd.Timestamp('2026-01-01 09:00'),
        pd.Timestamp('2026-01-01 09:20'),
    ]
    assert merged['duration'].tolist() == [5, 0]


def test_merge_stops_ping_table_supports_custom_columns_without_mutation():
    pings = pd.DataFrame(
        {
            'recorded_at': [0, 60],
            'building': ['home', 'home'],
        }
    )
    original = pings.copy()

    merged = merge_stops(
        pings,
        traj_cols={'timestamp': 'recorded_at', 'location_id': 'building'},
    )

    assert merged.to_dict('records') == [
        {
            'cluster': 0,
            'recorded_at': 0,
            'duration': 1,
            'building': 'home',
        }
    ]
    pd.testing.assert_frame_equal(pings, original)


def test_merge_stops_merges_gap_equal_to_threshold():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 40 * 60],
            'duration': [30, 20],
            'location_id': [4, 4],
        }
    )

    merged = merge_stops(stops, time_thresh=10)

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

    merged = merge_stops(stops, time_thresh=10)

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


def test_merge_stops_rejects_multiple_users():
    stops = pd.DataFrame(
        {
            'user_id': ['a', 'b'],
            'start_timestamp': [0, 60],
            'duration': [30, 30],
            'location_id': [4, 4],
        }
    )

    with pytest.raises(ValueError, match='one user per call'):
        merge_stops(stops)


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


def test_merge_stops_uses_furthest_end_for_nested_intervals():
    stops = pd.DataFrame(
        {
            'start_timestamp': [0, 10 * 60, 25 * 60],
            'end_timestamp': [30 * 60, 20 * 60, 40 * 60],
            'location_id': [4, 4, 4],
        }
    )

    merged = merge_stops(stops, time_thresh=5)

    assert len(merged) == 1
    assert merged['start_timestamp'].tolist() == [0]
    assert merged['end_timestamp'].tolist() == [40 * 60]


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
        'cluster': 0,
        'started': 0,
        'destination': 4,
        'person': 'a',
        'minutes': 55,
    }


def test_merge_stops_requires_monotonic_start_times():
    stops = pd.DataFrame(
        {
            'start_timestamp': [35 * 60, 0],
            'duration': [20, 30],
            'location_id': [4, 4],
        },
        index=[20, 10],
    )

    with pytest.raises(ValueError, match='monotonically increasing'):
        merge_stops(stops)


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


def test_merge_stops_returns_typed_empty_cluster():
    stops = pd.DataFrame(
        columns=['start_timestamp', 'duration', 'location_id']
    )

    merged = merge_stops(stops)

    expected = stops.copy()
    expected.insert(0, 'cluster', pd.Series(dtype='Int64'))

    pd.testing.assert_frame_equal(merged, expected)
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
