import pandas as pd
import pytest

from nomad.stop_detection.sequential_algs import lachesis
from nomad.visit_attribution.visit_attribution import lachesis_visits


@pytest.fixture
def interrupted_destination_trajectory():
    return pd.DataFrame(
        {
            'timestamp': [0, 300, 600, 1200, 1800, 2100, 2400],
            'x': [0.0, 0.0, 0.0, 1000.0, 1.0, 1.0, 1.0],
            'y': [0.0, 0.0, 0.0, 1000.0, 1.0, 1.0, 1.0],
        }
    )


def lachesis_options():
    return {'delta_roam': 5, 'dt_max': 10, 'dur_min': 5}


def test_lachesis_none_preserves_existing_output(interrupted_destination_trajectory):
    expected = lachesis(
        interrupted_destination_trajectory,
        **lachesis_options(),
    )

    result = lachesis_visits(
        interrupted_destination_trajectory,
        postprocessing='none',
        **lachesis_options(),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_lachesis_dbscan_clusters_and_merges_interrupted_visits(
    interrupted_destination_trajectory,
):
    result = lachesis_visits(
        interrupted_destination_trajectory,
        postprocessing='dbscan',
        postprocessing_kwargs={'epsilon': 5},
        merge_kwargs={'max_time_gap': '25min'},
        **lachesis_options(),
    )

    assert result['location_id'].tolist() == [0]
    assert result['timestamp'].tolist() == [0]
    assert result['duration'].tolist() == [40]
    assert result['x'].tolist() == [0.5]
    assert result['y'].tolist() == [0.5]


def test_lachesis_dbscan_respects_merge_threshold(
    interrupted_destination_trajectory,
):
    result = lachesis_visits(
        interrupted_destination_trajectory,
        postprocessing='dbscan',
        postprocessing_kwargs={'epsilon': 5},
        merge_kwargs={'max_time_gap': '15min'},
        **lachesis_options(),
    )

    assert len(result) == 2
    assert result['location_id'].tolist() == [0, 0]


def test_lachesis_visits_rejects_multiple_users(
    interrupted_destination_trajectory,
):
    first = interrupted_destination_trajectory.assign(user_id='a')
    second = interrupted_destination_trajectory.assign(user_id='b')
    data = pd.concat([first, second], ignore_index=True)

    with pytest.raises(ValueError, match='one user per call'):
        lachesis_visits(data, **lachesis_options())


def test_lachesis_visits_preserves_single_user_id(
    interrupted_destination_trajectory,
):
    data = interrupted_destination_trajectory.assign(user_id='a')

    result = lachesis_visits(
        data,
        postprocessing_kwargs={'epsilon': 5},
        merge_kwargs={'max_time_gap': '25min'},
        **lachesis_options(),
    )

    assert result['user_id'].tolist() == ['a']
    assert result['location_id'].tolist() == [0]


def test_lachesis_dbscan_supports_custom_columns():
    data = pd.DataFrame(
        {
            'seconds': [0, 300, 600, 1200, 1800, 2100, 2400],
            'east': [0.0, 0.0, 0.0, 1000.0, 1.0, 1.0, 1.0],
            'north': [0.0, 0.0, 0.0, 1000.0, 1.0, 1.0, 1.0],
        }
    )
    traj_cols = {
        'timestamp': 'seconds',
        'x': 'east',
        'y': 'north',
        'location_id': 'destination_id',
    }

    result = lachesis_visits(
        data,
        postprocessing='dbscan',
        postprocessing_kwargs={'epsilon': 5},
        merge_kwargs={'max_time_gap': '25min'},
        traj_cols=traj_cols,
        **lachesis_options(),
    )

    assert result['destination_id'].tolist() == [0]
    assert result['east'].tolist() == [0.5]
    assert result['north'].tolist() == [0.5]


def test_lachesis_dbscan_handles_no_detected_stops():
    data = pd.DataFrame(
        {
            'timestamp': [0, 60],
            'x': [0.0, 1000.0],
            'y': [0.0, 1000.0],
        }
    )

    result = lachesis_visits(
        data,
        postprocessing='dbscan',
        **lachesis_options(),
    )

    assert result.empty
    assert 'location_id' in result.columns


def test_lachesis_infomap_reports_unimplemented(
    interrupted_destination_trajectory,
):
    with pytest.raises(NotImplementedError, match='infomap'):
        lachesis_visits(
            interrupted_destination_trajectory,
            postprocessing='infomap',
            **lachesis_options(),
        )


def test_lachesis_rejects_unknown_postprocessing(
    interrupted_destination_trajectory,
):
    with pytest.raises(ValueError, match='postprocessing must be'):
        lachesis_visits(
            interrupted_destination_trajectory,
            postprocessing='kmeans',
            **lachesis_options(),
        )
