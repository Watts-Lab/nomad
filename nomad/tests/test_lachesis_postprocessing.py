import pandas as pd
import pytest

from nomad.stop_detection.sequential_algs import (
    lachesis,
    lachesis_labels,
    lachesis_per_user,
)


@pytest.fixture
def interrupted_destination_trajectory():
    return pd.DataFrame(
        {
            'timestamp': [0, 300, 600, 1200, 1800, 2100, 2400],
            'x': [0.0, 0.0, 0.0, 1000.0, 1.0, 1.0, 1.0],
            'y': [0.0, 0.0, 0.0, 1000.0, 1.0, 1.0, 1.0],
        }
    )


def lachesis_options(dt_max=10):
    return {'delta_roam': 5, 'dt_max': dt_max, 'dur_min': 5}


def test_lachesis_none_preserves_existing_output(interrupted_destination_trajectory):
    expected = lachesis(
        interrupted_destination_trajectory,
        **lachesis_options(),
    )

    result = lachesis(
        interrupted_destination_trajectory,
        postprocessing=None,
        **lachesis_options(),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_lachesis_dbscan_clusters_and_merges_interrupted_visits(
    interrupted_destination_trajectory,
):
    result = lachesis(
        interrupted_destination_trajectory,
        postprocessing='dbscan',
        eps=5,
        **lachesis_options(dt_max=25),
    )

    assert result['cluster'].tolist() == [0]
    assert result['timestamp'].tolist() == [0]
    assert result['duration'].tolist() == [40]
    assert result['x'].tolist() == [0.0]
    assert result['y'].tolist() == [0.0]
    assert 'location_id' not in result.columns


def test_lachesis_labels_applies_postprocessing(
    interrupted_destination_trajectory,
):
    labels = lachesis_labels(
        interrupted_destination_trajectory,
        delta_roam=5,
        dt_max=25,
        dur_min=5,
        postprocessing='dbscan',
        eps=5,
    )

    assert labels.tolist() == [0, 0, 0, -1, 0, 0, 0]


def test_lachesis_dbscan_respects_merge_threshold(
    interrupted_destination_trajectory,
):
    result = lachesis(
        interrupted_destination_trajectory,
        postprocessing='dbscan',
        eps=5,
        **lachesis_options(),
    )

    assert len(result) == 2
    assert result['cluster'].tolist() == [0, 1]


def test_lachesis_per_user_postprocesses_each_user_separately(
    interrupted_destination_trajectory,
):
    first = interrupted_destination_trajectory.assign(user_id='a')
    second = interrupted_destination_trajectory.assign(user_id='b')
    data = pd.concat([first, second], ignore_index=True)

    result = lachesis_per_user(
        data,
        postprocessing='dbscan',
        eps=5,
        **lachesis_options(dt_max=25),
    )

    assert result['user_id'].tolist() == ['a', 'b']
    assert result['cluster'].tolist() == [0, 0]
    assert 'location_id' not in result.columns


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

    result = lachesis(
        data,
        postprocessing='dbscan',
        eps=5,
        traj_cols=traj_cols,
        **lachesis_options(dt_max=25),
    )

    assert result['cluster'].tolist() == [0]
    assert result['east'].tolist() == [0.0]
    assert result['north'].tolist() == [0.0]
    assert 'destination_id' not in result.columns


def test_lachesis_dbscan_handles_no_detected_stops():
    data = pd.DataFrame(
        {
            'timestamp': [0, 60],
            'x': [0.0, 1000.0],
            'y': [0.0, 1000.0],
        }
    )

    result = lachesis(
        data,
        postprocessing='dbscan',
        eps=5,
        **lachesis_options(),
    )

    assert result.empty
    assert 'location_id' not in result.columns


def test_lachesis_dbscan_requires_eps(interrupted_destination_trajectory):
    with pytest.raises(ValueError, match='eps is required'):
        lachesis(
            interrupted_destination_trajectory,
            postprocessing='dbscan',
            **lachesis_options(),
        )


def test_lachesis_infomap_reports_unimplemented(
    interrupted_destination_trajectory,
):
    with pytest.raises(NotImplementedError, match='infomap'):
        lachesis(
            interrupted_destination_trajectory,
            postprocessing='infomap',
            **lachesis_options(),
        )


def test_lachesis_rejects_unknown_postprocessing(
    interrupted_destination_trajectory,
):
    with pytest.raises(NotImplementedError, match='kmeans'):
        lachesis(
            interrupted_destination_trajectory,
            postprocessing='kmeans',
            **lachesis_options(),
        )
