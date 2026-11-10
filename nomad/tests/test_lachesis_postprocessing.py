import pandas as pd
import pytest

from nomad.stop_detection.sequential_algs import (
    lachesis,
    lachesis_labels,
    lachesis_labels_per_user,
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


@pytest.fixture(params=['dbscan', 'infomap'])
def postprocessing_options(request):
    if request.param == 'infomap':
        pytest.importorskip('infomap')
        return {'postprocessing': 'infomap', 'eps': 5, 'infomap_kwargs': {'sigma': 2}}
    return {'postprocessing': 'dbscan', 'eps': 5}


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


def test_lachesis_postprocessing_clusters_and_merges_interrupted_visits(
    interrupted_destination_trajectory,
    postprocessing_options,
):
    result = lachesis(
        interrupted_destination_trajectory,
        **postprocessing_options,
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
    postprocessing_options,
):
    labels = lachesis_labels(
        interrupted_destination_trajectory,
        delta_roam=5,
        dt_max=25,
        dur_min=5,
        **postprocessing_options,
    )

    assert labels.tolist() == [0, 0, 0, -1, 0, 0, 0]


def test_lachesis_postprocessing_respects_merge_threshold(
    interrupted_destination_trajectory,
    postprocessing_options,
):
    result = lachesis(
        interrupted_destination_trajectory,
        **postprocessing_options,
        **lachesis_options(),
    )

    assert len(result) == 2
    assert result['cluster'].tolist() == [0, 1]


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_lachesis_per_user_postprocesses_each_user_separately(
    interrupted_destination_trajectory,
    postprocessing_options, n_jobs,
):
    first = interrupted_destination_trajectory.assign(user_id='a')
    second = interrupted_destination_trajectory.assign(user_id='b')
    data = pd.concat([first, second], ignore_index=True)

    result = lachesis_per_user(
        data, n_jobs=n_jobs,
        **postprocessing_options,
        **lachesis_options(dt_max=25),
    )

    assert result['user_id'].tolist() == ['a', 'b']
    assert result['cluster'].tolist() == [0, 0]
    assert 'location_id' not in result.columns


def test_lachesis_postprocessing_supports_custom_columns(postprocessing_options):
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
        **postprocessing_options,
        traj_cols=traj_cols,
        **lachesis_options(dt_max=25),
    )

    assert result['cluster'].tolist() == [0]
    assert result['east'].tolist() == [0.0]
    assert result['north'].tolist() == [0.0]
    assert 'destination_id' not in result.columns


def test_lachesis_postprocessing_handles_no_detected_stops(postprocessing_options):
    data = pd.DataFrame(
        {
            'timestamp': [0, 60],
            'x': [0.0, 1000.0],
            'y': [0.0, 1000.0],
        }
    )

    result = lachesis(
        data,
        **postprocessing_options,
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


def test_lachesis_infomap_requires_sigma(interrupted_destination_trajectory):
    pytest.importorskip('infomap')
    with pytest.raises(ValueError, match='sigma'):
        lachesis(
            interrupted_destination_trajectory,
            postprocessing='infomap', eps=5,
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


@pytest.mark.parametrize('datetime_input', [False, True])
@pytest.mark.parametrize('dt_max, expected', [
    (10, [0, 0, 0, -1, 1, 1, 1]),
    (25, [0, 0, 0, -1, 0, 0, 0]),
])
def test_lachesis_postprocessing_preserves_anchors_and_user_alignment(
    interrupted_destination_trajectory, postprocessing_options, dt_max, expected,
    datetime_input,
):
    data = pd.concat([
        interrupted_destination_trajectory.assign(user_id='a'),
        interrupted_destination_trajectory.assign(user_id='b'),
    ], ignore_index=True).sort_values('timestamp', kind='stable')
    if datetime_input:
        data['datetime'] = pd.to_datetime(data.timestamp, unit='s', utc=True)
        data = data.drop(columns='timestamp')
    original = lachesis_labels_per_user(
        data, return_anchors=True, **lachesis_options(dt_max),
    )
    result = lachesis_labels_per_user(
        data, return_anchors=True,
        **postprocessing_options, **lachesis_options(dt_max),
    )
    pd.testing.assert_series_equal(result['anchor_time'], original['anchor_time'])
    assert result.index.equals(data.index)
    for user in ['a', 'b']:
        assert result.loc[data.user_id == user, 'cluster'].tolist() == expected


def test_lachesis_infomap_does_not_merge_across_another_location():
    pytest.importorskip('infomap')
    data = pd.DataFrame({
        'timestamp': [0, 300, 600, 900, 1200, 1500],
        'x': [0., 0., 100., 100., 1., 1.], 'y': 0.,
    })
    result = lachesis_labels(
        data, postprocessing='infomap', eps=5, infomap_kwargs={'sigma': 2},
        **lachesis_options(dt_max=25),
    )
    assert result.tolist() == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize('options, message', [
    ({'postprocessing': 'dbscan', 'infomap_kwargs': {'sigma': 2}}, 'requires'),
    ({'postprocessing': 'infomap', 'infomap_kwargs': {'sigam': 2}}, 'Unknown'),
    ({'postprocessing': 'infomap', 'infomap_kwargs': {'sigma': 2}}, 'eps is required'),
])
def test_lachesis_rejects_invalid_infomap_options(
    interrupted_destination_trajectory, options, message,
):
    with pytest.raises(ValueError, match=message):
        lachesis(interrupted_destination_trajectory, **options, **lachesis_options())
