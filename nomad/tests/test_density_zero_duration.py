"""Optional zero-duration filtering, independent of duplicate-time graph support."""

import numpy as np
import pandas as pd
import pytest

from nomad.stop_detection import density_algs as density


ALGORITHMS = [
    (density.ta_dbscan_labels, density.ta_dbscan_labels_per_user),
    (density.dbstop_labels, density.dbstop_labels_per_user),
    (density.seqscan_labels, density.seqscan_labels_per_user),
]


def trajectory(seconds, time_kind):
    times = pd.Series(seconds)
    if time_kind != 'numeric':
        times = pd.to_datetime(times, unit='s', utc=True)
        if time_kind == 'naive':
            times = times.dt.tz_localize(None)
    return pd.DataFrame({
        'when': times.to_numpy() if time_kind != 'aware' else times,
        'east': [0, 100, 100, 200][:len(seconds)],
        'north': np.zeros(len(seconds)),
    }).set_axis([9, 3, 8, 2][:len(seconds)])


def options(fn, time_kind, return_cores):
    return dict(
        dist_thresh=5, min_pts=0, time_thresh=1, return_cores=return_cores,
        traj_cols={
            'timestamp' if time_kind == 'numeric' else 'datetime': 'when',
            'x': 'east', 'y': 'north',
        },
        **({'dur_min': 0} if fn is density.seqscan_labels else {}),
    )


@pytest.mark.parametrize('fn,wrapper', ALGORITHMS)
@pytest.mark.parametrize('time_kind', ['numeric', 'naive', 'aware'])
@pytest.mark.parametrize('return_cores', [False, True])
def test_zero_duration_filter_preserves_survivors(fn, wrapper, time_kind, return_cores):
    data = trajectory([0, 100, 110, 200], time_kind)
    kwargs = options(fn, time_kind, return_cores)
    baseline = fn(data, **kwargs)
    explicit_false = fn(data, remove_dur_zero_stops=False, **kwargs)
    filtered = fn(data, remove_dur_zero_stops=True, **kwargs)
    assert_equal = pd.testing.assert_frame_equal if return_cores else pd.testing.assert_series_equal
    assert_equal(baseline, explicit_false)
    assert filtered.index.equals(data.index)
    labels = filtered['cluster'] if return_cores else filtered
    assert labels.tolist() == [-1, 1, 1, -1]
    assert_equal(filtered.iloc[1:3], baseline.iloc[1:3])
    if return_cores:
        assert filtered['core'].tolist() == [-1, 1, 1, -1]
        assert filtered['promotion_time'].iloc[[0, 3]].isna().all()
        assert filtered.dtypes.equals(baseline.dtypes)
    else:
        assert filtered.dtype == baseline.dtype
        assert filtered.name == baseline.name


@pytest.mark.parametrize('fn,wrapper', ALGORITHMS)
@pytest.mark.parametrize('time_kind', ['numeric', 'aware'])
@pytest.mark.parametrize('return_cores', [False, True])
@pytest.mark.parametrize('case', ['excluded', 'noise', 'empty'])
def test_zero_duration_filter_degenerate_outputs(fn, wrapper, time_kind, return_cores, case):
    data = trajectory([0], time_kind)
    kwargs = options(fn, time_kind, return_cores)
    if case == 'noise':
        kwargs['min_pts'] = 1
    elif case == 'empty':
        data = data.iloc[:0]
    baseline = fn(data, **kwargs)
    result = fn(data, remove_dur_zero_stops=True, **kwargs)
    assert len(result) == len(data)
    if case == 'noise':
        if return_cores:
            pd.testing.assert_frame_equal(result, baseline)
        else:
            pd.testing.assert_series_equal(result, baseline)
        return
    if return_cores:
        assert result.dtypes.equals(baseline.dtypes)
        assert (result['cluster'] == -1).all()
        assert (result['core'] == -1).all()
        assert result['promotion_time'].isna().all()
    else:
        assert result.dtype == baseline.dtype
        assert (result == -1).all()


@pytest.mark.parametrize('fn,wrapper', ALGORITHMS)
@pytest.mark.parametrize('return_cores', [False, True])
def test_zero_duration_filter_per_user_forwarding(fn, wrapper, return_cores):
    first = trajectory([0, 100, 110, 200], 'numeric').assign(uid='a')
    second = first.copy().assign(uid='b').set_axis([19, 13, 18, 12])
    data = pd.concat([first, second]).iloc[[0, 4, 1, 5, 2, 6, 3, 7]]
    kwargs = options(fn, 'numeric', return_cores)
    kwargs['traj_cols']['user_id'] = 'uid'
    result = wrapper(data, remove_dur_zero_stops=True, **kwargs)
    expected = pd.concat([
        fn(first, remove_dur_zero_stops=True, **kwargs),
        fn(second, remove_dur_zero_stops=True, **kwargs),
    ]).reindex(data.index)
    if return_cores:
        pd.testing.assert_frame_equal(result, expected)
    else:
        pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize('time_kind', ['numeric', 'datetime'])
def test_zero_duration_mask_repeated_times_and_fractional_duration(time_kind):
    times = pd.Series([0, 0, 10, 10.001, 20, 30, 30])
    if time_kind == 'datetime':
        times = pd.to_datetime(times, unit='s', utc=True)
    labels = pd.Series([4, 4, 7, 7, 12, -1, -1], index=[9, 8, 7, 6, 5, 4, 3])
    mask = density._zero_duration_cluster_mask(labels, times)
    assert mask.tolist() == [True, True, False, False, True, False, False]


@pytest.mark.parametrize('fn,wrapper', ALGORITHMS)
def test_default_skips_zero_duration_helper(fn, wrapper, monkeypatch):
    def unexpected_call(*args):
        pytest.fail('The zero-duration helper must be skipped when disabled')
    monkeypatch.setattr(density, '_zero_duration_cluster_mask', unexpected_call)
    data = trajectory([0, 100, 110, 200], 'numeric')
    fn(data, **options(fn, 'numeric', False))
    fn(data, remove_dur_zero_stops=False, **options(fn, 'numeric', True))
