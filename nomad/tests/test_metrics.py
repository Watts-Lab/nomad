import pytest
import pandas as pd
import numpy as np
from nomad.metrics.metrics import rog, self_containment
from pathlib import Path
from nomad.io import base as loader
from nomad.stop_detection import lachesis as LACHESIS
import nomad.stop_detection.utils as utils

@pytest.fixture
def agent_traj_ground_truth():
    test_dir = Path(__file__).resolve().parent
    traj_path = test_dir.parent / "data" / "gc_3_stops.csv"
    df = loader.from_file(traj_path, timestamp='unix_timestamp', datetime='local_timestamp', user_id='identifier')
    # print(pd.to_datetime(df['local_timestamp'], unit='s').dt.date.unique())
    traj_cols = {'user_id':'identifier',
                 'x':'x',
                 'y':'y',
                 'timestamp':'unix_timestamp'}
    lachesis_out = LACHESIS.lachesis(
        data=df,
        dt_max=45,
        delta_roam=60,
        dur_min=3,
        traj_cols=traj_cols,
        complete_output=True,
        keep_col_names=False
    )
    # print(pd.to_datetime(lachesis_out['start_timestamp'], unit='s').dt.strftime('%Y-%m-%d %H').unique())
    # print(pd.to_datetime(lachesis_out['end_timestamp'], unit='s').dt.strftime('%Y-%m-%d %H').unique())
    return lachesis_out

def test_rog_output_size(agent_traj_ground_truth):
    # print(agent_traj_ground_truth)
    # Daily
    out_daily = rog(agent_traj_ground_truth,
                    agg_freq='d',
                    weighted=True,
                    traj_cols={'x':'x','y':'y','duration':'duration', 'user_id':'identifier'},
                    start_col="start_timestamp",
                    end_col="end_timestamp",
                    use_datetime=False)
    # print(out_daily)
    assert len(out_daily) == 1

    # Weekly
    out_weekly = rog(agent_traj_ground_truth,
                     agg_freq='w',
                     weighted=True,
                     traj_cols={'x':'x','y':'y','duration':'duration', 'user_id':'identifier'},
                     start_col="start_timestamp",
                     end_col="end_timestamp",
                     use_datetime=False)
    assert len(out_weekly) == 1

def test_rog_daily(agent_traj_ground_truth):
    # print(agent_traj_ground_truth)
    out_daily = rog(agent_traj_ground_truth,
                    agg_freq='d',
                    weighted=True,
                    traj_cols={'x':'x','y':'y','duration':'duration', 'user_id':'identifier'},
                    start_col="start_timestamp",
                    end_col="end_timestamp",
                    use_datetime=False)
    assert np.allclose(out_daily['rog'], 39.916789)

def test_rog_single_stop_straddling_days_and_weeks():
    # Stop starts Sunday 23:00, ends Monday 01:00
    stops = pd.DataFrame({
        'x': [10],
        'y': [10],
        'duration': [120],
        'start_timestamp': [pd.Timestamp('2024-06-02 23:00')],
        'end_timestamp': [pd.Timestamp('2024-06-03 01:00')],
        'user_id': [1]
    })

    # Daily
    out_daily = rog(
        stops,
        agg_freq='d',
        weighted=True,
        traj_cols={'x':'x','y':'y','duration':'duration','user_id':'user_id'},
        start_col="start_timestamp",
        end_col="end_timestamp",
        use_datetime=True,
        exploded=True
    )

    # print(out_daily)
    assert len(out_daily) == 2

    # Weekly
    out_weekly = rog(
        stops,
        agg_freq='w',
        weighted=True,
        traj_cols={'x':'x','y':'y','duration':'duration','user_id':'user_id'},
        start_col="start_timestamp",
        end_col="end_timestamp",
        use_datetime=True,
        exploded=True
    )
    print(out_weekly)

    assert len(out_weekly) == 2

def test_self_containment_basic():
    """Test basic self-containment calculation with simple data."""
    # Create test data with home at (0, 0) and activities at various distances
    stops = pd.DataFrame({
        'x': [0, 10, 20, 30],  # Home at 0, others at increasing distances
        'y': [0, 0, 0, 0],
        'duration': [60, 30, 30, 30],  # 60 min home, 30 min each for others
        'start_timestamp': pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 01:00',
            '2024-01-01 02:00',
            '2024-01-01 03:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2024-01-01 01:00',
            '2024-01-01 01:30',
            '2024-01-01 02:30',
            '2024-01-01 03:30'
        ]),
        'activity_type': ['Home', 'Work', 'Shopping', 'Restaurant'],
        'user_id': [1, 1, 1, 1]
    })

    # Threshold of 15 meters - should capture Work (10m) but not Shopping (20m) or Restaurant (30m)
    result = self_containment(
        stops,
        threshold=15,
        agg_freq='d',
        weighted=True,
        traj_cols={'x': 'x', 'y': 'y', 'duration': 'duration', 'user_id': 'user_id'},
        start_col='start_timestamp',
        end_col='end_timestamp',
        use_datetime=True,
        exploded=False
    )

    # Only Work (30 min) is within threshold out of 90 min total non-home time
    # Expected: 30/90 = 0.333...
    assert len(result) == 1
    assert np.allclose(result['self_containment'].values[0], 30/90)

def test_self_containment_unweighted():
    """Test unweighted self-containment (count of activities)."""
    stops = pd.DataFrame({
        'x': [0, 10, 20, 30],
        'y': [0, 0, 0, 0],
        'duration': [60, 10, 50, 30],  # Different durations
        'start_timestamp': pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 01:00',
            '2024-01-01 02:00',
            '2024-01-01 03:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2024-01-01 01:00',
            '2024-01-01 01:10',
            '2024-01-01 02:50',
            '2024-01-01 03:30'
        ]),
        'activity_type': ['Home', 'Work', 'Shopping', 'Restaurant'],
        'user_id': [1, 1, 1, 1]
    })

    result = self_containment(
        stops,
        threshold=15,
        agg_freq='d',
        weighted=False,  # Unweighted
        traj_cols={'x': 'x', 'y': 'y', 'duration': 'duration', 'user_id': 'user_id'},
        start_col='start_timestamp',
        end_col='end_timestamp',
        use_datetime=True,
        exploded=False
    )

    # 1 out of 3 non-home activities is within threshold
    assert len(result) == 1
    assert np.allclose(result['self_containment'].values[0], 1/3)

def test_self_containment_multi_user():
    """Test self-containment with multiple users."""
    stops = pd.DataFrame({
        'x': [0, 5, 0, 50],  # User 1: home at 0, work at 5; User 2: home at 0, work at 50
        'y': [0, 0, 0, 0],
        'duration': [60, 30, 60, 30],
        'start_timestamp': pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 01:00',
            '2024-01-01 00:00',
            '2024-01-01 01:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2024-01-01 01:00',
            '2024-01-01 01:30',
            '2024-01-01 01:00',
            '2024-01-01 01:30'
        ]),
        'activity_type': ['Home', 'Work', 'Home', 'Work'],
        'user_id': [1, 1, 2, 2]
    })

    result = self_containment(
        stops,
        threshold=10,  # User 1's work is within, User 2's is not
        agg_freq='d',
        weighted=True,
        traj_cols={'x': 'x', 'y': 'y', 'duration': 'duration', 'user_id': 'user_id'},
        start_col='start_timestamp',
        end_col='end_timestamp',
        use_datetime=True,
        exploded=False
    )

    assert len(result) == 2
    # User 1: all non-home time (30 min) is within threshold
    user1_result = result[result['user_id'] == 1]['self_containment'].values[0]
    assert np.allclose(user1_result, 1.0)

    # User 2: no non-home time is within threshold
    user2_result = result[result['user_id'] == 2]['self_containment'].values[0]
    assert np.allclose(user2_result, 0.0)

def test_self_containment_no_home():
    """Test self-containment when there's no home location."""
    stops = pd.DataFrame({
        'x': [10, 20, 30],
        'y': [0, 0, 0],
        'duration': [30, 30, 30],
        'start_timestamp': pd.to_datetime([
            '2024-01-01 01:00',
            '2024-01-01 02:00',
            '2024-01-01 03:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2024-01-01 01:30',
            '2024-01-01 02:30',
            '2024-01-01 03:30'
        ]),
        'activity_type': ['Work', 'Shopping', 'Restaurant'],
        'user_id': [1, 1, 1]
    })

    result = self_containment(
        stops,
        threshold=15,
        agg_freq='d',
        weighted=True,
        traj_cols={'x': 'x', 'y': 'y', 'duration': 'duration', 'user_id': 'user_id'},
        start_col='start_timestamp',
        end_col='end_timestamp',
        use_datetime=True,
        exploded=False
    )

    # Should return NaN when there's no home location
    assert len(result) == 1
    assert pd.isna(result['self_containment'].values[0])

def test_self_containment_all_home():
    """Test self-containment when all activities are at home."""
    stops = pd.DataFrame({
        'x': [0, 0, 0],
        'y': [0, 0, 0],
        'duration': [30, 30, 30],
        'start_timestamp': pd.to_datetime([
            '2024-01-01 01:00',
            '2024-01-01 02:00',
            '2024-01-01 03:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2024-01-01 01:30',
            '2024-01-01 02:30',
            '2024-01-01 03:30'
        ]),
        'activity_type': ['Home', 'Home', 'Home'],
        'user_id': [1, 1, 1]
    })

    result = self_containment(
        stops,
        threshold=15,
        agg_freq='d',
        weighted=True,
        traj_cols={'x': 'x', 'y': 'y', 'duration': 'duration', 'user_id': 'user_id'},
        start_col='start_timestamp',
        end_col='end_timestamp',
        use_datetime=True,
        exploded=False
    )

    # Should return NaN when there are no non-home activities
    assert len(result) == 1
    assert pd.isna(result['self_containment'].values[0])

def test_self_containment_with_time_weights():
    """Test self-containment with additional time weights."""
    stops = pd.DataFrame({
        'x': [0, 10, 20, 30],
        'y': [0, 0, 0, 0],
        'duration': [60, 30, 30, 30],
        'start_timestamp': pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 01:00',
            '2024-01-01 02:00',
            '2024-01-01 03:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2024-01-01 01:00',
            '2024-01-01 01:30',
            '2024-01-01 02:30',
            '2024-01-01 03:30'
        ]),
        'activity_type': ['Home', 'Work', 'Shopping', 'Restaurant'],
        'user_id': [1, 1, 1, 1]
    })

    # Create time weights that double the weight of the first activity
    time_weights = pd.Series([1.0, 2.0, 1.0, 1.0], index=stops.index)

    result = self_containment(
        stops,
        threshold=15,
        agg_freq='d',
        weighted=True,
        time_weights=time_weights,
        traj_cols={'x': 'x', 'y': 'y', 'duration': 'duration', 'user_id': 'user_id'},
        start_col='start_timestamp',
        end_col='end_timestamp',
        use_datetime=True,
        exploded=False
    )

    # Work (30 min * 2.0) = 60 weighted minutes within threshold
    # Total non-home: Work (60) + Shopping (30) + Restaurant (30) = 120 weighted minutes
    # Expected: 60/120 = 0.5
    assert len(result) == 1
    assert np.allclose(result['self_containment'].values[0], 60/120)

