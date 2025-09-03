import pytest
import pandas as pd
import numpy as np
from nomad.metrics.metrics import rog
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

