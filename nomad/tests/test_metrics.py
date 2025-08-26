import pytest
import pandas as pd
import numpy as np
from nomad.metrics.metrics import rog

def test_rog_latlon_daily_hourly():
    stops = pd.DataFrame({
        'latitude': [40.0, 40.0, 41.0, 41.0],
        'longitude': [-75.0, -75.0, -76.0, -76.0],
        'datetime': pd.to_datetime(['2025-01-01 08:00', '2025-01-01 12:00', '2025-01-05 08:00', '2025-01-05 12:00']),
        'duration': [3600, 3600, 3600, 3600],
        'user_id': [1, 1, 1, 1]
    })

    # Daily
    out_daily = rog(stops, agg_freq='D', weighted=True, traj_cols={'latitude':'latitude','longitude':'longitude','timestamp':'datetime','duration':'duration','user_id':'user_id'})
    assert len(out_daily) == 2

    # Hourly
    out_hourly = rog(stops, agg_freq='H', weighted=True, traj_cols={'latitude':'latitude','longitude':'longitude','timestamp':'datetime','duration':'duration','user_id':'user_id'})
    assert len(out_hourly) == 4

def test_rog_xy_daily_hourly():
    stops = pd.DataFrame({
        'x': [0, 0, 10, 10],
        'y': [0, 10, 0, 10],
        'timestamp': [0, 3600, 86400, 90000],
        'duration': [3600, 3600, 3600, 3600],
        'user_id': [1, 1, 1, 1]
    })

    # Daily
    out_daily = rog(stops, agg_freq='D', weighted=True, traj_cols={'x':'x','y':'y','timestamp':'timestamp','duration':'duration','user_id':'user_id'})
    assert len(out_daily) == 2
    
    # Hourly
    out_hourly = rog(stops, agg_freq='H', weighted=True, traj_cols={'x':'x','y':'y','timestamp':'timestamp','duration':'duration','user_id':'user_id'})
    assert len(out_hourly) == 4 

def test_rog_edge_cases():
    # Test for a single stop, rog = 0
    stops = pd.DataFrame({
        'x': [0], 'y': [0], 'timestamp': [0], 'duration': [3600], 'user_id': [1]
    })
    
    out = rog(stops, agg_freq='D', weighted=True, traj_cols={'x':'x','y':'y','timestamp':'timestamp','duration':'duration','user_id':'user_id'})
    assert np.allclose(out['rog'], 0)

    # Test for all stops at the same location, rog = 0
    stops = pd.DataFrame({
        'x': [1,1,1], 'y': [2,2,2], 'timestamp': [0,3600,7200], 'duration': [3600,3600,3600], 'user_id': [1,1,1]
    })
    out = rog(stops, agg_freq='D', weighted=True, traj_cols={'x':'x','y':'y','timestamp':'timestamp','duration':'duration','user_id':'user_id'})
    assert np.allclose(out['rog'], 0)

def test_rog_output_vs_expected():
    # two stops at (0,0) and (0,2), equal duration
    stops = pd.DataFrame({
        'x': [0,0], 'y': [0,2], 'timestamp': [0,3600], 'duration': [3600,3600], 'user_id': [1,1]
    })
    out = rog(stops, agg_freq='D', weighted=True, traj_cols={'x':'x','y':'y','timestamp':'timestamp','duration':'duration','user_id':'user_id'})

    # Centroid = (0,1), distances are 1, rog = sqrt(mean([1,1])) = 1
    assert np.allclose(out['rog'], 1)