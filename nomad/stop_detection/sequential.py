import pandas as pd
import nomad.stop_detection.lachesis as LACHESIS

'''
sd.sequential(user_data, traj_cols) # default to lachesis
sd.sequential(all_data, location_id='build_id', traj_cols) # defaults to grid_based, handles multi-user automatically
'''

def sequential(data, dt_max, delta_roam, dur_min, location_id=None, traj_cols=None, method='lachesis'):
    if data[traj_cols['user_id']].nunique() > 1: #multiple users
        if method == 'Lachesis':
            stop_table = data.groupby(['uid']).apply(lambda x: LACHESIS.lachesis(data=x,
                                                                             dt_max=dt_max,
                                                                             delta_roam=delta_roam,
                                                                             dur_min=dur_min,
                                                                             traj_cols=traj_cols))
        else:
            return None
    else: # only 1 user
        if method == 'Lachesis':
            stop_table = LACHESIS.lachesis(data=data,
                                       dt_max=dt_max,
                                       delta_roam=delta_roam,
                                       dur_min=dur_min,
                                       traj_cols=traj_cols)
    
    return stop_table