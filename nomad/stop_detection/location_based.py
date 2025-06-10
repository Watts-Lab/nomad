import nomad.visit_attribution as va
import pandas as pd

def location_based(traj, traj_cols=None, time_buffer = 0, buffer_pings = "none", complete_output=False, keep_col_names = False, **kwargs):
    # TO DO: consecutive entries with the same "location" column