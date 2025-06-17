import pandas as pd
import numpy as np
from functools import partial
import nomad.stop_detection.utils as utils
import nomad.stop_detection.grid_based as GRID_BASED 

def remove_overlaps(pred, time_thresh, traj_cols = None, post_processing = 'polygon'):
    if  post_processing == 'polygon':
        summarize_stops_with_loc = partial(utils.summarize_stop, x='x', y='y', keep_col_names=False, passthrough_cols = ['building_id'])
        labels = GRID_BASED.grid_based_labels(
                                data=pred.drop('cluster', axis=1),
                                time_thresh=time_thresh,
                                min_pts=0, #we allow stops of duration 0, patched later
                                location_id='building_id',
                                traj_cols=traj_cols)
                
        pred['cluster'] = labels
        stops = pred[pred.cluster!=-1].groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)
        return stops
    elif post_processing == 'cluster':
        summarize_stops_with_loc = partial(utils.summarize_stop, x='x', y='y', keep_col_names=False, passthrough_cols = ['cluster'])
        stops = pred[pred.cluster!=-1].groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)
        return stops
    elif post_processing == 'recurse':
        return


