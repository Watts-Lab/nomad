import pandas as pd
import numpy as np
from functools import partial
import nomad.stop_detection.utils as utils
import nomad.stop_detection.grid_based as GRID_BASED 

def remove_overlaps(pred, time_thresh, dur_min, min_pts, traj_cols = None, post_processing = 'polygon'):
    pred['building_id'] = pred['building_id'].fillna('None')

    if  post_processing == 'polygon':
        summarize_stops_with_loc = partial(utils.summarize_stop, x='x', y='y', keep_col_names=False, passthrough_cols = ['building_id'])
        labels = GRID_BASED.grid_based_labels(
                                data=pred.loc[pred.cluster!=-1],
                                time_thresh=time_thresh,
                                dur_min=dur_min,
                                min_cluster_size=min_pts,
                                location_id='building_id',
                                traj_cols=traj_cols)
                
        pred.loc[pred.cluster!=-1, 'cluster'] = labels
        stops = pred.loc[pred.cluster!=-1].groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)

    elif post_processing == 'cluster':
        labels = GRID_BASED.grid_based_labels(
                                data=pred.loc[pred.cluster!=-1],
                                time_thresh=time_thresh,
                                dur_min=dur_min,
                                min_cluster_size=min_pts,
                                location_id='cluster',
                                traj_cols=traj_cols)
        pred.loc[pred.cluster!=-1, 'cluster'] = labels
        stops = pred.groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)
    
    elif post_processing == 'recurse':
        pass

    return stops