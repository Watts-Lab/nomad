import pandas as pd
import numpy as np
from functools import partial
import nomad.stop_detection.utils as utils
import nomad.stop_detection.grid_based as GRID_BASED 
from nomad.contact_estimation import overlapping_visits

def remove_overlaps(pred, truth, time_thresh, traj_cols = None, poi_table = None, post_processing = 'polygon'):
    summarize_stops_with_loc = partial(utils.summarize_stop, x='x', y='y', keep_col_names=False, passthrough_cols = ['building_id'])

    if  post_processing == 'polygon':
        labels = GRID_BASED.grid_based_labels(
                                data=pred.drop('cluster', axis=1),
                                time_thresh=time_thresh,
                                min_pts=0, #we allow stops of duration 0, patched later
                                location_id='building_id',
                                traj_cols=traj_cols)
                
        pred['cluster'] = labels

        stops = pred[pred.cluster!=-1].groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)

        overlaps = overlapping_visits(left=stops,
                                      right=truth,
                                      location_id='building_id',
                                      match_location=True)
        return overlaps
    elif post_processing == 'cluster':
        labels = GRID_BASED.grid_based_labels(data=pred.drop('cluster', axis=1),
                                              time_thresh=time_thresh,
                                              min_pts=0, #we allow stops of duration 0, patched later
                                              location_id='cluster',
                                              traj_cols=traj_cols)
                
        pred['cluster'] = labels

        stops = pred[pred.cluster!=-1].groupby('cluster', as_index=False).apply(summarize_stops_with_loc, include_groups=False)
        
        overlaps = overlapping_visits(left=stops,
                                      right=truth,
                                      location_id='building_id',
                                      match_location=True)
    elif post_processing == 'recurse':
        return


