"""Various algorithms for estimating individual co-location"""

# Authors: Thomas Li and Francisco Barreras

import nomad.io.base as loader
import nomad.stop_detection.hdbscan as HDBSCAN
import pandas as pd
import pdb

def overlapping_visits(left, right, match_location=False, traj_cols=None, **kwargs):
    # Handle column names
    _ = loader._parse_traj_cols(right.columns, traj_cols, kwargs) # for warning
    traj_cols = loader._parse_traj_cols(left.columns, traj_cols, kwargs)

    # Has required columns
    uid_key = traj_cols['user_id']
    loc_key = traj_cols['location_id']
    e_t_key = traj_cols['end_timestamp']
    
    #check for keys
    for df_name, df in {'left':left, 'right':right}.items():
        loader._has_time_cols(df.columns, traj_cols)
        
        end_col_present = loader._has_end_cols(df.columns, traj_cols)
        duration_col_present = loader._has_duration_cols(df.columns, traj_cols)
        if not (end_col_present or duration_col_present):
            print(f"Missing required (end or duration) temporal columns for {df_name} visits dataframe in columns {df.columns}.")
            
        if uid_key in df.columns and not (df[uid_key].nunique() == 1):
            raise ValueError("Each visits dataframe must have at most one unique user_id")
            
        if traj_cols['location_id'] not in df:
            raise ValueError(
                "Could not find required location column in {}. The dataset must contain or map to "
                "a location column'.".format(list(df.columns)))

    # Non-na locations and end_timestamp on copy    
    left = left.loc[~left[loc_key].isna()].copy()
    right = right.loc[~right[loc_key].isna()].copy()
            
    keep_uid = (uid_key in left.columns and uid_key in right.columns )
    if keep_uid:
        same_id = (left[uid_key].iloc[0] == right[uid_key].iloc[0])
        uid = left[uid_key].iloc[0]
   
    # Possibly different start time keys are valid
    t_keys = []
    for df in [left, right]:
        if traj_cols['timestamp'] in df.columns:
            t_key = traj_cols['timestamp']
        elif traj_cols['start_timestamp'] in df.columns:
            t_key = traj_cols['start_timestamp']
        else:
            # TO DO: implement to timestamp conversion of datetime
            raise ValueError('Overlap of visits with datetime64 objects not yet implemented')

        t_keys += [t_key]
        if e_t_key not in df.columns:
            df[e_t_key] = df[t_key] + df[traj_cols['duration']]*60 # will fail if t_key is datetime!!
        
        cols = [t_key, e_t_key, loc_key]
        if keep_uid and not same_id:
            cols = [uid_key, t_key, e_t_key, loc_key]
        df.drop([col for col in df.columns if col not in cols], axis=1, inplace=True)

    # rename timekeys for now to avoid conflict
    right.rename({t_keys[1]:t_keys[0]}, axis=1, inplace=True)
    t_key = t_keys[0]
    
    if match_location:
        merged = left.merge(right, on=loc_key, suffixes=('_left','_right'))
    else:
        merged = left.merge(right, how='cross', suffixes=('_left','_right'))
    
    t_key_l = t_key+'_left'
    t_key_r = t_key+'_right'
    
    cond = ((merged[t_key_l] < merged[e_t_key+'_right']) &
            (merged[t_key_r] < merged[e_t_key+'_left']))
    merged = merged.loc[cond]

    start_max = merged[[t_key_l, t_key_r]].max(axis=1)
    end_min   = merged[[e_t_key+'_left',e_t_key+'_right']].min(axis=1)

    merged.drop([e_t_key+'_left', e_t_key+'_right'], axis=1)
    merged[traj_cols['duration']] = ((end_min - start_max) // 60).astype(int) #output names use traj_cols
    
    if keep_uid and same_id:
        merged[uid_key] = uid

    return merged.reset_index(drop=True)

def compute_visitation_errors(overlaps, true_visits, traj_cols=None, **kwargs):
    '''
    Assumes that the columns with the suffix '_right' in the overlaps dataframe correspond
    to columns from true_visits dataframe.
    ''' 
    true_visits = true_visits.dropna()
    
    stripped_col_names = [s.removesuffix('_left').removesuffix('_right') for s in overlaps.columns]
    
    _ = loader._parse_traj_cols(stripped_col_names, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(true_visits.columns, traj_cols, kwargs)

    loader._has_time_cols(true_visits.columns, traj_cols)
    if traj_cols['timestamp'] in true_visits.columns:
        t_key = traj_cols['timestamp']
    elif traj_cols['start_timestamp'] in true_visits.columns:
        t_key = traj_cols['start_timestamp']
    n_truth = true_visits[t_key].nunique()

    loader._has_time_cols(stripped_col_names, traj_cols)
    if traj_cols['timestamp'] in stripped_col_names:
        t_key_l = traj_cols['timestamp']+'_left'
        t_key_r = traj_cols['timestamp']+'_right'
    elif traj_cols['start_timestamp'] in stripped_col_names:
        t_key_l = traj_cols['start_timestamp']+'_left'
        t_key_r = traj_cols['start_timestamp']+'_right'

    loc_key_l = traj_cols['location_id']+'_left'
    loc_key_r = traj_cols['location_id']+'_right'

    # compute missed
    gt_overlapped = set(overlaps[t_key_r].unique()) # _right is for ground truth
    missed = (n_truth - len(gt_overlapped)) / n_truth

    # compute merging
    merged_ids = set()
    for pred_ts, group in overlaps.groupby(t_key_l):
        if group[loc_key_r].nunique() > 1:
            merged_ids.update(group[t_key_r].unique())
    merged = len(merged_ids) / n_truth

    # compute splitting
    split_ids = set()
    same_loc = overlaps[overlaps[loc_key_l] == overlaps[loc_key_r]] # _right corresponds to ground truth
    for gt_ts, group in same_loc.groupby(t_key_r): 
        if group[t_key_l].nunique() > 1:
            split_ids.add(gt_ts)
    split = len(split_ids) / n_truth

    return {'missed_fraction': missed,
            'merged_fraction': merged,
            'split_fraction': split}

def compute_precision_recall_f1(overlaps, pred_visits, true_visits, traj_cols=None, **kwargs):
    true_visits = true_visits.dropna()
    stripped_col_names = [s.removesuffix('_left').removesuffix('_right') for s in overlaps.columns]
    
    _ = loader._parse_traj_cols(stripped_col_names, traj_cols, kwargs)    
    _ = loader._parse_traj_cols(true_visits.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(pred_visits.columns, traj_cols, kwargs)

    d_key = traj_cols['duration']
    if d_key not in stripped_col_names:
        raise KeyError(f"Column {d_key} not found in columns {overlaps.columns} for 'overlaps' dataframe.")
    if d_key not in pred_visits.columns:
        raise KeyError(f"Column {d_key} not found in columns {pred_visits.columns} for 'pred_visits' dataframe.")
    if d_key not in true_visits.columns:
        raise KeyError(f"Column {d_key} not found in columns {true_visits.columns} for 'true_visits' dataframe.")
    
    
    total_pred = pred_visits[d_key].sum()
    total_truth = true_visits[d_key].sum()

    loc_key_r = traj_cols['location_id']+'_right'
    loc_key_l = traj_cols['location_id']+'_right'

    tp = overlaps.loc[overlaps[loc_key_r] == overlaps[loc_key_l], d_key].sum()
    
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_truth if total_truth > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)}
