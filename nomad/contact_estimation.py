"""Various algorithms for estimating individual co-location"""

# Authors: Thomas Li and Francisco Barreras

import nomad.io.base as loader
import nomad.stop_detection.hdbscan as HDBSCAN
import nomad.stop_detection.utils as utils
import pandas as pd
import numpy as np
import pdb

def overlapping_visits(left, right, match_location=False, traj_cols=None, **kwargs):
    if len(left) == 0 or len(right) == 0:
        return pd.DataFrame()
    
    left = left.copy()
    right = right.copy()
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
    left.rename({t_keys[0]:t_keys[1]}, axis=1, inplace=True)
    t_key = t_keys[1] # keep the t_key in right

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
    """
    Return missed / merged / split fractions for a set of ground-truth stops
    and an overlap table produced by `overlapping_visits`.

    The start-timestamp column in `true_visits` *must* be unique; if not,
    a ValueError is raised.  No gap filling is performed here: the caller
    must supply already-filled tables where appropriate.
    """

    # 0 – make sure ground truth rows are valid
    true_visits = true_visits.dropna()
        
    # 1 – decide the canonical start-time column once
    t_key, _ = loader._fallback_time_cols_dt(true_visits.columns, traj_cols, kwargs)

    # 2 – check uniqueness of that column in truth
    if true_visits[t_key].duplicated().any():
        dup_ts = true_visits.loc[true_visits[t_key].duplicated(), t_key].unique()
        raise ValueError(
            "Ground-truth stops share the same start time(s), which violates the "
            "per-stop key assumption.  Duplicated timestamps: " + repr(dup_ts)
        )
    n_truth = len(true_visits)

    # 3 – determine the canonical location column
    loc_key = loader._parse_traj_cols(true_visits.columns, traj_cols, kwargs)['location_id']

    # 4 – required column names in the overlap table
    t_left   = f"{t_key}_left"
    t_right  = f"{t_key}_right"
    loc_left = f"{loc_key}_left"
    loc_right = f"{loc_key}_right"

    for col in (t_left, t_right, loc_left, loc_right):
        if col not in overlaps.columns:
            raise ValueError(
                f"compute_visitation_errors: expected column '{col}' in overlaps but not found."
            )

    overlaps = overlaps.fillna({loc_left: 'Street'}) # So they count on merge and missing

    # 5 – error if the overlap table references a start time not in ground truth
    bad_ts = set(overlaps[t_right]) - set(true_visits[t_key])
    if bad_ts:
        raise ValueError(
            "compute_visitation_errors: overlap rows reference start times that "
            "do not exist in ground truth: " + repr(sorted(bad_ts)[:10])
        )

    diff_loc = overlaps[loc_left] != overlaps[loc_right] #because gt "street" segments are always under a minute, then predicted "street" can't ever be correct
    same_loc = ~diff_loc

    num_overlapped = overlaps[t_right].nunique()
    missed_fraction = 1 - num_overlapped / n_truth

    merged_fraction = diff_loc.groupby(overlaps[t_right]).any().mean()
    split_fraction  = overlaps[same_loc].groupby(t_right)[t_left].nunique().gt(1).mean()

    return {'missed_fraction': missed_fraction, 'merged_fraction': merged_fraction, 'split_fraction': split_fraction}

def precision_recall_f1_from_minutes(total_pred, total_truth, tp):
    """Compute P/R/F1 from minute totals."""
    precision = tp / total_pred if total_pred else np.nan
    recall    = tp / total_truth if total_truth else np.nan
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif precision + recall == 0:            # both are 0 → F1 = 0
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}

def compute_stop_detection_metrics(stops, truth, user_id=None, algorithm=None, traj_cols=None, **kwargs):
    """
    Compute stop detection metrics for a single user/algorithm combination.
    
    Parameters
    ----------
    stops : pd.DataFrame
        Predicted stops with columns: building_id, duration, start_timestamp, etc.
    truth : pd.DataFrame  
        Ground truth stops with columns: building_id, duration, timestamp, etc.
    user_id : str, optional
        User identifier for the results
    algorithm : str, optional
        Algorithm name for the results
    traj_cols : dict, optional
        Column name mappings
        
    Returns
    -------
    dict
        Dictionary with metrics: precision, recall, f1, missed_fraction, 
        merged_fraction, split_fraction, user_id, algorithm
    """
    # Handle empty stops case
    if len(stops) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'missed_fraction': 1.0, 'merged_fraction': 0.0, 'split_fraction': 0.0,
            'user_id': user_id, 'algorithm': algorithm
        }
    
    # Prepare data - fill missing building_ids with 'Street'
    stops_clean = stops.fillna({'building_id': 'Street'})
    truth_clean = truth.fillna({'building_id': 'Street'})
    truth_buildings = truth.dropna()  # Only actual buildings for error analysis
    
    # Compute overlaps
    overlaps = overlapping_visits(
        left=stops_clean, right=truth_clean, match_location=False,
        traj_cols=traj_cols, **kwargs
    )
    
    # Precision/Recall: only matching locations
    loc_key = loader._parse_traj_cols(stops_clean.columns, traj_cols, kwargs)['location_id']
    loc_left = f"{loc_key}_left"
    loc_right = f"{loc_key}_right"
    correct_overlaps = overlaps[overlaps[loc_left] == overlaps[loc_right]]
    total_pred = stops_clean['duration'].sum()
    total_truth = truth_clean['duration'].sum()
    tp = correct_overlaps['duration'].sum()
    prf_metrics = precision_recall_f1_from_minutes(total_pred, total_truth, tp)
    
    # Error metrics: compare against buildings only
    if len(truth_buildings) > 0:
        overlaps_err = overlapping_visits(
            left=stops_clean, right=truth_buildings, match_location=False,
            traj_cols=traj_cols, **kwargs
        )
        error_metrics = compute_visitation_errors(overlaps_err, truth_buildings, traj_cols, **kwargs)
    else:
        error_metrics = {'missed_fraction': 0.0, 'merged_fraction': 0.0, 'split_fraction': 0.0}
    
    return {**prf_metrics, **error_metrics, 'user_id': user_id, 'algorithm': algorithm}
