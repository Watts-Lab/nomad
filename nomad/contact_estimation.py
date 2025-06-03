"""Various algorithms for estimating individual co-location"""

# Authors: Thomas Li and Francisco Barreras

import nomad.io.base as loader
import nomad.stop_detection.hdbscan as HDBSCAN
import pandas as pd
import pdb

def overlapping_visits(df1, df2, match_location=False):
    df1 = df1.loc[~df1.location.isna()].copy()
    df2 = df2.loc[~df2.location.isna()].copy()
    # Raise error if either df has more than one distinct uid
    for df in (df1, df2):
        if 'uid' in df.columns and df['uid'].nunique() > 1:
            raise ValueError("Each dataframe must have at most one unique uid")
    keep_uid = ('uid' in df1.columns and 'uid' in df2.columns 
                and df1['uid'].iloc[0] != df2['uid'].iloc[0])

    df1['end_timestamp'] = df1['timestamp'] + df1['duration'] * 60
    df2['end_timestamp'] = df2['timestamp'] + df2['duration'] * 60

    if match_location:
        df1 = df1[df1['location'].notna()]
        df2 = df2[df2['location'].notna()]
        merged = df1.merge(df2, on='location', suffixes=('_left','_right'))
    else:
        merged = df1.merge(df2, how='cross', suffixes=('_left','_right'))

    cond = (
        (merged['timestamp_left'] < merged['end_timestamp_right']) &
        (merged['timestamp_right'] < merged['end_timestamp_left'])
    )
    merged = merged.loc[cond]

    start_max = merged[['timestamp_left','timestamp_right']].max(axis=1)
    end_min   = merged[['end_timestamp_left','end_timestamp_right']].min(axis=1)
    merged['overlap_duration'] = ((end_min - start_max) // 60).astype(int)

    if match_location:
        cols = ['timestamp_left','timestamp_right','location','overlap_duration']
    else:
        cols = ['timestamp_left','timestamp_right','location_left','location_right','overlap_duration']
    if keep_uid:
        cols += ['uid_left','uid_right']

    return merged[cols].reset_index(drop=True)

def compute_visitation_errors(overlaps, truth_df):
    n_truth = truth_df['timestamp'].nunique()
    gt_overlapped = set(overlaps['timestamp_right'].unique())
    missed = (n_truth - len(gt_overlapped)) / n_truth

    merged_ids = set()
    for pred_ts, group in overlaps.groupby('timestamp_left'):
        if group['location_right'].nunique() > 1:
            merged_ids.update(group['timestamp_right'].unique())
    merged = len(merged_ids) / n_truth

    split_ids = set()
    same_loc = overlaps[overlaps['location_left'] == overlaps['location_right']]
    for gt_ts, group in same_loc.groupby('timestamp_right'):
        if group['timestamp_left'].nunique() > 1:
            split_ids.add(gt_ts)
    split = len(split_ids) / n_truth

    return {'missed_fraction': missed,
            'merged_fraction': merged,
            'split_fraction': split}

    def overlapping_visits(df1, df2, match_location=False):
    df1 = df1.loc[~df1.location.isna()].copy()
    df2 = df2.loc[~df2.location.isna()].copy()
    # Raise error if either df has more than one distinct uid
    for df in (df1, df2):
        if 'uid' in df.columns and df['uid'].nunique() > 1:
            raise ValueError("Each dataframe must have at most one unique uid")
    keep_uid = ('uid' in df1.columns and 'uid' in df2.columns 
                and df1['uid'].iloc[0] != df2['uid'].iloc[0])

    df1['end_timestamp'] = df1['timestamp'] + df1['duration'] * 60
    df2['end_timestamp'] = df2['timestamp'] + df2['duration'] * 60

    if match_location:
        df1 = df1[df1['location'].notna()]
        df2 = df2[df2['location'].notna()]
        merged = df1.merge(df2, on='location', suffixes=('_left','_right'))
    else:
        merged = df1.merge(df2, how='cross', suffixes=('_left','_right'))

    cond = (
        (merged['timestamp_left'] < merged['end_timestamp_right']) &
        (merged['timestamp_right'] < merged['end_timestamp_left'])
    )
    merged = merged.loc[cond]

    start_max = merged[['timestamp_left','timestamp_right']].max(axis=1)
    end_min   = merged[['end_timestamp_left','end_timestamp_right']].min(axis=1)
    merged['overlap_duration'] = ((end_min - start_max) // 60).astype(int)

    if match_location:
        cols = ['timestamp_left','timestamp_right','location','overlap_duration']
    else:
        cols = ['timestamp_left','timestamp_right','location_left','location_right','overlap_duration']
    if keep_uid:
        cols += ['uid_left','uid_right']

    return merged[cols].reset_index(drop=True)


def compute_precision_recall_f1(overlaps, pred_df, truth_df):
    total_pred = pred_df['duration'].sum()
    total_truth = truth_df['duration'].sum()

    tp = overlaps.loc[
        overlaps['location_left'] == overlaps['location_right'],
        'overlap_duration'
    ].sum()

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_truth if total_truth > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {'precision': precision,
            'recall': recall,
            'f1': f1}

def prepare_stop_table(stop_table, diary, end_timestamp = None):
    """
    Map detected stops in `stop_table` to a list of true diary stops by overlapping location and timeframe.

    Parameters
    ----------
    stop_table : pd.DataFrame
        DataFrame of detected stops with at least 'local_timestamp', 'duration', and 'location' columns.
    diary : pd.DataFrame
        DataFrame of diary entries with at least 'local_timestamp', 'duration', and 'location' columns.
    
    Returns
    -------
    prepared_stop_table : pd.DataFrame
        The stop_table expanded to minute-level granularity.
    stop_table : pd.DataFrame
        The merged and updated stop_table before minute-level expansion.
    """

    # Compute end times of stops
    stop_table['end_time'] = stop_table['start_time'] + pd.to_timedelta(stop_table['duration'], unit='m')

    temp_df = stop_table.merge(diary, on="location", suffixes=("_st", "_d"))

    # Filter rows where time windows overlap
    temp_df = temp_df[
        (temp_df["start_time_st"] <= temp_df["end_time_d"]) &
        (temp_df["end_time_st"] >= temp_df["start_time_d"])
    ]

    # Aggregate all matching stop_ids into a list
    temp_df = (
        temp_df.groupby(["start_time_st", "end_time_st", "location"])['stop_id']
        .apply(list) 
        .reset_index()
        .rename(columns={'stop_id': 'mapped_stop_ids'})
    )

    return temp_df

    # Merge back with original stop_table
    stop_table = stop_table.merge(
        temp_df, 
        left_on=["start_time", "end_time", "location"],
        right_on=["start_time_st", "end_time_st", "location"],
        how="left"
    ).drop(columns=["start_time_st", "end_time_st"])

    return stop_table

def cluster_metrics(stop_table, agent, city):
    """
    Multiclass classification: compute precision, recall for each class separately,
    then use microaveraging to get the overall precision and recall.
    We could also try duration-weighted macroaveraging.
    """

    # Prepare diary: create stop IDs
    dt = agent.dt
    diary = prepare_diary(agent, dt, city)

    # Prepare stop table: map detected stops to diary stops
    stop_table = prepare_stop_table(stop_table, diary, dt)

    # Count number of rows in stop_table for each stop_id in diary
    stop_table_exploded = stop_table.explode("mapped_stop_ids").rename(columns={"mapped_stop_ids": "stop_id"})
    stop_counts = stop_table_exploded["stop_id"].value_counts().reset_index()
    stop_counts.columns = ["stop_id", "count"]

    # Compute the overlap in minutes between two time intervals.
    def compute_overlap(a_start, a_end, b_start, b_end):
        latest_start = max(a_start, b_start)
        earliest_end = min(a_end, b_end)
        overlap = (earliest_end - latest_start).total_seconds() / 60
        return max(0, overlap)

    tp_fp_fn = {}
    merging_dict = {}
    merged_stop_count_dict = {}

    # Efficient lookup
    diary_indexed = diary.set_index("stop_id")

    # First: true positive / false positive / false negative
    for stop_id in diary["stop_id"].unique():
        if stop_id == -1:
            continue

        diary_start = diary_indexed.loc[stop_id]["start_time"]
        diary_end = diary_indexed.loc[stop_id]["end_time"]
        
        diary_duration = (diary_end - diary_start).total_seconds() / 60
        
        if stop_id in stop_table_exploded["stop_id"].values:
            stop_row = stop_table_exploded[stop_table_exploded["stop_id"] == stop_id].iloc[0]
            stop_start = stop_row["start_time"]
            stop_end = stop_row["end_time"]
            stop_duration = (stop_end - stop_start).total_seconds() / 60

            tp = compute_overlap(stop_start, stop_end, diary_start, diary_end)
            fp = stop_duration - tp
            fn = diary_duration - tp
        else:
            tp = 0
            fp = diary_duration
            fn = 0

        tp_fp_fn[stop_id] = [tp, fp, fn]

    # Second: merging minutes per diary stop
    for _, d in diary[diary["stop_id"] != -1].iterrows():
        d_sid = d["stop_id"]
        d_start = d["start_time"]
        d_end = d["end_time"]

        overlaps = stop_table_exploded[
            (stop_table_exploded["end_time"] > d_start) &
            (stop_table_exploded["start_time"] < d_end)
        ]

        merging_minutes = 0
        merged_stop_ids_seen = set()

        for _, s in overlaps.iterrows():
            s_sid = s["stop_id"]
            if s_sid != d_sid:
                overlap_start = max(d_start, s["start_time"])
                overlap_end = min(d_end, s["end_time"])
                delta = (overlap_end - overlap_start).total_seconds() / 60
                minutes = max(0, delta)
            
                if minutes > 0:
                    merging_minutes += minutes
                    merged_stop_ids_seen.add(s_sid)

        if d_sid not in merging_dict:
            merging_dict[d_sid] = 0
        if d_sid not in merged_stop_count_dict:
            merged_stop_count_dict[d_sid] = 0

        merging_dict[d_sid] += merging_minutes
        merged_stop_count_dict[d_sid] += len(merged_stop_ids_seen)

    # Collect
    metrics_data = []
    for stop_id in diary["stop_id"].unique():
        if stop_id == -1:
            continue

        tp, fp, fn = tp_fp_fn.get(stop_id)
        mm = merging_dict.get(stop_id)
        sm = merged_stop_count_dict.get(stop_id)

        metrics_data.append({
            "stop_id": stop_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) != 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) != 0 else 0,
            "pings_merged": mm,
            "stops_merged": sm,
            "prop_merged": mm / (tp + fn) if (tp + fn) != 0 else 0,
        })

    # TODO: compute trip-related metrics
    # trips = metrics_data[0]
    # metrics_df = pd.DataFrame(metrics_data[1:])

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = diary.merge(metrics_df, on=['stop_id'], how='right')
    metrics_df = metrics_df.merge(stop_counts, on=['stop_id'], how='left')
    metrics_df['count'] = metrics_df['count'].fillna(0).astype(int)
    metrics_df = metrics_df.set_index('stop_id')

    tp = metrics_df['tp'].sum()
    fp = metrics_df['fp'].sum()
    fn = metrics_df['fn'].sum()
    stops_merged = metrics_df['stops_merged'].sum()
    n_stops = metrics_df.index.nunique()
    prop_stops_merged = stops_merged / (n_stops - 1)

    # Calculate micro-averaged precision and recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Calculate macro-averaged precision and merging, weighted by duration of stop
    total_duration = metrics_df['duration'].sum()  # total duration = tp + fn
    weighted_precision = (metrics_df['precision'] * metrics_df['duration']).sum() / total_duration
    weighted_merging = (metrics_df['prop_merged'] * metrics_df['duration']).sum() / total_duration

    # Count number of missed and split stops
    num_missed = metrics_df[metrics_df['count'] == 0].shape[0]
    num_split = metrics_df[metrics_df['count'] > 1].shape[0]

    metrics = {
        "Recall": float(recall),
        "Precision": float(precision),
        "Weighted Precision": float(weighted_precision),
        "Missed": int(num_missed),
        "Stops Merged": int(stops_merged),
        "Prop Stops Merged": float(prop_stops_merged),
        "Weighted Stop Merging": float(weighted_merging),
        "Split": int(num_split),
        "Stop Count": int(n_stops)
    }

    return metrics_df, metrics

if __name__ == '__main__':
    traj_cols = {'uid':'uid',
                 'x':'x',
                 'y':'y'}
    
    sparse_df = loader.from_file("data/sparse_traj/", format="parquet", traj_cols=traj_cols,
                          parse_dates=True)

    user = sparse_df.uid.iloc[0]
    stop_table = HDBSCAN.st_hdbscan(traj = sparse_df.loc[sparse_df.uid == user],
                                        is_long_lat = False,
                                        is_datetime = False,
                                        traj_cols = traj_cols,
                                        complete_output = True,
                                        time_thresh = 60,
                                        min_pts = 5,
                                        min_cluster_size = 10)

    diaries_df = loader.from_file("data/diaries/", format="parquet", traj_cols=traj_cols,
                           parse_dates=True)
    diaries_df = diaries_df.loc[diaries_df.uid == user]
    pdb.set_trace()
    output = prepare_stop_table(stop_table, diaries_df)
