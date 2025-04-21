import pandas as pd
import numpy as np

def identify_stop(alg_out, sparse_traj, stop_table, city, method='mode'):
    """
    Given the output of a stop detection algorithm, maps each cluster to a location
    by the method specified.
   
    Parameters
    ----------
    alg_out : pd.DataFrame
        DataFrame containing cluster assignments (one row per ping), indexed by ping ID.
        Must have a column 'cluster' indicating each ping's cluster.
    sparse_traj : pd.DataFrame
        DataFrame containing ping coordinates (x, y) by ping ID.
    stop_table : pd.DataFrame
        DataFrame containing stop clusters (one row per cluster), with a 'cluster_id' column.
    city : City
        An object providing the `get_block((x, y))` method returning a building or location with an `id`.
    method : str, optional
        The method to use for mapping clusters to locations. Options are:
        - 'mode': Assigns the most frequent location ID associated with each cluster.
        - 'centroid': Assigns the centroid of the cluster to the location.
    
    Returns
    -------
    pd.DataFrame
        Updated stop_table with a 'location' column indicating location associated with each cluster.
    """

    # If either alg_out or stop_table is empty, there's nothing to do
    if alg_out.empty or stop_table.empty:
        stop_table['location'] = pd.Series(dtype='object')
        return stop_table

    # Merge the cluster assignments with coordinates
    alg_out.index = alg_out.index.map(lambda x: int(x.timestamp()))  # use unix timestamps as index
    merged_df = alg_out.merge(sparse_traj, left_index=True, right_index=True)

    # Compute the location for each cluster
    if method == 'centroid':
        centroids = merged_df.groupby('cluster')[['x', 'y']].mean()
        locations = centroids.apply(lambda row: city.get_block((row['x'], row['y'])).id, axis=1)
    else:  # method == 'mode'
        # Extract location IDs for each ping
        merged_df['location'] = [
            city.get_block((x, y)).id 
            for x, y in zip(merged_df['x'], merged_df['y'])
        ]
        locations = merged_df.groupby('cluster')['location'].agg(lambda x: x.mode().iat[0])

    # Map the mode location back to the stop_table
    stop_table['location'] = stop_table.index.map(locations)

    return stop_table


def mode(alg_out, sparse_traj, stop_table, city, method='mode'):
    """
    Given the output of a stop detection algorithm, maps each cluster to a location
    by the method specified.
   
    Parameters
    ----------
    alg_out : pd.DataFrame
        DataFrame containing cluster assignments (one row per ping), indexed by ping ID.
        Must have a column 'cluster' indicating each ping's cluster.
    sparse_traj : pd.DataFrame
        DataFrame containing ping coordinates (x, y) by ping ID.
    stop_table : pd.DataFrame
        DataFrame containing stop clusters (one row per cluster), with a 'cluster_id' column.
    city : City
        An object providing the `get_block((x, y))` method returning a building or location with an `id`.
    method : str, optional
        The method to use for mapping clusters to locations. Options are:
        - 'mode': Assigns the most frequent location ID associated with each cluster.
        - 'centroid': Assigns the centroid of the cluster to the location.
    
    Returns
    -------
    pd.DataFrame
        Updated stop_table with a 'location' column indicating location associated with each cluster.
    """

    # If either alg_out or stop_table is empty, there's nothing to do
    if alg_out.empty or stop_table.empty:
        stop_table['location'] = pd.Series(dtype='object')
        return stop_table

    # Merge the cluster assignments with coordinates
    alg_out.index = alg_out.index.map(lambda x: int(x.timestamp()))  # use unix timestamps as index
    merged_df = alg_out.merge(sparse_traj, left_index=True, right_index=True)

    # Compute the location for each cluster
    if method == 'centroid':
        centroids = merged_df.groupby('cluster')[['x', 'y']].mean()
        locations = centroids.apply(lambda row: city.get_block((row['x'], row['y'])).id, axis=1)
    else:  # method == 'mode'
        # Extract location IDs for each ping
        merged_df['location'] = [
            city.get_block((x, y)).id 
            for x, y in zip(merged_df['x'], merged_df['y'])
        ]
        locations = merged_df.groupby('cluster')['location'].agg(lambda x: x.mode().iat[0])

    # Map the mode location back to the stop_table
    stop_table['location'] = stop_table.index.map(locations)

    return stop_table

def nearest(df, poi_table, traj_cols, **kwargs):
    """
    Given the output of a stop detection algorithm, maps each cluster to a location
    by the method specified.
   
    Parameters
    ----------
    alg_out : pd.DataFrame
        DataFrame containing cluster assignments (one row per ping), indexed by ping ID.
        Must have a column 'cluster' indicating each ping's cluster.
    sparse_traj : pd.DataFrame
        DataFrame containing ping coordinates (x, y) by ping ID.
    stop_table : pd.DataFrame
        DataFrame containing stop clusters (one row per cluster), with a 'cluster_id' column.
    city : City
        An object providing the `get_block((x, y))` method returning a building or location with an `id`.
    method : str, optional
        The method to use for mapping clusters to locations. Options are:
        - 'mode': Assigns the most frequent location ID associated with each cluster.
        - 'centroid': Assigns the centroid of the cluster to the location.
    
    Returns
    -------
    pd.DataFrame
        Updated stop_table with a 'location' column indicating location associated with each cluster.
    """

    # If either alg_out or stop_table is empty, there's nothing to do
    if df.empty or stop_table.empty:
        stop_table['location'] = pd.Series(dtype='object')
        return stop_table

    # Merge the cluster assignments with coordinates
    alg_out.index = alg_out.index.map(lambda x: int(x.timestamp()))  # use unix timestamps as index
    merged_df = alg_out.merge(sparse_traj, left_index=True, right_index=True)

    # Compute the location for each cluster
    if method == 'centroid':
        centroids = merged_df.groupby('cluster')[['x', 'y']].mean()
        locations = centroids.apply(lambda row: city.get_block((row['x'], row['y'])).id, axis=1)

    # Map the mode location back to the stop_table
    stop_table['location'] = stop_table.index.map(locations)

    return stop_table


def q_stat(agent):
    pass


def radius_of_gyration(df):
    rog = np.sqrt(np.nanmean(np.nansum((df.to_numpy() - np.nanmean(df.to_numpy(), axis=0))**2, axis=1)))
    return rog


def prepare_diary(agent, dt, city, keep_all_stops=True):
    """
    Create stop IDs and optionally filter for stops that have at least one ping
    in agent.sparse_traj

    Parameters
    ----------
    agent : Agent
        The agent object containing 'diary' and 'sparse_traj'.
    dt : float
    keep_all_stops : bool
        If False, only retains stops that have at least one ping in agent.sparse_traj

    Returns
    -------
    diary : pd.DataFrame
        A copy of the original diary with stop_id assignments.
    """

    # Copy the agent's diary to avoid modifying it in place
    diary = agent.diary.copy()

    # Compute end times of stops
    diary.rename(columns={'local_timestamp': 'start_time'}, inplace=True)
    diary['end_time'] = diary['start_time'] + pd.to_timedelta(diary['duration'], unit='m')

    # Add columns for x, y coordinates for each location
    def get_x(b):
        return city.buildings[b].geometry.centroid.x if b is not None else None
    def get_y(b):
        return city.buildings[b].geometry.centroid.y if b is not None else None
    diary['x'] = diary['location'].apply(get_x)
    diary['y'] = diary['location'].apply(get_y)

    # Assign stop_id: -1 for trips, increments for each stop
    is_stop = diary['location'].notna()
    diary['stop_id'] = np.where(is_stop, np.cumsum(is_stop) - 1, -1)
    diary.drop(columns=['x', 'y'], inplace=True)

    if not keep_all_stops:
        sparse_traj = agent.sparse_traj
        stops_with_pings = diary[
            diary.apply(lambda row: sparse_traj['local_timestamp']
                        .between(row['start_time'], row['end_time']).any(), axis=1)
        ]["stop_id"].tolist()
        # Include -1 to ensure trips are also kept
        stops_to_keep = np.concatenate([stops_with_pings, [-1]])
        diary = diary[diary['stop_id'].isin(stops_to_keep)]

    return diary


def prepare_stop_table(stop_table, diary, dt):
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
    stop_table_exploded = stop_table.explode("mapped_stop_ids")
    stop_counts = stop_table_exploded["mapped_stop_ids"].value_counts().reset_index()
    stop_counts.columns = ["stop_id", "count"]

    # loop over all (true) stops
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

        if stop_id not in tp_fp_fn:
            tp_fp_fn[stop_id] = [0, 0, 0]
        tp_fp_fn[stop_id][0] += tp
        tp_fp_fn[stop_id][1] += fp
        tp_fp_fn[stop_id][2] += fn

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

    stops_split = stop_counts[stop_counts["count"] > 1]

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
        })

    # TODO: BELOW

    trips = metrics_data[0]
    metrics_df = pd.DataFrame(metrics_data[1:])
    metrics_df = diary.merge(metrics_df, on=['stop_id'], how='right')
    metrics_df = metrics_df.merge(stop_counts, on=['stop_id'], how='left')
    metrics_df = metrics_df.set_index('stop_id')
    metrics_df['stop_count'] = metrics_df['stop_count'].fillna(0).astype(int)

    tp = metrics_df['tp'].sum()
    fp = metrics_df['fp'].sum()
    fn = metrics_df['fn'].sum()
    stops_merged = metrics_df['stops_merged'].sum()
    prop_stops_merged = stops_merged / (n_stops - 1)

    # Calculate micro-averaged precision and recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Calculate macro-averaged precision and merging, weighted by duration of stop
    total_duration = metrics_df['duration'].sum()  # total duration = tp + fn
    weighted_precision = (metrics_df['precision'] * metrics_df['duration']).sum() / total_duration
    weighted_merging = (metrics_df['prop_merged'] * metrics_df['duration']).sum() / total_duration

    # Count number of missed and split stops
    num_missed = metrics_df[metrics_df['stop_count'] == 0].shape[0]
    num_split = metrics_df[metrics_df['stop_count'] > 1].shape[0]

    metrics = {
        "Recall": recall,
        "Precision": precision,
        "Weighted Precision": weighted_precision,
        "Missed": num_missed,
        "Stops Merged": stops_merged,
        "Prop Stops Merged": prop_stops_merged,
        "Weighted Stop Merging": weighted_merging,
        "Trip Merging": trips['prop_merged'],
        "Split": num_split,
        "Stop Count": n_stops
    }

    return metrics_df, metrics