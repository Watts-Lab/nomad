import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from zoneinfo import ZoneInfo
import warnings

import pdb


def poi_map(traj, poi_table, traj_cols=None, max_distance=1, **kwargs):
    """
    Map pings in the trajectory to the POI table.

    Parameters
    ----------
    traj : pd.DataFrame
        The trajectory DataFrame containing x and y coordinates.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
    traj_cols : list
        The columns in the trajectory DataFrame to be used for mapping.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pd.Series
        A Series containing the building IDs corresponding to the pings in the trajectory.
    """

    # TODO traj_cols support

    # Build pings GeoDataFrame
    pings_df = traj[['x', 'y']].copy()
    pings_df["pings_geometry"] = pings_df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
    pings_df = gpd.GeoDataFrame(pings_df, geometry="pings_geometry", crs=poi_table.crs)
    
    # First spatial join (within)
    pings_df = gpd.sjoin(pings_df, poi_table, how="left", predicate="within")
    
    # Identify unmatched pings
    unmatched_mask = pings_df["building_id"].isna()
    unmatched_pings = pings_df[unmatched_mask].drop(columns=["building_id", "index_right"])
    
    if not unmatched_pings.empty:
        # Nearest spatial join for unmatched pings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
            nearest = gpd.sjoin_nearest(unmatched_pings, poi_table, how="left", max_distance=max_distance)

        # Keep only the first match for each original ping
        nearest = nearest.groupby(nearest.index).first()

        # Update original DataFrame with nearest matches
        pings_df.loc[unmatched_mask, "building_id"] = nearest["building_id"].values

    return pings_df["building_id"]


def identify_stop(alg_out, traj, stop_table, poi_table, method='mode'):
    """
    Given the output of a stop detection algorithm, maps each cluster to a location
    by the method specified.
   
    Parameters
    ----------
    alg_out : pd.DataFrame
        DataFrame containing cluster assignments (one row per ping), indexed by ping ID.
        Must have a column 'cluster' indicating each ping's cluster.
    traj : pd.DataFrame
        DataFrame containing ping coordinates (x, y) by ping ID.
    stop_table : pd.DataFrame
        DataFrame containing stop clusters (one row per cluster), with a 'cluster_id' column.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
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
        tz = ZoneInfo("America/New_York")
        stop_table = pd.DataFrame({
            'start_time': pd.Series(dtype='datetime64[ns, America/New_York]'),
            'duration': pd.Series(dtype='float'),
            'x': pd.Series(dtype='float'),
            'y': pd.Series(dtype='float'),
            'location': pd.Series(dtype='object'),
            'end_time': pd.Series(dtype='datetime64[ns, America/New_York]')
        })
        return stop_table

    merged_df = traj.copy()
    merged_df['cluster'] = alg_out

    # Compute the location for each cluster
    if method == 'centroid':
        pings_df = merged_df.groupby('cluster')[['x', 'y']].mean()
        locations = poi_map(pings_df, poi_table)

    elif method == 'mode':
        pings_df = merged_df[['x', 'y']].copy()
        merged_df["building_id"] = poi_map(pings_df, poi_table)
        locations = merged_df.groupby('cluster')['building_id'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mode' or 'centroid'.")

    # Map the mode location back to the stop_table
    stop_table['location'] = locations

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