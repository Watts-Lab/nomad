"""Various algorithms for estimating individual co-location"""

# Authors: Thomas Li and Francisco Barreras

import nomad.io.base as loader
import nomad.stop_detection.hdbscan as HDBSCAN
import pandas as pd
import pdb

def prepare_stop_table(stop_table, diary):
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
    pdb.set_trace()
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

if __name__ == '__main__':
    traj_cols = {'uid':'uid',
                 'x':'x',
                 'y':'y',
                 "start_timestamp":"start_time"}
    
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

    output = prepare_stop_table(stop_table, diaries_df)
    pdb.set_trace()