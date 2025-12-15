import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm


def haversine_dist(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance between two points in meters.

    Parameters
    ----------
    lat1, lon1 : float or array-like
        Latitude and longitude of first point(s)
    lat2, lon2 : float or array-like
        Latitude and longitude of second point(s)

    Returns
    -------
    float or array-like
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def _explode_agg(col_agg, col_id, df_left, df_right):
    """
    Explode aggregated column and merge with original dataframe.

    Parameters
    ----------
    col_agg : str
        Name of column containing aggregated values
    col_id : str
        Name of ID column to assign
    df_left : pd.DataFrame
        DataFrame to update
    df_right : pd.DataFrame
        DataFrame containing aggregated data

    Returns
    -------
    pd.DataFrame
        Updated dataframe with exploded values
    """
    # Create a mapping from original indices to staypoint IDs
    exploded = df_right[[col_agg, col_id]].explode(col_agg)
    exploded = exploded[exploded[col_agg].notna()]

    # Create a series mapping pfs_id to staypoint_id
    mapping = pd.Series(
        exploded[col_id].values,
        index=exploded[col_agg].values
    )

    # Map to original dataframe
    df_left[col_id] = df_left.index.map(mapping)

    return df_left


def _generate_staypoints_sliding_user(
    pfs_user,
    geo_col='geometry',
    elevation_flag=False,
    dist_threshold=100,
    time_threshold=5.0,
    gap_threshold=15.0,
    distance_metric='haversine',
    include_last=False
):
    """
    Generate staypoints for a single user using sliding stop detection.

    Parameters
    ----------
    pfs_user : tuple
        (user_id, DataFrame) from groupby operation
    geo_col : str
        Name of geometry column
    elevation_flag : bool
        Whether elevation data is present
    dist_threshold : float
        Distance threshold in meters
    time_threshold : float
        Time threshold in minutes
    gap_threshold : float
        Maximum gap threshold in minutes
    distance_metric : str
        Distance metric to use
    include_last : bool
        Whether to include last staypoint

    Returns
    -------
    pd.DataFrame
        Staypoints for this user
    """
    user_id, pfs = pfs_user

    if len(pfs) == 0:
        return pd.DataFrame()

    # Sort by time but preserve original index
    pfs = pfs.sort_values('tracked_at').reset_index(drop=False)
    original_index_col = pfs.columns[0]  # The original index is now the first column

    staypoints = []
    pfs_ids = []

    i = 0
    n = len(pfs)

    while i < n:
        j = i + 1

        # Get coordinates of starting point
        start_geom = pfs.iloc[i][geo_col]
        start_lat = start_geom.y
        start_lon = start_geom.x
        start_time = pfs.iloc[i]['tracked_at']

        # Slide window forward
        while j < n:
            # Check for temporal gap
            time_gap = (pfs.iloc[j]['tracked_at'] - pfs.iloc[j-1]['tracked_at']).total_seconds() / 60
            if time_gap > gap_threshold:
                break

            # Calculate distance from start point
            curr_geom = pfs.iloc[j][geo_col]
            curr_lat = curr_geom.y
            curr_lon = curr_geom.x

            if distance_metric == 'haversine':
                dist = haversine_dist(start_lat, start_lon, curr_lat, curr_lon)
            else:
                dist = start_geom.distance(curr_geom)

            # Check if moved beyond distance threshold
            if dist > dist_threshold:
                break

            j += 1

        # Check if we have a valid staypoint (enough time spent)
        time_spent = (pfs.iloc[j-1]['tracked_at'] - start_time).total_seconds() / 60

        if time_spent >= time_threshold:
            # Create staypoint
            sp_pfs = pfs.iloc[i:j]

            # Calculate centroid
            lats = [p.y for p in sp_pfs[geo_col]]
            lons = [p.x for p in sp_pfs[geo_col]]

            if distance_metric == 'haversine':
                # Use spherical mean for geographic coordinates
                lats_rad = np.radians(lats)
                lons_rad = np.radians(lons)

                x = np.cos(lats_rad) * np.cos(lons_rad)
                y = np.cos(lats_rad) * np.sin(lons_rad)
                z = np.sin(lats_rad)

                mean_x = np.mean(x)
                mean_y = np.mean(y)
                mean_z = np.mean(z)

                mean_lon = np.arctan2(mean_y, mean_x)
                hyp = np.sqrt(mean_x**2 + mean_y**2)
                mean_lat = np.arctan2(mean_z, hyp)

                center_lat = np.degrees(mean_lat)
                center_lon = np.degrees(mean_lon)
            else:
                center_lat = np.mean(lats)
                center_lon = np.mean(lons)

            sp_dict = {
                'user_id': user_id,
                'started_at': sp_pfs.iloc[0]['tracked_at'],
                'finished_at': sp_pfs.iloc[-1]['tracked_at'],
                geo_col: Point(center_lon, center_lat),
                'pfs_id': list(sp_pfs[original_index_col])  # Use original index
            }

            if elevation_flag and 'elevation' in sp_pfs.columns:
                sp_dict['elevation'] = sp_pfs['elevation'].mean()

            staypoints.append(sp_dict)

            # Move to the point that broke the staypoint
            i = j
        else:
            # Not enough time spent, move to next point
            i += 1

    # Handle last staypoint if requested
    if include_last and i < n:
        sp_pfs = pfs.iloc[i:]
        if len(sp_pfs) > 0:
            lats = [p.y for p in sp_pfs[geo_col]]
            lons = [p.x for p in sp_pfs[geo_col]]

            if distance_metric == 'haversine':
                lats_rad = np.radians(lats)
                lons_rad = np.radians(lons)

                x = np.cos(lats_rad) * np.cos(lons_rad)
                y = np.cos(lats_rad) * np.sin(lons_rad)
                z = np.sin(lats_rad)

                mean_x = np.mean(x)
                mean_y = np.mean(y)
                mean_z = np.mean(z)

                mean_lon = np.arctan2(mean_y, mean_x)
                hyp = np.sqrt(mean_x**2 + mean_y**2)
                mean_lat = np.arctan2(mean_z, hyp)

                center_lat = np.degrees(mean_lat)
                center_lon = np.degrees(mean_lon)
            else:
                center_lat = np.mean(lats)
                center_lon = np.mean(lons)

            sp_dict = {
                'user_id': user_id,
                'started_at': sp_pfs.iloc[0]['tracked_at'],
                'finished_at': sp_pfs.iloc[-1]['tracked_at'],
                geo_col: Point(center_lon, center_lat),
                'pfs_id': list(sp_pfs[original_index_col])  # Use original index
            }

            if elevation_flag and 'elevation' in sp_pfs.columns:
                sp_dict['elevation'] = sp_pfs['elevation'].mean()

            staypoints.append(sp_dict)

    return pd.DataFrame(staypoints)


def applyParallel(groups, func, n_jobs=1, print_progress=False, **kwargs):
    """
    Apply function to groups in parallel.

    Parameters
    ----------
    groups : DataFrameGroupBy
        Grouped dataframe
    func : callable
        Function to apply to each group
    n_jobs : int
        Number of parallel jobs
    print_progress : bool
        Whether to show progress bar
    **kwargs
        Additional arguments to pass to func

    Returns
    -------
    pd.DataFrame
        Concatenated results
    """
    if n_jobs == 1:
        # Sequential processing
        if print_progress:
            results = [func(group, **kwargs) for group in tqdm(groups, desc="Processing users")]
        else:
            results = [func(group, **kwargs) for group in groups]
    else:
        # Parallel processing
        group_list = list(groups)
        if print_progress:
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(group, **kwargs) for group in tqdm(group_list, desc="Processing users")
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(group, **kwargs) for group in group_list
            )

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def generate_staypoints(
    positionfixes,
    method="sliding",
    distance_metric="haversine",
    dist_threshold=100,
    time_threshold=5.0,
    gap_threshold=15.0,
    include_last=False,
    print_progress=False,
    exclude_duplicate_pfs=True,
    n_jobs=1,
):
    """
    Generate staypoints from positionfixes.

    Parameters
    ----------
    positionfixes : pd.DataFrame or gpd.GeoDataFrame
        Position fixes with columns: user_id, tracked_at, geometry

    method : {'sliding'}
        Method to create staypoints. 'sliding' applies a sliding window over the data.

    distance_metric : {'haversine'}
        The distance metric used by the applied method.

    dist_threshold : float, default 100
        The distance threshold for the 'sliding' method, i.e., how far someone has to travel to
        generate a new staypoint. Units depend on the dist_func parameter. If 'distance_metric' is 'haversine' the
        unit is in meters

    time_threshold : float, default 5.0 (minutes)
        The time threshold for the 'sliding' method in minutes.

    gap_threshold : float, default 15.0 (minutes)
        The time threshold of determine whether a gap exists between consecutive pfs. Consecutive pfs with
        temporal gaps larger than 'gap_threshold' will be excluded from staypoints generation.
        Only valid in 'sliding' method.

    include_last: boolean, default False
        The algorithm in Li et al. (2008) only detects staypoint if the user steps out
        of that staypoint. This will omit the last staypoint (if any). Set 'include_last'
        to True to include this last staypoint.

    print_progress: boolean, default False
        Show per-user progress if set to True.

    exclude_duplicate_pfs: boolean, default True
        Filters duplicate positionfixes before generating staypoints. Duplicates can lead to problems in later
        processing steps (e.g., when generating triplegs). It is not recommended to set this to False.

    n_jobs: int, default 1
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all.

    Returns
    -------
    pfs: pd.DataFrame or gpd.GeoDataFrame
        The original positionfixes with a new column ``[`staypoint_id`]``.

    sp: gpd.GeoDataFrame
        The generated staypoints.

    References
    ----------
    Zheng, Y. (2015). Trajectory data mining: an overview. ACM Transactions on Intelligent Systems
    and Technology (TIST), 6(3), 29.

    Li, Q., Zheng, Y., Xie, X., Chen, Y., Liu, W., & Ma, W. Y. (2008, November). Mining user
    similarity based on location history. In Proceedings of the 16th ACM SIGSPATIAL international
    conference on Advances in geographic information systems (p. 34). ACM.
    """
    # Validate required columns
    required_cols = ['user_id', 'tracked_at']
    for col in required_cols:
        if col not in positionfixes.columns:
            raise ValueError(f"Missing required column: {col}")

    if not isinstance(positionfixes, gpd.GeoDataFrame):
        raise TypeError("positionfixes must be a GeoDataFrame with geometry column")

    # Copy the original pfs for adding 'staypoint_id' column
    pfs = positionfixes.copy()

    if exclude_duplicate_pfs:
        len_org = pfs.shape[0]
        pfs = pfs.drop_duplicates()
        nb_dropped = len_org - pfs.shape[0]
        if nb_dropped > 0:
            warn_str = (
                f"{nb_dropped} duplicates were dropped from your positionfixes. Dropping duplicates is"
                + " recommended but can be prevented using the 'exclude_duplicate_pfs' flag."
            )
            warnings.warn(warn_str)

    # If the positionfixes already have a column "staypoint_id", we drop it
    if "staypoint_id" in pfs.columns:
        pfs.drop(columns="staypoint_id", inplace=True)

    elevation_flag = "elevation" in pfs.columns  # if there is elevation data

    geo_col = pfs.geometry.name
    if elevation_flag:
        sp_column = ["user_id", "started_at", "finished_at", "elevation", geo_col]
    else:
        sp_column = ["user_id", "started_at", "finished_at", geo_col]

    if method == "sliding":
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        sp = applyParallel(
            pfs.groupby("user_id", as_index=False),
            _generate_staypoints_sliding_user,
            n_jobs=n_jobs,
            print_progress=print_progress,
            geo_col=geo_col,
            elevation_flag=elevation_flag,
            dist_threshold=dist_threshold,
            time_threshold=time_threshold,
            gap_threshold=gap_threshold,
            distance_metric=distance_metric,
            include_last=include_last,
        ).reset_index(drop=True)

        # Index management
        sp["staypoint_id"] = sp.index
        sp.index.name = "id"

        if "pfs_id" not in sp.columns:
            sp["pfs_id"] = None
        pfs = _explode_agg("pfs_id", "staypoint_id", pfs, sp)
    else:
        raise ValueError(f"Unknown method: {method}. Only 'sliding' is supported.")

    sp = gpd.GeoDataFrame(sp, columns=sp_column, geometry=geo_col, crs=pfs.crs)

    # dtype consistency
    # sp id (generated by this function) should be int64
    sp.index = sp.index.astype("int64")
    # pfs['staypoint_id'] should be Int64 (missing values)
    pfs["staypoint_id"] = pfs["staypoint_id"].astype("Int64")

    # user_id of sp should be the same as pfs
    sp["user_id"] = sp["user_id"].astype(pfs["user_id"].dtype)

    if len(sp) == 0:
        warnings.warn("No staypoints can be generated, returning empty sp.")
        return pfs, sp

    return pfs, sp
