import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import math
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
import daphmeIO as loader
import constants as constants


##########################################
### COORDINATE CONVERSION FUNCTIONS ######
##########################################

def ConvertCoordinate_3587(lat, lon):
    '''
    conversion from crs:4326 to crs:3587
    '''
    latInEPSG3857 = (math.log(math.tan((90 + lat) * math.pi / 360)) / (
                math.pi / 180)) * (20037508.34 / 180)
    lonInEPSG3857 = (lon * 20037508.34 / 180)

    return latInEPSG3857, lonInEPSG3857


def ConvertCoordinate_4326(lat, lon):
    '''
    function for conversion from crs:3857 to crs:4326
    '''
    latInEPSG4326 = (180 / math.pi) * (
                2 * math.atan(math.exp(lat * math.pi / 20037508.34)) - (
                    math.pi / 2))
    lonInEPSG4326 = lon / (20037508.34 / 180)

    return latInEPSG4326, lonInEPSG4326


def convert_df_coord_4326(df_loc, lat='lat', lon='lon'):
    '''
    convert df coordinate from crs:3857 to crs:4326
    '''
    df_loc[[lat, lon]] = df_loc.apply(
        lambda row: ConvertCoordinate_4326(row[lat], row[lon]), axis=1,
        result_type='expand')


def convert_df_coord_3587(df_loc, lat='lat', lon='lon'):
    '''
    convert df coordinate from crs:4326 to crs:3587
    '''
    df_loc[[lat, lon]] = df_loc.apply(
        lambda row: ConvertCoordinate_3587(row[lat], row[lon]), axis=1,
        result_type='expand')


##########################################
######## STOP DETECTION FUNCTIONS ########
##########################################

##########################################
########        Lachesis          ########
##########################################

def diameter(coords, metric='euclidean'):
    if len(coords) < 2:
        return 0
        
    if metric == 'haversine':
        coords = np.radians(coords)
        pairwise_dists = pdist(coords,
                               metric=lambda u, v: haversine_distance(u, v))
        return np.max(pairwise_dists)
    return np.max(pdist(coords, metric=metric))


def medoid(coords, metric='euclidean'):
    if len(coords) < 2:
        return coords[0]
    
    if metric == 'haversine':
        coords = np.radians(coords)
        distances = pairwise_haversine(coords)
    else:
        distances = cdist(coords, coords, metric=metric)
    
    sum_distances = np.sum(distances, axis=1)
    medoid_index = np.argmin(sum_distances)
    return coords[medoid_index, :]


def haversine_distance(coord1, coord2):
    """
    Compute the haversine distance between two points on Earth.

    Parameters:
        coord1: [lat1, lon1] in radians
        coord2: [lat2, lon2] in radians

    Returns:
        Distance in meters.
    """
    earth_radius_meters = 6371000  # Earth's radius in meters
    delta_lat = coord2[0] - coord1[0]
    delta_lon = coord2[1] - coord1[1]
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(coord1[0]) * np.cos(
        coord2[0]) * np.sin(delta_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_meters * c  # Distance in meters


def pairwise_haversine(coords):
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = haversine_distance(coords[i], coords[j])
            distances[j, i] = distances[i, j]
    return distances

    
def update_diameter(c_j, coords_prev, D_prev, metric='euclidean'):
    """
    Update the diameter of a cluster, supporting both euclidean and haversine.

    Parameters:
    - c_j: New point's coordinate [x, y] or [latitude, longitude].
    - coords_prev: Array of previous coordinates [[x1, y1], [x2, y2], ...] or [[lat1, lon1], [lat2, lon2], ...].
    - D_prev: Previously computed diameter.
    - metric: Coordinate system, 'euclidean' (default) or 'haversine'.

    Returns:
    - Updated diameter.
    """
    if metric == 'euclidean':
        X_prev = coords_prev[:, 0]
        Y_prev = coords_prev[:, 1]
        x_j, y_j = c_j[0], c_j[1]
        new_dists = np.sqrt((X_prev - x_j) ** 2 + (Y_prev - y_j) ** 2)

    elif metric == 'haversine':
        coords_prev = np.radians(coords_prev)
        c_j = np.radians(c_j)

        lat_j, lon_j = c_j[0], c_j[1]
        new_dists = np.array([
            haversine_distance([lat_j, lon_j], [lat_i, lon_i])
            for lat_i, lon_i in coords_prev
        ])

    else:
        raise ValueError("metric must be 'euclidean' or 'haversine'")

    D_i_jp1 = np.max([D_prev, np.max(new_dists)])

    return D_i_jp1


def lachesis(traj,
             dur_min,
             dt_max,
             delta_roam,
             traj_cols=None,
             complete_output=False,
             **kwargs):
    """
    Extract stays from raw location data.
    Parameters
    ----------
    traj: numpy array - simulated trajectory from simulate_traj.
        - (lat,lon) : 'y', 'x' in crs:3586
        - time is 'unix_timestamp' in 'timestamp' format
        - Longitude (long) corresponds to the x-coordinate (horizontal axis).
        - Latitude (lat) corresponds to the y-coordinate (vertical axis).
    dur_min [minutes]   : float - minimum duration for a stay (stay duration).
    dt_max  [minutes]   : float - maximum duration permitted between consecutive pings in a stay. dt_max should be greater than dur_min
    delta_roam [meters] : float - maximum roaming distance for a stay (roaming distance).

    example:
    lachesis(traj, dur_min, dt_max, delta_roam, traj_cols=None, x = 'x_coord', y = 'y_coord')

    Returns
    """
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs[
        'latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in traj.column

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns:
        long_lat = False
        coords = traj[[traj_cols['x'], traj_cols['y']]].to_numpy()
    else:
        long_lat = True
        coords = traj[[traj_cols['longitude'], traj_cols['latitude']]].to_numpy()

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in traj.columns:
        datetime = False
        timestamp_col = traj_cols['timestamp']
        traj = traj.set_index(timestamp_col, drop=False)
    else:
        datetime = True
        datetime_col = traj_cols['datetime']
        traj = traj.set_index(datetime_col, drop=False)

    Stays = np.empty((0, 6))

    # [STEP0] starting the search
    i = 0

    while i < len(traj) - 1:

        # [STEP1] - find the least amount of pings over a timerange > dur_min
        # j_star is the candidate 'stop end'
        time_i = traj[timestamp_col].iat[i] if not datetime else \
        traj[datetime_col].iat[i]

        if not datetime:
            j_star = next((j for j in range(i, len(traj)) if (
                        traj[timestamp_col].iat[j] - time_i) >= dur_min * 60),
                          -1)
        else:
            j_star = next((j for j in range(i, len(traj)) if
                           (traj[datetime_col].iat[j] - time_i) >= timedelta(
                               minutes=dur_min)), -1)

        # initial diameter
        D_start = diameter(coords[i:j_star + 1],
                           metric='haversine' if long_lat else 'euclidean')

        # CONDITIONS BLOCKING THE SEARCH
        # conditions to block the j_star search
        Cond_exhausted_traj = j_star == -1
        # condition that diameter is over the delta_roam threshold
        Cond_diam_OT = D_start > delta_roam
        # condition that there is at least a consecutive ping pair that has a time-separation greater than dt_max
        if not datetime:
            Cond_cc_diff_OT = (traj[timestamp_col][
                               i:j_star + 1].diff().dropna() >= dt_max * 60).any()
        else:
            Cond_cc_diff_OT = (traj[datetime_col][
                               i:j_star + 1].diff().dropna().dt.total_seconds() >= dt_max * 60).any()

        # [STEP2] - decide whether index i is a 'stop' or 'trip' ping
        if Cond_exhausted_traj or Cond_diam_OT or Cond_cc_diff_OT:
            # DISCARD i and j_star as candidate 'stop start' and 'stop end' pings
            # move forward and update i, starting from scratch
            i += 1
        else:
            # SELECT i as a 'stop start' ping AND
            # [STEP3] proceed with the iterative search of 'stop end'
            COND_found_jstar = False
            for j in range(j_star, len(traj) - 1):

                # update diameter
                D_update = update_diameter(coords[j], coords[i:j], D_start,
                                           metric='haversine' if long_lat else 'euclidean')

                # compute the conescutive ping's time difference
                if datetime:
                    cc_diff = (traj[datetime_col].iat[j] -
                               traj[datetime_col].iat[j - 1]).total_seconds()
                else:
                    cc_diff = traj[timestamp_col].iat[j] - \
                              traj[timestamp_col].iat[j - 1]

                # verify that the new ping does not break the rule
                COND_j = (D_update > delta_roam) or (cc_diff > dt_max * 60)

                if COND_j:
                    # new ping broke the rule and the stop detection is completed
                    j_star = j - 1
                    COND_found_jstar = True
                    break
                else:
                    # the stop detection proceeds - update the diameter
                    D_start = D_update
            # handle the case in which no further pings
            if not COND_found_jstar:
                j_star = len(traj) - 1

            # COLLECT STOP INFORMATION
            start, end = (traj[datetime_col].iat[i],
                          traj[datetime_col].iat[j_star]) if datetime else (
            traj[timestamp_col].iat[i], traj[timestamp_col].iat[j_star])
            stay_medoid = medoid(coords[i:j_star + 1])
            n_pings = j_star - i + 1
            stay = np.array([[start, end, stay_medoid[0], stay_medoid[1],
                              D_start, n_pings]])

            # UPDATE THE STAY DATAFRAME
            Stays = np.concatenate((Stays, stay), axis=0)

            # updating start index of the stop
            # proceed the search
            i = j_star + 1

    Cols_ = ['start_time', 'end_time', 'x', 'y', 'diameter', 'n_pings']
    stays = pd.DataFrame(Stays, columns=Cols_)

    # compute stop duration
    stays['duration'] = stays['end_time'] - stays['start_time']

    if complete_output:
        return stays
    else:
        stays.drop(columns=['end_time', 'diameter', 'n_pings'], inplace=True)
        stays = stays[['start_time', 'duration', 'x', 'y']]
        return stays


##########################################
########         DBSCAN           ########
##########################################

def extract_middle(data):
    """
    TODO

    Parameters
    ----------
    df : pandas.DataFrame
        User pings with 'x' (EPSG:3857), 'y' (EPSG:3857) columns, indexed by 'unix_timestamp'

    Returns
    -------
    tuple (i,j)
        First and last indices of the "middle" of the cluster
    """
    current = data.iloc[0]['cluster']
    x = (data.cluster != current).values
    if len(np.where(x)[0]) == 0:  # There is no inbetween
        return (len(data), len(data))
    else:
        i = np.where(x)[0][
            0]  # First index where the cluster is not the value of the first entry's cluster
    if len(np.where(~x[i:])[
               0]) == 0:  # There is no current again (i.e., the first cluster does not reappear, so the middle is actually the tail)
        return (i, len(data))
    else:  # Current reappears
        j = i + np.where(~x[i:])[0][0]
    return (i, j)


def find_neighbors(data, time_thresh, dist_thresh, long_lat,
                   datetime, traj_cols):
    """
    Identifies neighboring pings for each user ping within specified time and distance thresholds.

    Parameters
    ----------
    data : pandas.DataFrame
        User pings with spatial and temporal columns.
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.
    traj_cols : dict
        Dictionary mapping trajectory column names.
    long_lat : bool
        Whether the data uses longitude and latitude for spatial coordinates.
    datetime : bool
        Whether the data uses datetime instead of unix timestamps for time.

    Returns
    -------
    dict
        Neighbors indexed by time, with values as sets of neighboring times.
    """
    if long_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values

    if datetime:
        times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
    else:
        times = data[traj_cols['timestamp']].values

    # Pairwise time differences
    time_diffs = np.abs(times[:, np.newaxis] - times)
    time_diffs = time_diffs.astype(int)

    # Time threshold calculation using broadcasting
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1)
    time_pairs = np.where(within_time_thresh)
    
    # Distance calculation
    if long_lat:
        distances = np.array([haversine_distance(coords[i], coords[j]) for i, j in zip(*time_pairs)])
    else:
        distances_sq = (coords[time_pairs[0], 0] - coords[time_pairs[1], 0])**2 + \
                       (coords[time_pairs[0], 1] - coords[time_pairs[1], 1])**2
        distances = np.sqrt(distances_sq)

    # Filter by distance threshold
    neighbor_pairs = distances < dist_thresh

    # Building the neighbor dictionary
    neighbor_dict = defaultdict(set)
    for i, j in zip(time_pairs[0][neighbor_pairs], time_pairs[1][neighbor_pairs]):
        neighbor_dict[times[i]].add(times[j])
        neighbor_dict[times[j]].add(times[i])

    return neighbor_dict


def dbscan(data, time_thresh, dist_thresh, min_pts, long_lat, datetime, traj_cols, neighbor_dict=None):
    """
    Implements DBSCAN.

    Parameters
    ----------
    data : pandas.DataFrame
        User pings with 'x' (EPSG:3857), 'y' (EPSG:3857) columns, indexed by 'unix_timestamp'
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.
    min_pts: int
        A cluster must have at least (min_pts+1) points to be considered a cluster.

    Returns
    -------
    pandas.DataFrame
        Contains two columns 'cluster' (int), 'core' (int) labeling each ping with their cluster id and core id or noise, indexed by 'unix_timestamp'
    """
    if not neighbor_dict:
        neighbor_dict = find_neighbors(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols)
    else:
        valid_times = set(data.index)
        neighbor_dict = defaultdict(set,
                                    {k: v.intersection(valid_times) for k, v in
                                     neighbor_dict.items() if
                                     k in valid_times})

    cluster_df = pd.Series(-2, index=data.index, name='cluster')
    core_df = pd.Series(-1, index=data.index, name='core')

    cid = -1  # Initialize cluster label

    for i, cluster in cluster_df.items():
        if cluster < 0:  # Check if point is not yet in a cluster
            if len(neighbor_dict[i]) < min_pts:
                cluster_df[i] = -1  # Mark as noise if below min_pts
            else:
                cid += 1
                cluster_df[i] = cid  # Assign new cluster label
                core_df[i] = cid  # Assign new core label
                S = list(neighbor_dict[i])  # Initialize stack with neighbors

                while S:
                    j = S.pop()
                    if cluster_df[j] < 0:  # Process if not yet in a cluster
                        cluster_df[j] = cid
                        if len(neighbor_dict[j]) >= min_pts:
                            core_df[j] = cid  # Assign core label
                            for k in neighbor_dict[j]:
                                if cluster_df[k] < 0:
                                    S.append(k)  # Add new neighbors

    return pd.DataFrame({'cluster': cluster_df, 'core': core_df})


def process_clusters(data,
                     time_thresh,
                     dist_thresh,
                     min_pts,
                     output,
                     long_lat,
                     datetime,
                     traj_cols,
                     cluster_df=None,
                     neighbor_dict=None,
                     min_duration=4):
    """
    TODO

    Parameters
    ----------
    data : pandas.DataFrame
        User pings with 'unix_timestamp' (integer), 'x' (EPSG:3857), 'y' (EPSG:3857) columns, indexed by 'unix_timestamp'
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.
    min_pts: int
        A cluster must have at least (min_pts+1) points to be considered a cluster.
    output: pandas.DataFrame
        TODO
    cluster_df : pandas.DataFrame
        Output of dbscan
    neighbor_dict: dictionary
        TODO
    min_duration: int
        A cluster must have duration at least 'min_duration' to be considered a cluster.

    Returns
    -------
    List
        (start, duration, x_mean, y_mean, n, max_gap, radius) of each (post-processed) cluster
    """
    if not neighbor_dict:
        neighbor_dict = find_neighbors(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols)
    if cluster_df is None:
        cluster_df = dbscan(data, time_thresh, dist_thresh, min_pts, long_lat,
                            datetime, traj_cols, neighbor_dict=neighbor_dict)
    if len(cluster_df) < min_pts:
        return False

    cluster_df = cluster_df[cluster_df['cluster'] != -1]  # Remove noise pings

    # All pings are in the same cluster
    if len(cluster_df['cluster'].unique()) == 1:
        # We rerun dbscan because possibly these points no longer hold their own
        x = dbscan(data = data.loc[cluster_df.index], time_thresh = time_thresh, dist_thresh = dist_thresh,
                   min_pts = min_pts, long_lat = long_lat, datetime = datetime, traj_cols = traj_cols, neighbor_dict = neighbor_dict)
        
        y = x.loc[x['cluster'] != -1]
        z = x.loc[x['core'] != -1]

        # if len(y['cluster'].unique()) > 0: CHANGED
        if len(y) > 0:
            duration = int((y.index.max() - y.index.min()) // 60) # Assumes unix_timestamp is in seconds

            if duration > min_duration:
                cid = max(output['cluster']) + 1 # Create new cluster id
                output.loc[y.index, 'cluster'] = cid
                output.loc[z.index, 'core'] = cid
            
            return True
        elif len(y) == 0: # The points in df, despite originally being part of a cluster, no longer hold their own
            return False

    # There are no clusters
    elif len(cluster_df['cluster'].unique()) == 0:
        return False
   
    # There is more than one cluster
    elif len(cluster_df['cluster'].unique()) > 1:
        i, j = extract_middle(cluster_df)  # Indices of the "middle" of the cluster
        
        # Recursively processes clusters
        if process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                            long_lat, datetime, traj_cols, cluster_df = cluster_df[i:j]):  # Valid cluster in the middle
            process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             long_lat, datetime, traj_cols, cluster_df = cluster_df[:i])  # Process the initial stub
            process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             long_lat, datetime, traj_cols, cluster_df = cluster_df[j:])  # Process the "tail"
            return True
        else:  # No valid cluster in the middle
            return process_clusters(data, time_thresh, dist_thresh, min_pts,
                                    output, long_lat, datetime, traj_cols, pd.concat(
                    [cluster_df[:i],
                     cluster_df[j:]]))  # what if this is out of bounds?

def temporal_dbscan(data,
                    time_thresh,
                    dist_thresh,
                    min_pts,
                    complete_output=False,
                    traj_cols=None,
                    **kwargs):
    
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs[
        'latitude'] in data.columns and kwargs['longitude'] in data.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in data.column

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns:
        long_lat = False
        coords = data[[traj_cols['x'], traj_cols['y']]].to_numpy()
    else:
        long_lat = True
        coords = data[[traj_cols['longitude'], traj_cols['latitude']]].to_numpy()

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in data.columns:
        datetime = False
        data = data.set_index(traj_cols['timestamp'], drop=False)
    else:
        datetime = True
        data = data.set_index(traj_cols['datetime'], drop=False)

    output = pd.DataFrame({'cluster': -1, 'core': -1}, index=data.index)

    process_clusters(data, time_thresh, dist_thresh, min_pts, output, long_lat, datetime, traj_cols, min_duration=4)

    # output = _temporal_dbscan_labels(data, time_thresh, dist_thresh, min_pts, long_lat, datetime, traj_cols)
    
    # complete_data = data.join(output, how='inner')
    
    # stop_table = complete_data.groupby('cluster').apply(lambda group: _stop_metrics(group, long_lat, datetime, complete_output, traj_cols))
    
    # stop_table = stop_table.reset_index()
    # stop_table = stop_table.drop(columns=['level_1'])
    # stop_table = stop_table[stop_table['cluster'] != -1]
    # stop_table.set_index(['cluster'], inplace=True)

    return output

def _stop_metrics(grouped_data,
                     long_lat,
                     datetime,
                     complete_output,
                     traj_cols):

    col_names = ['start_time', 'end_time', 'x', 'y', 'diameter', 'n_pings']

    # Coordinates array and distance metrics
    if long_lat:
        coords = grouped_data[[traj_cols['longitude'], traj_cols['latitude']]].to_numpy()
        stop_medoid = medoid(coords, metric='haversine')
        diameter_m = diameter(coords, metric='haversine')
    else:
        coords = grouped_data[[traj_cols['x'], traj_cols['y']]].to_numpy()
        stop_medoid = medoid(coords, metric='euclidean')
        diameter_m = diameter(coords, metric='euclidean')


    # Compute start and end time of stop
    if datetime:
        start_time = grouped_data[traj_cols['datetime']].min()
        end_time = grouped_data[traj_cols['datetime']].max()
    else:
        start_time = grouped_data[traj_cols['timestamp']].min()
        end_time = grouped_data[traj_cols['timestamp']].max()

    # Number of pings in stop
    n_pings = len(grouped_data)

    # Create output DataFrame
    stop = pd.DataFrame(np.array([[start_time, end_time, stop_medoid[0],
                                   stop_medoid[1], diameter_m, n_pings]]),
                        columns=col_names)

    # stop duration
    stop['duration'] = stop['end_time'] - stop['start_time']

    if complete_output:
        return stop
    else:
        stop = stop.drop(columns=['end_time', 'diameter', 'n_pings'])
        stop = stop[['start_time', 'duration', 'x', 'y']]
        return stop