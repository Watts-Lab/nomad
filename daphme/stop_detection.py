from collections import defaultdict
import pandas as pd
import numpy as np
import numpy.random as npr

# =========================== #
#    DBSCAN IMPLEMENTATION    #
# =========================== #

'''
min_num_elements: A cluster must have at least (min_num_elements+1) points to be considered a cluster
'''
def extract_middle(data):
    current = data.iloc[0]['cluster']     # Test case: current = 1
    x = (data.cluster != current).values  # Test case: [False, False, True]
    if len(np.where(x)[0]) == 0:          # there is no inbetween
        return(len(data), len(data))
    else:
        i = np.where(x)[0][0]   # first index where the cluster is not the value of the first entry's cluster
        
    if len(np.where(~x[i:])[0]) == 0:   # there is no current again (i.e., the first cluster does not reappear, so the middle is actually the tail)
        return(i, len(data))
    else:   # current reappears
        j = i + np.where(~x[i:])[0][0]
    return(i, j)

def find_neighbors(df, time_thresh, dist_thresh):
    """
    Identifies neighboring pings for each user ping within specified time and distance thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        User pings with 'unix_timestamp' (integer), 'x' (EPSG:3857), 'y' (EPSG:3857) columns.
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.

    Returns
    -------
    dict
        Neighbors indexed by unix timestamp, with values as sets of neighboring unix timestamps.
    """    
    unix_timestamps, x, y = df[['unix_timestamp', 'x', 'y']].values.T

    # Time threshold calculation using broadcasting
    within_threshold = np.triu(np.abs(unix_timestamps[:, np.newaxis] - unix_timestamps) <= (time_thresh * 60), k=1)
    t_pairs = np.where(within_threshold)

    # Distance calculation
    distances_sq = (x[t_pairs[0]] - x[t_pairs[1]])**2 + (y[t_pairs[0]] - y[t_pairs[1]])**2
    neighbor_pairs = distances_sq < dist_thresh**2

    # Building the neighbor dictionary
    neighbor_dict = defaultdict(set)
    for i, j in zip(t_pairs[0][neighbor_pairs], t_pairs[1][neighbor_pairs]):
        neighbor_dict[unix_timestamps[i]].add(unix_timestamps[j])

    return neighbor_dict


def dbscan(data, time_thresh, dist_thresh, min_pts):
    df = data.copy(deep=True)
    c = -1
    neighbor_dict = find_neighbors(df, time_thresh, dist_thresh)
    
    df['cluster'] = -2
    for idx in df.index:
        time = df.unix_timestamp[idx]
        if df.cluster[idx] == -2:   # unprocessed point
            if len(neighbor_dict[time]) < min_pts:
                df.at[idx, 'cluster'] = -1
            else:
                c = c + 1
                df.at[idx, 'cluster'] = c
                S = neighbor_dict[time].copy()
                while len(S)>0:
                    point = S.pop()
                    point = df.index[df['unix_timestamp'] == point][0]
                    if df.loc[point, 'cluster'] >= 0:   # point is already a core/border point
                        continue
                    
                    df.loc[point, 'cluster'] = c
                    if len(neighbor_dict[point]) >= min_pts:
                        S = S.union(neighbor_dict[point].copy())
                        
    return df[['identifier','local_timestamp','unix_timestamp','x','y', 'cluster']]

def process_clusters(df, time_thresh, dist_thresh, min_pts, output, min_duration=4):
    try:
        df = df.loc[df.cluster!=-1]
        if len(df) == 0:
            return False
        elif len(df.cluster.unique()) == 1:   #recognize that they are all same value
            x = dbscan(data=df, time_thresh=time_thresh, dist_thresh=dist_thresh, min_pts=min_pts)   # we rerun dbscan because possibly these points no longer hold their own
            y = x.loc[x.cluster != -1] 
            if len(y) > 0:   # there is exactly 1 cluster of all the same values
                start = y.local_timestamp.min().isoformat()
                duration = int((y.local_timestamp.max() - y.local_timestamp.min()).total_seconds() // 60)
                n = len(y)
                x_mean = np.mean(y.x)
                y_mean = np.mean(y.y)
                radius = np.sqrt(np.mean(np.sum((y[['x','y']].to_numpy() - np.mean(y[['x','y']].to_numpy(), axis=0))**2, axis=1)))
                max_gap = int(y.local_timestamp.diff().max().total_seconds() // 60)
                if duration > min_duration:
                    output += [(start, duration, x_mean, y_mean, n, max_gap, radius)]
                return True
            elif len(y)==0: # the points in df, despite originally being part of a cluster, no longer hold their own
                return False
        else: # >1 unique value for cluster
            i, j = extract_middle(df) # indices of the "middle" of the cluster (i.e., the head is the first contiguous cluster, and the middle follows that)
            # print(i,j)
            # recurisvely processes clusters
            if process_clusters(df[i:j].copy(), time_thresh, dist_thresh, min_pts, output): # valid cluster in the middle
                process_clusters(df[:i].copy(), time_thresh, dist_thresh, min_pts, output) #we process the initial stub
                process_clusters(df[j:].copy(), time_thresh, dist_thresh, min_pts, output) #we process the "tail"
                return True
            else: # no valid cluster in the middle
                return process_clusters(pd.concat( [df[:i].copy(),df[j:].copy()] ), time_thresh, dist_thresh, min_pts, output) #what if this is out of bounds?
    except Exception as e:
        sys.exit(f'{e}')
        
def temporal_dbscan(data, time_thresh, dist_thresh, min_pts):
    output = []
    #data['local_timestamp'] = data['local_timestamp'].apply(lambda x: parser.parse(x))
    data['local_timestamp'] = pd.to_datetime(data['local_timestamp']) #the parser was giving me issues
    data = data.sort_values('local_timestamp')
    data = data.set_index('local_timestamp', drop=False)
    data.index.name = 'index'
    first_clusters = dbscan(data=data, time_thresh=time_thresh, dist_thresh=dist_thresh, min_pts=min_pts)
    process_clusters(df=first_clusters, time_thresh=time_thresh, dist_thresh=dist_thresh, min_pts=min_pts, output=output, min_duration=4)
    pdf = pd.DataFrame(output, columns=['start','duration','x','y','num_points', 'max_gap', 'radius'])
    pdf['identifier'] = str(data.identifier.iloc[0])
    
    return pdf

def dbscan_patches(df, i, seed=None):
    #i = 0 is coarse dbscan, i = 1 is fine dbscan
    if seed:
        npr.seed(seed)
    else:
        seed = npr.randint(0,1000,1)[0]
        npr.seed(seed)
        print(seed)
    
    params = [(480, 120, 2), (110, 70, 3)][i]    
    
    #Compute clusters
    clusters = temporal_dbscan(df.copy(), *params)
    
    clusters['start'] = pd.to_datetime(clusters.start)
    clusters['end'] = clusters.start + pd.to_timedelta(clusters.duration,'m')
    df['local_timestamp'] = pd.to_datetime(df.local_timestamp)

    #Add patches for each cluster
    cluster_pings = []
    for idx, row in clusters.iterrows():
        cluster_pings += [df.loc[(df.local_timestamp>=row.start)&(df.local_timestamp<=row.end), ['x','y']]]
        
    return(cluster_pings)

# =========================== #
#   LACHESIS IMPLEMENTATION   #
# =========================== #

def medoid(coords):
    """
    Computes the medoid of a set of coordinates. The medoid is defined to be the coordinate 
    in a set that minimizes the maximum distance to every other coordinate in the set.

    Parameters
    ----------
    coords: numpy array
        n x 2 array of pings (x, y).

    Returns
    -------
    numpy array of shape (1,2) denoting medroid coordinates.
    """
    
    # Create matrix of all pairwise distances
    x = coords[:, 0]
    y = coords[:, 1]
    x_diff = x[:, np.newaxis] - x
    y_diff = y[:, np.newaxis] - y
    distances = np.sqrt(x_diff**2 + y_diff**2)
    
    max_distances = np.amax(distances, axis=1)
    medoid_index = np.argmin(max_distances)
    
    return coords[medoid_index,:]

def diameter(coords):
    """
    Computes the diameter of a set of coordinates, where the diameter is defined to be the greatest 
    distance between any two coordinates in a set.

    Parameters
    ----------
    coords: numpy array
        n x 2 array of pings (x, y).

    Returns
    -------
    float denoting diameter.
    """
    
    # Create matrix of all pairwise distances
    x = coords[:, 0]
    y = coords[:, 1]
    x_diff = x[:, np.newaxis] - x
    y_diff = y[:, np.newaxis] - y
    distances = np.sqrt(x_diff**2 + y_diff**2)
    
    return np.max(distances)

def lachesis(traj, delta_dur, delta_roam):
    """
    Extracts stays from raw location data.

    Parameters
    ----------
    traj: numpy array
        simulated trajectory from simulate_traj.
    
    delta_dur: float
        Minimum duration for a stay (stay duration).
        
    delta_roam: float
        Maximum roaming distance for a stay (roaming distance).

    Returns
    -------
    pandas array with columns 'medoid_x', 'medoid_y', 'start_time', 'end_time'
    """
    
    coords = traj[['x', 'y']].to_numpy()
    stays = np.empty((0,4))
    i = 0
    while i < len(traj)-1:
        
        j_star = next((j for j in range(i, len(traj)) if 
                       traj['unix_timestamp'].iloc[j] - traj['unix_timestamp'].iloc[i] >= delta_dur), -1)
        
        if j_star == -1 or diameter(coords[i:j_star+1]) > delta_roam:
            i += 1
        else:
            j_star = next((j for j in range(j_star, len(traj)) if diameter(coords[i:j+1]) > delta_roam), len(traj)) - 1
            
            stay_medoid = medoid(coords[i:j_star+1])
            start = traj['local_timestamp'].iloc[i]
            end   = traj['local_timestamp'].iloc[j_star]
            stay  = np.array([[stay_medoid[0], stay_medoid[1], start, end]])
            stays = np.concatenate((stays, stay), axis=0)
            
            i = j_star + 1
            
    stays = pd.DataFrame(stays, columns = ['medoid_x', 'medoid_y', 'start_time', 'end_time'])
    return stays

def lachesis_patches(traj, i):
    
    #i = 0 is coarse, i = 1 is fine
    
    params = [(10*60, 400), (10*60, 200)][i]    #TODO 
    
    #Compute clusters
    clusters = lachesis(traj, *params)

    #Add patches for each cluster
    cluster_pings = []
    for idx, row in clusters.iterrows():
        cluster_pings += [traj.loc[(traj.local_timestamp>=row.start_time)&(traj.local_timestamp<=row.end_time), ['x','y']]]
        
    return(cluster_pings)