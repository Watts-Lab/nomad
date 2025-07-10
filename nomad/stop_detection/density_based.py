import nomad.stop_detection.hdbscan as HDBSCAN
import nomad.stop_detection.ta_dbscan as TADBSCAN

'''
sd.density_based(data, method='hdbscan', traj_cols, other_parameter_1=x, other_parameter_2=y)
_temporal_dbscan_labels(data, time_thresh, dist_thresh, min_pts, return_cores=False, traj_cols=None, **kwargs)
'''

def density_based(data, time_thresh, dist_thresh, min_pts, location_id=None, traj_cols=None, method='hdbscan'):
    if data[traj_cols['user_id']].nunique() > 1:
        if method == 'hdbscan':
            stop_table = data.groupby(['uid']).apply(lambda x: HDBSCAN.st_hdbscan(data=x,
                                                                                  time_thresh=time_thresh,
                                                                                  min_pts=min_pts,
                                                                                  min_cluster_size=3,
                                                                                  traj_cols=traj_cols))
        elif method == 'ta-dbscan':
            stop_table = data.groupby(['uid']).apply(lambda x: TADBSCAN.temporal_dbscan(data=x,
                                                                                        time_thresh=time_thresh,
                                                                                        dist_thresh=dist_thresh,
                                                                                        min_pts=min_pts,
                                                                                        traj_cols=traj_cols))
    else: # only 1 user
        if method == 'hdbscan':
            stop_table = HDBSCAN.st_hdbscan(data=data,
                                            time_thresh=time_thresh,
                                            min_pts=min_pts,
                                            min_cluster_size=3,
                                            traj_cols=traj_cols)
        elif method == 'ta-dbscan':
            stop_table =  TADBSCAN.temporal_dbscan(data=data,
                                                   time_thresh=time_thresh,
                                                   dist_thresh=dist_thresh,
                                                   min_pts=min_pts,
                                                   traj_cols=traj_cols)
    return stop_table