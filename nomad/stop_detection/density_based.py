import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import pdb

##todo: impl secscan by following dbscan.py
def seqscan_labels(
    df,
    eps,
    min_pts=3,
    user_id=None,
    x_col="x",
    y_col="y",
    t_col="timestamp",
):
    """
    start, end = [t_0, -inf] (time context)
    active_cluster = set()
    
    For t in df.timestamp:
      - Run DBSCAN on [start, t]
      - when DBSCAN produces 1 non-noise cluster -> end = t, and set active cluster 
      - Once there is an active cluster, the next ping might be part of it (core) or not
      - If t in active_cluster:
          * end = t
      - If t is not in active cluster
          * run DBSCAN on [end+1, t] and see if we change to new cluster and close previous cluster

    Note:
      - Otherwise, follow conventions of dbscan.py
    """
    data = df.copy()

    if user_id is not None:
        data = data.loc[data["user_id"] == user_id].copy()

    n = len(data)
    if n == 0:
        data["sw_label"] = []
        data["sw_dbscan_last"] = []
        return data

    X = data[[x_col, y_col]].to_numpy()

    # perm = np.full(n, -1, dtype=int)
    perm = pd.Series(-1, index=data.index, name='cluster')
    last_db = np.full(n, -2, dtype=int)

    start = 0
    end = 0
    current_label = 0

    while start < n:
        # print("Entering while")

        if end < start:
            end = start
        if end >= n:
            break

        # 1) DBSCAN on window W=[start,end]
        Xw = X[start:end+1]
        db = DBSCAN(eps=eps, min_samples=min_pts + 1)
        labels = db.fit_predict(Xw)
        last_db[start:end+1] = labels + current_label

        # 2) Determine how many clusters exist (excluding noise=-1)
        # clusters = sorted([c for c in set(labels) if c != -1])

        # grow window
        if not(labels > 0).any():
            end += 1
            continue

        # Else: "cut and log" first cluster (prefer label 0)

        # TODO: labels = an array of len w, core_indices also array, we want to find np.where(labels[core_indices]==0[-1]])
        # want the last core indices that is value 0
        in_target = (labels == 0)

        # Core points are indices in the window that are core samples
        core_rel = set(getattr(db, "core_sample_indices_", []))

        # Latest core point inside the target cluster (by window index)
        core_in_target = [i for i in np.where(in_target)[0] if i in core_rel]
        # TODO: consider [-1] instead of max, for computational reasons
        cut_rel = max(core_in_target)

        cut_global = start + cut_rel

        # TODO: start after the cut maybe?
        # Permanently label all points in target cluster strictly before cut
        # TODO: rewrite to be simpler, is sorted so should be easier
        finalize_rel = np.where(in_target & (np.arange(len(labels)) < cut_rel))[0]
        print("finalizerel", finalize_rel)
        finalize_global = start + finalize_rel
        print("finalize global", finalize_global)

        # assign only those not already assigned
        mask = (perm.iloc[finalize_global] == -1)
        print("mask:", mask)
        perm.iloc[finalize_global[mask]] = current_label

        # move window start to cut point; reset window
        start = cut_global
        end = start
        current_label += 1
    print("finished while")

    out = data.copy()
    print("out", out)
    out["sw_label"] = perm
    out["sw_dbscan_last"] = last_db
    return perm

#TODO: dont use np.where, key can use same indices 1-n, look at exp_hdbscan_paper dbscan impl 
#TODO: perm should be labels
#TODO: PUSH TO NEW BRANCH, change back to hdbscan, git force pull, switch back to new branch (git checkout), git merge
