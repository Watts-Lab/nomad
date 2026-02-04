import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import pdb

##todo: impl secscan by following dbscan.py
def seqscan_labels(
    df,
    eps,
    delta, # presence treshold: min time to count as stay
    min_pts=3,
    user_id=None,
    x_col="x",
    y_col="y",
    t_col="timestamp",
):
    data = df.copy()

    if user_id is not None:
        data = data.loc[data["user_id"] == user_id].copy()

    # not necessary?
    # sort points and keep orig idex so can return aligned labels
    # data = data.sort_values(t_col, kind="mergesort").reset_index(drop=False)
    orig_idx = data["index"].to_numpy()

    X = data[[x_col, y_col]].to_numpy()
    t = data[t_col].to_numpy().astype(float)
    n = len(data)
    out = np.full(n, -1, dtype=int)

    # presence def: stay must satisfy a minimum time spent in the region
    def presence(curr_window, lo, hi):
        total = 0.0
        for k in range(lo, hi):
            if curr_window[k - lo] and curr_window[k + 1 - lo]:
                total += (t[k + 1] - t[k])
        return total
    
    def expand(active_cluster, lo, hi, newPoint):
        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X[lo:hi+1])
        return labels, (labels[newPoint - lo] == active_cluster)

    # SeqScan main loop start
    start = 0        # current time context ind
    end = -1         # last ind in active stay
    active_cid = None  # which DBSCAN cluster id is curr stay
    labels = 0      # permanent stay label

    for c in range(n):
        lo, hi = start, c # curr timeContext

        # if already have active stay,, expand
        if active_cid is not None:
            _, ok = expand(active_cid, lo, hi, c)
            if ok:
                end = c
                continue

        # if can't expand, look for new stay in segment after end of last stay
        seg_lo, seg_hi = end + 1, c
        next_cid = None

        if seg_lo <= seg_hi:
            # Run DBSCNA on potential segment
            labels_seg = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X[seg_lo:seg_hi+1])

            # Find first non-noise cluster >= delta
            #sorted?
            for cid in sorted(set(labels_seg)):
                if cid == -1:
                    continue
                curr_window = (labels_seg == cid)
                if (presence(curr_window, seg_lo, seg_hi) >= delta):
                    next_cid = cid
                    break
        
        # if find next stay cluster, close old stay
        if next_cid is not None:
            # close old stay, assign perm label
            if active_cid is not None:
                labels_old = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X[start:end+1])
                for i in range(start, end + 1):
                    if labels_old[i - start] == active_cid:
                        out[i] = labels
                # move stay labels
                labels += 1
                start = end + 1
        # switch active stay
            active_cid = next_cid
            end = c

    return pd.Series(out, index=orig_idx, name="seqscan")
        

    # while start < n:
    #     # print("Entering while")

    #     if end < start:
    #         end = start
    #     if end >= n:
    #         break

    #     # 1) DBSCAN on window W=[start,end]
    #     Xw = X[start:end+1]
    #     db = DBSCAN(eps=eps, min_samples=min_pts + 1)
    #     labels = db.fit_predict(Xw)
    #     last_db[start:end+1] = labels + current_label

    #     # 2) Determine how many clusters exist (excluding noise=-1)
    #     # clusters = sorted([c for c in set(labels) if c != -1])

    #     # grow window
    #     if not(labels > 0).any():
    #         end += 1
    #         continue

    #     # Else: "cut and log" first cluster (prefer label 0)

    #     # TODO: labels = an array of len w, core_indices also array, we want to find np.where(labels[core_indices]==0[-1]])
    #     # want the last core indices that is value 0
    #     in_target = (labels == 0)

    #     # Core points are indices in the window that are core samples
    #     core_rel = set(getattr(db, "core_sample_indices_", []))

    #     # Latest core point inside the target cluster (by window index)
    #     core_in_target = [i for i in np.where(in_target)[0] if i in core_rel]
    #     # TODO: consider [-1] instead of max, for computational reasons
    #     cut_rel = max(core_in_target)

    #     cut_global = start + cut_rel

    #     # TODO: start after the cut maybe?
    #     # Permanently label all points in target cluster strictly before cut
    #     # TODO: rewrite to be simpler, is sorted so should be easier
    #     finalize_rel = np.where(in_target & (np.arange(len(labels)) < cut_rel))[0]
    #     print("finalizerel", finalize_rel)
    #     finalize_global = start + finalize_rel
    #     print("finalize global", finalize_global)

    #     # assign only those not already assigned
    #     mask = (perm.iloc[finalize_global] == -1)
    #     print("mask:", mask)
    #     perm.iloc[finalize_global[mask]] = current_label

    #     # move window start to cut point; reset window
    #     start = cut_global
    #     end = start
    #     current_label += 1
    # print("finished while")

    # out = data.copy()
    # print("out", out)
    # out["sw_label"] = perm
    # out["sw_dbscan_last"] = last_db
    # return perm

#TODO: dont use np.where, key can use same indices 1-n, look at exp_hdbscan_paper dbscan impl 
#TODO: perm should be labels
#TODO: PUSH TO NEW BRANCH, change back to hdbscan, git force pull, switch back to new branch (git checkout), git merge
