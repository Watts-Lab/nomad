import pandas as pd
import numpy as np
import geopandas as gpd
import nomad.io.base as loader
import nomad.stop_detection.utils as utils
from nomad.filters import to_timestamp

##########################################
########         SeqScan          ########
##########################################

def _dist_xy(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx + dy * dy) ** 0.5


def _is_core(nbrs, idx, min_pts):
    return len(nbrs[idx]) >= min_pts


def _presence(members, times):
    """
    members: set of indices in current window
    times: original times array

    Return:
    sum time gaps for consecutive indices that are both members
    """
    if not members:
        return 0.0

    idxs = sorted(members)
    mem = set(idxs)

    presence = 0.0

    for idx in idxs:
        if idx + 1 in mem:
            presence += times[idx + 1] - times[idx]
    
    return float(presence)


def _assign(region_list, regions_dict, idx, region):
    '''
    Assign point idx to region_id.

                    0  1  2   3  4
    region_list = [-1, 0, 0, -1, 1]

    regions_dict = {0: {1, 2},
                    1: {4}}
    '''
    region_list[idx] = region # Point at idx now belongs to region
    regions_dict[region].add(idx) # Region now contains point at idx

def _merge(region_list, regions_dict, keep_region, remove_region):
    '''
    Merge region 'remove' into region 'keep'

    rid_list = [-1, 3, 3, -1, -1, 3, -1, -1, 7, 7]
    regions_dict = {3: {1,2,5}, 7: {8,9}}

    _merge(rid_list, regions_dict, keep_region=3, remove_region=7) => rid_list = [-1, 3, 3, -1, -1, 3, -1, -1, 3, 3], regions_dict = {3: {1,2,5,8,9}}
    '''
    if keep_region == remove_region:
        return keep_region
    
    if remove_region not in regions_dict:
        return keep_region
    
    for u in regions_dict[remove_region]:
        region_list[u] = keep_region

    regions_dict[keep_region] |= regions_dict[remove_region] # Add all points from remove_region into keep_region.

    del regions_dict[remove_region]

    return keep_region


def _region_for_new_core(nbrs, region_list, regions_dict, idx, min_pts, next_region_id):
    """
    when a point becomes a core point, make sure it creates/joins a region and pulls in border points

    If idx is core and unassigned:
      - create new region if no neighbor has region
      - else adopt/merge neighbor regions
      - assign idx and its unassigned neighbors as border points
    next_region_id is a 1-element list holding next_region_id (so we can mutate it).
    """
    if region_list[idx] != -1: # if point already assigned to region, dont reassign
        return
    
    if not _is_core(nbrs, idx, min_pts): # if point is not core, do nothing
        return

    # get region ids of neighbors
    neighbor_region_ids = {region_list[v] for v in nbrs[idx] if region_list[v] != -1}

    if not neighbor_region_ids: # no neighbor regions, create new
        target_region = next_region_id[0]
        next_region_id[0] += 1
        regions_dict[target_region] = set()
    else: # if neighbors already in regions, get the earliest region and merge the rest
        target_region = min(neighbor_region_ids)

        for other in sorted(neighbor_region_ids):
            if other != target_region:
                _merge(region_list, regions_dict, target_region, other)

    _assign(region_list, regions_dict, idx, target_region)

    for v in nbrs[idx]:
        if region_list[v] == -1:
            _assign(region_list, regions_dict, v, target_region)


def _add_point_to_window(q, start, xs, ys, dist_thresh, min_pts, nbrs, region_list, regions_dict, next_region_id):
    """
    Incrementally add global point q into the current SeqScan window [start..q].
    Mutates nbrs / region_list / regions_dict / next_region_id in-place.
    """

    # Ensure q has a neighbor set (include itself)
    if q not in nbrs:
        nbrs[q] = set()
    
    nbrs[q].add(q)

    # Compute neighbors of q against prior points in the CURRENT WINDOW: j in [start..q-1]
    xq, yq = xs[q], ys[q]

    for j in range(start, q):
        if _dist_xy(xq, yq, xs[j], ys[j]) <= dist_thresh:
            nbrs[q].add(j)

            if j not in nbrs:
                nbrs[j] = set([j])
            
            nbrs[j].add(q)

    # 1) if any neighbor p is already in a region, pull q into it (or merge)
    # 2) adding q may make neighbors become core; if so, create/adopt/merge regions & absorb borders
    for p in list(nbrs[q]):
        if p == q:
            continue

        p_region = region_list[p]

        if p_region != -1:
            q_region = region_list[q]
            if q_region == -1:
                _assign(region_list, regions_dict, q, p_region)
            elif q_region != p_region:
                keep = min(q_region, p_region)
                kill = max(q_region, p_region)
                _merge(region_list, regions_dict, keep, kill)

        # neighbor might have become core due to q being added
        _region_for_new_core(nbrs, region_list, regions_dict, p, min_pts, next_region_id)

    # q itself might be core now
    _region_for_new_core(nbrs, region_list, regions_dict, q, min_pts, next_region_id)


def _pick_earliest_persistent_region(regions_dict, times, min_dur):
    """
    regions_dict: {region_id: set(global_indices)}
    """
    best_region = None
    best_first_idx = None

    for region_id, members in regions_dict.items():
        if _presence(members, times) > min_dur:
            first_idx = min(members)

            if best_first_idx is None or first_idx < best_first_idx:
                best_first_idx = first_idx
                best_region = region_id

    return best_region

def _find_first_persistent_in_suffix(suf_start, suf_end, xs, ys, times, dist_thresh, min_pts, min_dur):
    """
    Build a fresh (temporary) incremental-DBSCAN state over suffix [suf_start..suf_end]
    and return the earliest persistent region's
    member indices (sorted) as soon as any persistent region appears.

    Returns:
      sorted list of global indices (members of earliest persistent region), or None
    """
    if suf_end < suf_start:
        return None

    # Fresh state just for the suffix
    nbrs = {}
    region_list = region_list = [-1] * len(xs)
    regions_dict = {}
    next_region_id = [0]

    # Incrementally add points in the suffix window
    for q in range(suf_start, suf_end + 1):
        _add_point_to_window(q, suf_start, xs, ys, dist_thresh, min_pts, nbrs, region_list, regions_dict, next_region_id)

        # check if any region is persistent yet; if yes, return earliest one
        best_region = _pick_earliest_persistent_region(regions_dict, times, min_dur)

        if best_region is not None:
            return sorted(regions_dict[best_region])

    return None

def seqscan_labels(data, dist_thresh, min_pts, min_dur, traj_cols=None, **kwargs):
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    if data.empty:
        return pd.DataFrame()

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)

    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    xs = data[traj_cols[coord_key1]].to_numpy(dtype=float)
    ys = data[traj_cols[coord_key2]].to_numpy(dtype=float)

    times = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]

    n = len(data)
    labels = pd.Series(-1, index=data.index, name="cluster")

    # SeqScan pointers (global indices)
    start = 0
    end = -1
    active_region = None  # region id within current window-state
    c = 0

    # Current window state (global-indexed)
    nbrs = {}
    region_list = [-1] * n
    regions_dict = {}
    next_region_id = [0]

    # debug
    stay_regions = []
    noise_positions = []

    while c < n:
        # update state with point c within current window [start..c]
        _add_point_to_window(c, start, xs, ys, dist_thresh, min_pts, nbrs, region_list, regions_dict, next_region_id)

        if active_region is None:
            chosen = _pick_earliest_persistent_region(regions_dict, times, min_dur)
            if chosen is not None:
                active_region = chosen
                end = c
        else:
            # is point c in the active region?
            if region_list[c] == active_region:
                end = c
            else:
                # suffix search [end+1 .. c]
                suf_members = _find_first_persistent_in_suffix(end + 1, c, xs, ys, times, dist_thresh, min_pts, min_dur)

                if suf_members is not None:
                    # close current active region
                    active_members = regions_dict.get(active_region, set())
                    
                    if active_members:
                        rid_out = len(stay_regions)
                        labels.iloc[sorted(active_members)] = rid_out

                        stay_regions.append({
                            "rid": rid_out,
                            "members": sorted(active_members),
                            "start_idx": min(active_members),
                            "end_idx": max(active_members),
                            "presence": _presence(active_members, times),
                        })

                        # mark noise inside [start..end] excluding active members
                        mem_set = set(active_members)
                        for gi in range(start, end + 1):
                            if gi not in mem_set:
                                noise_positions.append(gi)

                    # move window start
                    start = end + 1

                    # reset window state (rebuild fresh for new window starting at start)
                    nbrs = {}
                    region_list = [-1] * n
                    regions_dict = {}
                    next_region_id = [0]
                    active_region = None
                    end = -1

                    # rebuild state from [start..c] (inclusive), stop once first persistent region appears
                    for g in range(start, c + 1):
                        _add_point_to_window(g, start, xs, ys, dist_thresh, min_pts, nbrs, region_list, regions_dict, next_region_id)
                        chosen = _pick_earliest_persistent_region(regions_dict, times, min_dur)

                        if chosen is not None:
                            active_region = chosen
                            end = g
                            break

        c += 1

    # finalize last active region
    if active_region is not None:
        active_members = regions_dict.get(active_region, set())
        if active_members:
            rid_out = len(stay_regions)
            labels.iloc[sorted(active_members)] = rid_out

            stay_regions.append({
                "rid": rid_out,
                "members": sorted(active_members),
                "start_idx": min(active_members),
                "end_idx": max(active_members),
                "presence": _presence(active_members, times),
            })

            mem_set = set(active_members)
            for gi in range(start, n):
                if gi not in mem_set:
                    noise_positions.append(gi)

    return labels