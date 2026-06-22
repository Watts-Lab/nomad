"""Various algorithms for estimating individual co-location"""

# Authors: Thomas Li and Francisco Barreras

import nomad.filters as filters
import nomad.io.base as loader
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KDTree


_EARTH_RADIUS_M = 6_371_000
_TEMPORAL_BLOCK_SECONDS = 60 * 60


def _temporal_blocks(start, end):
    """Return stop-to-hour block memberships for non-empty intervals."""
    valid = end > start
    if not valid.any():
        return pd.DataFrame(columns=["block", "stop"])

    stop = np.flatnonzero(valid)
    start_bin = np.floor_divide(start[valid], _TEMPORAL_BLOCK_SECONDS).astype(np.int64)
    end_bin = np.floor_divide(end[valid] - 1, _TEMPORAL_BLOCK_SECONDS).astype(np.int64)
    counts = end_bin - start_bin + 1
    offsets = np.repeat(np.r_[0, counts.cumsum()[:-1]], counts)
    block = np.repeat(start_bin, counts) + np.arange(counts.sum()) - offsets
    return pd.DataFrame({"block": block, "stop": np.repeat(stop, counts)})


def _pair_distances(query_coords, stop_1, stop_2, use_lon_lat):
    """Return distances for stop pairs."""
    if use_lon_lat:
        lat_1, lon_1 = query_coords[stop_1].T
        lat_2, lon_2 = query_coords[stop_2].T
        dlat = lat_2 - lat_1
        dlon = lon_2 - lon_1
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat_1) * np.cos(lat_2) * np.sin(dlon / 2) ** 2
        )
        return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return np.linalg.norm(query_coords[stop_1] - query_coords[stop_2], axis=1)


def _radius_candidates(
    stops,
    traj_cols,
    input_traj_cols,
    kwargs,
    distance_threshold,
    start,
    end,
    users,
):
    """Return stop row pairs within distance_threshold and their distances."""
    coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(
        stops.columns,
        input_traj_cols,
        kwargs,
    )

    if use_lon_lat:
        # Haversine uses (lat, lon) in radians.
        coords = stops[[traj_cols["latitude"], traj_cols["longitude"]]].to_numpy()
        radius = distance_threshold / _EARTH_RADIUS_M
        query_coords = np.radians(coords)
        tree_class = BallTree
        tree_kwargs = {"metric": "haversine"}
    else:
        coords = stops[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy()
        radius = distance_threshold
        query_coords = coords
        tree_class = KDTree
        tree_kwargs = {}

    blocks = _temporal_blocks(start, end)
    if blocks.empty:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)

    rows = []
    cols = []
    for _, block_stops in blocks.groupby("block", sort=False)["stop"]:
        block_stops = block_stops.to_numpy()
        if len(block_stops) < 2:
            continue

        tree = tree_class(query_coords[block_stops], **tree_kwargs)
        indices = tree.query_radius(
            query_coords[block_stops],
            r=radius,
            return_distance=False,
            sort_results=False,
        )
        counts = np.array([len(idx) for idx in indices])
        row = block_stops[np.repeat(np.arange(len(block_stops)), counts)]
        col = block_stops[np.concatenate(indices).astype(int)]
        keep = row < col
        rows.append(row[keep])
        cols.append(col[keep])

    if not rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)

    pairs = pd.DataFrame(
        {"stop_1": np.concatenate(rows), "stop_2": np.concatenate(cols)}
    ).drop_duplicates()
    stop_1 = pairs["stop_1"].to_numpy()
    stop_2 = pairs["stop_2"].to_numpy()
    keep = (
        (users[stop_1] != users[stop_2])
        & (start[stop_1] < end[stop_2])
        & (start[stop_2] < end[stop_1])
    )
    stop_1, stop_2 = stop_1[keep], stop_2[keep]
    if len(stop_1) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)

    distance = _pair_distances(query_coords, stop_1, stop_2, use_lon_lat)
    keep = distance <= distance_threshold
    return stop_1[keep], stop_2[keep], distance[keep]


def estimate_contacts(stops, distance_threshold=None, traj_cols=None, **kwargs):
    """
    Estimate undirected co-location/contact events from a stop table.

    With ``distance_threshold=None``, contacts are stops from different users at
    the same ``location_id`` with strictly overlapping times. With a distance
    threshold, contacts are stops from different users within that radius and
    with strictly overlapping times. Latitude/longitude thresholds are meters;
    projected x/y thresholds use the coordinate units.

    Parameters
    ----------
    stops : pd.DataFrame
        Stop table with user_id, a start time (datetime or timestamp), an end
        time or duration, and either location_id (exact mode) or coordinates
        (radius mode).
    distance_threshold : float, optional
        Contact radius. None selects exact-location mode. Meters for lat/lon,
        coordinate units for projected x/y.
    traj_cols : dict, optional
        Mapping for user_id, time, duration, location_id, and coordinates.
    **kwargs
        Column-name overrides forwarded to the traj_cols resolver.

    Returns
    -------
    pd.DataFrame
        Columns = [user_id_1, user_id_2, contact_start, contact_end,
        overlap_duration, location_id or distance]. overlap_duration is in
        minutes. Contact times are returned as datetime for datetime input and
        as Unix seconds for timestamp input; timezone metadata on datetime
        input is not preserved (output is timezone-naive).
    """
    contact_col = "location_id" if distance_threshold is None else "distance"
    output_cols = [
        "user_id_1",
        "user_id_2",
        "contact_start",
        "contact_end",
        "overlap_duration",
        contact_col,
    ]
    if len(stops) == 0:
        return pd.DataFrame(columns=output_cols)

    input_traj_cols = traj_cols
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)

    uid_col = traj_cols["user_id"]
    if uid_col not in stops.columns:
        raise ValueError(f"Missing required user_id column '{uid_col}'.")

    t_key, use_datetime = loader._fallback_time_cols_dt(
        stops.columns,
        input_traj_cols,
        kwargs,
    )
    start_col = traj_cols[t_key]
    end_key = "end_datetime" if use_datetime else "end_timestamp"
    end_col = traj_cols[end_key]
    if end_col in stops.columns:
        end_time = stops[end_col]
    elif loader._has_duration_cols(stops.columns, traj_cols):
        if use_datetime:
            end_time = stops[start_col] + pd.to_timedelta(
                stops[traj_cols["duration"]],
                unit="m",
            )
        else:
            end_time = stops[start_col] + stops[traj_cols["duration"]] * 60
    else:
        raise ValueError(
            "Contact estimation requires an end time or duration column."
        )

    if use_datetime:
        start = filters.to_timestamp(stops[start_col]).to_numpy()
        end = filters.to_timestamp(end_time).to_numpy()
    else:
        start = stops[start_col].to_numpy()
        end = end_time.to_numpy()

    users = stops[uid_col].to_numpy()

    if distance_threshold is None:
        loc_col = traj_cols["location_id"]
        if loc_col not in stops.columns:
            raise ValueError(
                "Exact-location contact estimation requires a location_id column."
            )
        if stops[loc_col].isna().any():
            raise ValueError(
                "Exact-location contact estimation requires non-missing location_id values."
            )
        location = stops[loc_col].to_numpy()
        # Pair stops sharing a location, then keep distinct users with overlap.
        candidates = pd.DataFrame({"stop": np.arange(len(stops)), "loc": location})
        pairs = candidates.merge(candidates, on="loc", suffixes=("_1", "_2"))
        stop_1 = pairs["stop_1"].to_numpy()
        stop_2 = pairs["stop_2"].to_numpy()
        keep = (
            (stop_1 < stop_2)
            & (users[stop_1] != users[stop_2])
            & (start[stop_1] < end[stop_2])
            & (start[stop_2] < end[stop_1])
        )
        stop_1, stop_2 = stop_1[keep], stop_2[keep]
        contact_values = location[stop_1]
    else:
        stop_1, stop_2, distance = _radius_candidates(
            stops,
            traj_cols,
            input_traj_cols,
            kwargs,
            distance_threshold,
            start,
            end,
            users,
        )
        contact_values = distance

    contact_start = np.maximum(start[stop_1], start[stop_2])
    contact_end = np.minimum(end[stop_1], end[stop_2])
    return pd.DataFrame(
        {
            "user_id_1": users[stop_1],
            "user_id_2": users[stop_2],
            "contact_start": (
                pd.to_datetime(contact_start, unit="s") if use_datetime else contact_start
            ),
            "contact_end": (
                pd.to_datetime(contact_end, unit="s") if use_datetime else contact_end
            ),
            "overlap_duration": ((contact_end - contact_start) // 60).astype(int),
            contact_col: contact_values,
        },
        columns=output_cols,
    )


def compute_contact_weights(
    contacts,
    method="duration",
    distance_threshold=None,
    overlap_duration_col="overlap_duration",
    distance_col="distance",
):
    """
    Compute contact weights from a contact event table.

    Supported methods are ``"duration"`` and ``"linear_distance"``. Linear
    distance weighting uses ``overlap_duration * max(0, 1 - distance / threshold)``.

    Parameters
    ----------
    contacts : pd.DataFrame
        Contact event table from estimate_contacts.
    method : str
        Either 'duration' or 'linear_distance'.
    distance_threshold : float, optional
        Contact radius used by 'linear_distance'; required for that method.
    overlap_duration_col : str
        Column holding overlap duration in minutes.
    distance_col : str
        Column holding contact distance; required for 'linear_distance'.

    Returns
    -------
    pandas.Series
        Series named ``contact_weight`` and indexed like ``contacts``.
    """
    if overlap_duration_col not in contacts.columns:
        raise ValueError(
            f"Missing required overlap duration column '{overlap_duration_col}'."
        )

    method = method.lower()

    if method == "duration":
        return contacts[overlap_duration_col].rename("contact_weight")

    if method != "linear_distance":
        raise ValueError("method must be one of 'duration' or 'linear_distance'.")

    if distance_col not in contacts.columns:
        raise ValueError(
            f"method='linear_distance' requires a '{distance_col}' column."
        )
    if distance_threshold is None:
        raise ValueError(
            "method='linear_distance' requires an explicit distance_threshold."
        )

    return (
        contacts[overlap_duration_col]
        * np.maximum(0, 1 - contacts[distance_col] / distance_threshold)
    ).rename("contact_weight")


def overlapping_visits(left, right, match_location=False, traj_cols=None, right_traj_cols=None, **kwargs):
    if len(left) == 0 or len(right) == 0:
        return pd.DataFrame()

    left = left.copy()
    right = right.copy()
    input_traj_cols = traj_cols
    traj_cols = loader._parse_traj_cols(left.columns, traj_cols, kwargs)
    if right_traj_cols is None:
        right_schema_input = input_traj_cols
        right_kwargs = kwargs
    else:
        right_schema_input = right_traj_cols
        right_kwargs = {}
    temp_traj_cols = loader._parse_traj_cols(right.columns, right_schema_input, right_kwargs)
    right_schema_hint = ""
    if right_traj_cols is None:
        right_schema_hint = (
            " The shared traj_cols/kwargs mapping appears not to fit the right table. "
            "Pass right_traj_cols for the right table, or make both tables use the same relevant "
            "column names for time, duration, and location."
        )

    left_uid_key = traj_cols["user_id"]
    right_uid_key = temp_traj_cols["user_id"]
    left_loc_key = traj_cols["location_id"]
    right_loc_key = temp_traj_cols["location_id"]

    loader._has_time_cols(left.columns, traj_cols)
    if right_traj_cols is None:
        time_keys = ["datetime", "start_datetime", "timestamp", "start_timestamp"]
        if "timestamp" in kwargs or "start_timestamp" in kwargs:
            time_keys = ["timestamp", "start_timestamp", "datetime", "start_datetime"]
        if "datetime" in kwargs or "start_datetime" in kwargs:
            time_keys = ["datetime", "start_datetime", "timestamp", "start_timestamp"]

        right_has_time = any(
            temp_traj_cols[key] in right.columns
            for key in time_keys
        )
        if not right_has_time:
            raise ValueError(
                "Could not find required temporal columns in {}. The dataset must contain or map to "
                "at least one of 'datetime', 'timestamp', 'start_datetime', or 'start_timestamp'.".format(
                    list(right.columns)
                )
                + right_schema_hint
            )
    else:
        loader._has_time_cols(right.columns, temp_traj_cols)

    if left_loc_key not in left.columns:
        raise ValueError(
            "Could not find required location column in {}. The dataset must contain or map to "
            "a location column.".format(list(left.columns))
        )
    if right_loc_key not in right.columns:
        raise ValueError(
            "Could not find required location column in {}. The dataset must contain or map to "
            "a location column.{}".format(list(right.columns), right_schema_hint if right_traj_cols is None else "")
        )

    if left_uid_key in left.columns and not (left[left_uid_key].nunique() == 1):
        raise ValueError("Each visits dataframe must have at most one unique user_id")
    if right_uid_key in right.columns and not (right[right_uid_key].nunique() == 1):
        raise ValueError("Each visits dataframe must have at most one unique user_id")

    keep_uid = (left_uid_key in left.columns and right_uid_key in right.columns)
    if keep_uid:
        same_id = (left[left_uid_key].iloc[0] == right[right_uid_key].iloc[0])
        uid = left[left_uid_key].iloc[0]
    else:
        same_id = False

    left_t_name, left_use_datetime = loader._fallback_time_cols_dt(left.columns, input_traj_cols, kwargs)
    left_t_key = traj_cols[left_t_name]
    left_e_t_key = traj_cols["end_datetime" if left_use_datetime else "end_timestamp"]

    right_t_name, right_use_datetime = loader._fallback_time_cols_dt(right.columns, right_schema_input, right_kwargs)
    right_t_key = temp_traj_cols[right_t_name]
    right_e_t_key = temp_traj_cols["end_datetime" if right_use_datetime else "end_timestamp"]

    if not (loader._has_end_cols(left.columns, traj_cols) or loader._has_duration_cols(left.columns, traj_cols)):
        raise ValueError(
            "Missing required (end or duration) temporal columns for left visits dataframe in columns {}.".format(
                list(left.columns)
            )
        )
    if not (loader._has_end_cols(right.columns, temp_traj_cols) or loader._has_duration_cols(right.columns, temp_traj_cols)):
        raise ValueError(
            "Missing required (end or duration) temporal columns for right visits dataframe in columns {}.{}".format(
                list(right.columns),
                right_schema_hint if right_traj_cols is None else "",
            )
        )

    if left_e_t_key not in left.columns:
        if left_use_datetime:
            left[left_e_t_key] = left[left_t_key] + pd.to_timedelta(left[traj_cols["duration"]], unit="m")
        else:
            left[left_e_t_key] = left[left_t_key] + left[traj_cols["duration"]] * 60
    if right_e_t_key not in right.columns:
        if right_use_datetime:
            right[right_e_t_key] = right[right_t_key] + pd.to_timedelta(right[temp_traj_cols["duration"]], unit="m")
        else:
            right[right_e_t_key] = right[right_t_key] + right[temp_traj_cols["duration"]] * 60

    if left_use_datetime:
        left["temp_t_key"] = filters.to_timestamp(left[left_t_key])
        left["temp_e_t_key"] = filters.to_timestamp(left[left_e_t_key])
    else:
        left["temp_t_key"] = left[left_t_key]
        left["temp_e_t_key"] = left[left_e_t_key]

    if right_use_datetime:
        right["temp_t_key"] = filters.to_timestamp(right[right_t_key])
        right["temp_e_t_key"] = filters.to_timestamp(right[right_e_t_key])
    else:
        right["temp_t_key"] = right[right_t_key]
        right["temp_e_t_key"] = right[right_e_t_key]

    left_cols = [left_t_key, left_e_t_key, left_loc_key, "temp_t_key", "temp_e_t_key"]
    right_cols = [right_t_key, right_e_t_key, right_loc_key, "temp_t_key", "temp_e_t_key"]
    if keep_uid and not same_id:
        left_cols = [left_uid_key] + left_cols
        right_cols = [right_uid_key] + right_cols
    left.drop([col for col in left.columns if col not in left_cols], axis=1, inplace=True)
    right.drop([col for col in right.columns if col not in right_cols], axis=1, inplace=True)

    if match_location:
        if left_loc_key == right_loc_key:
            merged = left.merge(right, on=left_loc_key, suffixes=("_left", "_right"))
        else:
            merged = left.merge(right, left_on=left_loc_key, right_on=right_loc_key, suffixes=("_left", "_right"))
    else:
        merged = left.merge(right, how="cross", suffixes=("_left", "_right"))

    cond = (
        (merged["temp_t_key_left"] < merged["temp_e_t_key_right"])
        & (merged["temp_t_key_right"] < merged["temp_e_t_key_left"])
    )
    merged = merged.loc[cond]

    start_max = merged[["temp_t_key_left", "temp_t_key_right"]].max(axis=1)
    end_min = merged[["temp_e_t_key_left", "temp_e_t_key_right"]].min(axis=1)
    merged[traj_cols["duration"]] = ((end_min - start_max) // 60).astype(int)

    if keep_uid and same_id:
        merged[left_uid_key] = uid

    rename_cols = {}
    if left_t_key != right_t_key:
        rename_cols[left_t_key] = f"{left_t_key}_left"
        rename_cols[right_t_key] = f"{right_t_key}_right"
    if left_e_t_key != right_e_t_key:
        rename_cols[left_e_t_key] = f"{left_e_t_key}_left"
        rename_cols[right_e_t_key] = f"{right_e_t_key}_right"
    if left_loc_key != right_loc_key:
        rename_cols[left_loc_key] = f"{left_loc_key}_left"
        rename_cols[right_loc_key] = f"{right_loc_key}_right"
    if keep_uid and not same_id and left_uid_key != right_uid_key:
        rename_cols[left_uid_key] = f"{left_uid_key}_left"
        rename_cols[right_uid_key] = f"{right_uid_key}_right"
    merged.rename(rename_cols, axis=1, inplace=True)

    merged.drop(["temp_t_key_left", "temp_e_t_key_left", "temp_t_key_right", "temp_e_t_key_right"], axis=1, inplace=True)

    return merged.reset_index(drop=True)

def precision_recall_f1_from_minutes(total_pred, total_truth, tp):
    """Compute P/R/F1 from minute totals."""
    precision = tp / total_pred if total_pred else np.nan
    recall    = tp / total_truth if total_truth else np.nan
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif precision + recall == 0:            # both are 0 → F1 = 0
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}
