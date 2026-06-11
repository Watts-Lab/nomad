"""Various algorithms for estimating individual co-location"""

# Authors: Thomas Li and Francisco Barreras

import nomad.filters as filters
import nomad.io.base as loader
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KDTree


_EARTH_RADIUS_M = 6_371_000
_STOP_ID = "_nomad_stop_id"
_ROW = "_nomad_row"
_TEMP_START = "_nomad_start"
_TEMP_END = "_nomad_end"


def _empty_contacts(mode):
    columns = [
        "user_id_1",
        "user_id_2",
        "contact_start",
        "contact_end",
        "overlap_duration",
        "stop_id_1",
        "stop_id_2",
    ]
    if mode == "exact":
        columns.append("location_id")
    if mode == "radius":
        columns.append("distance")
    return pd.DataFrame(columns=columns)


def _to_seconds(values, use_datetime):
    if use_datetime or pd.api.types.is_datetime64_any_dtype(values):
        seconds = filters.to_timestamp(values)
    else:
        seconds = pd.to_numeric(values, errors="raise")

    seconds = pd.Series(seconds, index=values.index).astype(float)
    if seconds.isna().any():
        raise ValueError("Temporal columns cannot contain missing values.")
    return seconds


def _prepare_contact_stops(stops, traj_cols, kwargs):
    input_traj_cols = traj_cols
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)

    uid_col = traj_cols["user_id"]
    if uid_col not in stops.columns:
        raise ValueError(f"Missing required user_id column '{uid_col}'.")

    t_name, use_datetime = loader._fallback_time_cols_dt(
        stops.columns, input_traj_cols, kwargs
    )
    start_col = traj_cols[t_name]
    end_col = traj_cols["end_datetime" if use_datetime else "end_timestamp"]
    duration_col = traj_cols["duration"]

    if end_col not in stops.columns and duration_col not in stops.columns:
        raise ValueError(
            "Missing required end time or duration column for contact estimation."
        )

    normalized = stops.copy()
    if normalized[uid_col].isna().any():
        raise ValueError("The user_id column cannot contain missing values.")

    normalized[_STOP_ID] = stops.index.to_numpy()
    normalized[_ROW] = np.arange(len(stops))
    normalized[_TEMP_START] = _to_seconds(normalized[start_col], use_datetime).to_numpy()

    if end_col in normalized.columns:
        normalized[_TEMP_END] = _to_seconds(normalized[end_col], use_datetime).to_numpy()
    else:
        duration = pd.to_numeric(normalized[duration_col], errors="raise")
        if duration.isna().any():
            raise ValueError("The duration column cannot contain missing values.")
        normalized[_TEMP_END] = normalized[_TEMP_START] + duration.to_numpy() * 60

    if (normalized[_TEMP_END] <= normalized[_TEMP_START]).any():
        raise ValueError("Stop end times must be after start times.")

    return normalized, traj_cols, uid_col, use_datetime, input_traj_cols


def _contact_time(seconds, use_datetime):
    if use_datetime:
        return pd.to_datetime(seconds, unit="s")

    seconds = np.asarray(seconds)
    if np.all(np.isfinite(seconds)) and np.all(seconds == np.floor(seconds)):
        return seconds.astype(int)
    return seconds


def _make_contacts(normalized, row, col, uid_col, use_datetime, mode, values=None):
    if len(row) == 0:
        return _empty_contacts(mode)

    users = normalized[uid_col].to_numpy()
    start = normalized[_TEMP_START].to_numpy()
    end = normalized[_TEMP_END].to_numpy()
    stop_id = normalized[_STOP_ID].to_numpy()
    contact_start = np.maximum(start[row], start[col])
    contact_end = np.minimum(end[row], end[col])

    contacts = pd.DataFrame(
        {
            "user_id_1": users[row],
            "user_id_2": users[col],
            "contact_start": _contact_time(contact_start, use_datetime),
            "contact_end": _contact_time(contact_end, use_datetime),
            "overlap_duration": ((contact_end - contact_start) // 60).astype(int),
            "stop_id_1": stop_id[row],
            "stop_id_2": stop_id[col],
        }
    )
    if mode == "exact":
        contacts["location_id"] = values
    if mode == "radius":
        contacts["distance"] = values

    return contacts.reset_index(drop=True)


def _radius_candidates(normalized, traj_cols, input_traj_cols, kwargs, distance_threshold):
    coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(
        normalized.columns,
        input_traj_cols,
        kwargs,
    )

    if use_lon_lat:
        coord_cols = [traj_cols["latitude"], traj_cols["longitude"]]
        coords = normalized[coord_cols].astype(float)
    else:
        coord_cols = [traj_cols[coord_key1], traj_cols[coord_key2]]
        coords = normalized[coord_cols].astype(float)

    if coords.isna().any().any():
        raise ValueError("Spatial coordinate columns cannot contain missing values.")

    if use_lon_lat:
        radius = distance_threshold / _EARTH_RADIUS_M
        query_coords = np.radians(coords.to_numpy())
        tree = BallTree(query_coords, metric="haversine")
    else:
        radius = distance_threshold
        query_coords = coords.to_numpy()
        tree = KDTree(query_coords)

    indices, distances = tree.query_radius(
        query_coords,
        r=radius,
        return_distance=True,
        sort_results=True,
    )
    counts = np.array([len(idx) for idx in indices])
    if counts.sum() == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)

    row = np.repeat(np.arange(len(normalized)), counts)
    col = np.concatenate(indices).astype(int)
    distance = np.concatenate(distances).astype(float)
    if use_lon_lat:
        distance = distance * _EARTH_RADIUS_M

    keep = row < col
    return row[keep], col[keep], distance[keep]


def estimate_contacts(stops, distance_threshold=None, traj_cols=None, **kwargs):
    """
    Estimate undirected co-location/contact events from a stop table.

    With ``distance_threshold=None``, contacts are stops from different users at
    the same ``location_id`` with strictly overlapping times. With a distance
    threshold, contacts are stops from different users within that radius and
    with strictly overlapping times. Latitude/longitude thresholds are meters;
    projected x/y thresholds use the coordinate units.
    """
    if len(stops) == 0:
        return _empty_contacts(
            "exact" if distance_threshold is None else "radius"
        )

    if distance_threshold is not None and distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative.")

    normalized, traj_cols, uid_col, use_datetime, input_traj_cols = _prepare_contact_stops(
        stops,
        traj_cols,
        kwargs,
    )

    if distance_threshold is None:
        loc_col = traj_cols["location_id"]
        if loc_col not in normalized.columns:
            raise ValueError(
                "Exact-location contact estimation requires a location_id column."
            )

        data = normalized.dropna(subset=[loc_col])
        cols = [uid_col, loc_col, _ROW, _TEMP_START, _TEMP_END]
        pairs = data[cols].merge(data[cols], on=loc_col, suffixes=("_1", "_2"))
        keep = (
            (pairs[f"{_ROW}_1"] < pairs[f"{_ROW}_2"])
            & (pairs[f"{uid_col}_1"] != pairs[f"{uid_col}_2"])
            & (pairs[f"{_TEMP_START}_1"] < pairs[f"{_TEMP_END}_2"])
            & (pairs[f"{_TEMP_START}_2"] < pairs[f"{_TEMP_END}_1"])
        )
        pairs = pairs.loc[keep]
        return _make_contacts(
            normalized,
            pairs[f"{_ROW}_1"].to_numpy(),
            pairs[f"{_ROW}_2"].to_numpy(),
            uid_col,
            use_datetime,
            "exact",
            values=pairs[loc_col].to_numpy(),
        )

    row, col, distance = _radius_candidates(
        normalized,
        traj_cols,
        input_traj_cols,
        kwargs,
        distance_threshold,
    )
    users = normalized[uid_col].to_numpy()
    start = normalized[_TEMP_START].to_numpy()
    end = normalized[_TEMP_END].to_numpy()
    keep = (
        (users[row] != users[col])
        & (start[row] < end[col])
        & (start[col] < end[row])
    )
    return _make_contacts(
        normalized,
        row[keep],
        col[keep],
        uid_col,
        use_datetime,
        "radius",
        values=distance[keep],
    )


def compute_contact_weights(
    contacts,
    method="duration",
    distance_threshold=None,
    overlap_duration_col="overlap_duration",
    distance_col="distance",
):
    """
    Add a contact_weight column to a contact event table.

    Supported methods are ``"duration"`` and ``"linear_distance"``. Linear
    distance weighting uses ``overlap_duration * max(0, 1 - distance / threshold)``.
    """
    if overlap_duration_col not in contacts.columns:
        raise ValueError(
            f"Missing required overlap duration column '{overlap_duration_col}'."
        )

    method = method.lower()
    weighted = contacts.copy()

    if method == "duration":
        weighted["contact_weight"] = weighted[overlap_duration_col]
        return weighted

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
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive for linear_distance.")

    distance_factor = np.maximum(
        0,
        1 - weighted[distance_col] / distance_threshold,
    )
    weighted["contact_weight"] = weighted[overlap_duration_col] * distance_factor
    return weighted


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
