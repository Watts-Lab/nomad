import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

import geopandas as gpd
import pyproj
from shapely.geometry import Polygon, Point
from shapely import wkt
import warnings

import nomad.io.base as loader
from nomad.constants import DEFAULT_SCHEMA, SEC_PER_UNIT
import pdb


def _timestamp_handling(
    ts,
    output_type,
    timezone=None
):
    """
    Convert timestamp to either pandas Timestamp or UNIX timestamp, with optional timezone handling.

    Parameters
    ----------
    ts : str, int, float, pd.Timestamp, or np.datetime64
        The input timestamp to be converted.
    output_type : str
        Desired output type: "pd.timestamp" or "unix".
    timezone : str, optional
        Timezone to localize or convert the timestamp to. If None, no timezone conversion is applied.

    Returns
    -------
    pd.Timestamp or int
    Converted timestamp in the desired format.

    Notes
    ------
    Using tz_localize is intentional. It will raise a TypeError
    if the timestamp is already timezone-aware, which correctly signals
    a caller error (e.g., providing a timezone for data that already has one).
    """
    if ts is None: 
        return None
        
    if isinstance(ts, str):
        ts = pd.to_datetime(ts, errors="coerce")
    elif isinstance(ts, (pd.Timestamp, np.datetime64)):
        ts = pd.to_datetime(ts)
    elif isinstance(ts, (int, np.integer, float, np.floating)):
        ts = pd.to_datetime(ts, unit='s', errors="coerce")
    else:
        raise TypeError("Unsupported input type for timestamp conversion.")

    if timezone:
        ts = ts.tz_localize(timezone)

    if output_type == "pd.timestamp":
        return ts
    elif output_type == "unix":
        return int(ts.timestamp())
    else:
        raise ValueError("Invalid ts_output value. Use 'pd.timestamp' or 'unix'.")



def to_timestamp(datetime, tz_offset=None):
    """
    Convert a datetime series into UNIX timestamps (seconds).
    
    Parameters
    ----------
    datetime : pd.Series
    tz_offset : pd.Series, optional

    Returns
    -------
    pd.Series
        UNIX timestamps as int64 values (seconds since epoch).
    """
    # Validate input type
    if not (
        
        pd.api.types.is_datetime64_any_dtype(datetime) or
        pd.api.types.is_string_dtype(datetime) or
        (pd.api.types.is_object_dtype(datetime) and loader._is_series_of_timestamps(datetime))
    ):
        raise TypeError(
            f"Input must be of type datetime64, string, or an array of Timestamp objects, "
            f"but it is of type {datetime.dtype}."
        )
    
    if tz_offset is not None:
        if not is_integer_dtype(tz_offset):
            tz_offset = tz_offset.astype('int64')

    if isinstance(datetime.dtype, pd.DatetimeTZDtype):
        return datetime.astype("int64") // 10**9
    
    # datetime without timezone
    elif pd.api.types.is_datetime64_dtype(datetime):
        if tz_offset is not None:
            return datetime.astype("int64") // 10**9 - tz_offset
        warnings.warn(
                f"The input is timezone-naive. UTC will be assumed."
                "Consider localizing to a timezone or passing a timezone offset column.")
        return datetime.astype("int64") // 10**9
    
    # datetime as string
    elif pd.api.types.is_string_dtype(datetime):
        result = pd.to_datetime(datetime, errors="coerce", utc=True)

        # contains timezone e.g. '2024-01-01 12:29:00-02:00'
        if datetime.str.contains(r'(?:Z|[+\-]\d{2}:\d{2})$', regex=True, na=False).any():
            return result.astype('int64') // 10**9
        else:
            # naive e.g. "2024-01-01 12:29:00"
            if tz_offset is not None and not tz_offset.empty:
                return result.astype('int64') // 10**9 - tz_offset
            else:
                warnings.warn(
                    f"The input is timezone-naive. UTC will be assumed."
                    "Consider localizing to a timezone or passing a timezone offset column.")
                return result.astype('int64') // 10**9
                
    # datetime is a series of pandas.Timestamp object. Always has unix timestamp in value
    else:
        f = np.frompyfunc(lambda x: x.timestamp(), 1, 1)
        return pd.Series(f(datetime).astype("int64"), index=datetime.index)

def _dup_per_freq_mask(sec, periods=1, freq='min', keep='first'): 
        
    bins = sec // (periods * SEC_PER_UNIT[freq])
    if isinstance(sec, pd.Series):
        return ~pd.Series(bins, index=sec.index).duplicated(keep=keep)
    return ~pd.Series(bins).duplicated(keep=keep).to_numpy()

def _fmt_from_freq(f):
    return {"s": "%Y-%m-%d %H:%M:%S",
            "min": "%Y-%m-%d %H:%M",
            "h": "%Y-%m-%d %H:00",
            "d": "%Y-%m-%d",
            "w": "%Y-%m-%d"}.get(f.lower(), "%Y-%m-%d %H:%M:%S")
    
def downsample(df,
               periods=1,
               freq='min',
               keep='first',
               traj_cols=None,
               verbose=False,
               **kwargs):
    """
    Down-sample *df* so that each user contributes at most one row
    in every consecutive ``periods × freq`` window.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    periods : int, default 1
        Size of the window expressed in multiples of *freq*; must be ≥ 1.
    freq : {'s', 'min', 'h', 'd', 'w'}, default 'min'
        Unit of the window: second, minute, hour, day, or week
        (lower-case aliases).
    keep : {'first', 'last', False}, default 'first'
        Which duplicate inside each window to retain, matching
        ``pandas.Series.duplicated`` semantics.
    traj_cols : dict, optional
        Mapping from the standard keys `'timestamp'`, `'datetime'`,
        `'user_id'`, and `'tz_offset'` to the actual column names in *df*.
        Any key may be absent if the corresponding column is not present.
    verbose : bool, default False
        When True, prints the fraction of rows removed and the window size.
    **kwargs
        Shorthand overrides for entries in *traj_cols* 

    Returns
    -------
    pandas.DataFrame
        A view of *df* containing the surviving rows.

    Raises
    ------
    ValueError
        If *periods* is not a positive integer or *freq* is invalid.
    KeyError
        If no suitable time column is found after parsing *traj_cols*.
    """
    if not isinstance(periods, (int, np.integer)) or periods < 1:
        raise ValueError("periods must be an integer ≥ 1")
    freq = freq.lower()
    if freq not in SEC_PER_UNIT:
        raise ValueError("freq must be one of 's', 'min', 'h', 'd', 'w'")

    t_key, use_dt = loader._fallback_time_cols_dt(df.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(df.columns, traj_cols, kwargs)
    loader._has_time_cols(df.columns, traj_cols)

    uid = traj_cols['user_id']
    multi = uid in df.columns and df[uid].nunique() > 1

    if use_dt:
        window = f"{periods}{freq}"
        if multi:
            mask = df.groupby(uid)[traj_cols[t_key]].transform(
                lambda s: ~s.dt.floor(window).duplicated(keep=keep))
        else:
            mask = ~df[traj_cols[t_key]].dt.floor(window).duplicated(keep=keep)
    else:
        sec = df[traj_cols[t_key]]
        if traj_cols['tz_offset'] in df.columns:
            sec = sec + df[traj_cols['tz_offset']]
        if multi:
            mask = sec.groupby(df[uid]).transform(
                lambda s: _dup_per_freq_mask(s, periods, freq, keep))
        else:
            mask = _dup_per_freq_mask(sec, periods, freq, keep)

    if verbose:
        pct = 100 * (1 - mask.sum() / len(mask))
        print(f"{pct:.3f}% of rows removed by downsampling to {periods}{freq} windows per user.")

    return df[mask]

def _to_projection(
    traj,
    input_crs,
    output_crs,
    input_x_col,
    input_y_col
):
    """
    Helper function to project latitude/longitude columns to a new CRS.
    """
    gdf = gpd.GeoSeries(gpd.points_from_xy(traj[input_x_col], traj[input_y_col]), crs=input_crs)
    projected = gdf.to_crs(output_crs)
    return projected.x, projected.y

def _filtered_users(
    traj,
    start_time,
    end_time,
    polygon,
    min_active_days,
    min_pings_per_day,
    traj_cols,
    input_x,
    input_y,
    time_col,
    crs
):
    """
    Helper function that returns a series containing users who have at least 
    k distinct days with at least m pings in the polygon within the timeframe T0 to T1.
    """
    # Filter by time range (this logic would not necessarily remove pings outside the timeframe.
    # Rather, it use pings inside the timeframe to determine whether a user is sufficient "complete".)
    # traj_filtered = traj[(traj[time_col] >= start_time) & (traj[time_col] <= end_time)].copy()
    # traj_filtered[time_col] = pd.to_datetime(traj_filtered[time_col])
    traj_filtered = traj.copy()

    if traj_filtered.empty:
        return pd.Series()

    # Filter points inside the polygon
    if polygon is not None:
        traj_filtered = _in_geo(traj_filtered, input_x, input_y, polygon, crs)
    else:
        traj_filtered['in_geo'] = True
    
    if traj_cols['tz_offset'] not in traj_filtered.columns:
        traj_filtered[traj_cols['tz_offset']] = 0
        warnings.warn(
            f"The trajectory dataframe does not have a tz_offset (timezone offset) column."
            "UTC (tz_offset=0) will be assumed.")

    if traj_cols['datetime'] not in traj_filtered.columns:
        traj_filtered[traj_cols['datetime']] = loader.naive_datetime_from_unix_and_offset(
            traj_filtered[traj_cols[time_col]], traj_filtered[traj_cols['tz_offset']]
        )

    traj_filtered['date'] = pd.to_datetime(traj_filtered[traj_cols['datetime']].dt.date)

    # Count pings per user per date inside the polygon
    daily_ping_counts = (
        traj_filtered[traj_filtered['in_geo']]
        .groupby([traj_cols['user_id'], 'date'])
        .size()
        .reset_index(name='ping_count')
    )

    # Filter users who have at least `m` (`min_pings_per_day`) pings on a given day
    users_with_m_pings = daily_ping_counts[daily_ping_counts['ping_count'] >= min_pings_per_day]

    # Count distinct days per user that satisfy the `m` pings condition
    users_with_k_days = (
        users_with_m_pings
        .groupby(traj_cols['user_id'])['date']
        .nunique()
        .reset_index(name='days_in_polygon')
    )

    # Select users who have at least `k` (`min_active_days`) such days
    filtered_users = users_with_k_days[users_with_k_days['days_in_polygon'] >= min_active_days][traj_cols['user_id']]

    return filtered_users

def q_filter(df: pd.DataFrame,
             qbar: float,
             traj_cols: dict = None,
             user_id: str = DEFAULT_SCHEMA["user_id"],
             timestamp: str = DEFAULT_SCHEMA["timestamp"]):
    """
    Computes the q statistic for each user as the proportion of unique hours with pings 
    over the total observed hours (last hour - first hour) and filters users where q > qbar.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with user_id and timestamp columns.
    qbar : float
        The threshold q value; users with q > qbar will be retained.
    traj_cols : dict, optional
        Dictionary containing column mappings, 
        e.g., {"user_id": "user_id", "timestamp": "timestamp"}.
    user_id : str, optional
        Name of the user_id column (default is "user_id").
    timestamp : str, optional
        Name of the timestamp column (default is "timestamp").

    Returns
    -------
    pd.Series
        A Series containing the user IDs for users whose q_stat > qbar.
    """
    user_col = traj_cols.get("user_id", user_id) if traj_cols else user_id
    datetime_col = traj_cols.get("timestamp", timestamp) if traj_cols else timestamp

    user_q_stats = df.groupby(user_col).apply(
        lambda group: _compute_q_stat(group, datetime_col)
    ).reset_index(name='q_stat')

    # Filter users where q > qbar
    filtered_users = user_q_stats[user_q_stats['q_stat'] > qbar][user_col]

    return filtered_users


def within(
    df,
    within,
    poly_crs=None,
    data_crs=None,
    traj_cols=None,
    **kwargs
):
    """
    Return a filtered DataFrame containing only rows within the given polygon.

    All arguments are passed to is_within.
    """
    mask = is_within(
        df,
        within,
        traj_cols=traj_cols,
        poly_crs=poly_crs,
        data_crs=data_crs,
        **kwargs
    )
    return df.loc[mask]

def is_within(
    df,
    within,
    poly_crs=None,
    data_crs=None,
    traj_cols=None,
    **kwargs,
):
    """
    Filter a DataFrame to include only points within the given polygon.

    Parameters
    ----------
    df : pd.DataFrame or GeoDataFrame
        Trajectory data.
    within : shapely Polygon/MultiPolygon or WKT string
        Polygon defining the spatial filter.
    traj_cols : dict, optional
        Mapping of logical trajectory column names to actual columns.
    poly_crs : CRS or str, optional
        CRS of the polygon.
    data_crs : CRS or str, optional
        CRS of the DataFrame coordinates.
    **kwargs
        Additional parameters for trajectory columns resolution.

    Returns
    -------
    pd.Series
        Boolean mask for which points are in the polygon within
    """
    if within is None:
        raise ValueError("A polygon or WKT string must be provided in 'within'.")

    # Normalize polygon geometry
    if isinstance(within, str):
        poly = wkt.loads(within)
    elif isinstance(within, Polygon):
        poly = within
    elif isinstance(within, gpd.GeoSeries):
        poly = within.unary_union
    elif isinstance(within, gpd.GeoDataFrame):
        poly = within.geometry.unary_union
    else:
        raise TypeError("within must be WKT, shapely Polygon, GeoSeries or GeoDataFrame.")

    # Determine coordinate columns
    coord_key1, coord_key2, use_lat_lon = loader._fallback_spatial_cols(df.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(df.columns, traj_cols, kwargs)

    # Handle CRS defaults
    if data_crs is None:
        if use_lat_lon:
            data_crs = "EPSG:4326"
            warnings.warn("data_crs not provided; assuming EPSG:4326 for lon/lat.")
        else:
            raise ValueError(
                "data_crs must be supplied for projected x/y columns or provide lat/lon instead."
            )
    data_crs = pyproj.CRS(data_crs)

    poly_crs_final = getattr(within, "crs", None) or poly_crs

    if poly_crs_final is not None:
        src_crs = pyproj.CRS(poly_crs_final)
        if not src_crs.equals(data_crs):
            poly = gpd.GeoSeries([poly], crs=src_crs).to_crs(data_crs).iloc[0]
    else:
        warnings.warn("Polygon CRS unspecified; assuming it matches data_crs.")

    # Construct GeoSeries of points from df coordinates
    pts = gpd.GeoSeries(gpd.points_from_xy(df[coord_key1], df[coord_key2]), crs=data_crs)
    
    return pts.within(poly).set_axis(df.index)


def coverage_matrix(data,
                    periods=1,
                    freq="h",
                    start=None,
                    end=None,
                    offset_col=0,
                    relative=False,
                    str_from_time=False,
                    traj_cols=None,
                    **kwargs):
    """
    Matrix of 0/1 flags; rows=user (or the single Series), columns=bucket start.
    """
    if isinstance(data, pd.Series):
        return _q_series(data, periods, freq, start, end, offset_col=offset_col)
        
    if isinstance(data, pd.Series):
        hits = _q_series(data, periods, freq, start, end, offset_col=offset_col).astype(int)
        if str_from_time:
            if is_integer_dtype(hits.index.dtype):
                hits.index = pd.to_datetime(hits.index, unit='s')
            hits.index = hits.index.strftime(_fmt_from_freq(freq))
        return hits

    df = data
    t_key, _ = loader._fallback_time_cols_dt(df.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(df.columns, traj_cols, kwargs)
    loader._has_time_cols(df.columns, traj_cols)

    uid  = traj_cols["user_id"]
    ts   = traj_cols[t_key]
    off_name = traj_cols["tz_offset"]
    if off_name in df.columns:
        offset_col = df[off_name]

    if (start is not None) or (end is not None):
        relative = False
    if not relative and (start is None or end is None):
        start = start or df[ts].min()
        end   = end   or df[ts].max()

    hit_map = {}
    for user, grp in df.groupby(uid, sort=False):
        off = offset_col.loc[grp.index] if isinstance(offset_col, pd.Series) else offset_col
        s = None if relative else start
        e = None if relative else end
        hit_map[user] = _q_series(grp[ts], periods, freq, s, e, offset_col=off)
        
    if not hit_map:                     # empty dataset edge-case
        return pd.DataFrame(dtype=int)

    hit_df = pd.concat(hit_map, axis=1).T.astype(int)   # rows=user
    
    if str_from_time:
        if is_integer_dtype(hit_df.columns.dtype):
            hit_df.columns = pd.to_datetime(hit_df.columns, unit='s')
        hit_df.columns = hit_df.columns.strftime(_fmt_from_freq(freq))
    return hit_df


def completeness(data,
                 periods=1,
                 freq="h",
                 *,
                 start=None,
                 end=None,
                 offset_col=0,
                 relative=False,
                 str_from_time=True,
                 agg_freq=None,
                 traj_cols=None,
                 **kwargs):
    """
    Measure trajectory completeness as the fraction of expected time intervals
    ('buckets') containing at least one observation.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Trajectory data containing timestamps, either as:
        - A pandas Series of Unix-second integers or datetime64 values.
        - A DataFrame, from which timestamp and user columns are identified
          via `traj_cols` or default column naming conventions.
    periods : int, default 1
        Number of units of `freq` per bucket (must be ≥ 1). For example,
        `periods=3, freq='h'` results in 3-hour buckets.
    freq : {'s', 'min', 'h', 'd', 'w'}, default 'h'
        Time resolution used to define buckets: seconds ('s'), minutes ('min'),
        hours ('h'), days ('d'), or weeks ('w').
    start, end : scalar, optional
        Explicit time bounds to define the bucket range. If either is omitted,
        the range is inferred from the data. Ignored if `relative=True`.
    relative : bool, default False
        If False, completeness is measured within a common time span shared
        by all users. If True, each user's completeness is computed only within
        their own individual time span (from their first to their last record).
    offset_col : pandas.Series or int, default 0
        Offset in seconds to apply to timestamps (useful for handling time zones).
        If a `tz_offset` column is present in the data and indicated via
        `traj_cols` or `kwargs`, this argument is ignored.
    agg_freq : str, optional
        Aggregation frequency (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly).
        If specified, returns completeness aggregated at this frequency instead
        of overall completeness.
    traj_cols : dict, optional
        Mapping from standard keys ('timestamp', 'datetime', 'user_id',
        'tz_offset') to column names in `data`. If omitted, defaults are used.
    **kwargs
        Shorthand overrides for entries in `traj_cols`.

    Returns
    -------
    float or pandas.Series or pandas.DataFrame
        - If input is a single Series and `agg_freq=None`, returns a single float.
        - If input is a DataFrame and `agg_freq=None`, returns a Series indexed by user_id.
        - If `agg_freq` is specified, returns completeness aggregated by the specified
          frequency, either as a Series (single user) or DataFrame (rows per user,
          columns per aggregation bucket).
    """    
    hits = coverage_matrix(
        data, periods, freq,
        start=start, end=end,
        offset_col=offset_col,
        relative=relative,
        traj_cols=traj_cols,
        **kwargs
    )

    if agg_freq is None:
        return hits.mean(axis=1) if isinstance(hits, pd.DataFrame) else hits.mean()

    if isinstance(hits.columns, pd.DatetimeIndex):
        buckets = hits.columns.floor(agg_freq)
    else:  # unix seconds (integers)
        agg_step = SEC_PER_UNIT[agg_freq.lower()]
        buckets = (hits.columns // agg_step) * agg_step

    # Single series input
    if isinstance(hits, pd.Series):
        return hits.groupby(buckets).mean()

    # DataFrame input
    hits.columns = buckets
    hit_df = hits.T.groupby(hits.columns).mean().T

    if str_from_time:
        if is_integer_dtype(hit_df.columns.dtype):
            hit_df.columns = pd.to_datetime(hit_df.columns, unit='s')
        hit_df.columns = hit_df.columns.strftime(_fmt_from_freq(agg_freq))
    return hit_df

def _q_series(time_col, periods, freq, start=None, end=None, offset_col=0):
    """Return the per-bucket Boolean hits array for a single Series of timestamps."""
    if is_integer_dtype(time_col):                  # unix seconds path
        if not time_col.is_monotonic_increasing:
            raise ValueError("time_col must be sorted in ascending order.")        
        sec   = time_col + offset_col                                  # offset may be scalar or vector
        s_min = _timestamp_handling(start, "unix") or int(sec.min())
        s_max = _timestamp_handling(end,   "unix") or int(sec.max())
        return _q_array(sec, s_min, s_max, periods, freq)

    # datetime64 path (tz-aware respected by .floor)
    tz     = getattr(time_col.dt, "tz", None)
    start  = _timestamp_handling(start, "pd.timestamp", tz) or time_col.min()
    end    = _timestamp_handling(end,   "pd.timestamp", tz) or time_col.max()
    window = f"{periods}{freq}"
    bucket_start = pd.date_range(start.floor(window), end, freq=window, inclusive="left")
    hits = pd.Series(False, index=bucket_start)
    active = time_col[(time_col >= start) & (time_col < end)].dt.floor(window).unique()
    hits.loc[active] = True
    return hits

def _q_array(sec, start_timestamp, end_timestamp, periods=1, freq='h'):
    """True/False array: one flag per (periods×freq) bucket."""
    step = periods * SEC_PER_UNIT[freq]
    first = (start_timestamp // step) * step
    bin_starts = np.arange(first, end_timestamp, step)

    pos  = np.searchsorted(sec, bin_starts, side='left')
    all_pos = np.append(pos, len(sec))
    # np.diff computes `pos[i+1] - pos[i]`. If > 0, the bucket had data.
    hits = np.diff(all_pos) > 0
    return pd.Series(hits, index=bin_starts) if isinstance(sec, pd.Series) else hits