import pandas as pd
import nomad.io.base as loader
from nomad.stop_detection.utils import _fallback_time_cols
from datetime import datetime, time, timedelta


def nocturnal_stops(
    stops_table,
    dusk_hour=19,
    dawn_hour=6,
    start_datetime="start_datetime",
    end_datetime="end_datetime",
):
    """
    Slice each stop so that only its *night-time* portion—defined by a daily
    window from ``dusk_hour`` to ``dawn_hour``—is retained.

    The helper assumes that ``stops_table`` already contains fully parsed
    timezone-aware datetimes.  For every stop it constructs the set of
    candidate night windows it might intersect, clips the stop to those
    windows, recomputes the duration, and drops rows whose clipped duration
    is zero.

    Parameters
    ----------
    stops_table : pandas.DataFrame
        Output of a stop-detection algorithm with at least

        * a *start* column named ``start_datetime`` (default
          ``"start_datetime"``) and
        * an *end* column named ``end_datetime``   (default
          ``"end_datetime"``).

        Both must be ``datetime64[ns, tz]``.
    dusk_hour : int, default 19
        Local hour (0–23) that marks the beginning of the nocturnal window
        **on the same calendar day**.
    dawn_hour : int, default 6
        Local hour (0–23) that marks the end of the nocturnal window
        **on the following calendar day**.
    start_datetime, end_datetime : str, optional
        Column names if different from the defaults.

    Returns
    -------
    pandas.DataFrame
        Same columns as the input plus an updated ``duration`` (integer minutes)
        and *only* those rows whose clipped duration is positive.  Temporary
        helper columns are removed.

    Notes
    -----
    * A stop that spans several nights is exploded so that **one row per night**
      is produced.
    * The duration is integer-divided by 60 s → minutes.
    * Time-zone information is preserved.
    """

    df = stops_table.copy()

    # Build candidate night windows for every stop
    df["_night_start"] = df.apply(
        lambda r: [
            pd.Timestamp(datetime.combine(d, time(dusk_hour)), tz=r[start_datetime].tzinfo)
            for d in pd.date_range(
                (r[start_datetime] - timedelta(days=1)).date(),
                r[end_datetime].date(),
                freq="D",
            )
        ],
        axis=1,
    )

    df = df.explode("_night_start", ignore_index=True)
    df["_night_end"] = df["_night_start"] + timedelta(hours=(24 - dusk_hour + dawn_hour))

    # Clip the stop to the nightly interval
    df[start_datetime] = df[[start_datetime, "_night_start"]].max(axis=1)
    df[end_datetime] = df[[end_datetime, "_night_end"]].min(axis=1)

    df["duration"] = (
        (df[end_datetime] - df[start_datetime]).dt.total_seconds() // 60
    ).astype(int)

    return df[df["duration"] > 0].drop(columns=["_night_start", "_night_end"])


def compute_candidate_homes(
    stops_table,
    dusk_hour=19,
    dawn_hour=6,
    traj_cols=None,
    **kwargs,
):
    """
    Aggregate night-time stop statistics that serve as features for home-location
    inference.

    Internally this calls :func:`nocturnal_stops` to keep only night portions of
    the stops, then counts how many distinct nights and ISO-8601 calendar weeks
    each user spent at each location.

    Parameters
    ----------
    stops_table : pandas.DataFrame
        A stop table with at least

        * temporal columns (``start_*`` and either ``end_*`` **or**
          ``duration``),
        * one user identifier, and
        * one location identifier.

        Column names can be supplied/overridden via *traj_cols* or *kwargs*.
    dusk_hour, dawn_hour : int, optional
        Same semantics as in :func:`nocturnal_stops`.
    traj_cols : dict, optional
        Mapping from canonical names (``"user_id"``, ``"location_id"``,
        ``"start_timestamp"``/``"start_datetime"``, etc.) to the actual column
        names in *stops_table*.
    **kwargs
        Column-overrides written as ``<canonical>=<actual>``; passed through to
        the NOMAD I/O helpers.

    Returns
    -------
    pandas.DataFrame
        Columns:

        * ``user_id``        – user identifier  
        * ``location_id``    – candidate home location  
        * ``num_nights``     – unique nights present at the location  
        * ``num_weeks``      – unique ISO weeks present at the location  
        * ``total_duration`` – aggregated night-time minutes

    Raises
    ------
    ValueError
        If neither an *end* column nor a *duration* column is present.
    """

    stops = stops_table.copy()

    # Resolve column names
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)
    loader._has_time_cols(stops.columns, traj_cols)

    t_key, use_datetime = _fallback_time_cols(stops.columns, traj_cols, kwargs)
    end_t_key = "end_datetime" if use_datetime else "end_timestamp"

    # Ensure we can compute an end time
    end_col_present = loader._has_end_cols(stops.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stops.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        raise ValueError("stops_table must provide either an end time or a duration.")

    if not end_col_present:
        dur_col = traj_cols["duration"]
        if use_datetime:
            stops[end_t_key] = stops[traj_cols[t_key]] + pd.to_timedelta(stops[dur_col], unit="m")
        else:
            stops[end_t_key] = stops[traj_cols[t_key]] + stops[dur_col] * 60

    # Nocturnal clipping
    stops_night = nocturnal_stops(
        stops,
        dusk_hour=dusk_hour,
        dawn_hour=dawn_hour,
        start_datetime=traj_cols[t_key],
        end_datetime=end_t_key,
    )

    # Dates and ISO weeks (convert timestamps if needed)
    if use_datetime:
        dt = stops_night[traj_cols[t_key]]
    else:
        dt = pd.to_datetime(stops_night[traj_cols[t_key]], unit="s", utc=True)

    stops_night["_date"] = dt.dt.date
    stops_night["_iso_week"] = dt.dt.isocalendar().week

    out = (
        stops_night.groupby([traj_cols["user_id"], traj_cols["location_id"]], as_index=False)
        .agg(
            num_nights=("_date", "nunique"),
            num_weeks=("_iso_week", "nunique"),
            total_duration=(traj_cols["duration"], "sum"),
        )
    )

    return out

def select_home(
    candidate_homes,
    stops_table,
    min_days,
    min_weeks,
    traj_cols=None,
    **kwargs,
):
    """
    Pick **one** home location per user from pre-computed nightly aggregates.

    The function filters *candidate_homes* by minimum presence thresholds,
    ranks the surviving locations by

    1. number of nights (descending) and
    2. total night-time duration (descending),

    and finally returns the top-ranked location together with the date of the
    last stop observation.

    Parameters
    ----------
    candidate_homes : pandas.DataFrame
        Output of :func:`compute_candidate_homes`.
    stops_table : pandas.DataFrame
        Full stop table (not night-clipped) used only to determine the most
        recent observation date.
    min_days : int
        Minimum number of distinct nights required for a location to qualify
        as home.
    min_weeks : int
        Minimum number of distinct ISO weeks required for a location to qualify
        as home.
    traj_cols : dict, optional
        Column mapping overrides, as in other NOMAD helpers.
    **kwargs
        Additional ``<canonical>=<actual>`` column overrides.

    Returns
    -------
    pandas.DataFrame
        Columns ``['user_id', 'location_id', 'home_date']`` with exactly one
        row per user.  *home_date* equals the date (YYYY-MM-DD) of the most
        recent stop in *stops_table*.

    Notes
    -----
    * Ties beyond the ranking rules are broken by **first occurrence** in
      *candidate_homes* (i.e. deterministic given the input order).
    * Users with no location meeting the thresholds are omitted from the
      output.
    """

    traj_cols = loader._parse_traj_cols(candidate_homes.columns, traj_cols, kwargs)

    # Last observation date
    t_key, use_datetime = _fallback_time_cols(stops_table.columns, traj_cols, kwargs)
    dt_series = (
        stops_table[traj_cols[t_key]]
        if use_datetime
        else pd.to_datetime(stops_table[traj_cols[t_key]], unit="s", utc=True)
    )
    last_date = dt_series.dt.date.max()

    # Filter and rank
    filtered = (
        candidate_homes.loc[
            (candidate_homes["num_nights"] >= min_days)
            & (candidate_homes["num_weeks"] >= min_weeks)
        ]
        .sort_values(
            [traj_cols["user_id"], "num_nights", "total_duration"],
            ascending=[True, False, False],
        )
    )

    best = (
        filtered.drop_duplicates(traj_cols["user_id"], keep="first")
        .assign(home_date=last_date)
        .reset_index(drop=True)
    )

    return best[[traj_cols["user_id"], traj_cols["location_id"], "home_date"]]

def workday_stops(
    stops_table,
    work_start_hour=9,
    work_end_hour=17,
    include_weekends=False,
    start_datetime="start_datetime",
    end_datetime="end_datetime",
):
    """
    Clip each stop to the daily work-hour window ``work_start_hour`` →
    ``work_end_hour`` and return only the portions that fall on workdays.

    Stops that span several calendar days are *exploded* into one row per day
    (exactly like the night-time helper) so long multi-day artefacts do **not**
    crash the logic.

    Parameters
    ----------
    stops_table : pandas.DataFrame
        At minimum the two datetime columns given in *start_datetime* and
        *end_datetime* (timezone-aware).
    work_start_hour, work_end_hour : int, optional
        Local hours (0–23) delimiting the daily work block.  ``work_start_hour``
        must be strictly < ``work_end_hour``; otherwise a ``ValueError`` is
        raised.
    include_weekends : bool, default False
        If *False* (default) rows whose clipped interval falls entirely on
        Saturday or Sunday are dropped.
    start_datetime, end_datetime : str, optional
        Column names overriding the defaults.

    Returns
    -------
    pandas.DataFrame
        Same schema as the input plus an updated integer-minute ``duration`` and
        only rows whose clipped duration is positive.
    """
    if work_start_hour >= work_end_hour:
        raise ValueError("work_start_hour must be < work_end_hour within the same day.")

    df = stops_table.copy()

    # Build candidate day windows for every stop
    df["_work_start"] = df.apply(
        lambda r: [
            pd.Timestamp(
                datetime.combine(d, time(work_start_hour)),
                tz=r[start_datetime].tzinfo,
            )
            for d in pd.date_range(
                r[start_datetime].date(), r[end_datetime].date(), freq="D"
            )
        ],
        axis=1,
    )

    df = df.explode("_work_start", ignore_index=True)
    df["_work_end"] = df["_work_start"] + timedelta(hours=work_end_hour - work_start_hour)

    # Clip the stop to the workday interval
    df[start_datetime] = df[[start_datetime, "_work_start"]].max(axis=1)
    df[end_datetime] = df[[end_datetime, "_work_end"]].min(axis=1)

    df["duration"] = (
        (df[end_datetime] - df[start_datetime]).dt.total_seconds() // 60
    ).astype(int)

    # Weekend filter
    if not include_weekends:
        df = df[df[start_datetime].dt.dayofweek < 5]  # 0=Mon … 4=Fri

    return df[df["duration"] > 0].drop(columns=["_work_start", "_work_end"])


# ---------------------------------------------------------------------------
# 2.  Aggregate daily work-presence statistics
# ---------------------------------------------------------------------------
def compute_candidate_workplaces(
    stops_table,
    work_start_hour=9,
    work_end_hour=17,
    include_weekends=False,
    traj_cols=None,
    **kwargs,
):
    """
    Build per-location daytime presence features for workplace inference.

    The logic is identical to *compute_candidate_homes* but uses
    :pyfunc:`workday_stops` to keep only weekday work-hour intervals.

    Returns a table with the columns

    * ``user_id``
    * ``location_id``
    * ``num_work_days``   – distinct workdays present at the location
    * ``num_weeks``       – distinct ISO-weeks present
    * ``total_duration``  – total minutes inside the work window
    """
    stops = stops_table.copy()

    # Resolve column names
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)
    loader._has_time_cols(stops.columns, traj_cols)

    t_key, use_datetime = _fallback_time_cols(stops.columns, traj_cols, kwargs)
    end_t_key = "end_datetime" if use_datetime else "end_timestamp"

    # Ensure an end time exists
    end_col_present = loader._has_end_cols(stops.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stops.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        raise ValueError("stops_table must provide either an end time or a duration.")

    if not end_col_present:
        dur_col = traj_cols["duration"]
        if use_datetime:
            stops[end_t_key] = stops[traj_cols[t_key]] + pd.to_timedelta(stops[dur_col], unit="m")
        else:
            stops[end_t_key] = stops[traj_cols[t_key]] + stops[dur_col] * 60

    # Daytime clipping
    stops_work = workday_stops(
        stops,
        work_start_hour=work_start_hour,
        work_end_hour=work_end_hour,
        include_weekends=include_weekends,
        start_datetime=traj_cols[t_key],
        end_datetime=end_t_key,
    )

    # Dates and ISO weeks
    if use_datetime:
        dt = stops_work[traj_cols[t_key]]
    else:
        dt = pd.to_datetime(stops_work[traj_cols[t_key]], unit="s", utc=True)

    stops_work["_date"] = dt.dt.date
    stops_work["_iso_week"] = dt.dt.isocalendar().week

    out = (
        stops_work.groupby([traj_cols["user_id"], traj_cols["location_id"]], as_index=False)
        .agg(
            num_work_days=("_date", "nunique"),
            num_weeks=("_iso_week", "nunique"),
            total_duration=(traj_cols["duration"], "sum"),
        )
    )

    return out


# ---------------------------------------------------------------------------
# 3.  Select one workplace per user
# ---------------------------------------------------------------------------
def select_workplace(
    candidate_workplaces,
    stops_table,
    min_days,
    min_weeks,
    traj_cols=None,
    **kwargs,
):
    """
    Choose the single most plausible workplace for each user.

    Ranking is identical to *select_home*:

    1. highest ``num_work_days``
    2. highest ``total_duration``

    Parameters
    ----------
    candidate_workplaces : pandas.DataFrame
        Output of :pyfunc:`compute_candidate_workplaces`.
    stops_table : pandas.DataFrame
        Full stop table used only to derive the date of the last observation.
    min_days, min_weeks : int
        Presence thresholds.
    """
    traj_cols = loader._parse_traj_cols(candidate_workplaces.columns, traj_cols, kwargs)

    # Last observation date
    t_key, use_datetime = _fallback_time_cols(stops_table.columns, traj_cols, kwargs)
    dt_series = (
        stops_table[traj_cols[t_key]]
        if use_datetime
        else pd.to_datetime(stops_table[traj_cols[t_key]], unit="s", utc=True)
    )
    last_date = dt_series.dt.date.max()

    # Filter and rank
    filtered = (
        candidate_workplaces.loc[
            (candidate_workplaces["num_work_days"] >= min_days)
            & (candidate_workplaces["num_weeks"] >= min_weeks)
        ]
        .sort_values(
            [traj_cols["user_id"], "num_work_days", "total_duration"],
            ascending=[True, False, False],
        )
    )

    best = (
        filtered.drop_duplicates(traj_cols["user_id"], keep="first")
        .assign(workplace_date=last_date)
        .reset_index(drop=True)
    )

    return best[[traj_cols["user_id"], traj_cols["location_id"], "workplace_date"]]