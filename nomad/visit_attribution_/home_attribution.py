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
    """Clip each stop to the nocturnal window between *dusk_hour* and *dawn_hour*.

    This helper assumes the caller already provides proper datetime columns. It
    merely slices the stop to the relevant night portion and recomputes the
    duration, dropping rows that do not intersect the night at all.
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
    """Aggregate nightly presence statistics for home inference.

    Column names are resolved through *traj_cols* or keyword overrides and no
    type coercion beyond what is strictly necessary for the calculation is
    performed.
    """

    stops = stops_table.copy()

    # Resolve column names
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)
    loader._has_time_cols(stops.columns, traj_cols)

    t_key, use_datetime = _fallback_time_cols(stops.columns, traj_cols, kwargs)
    if not use_datetime:
        raise ValueError(
            "zoned datetime column is required for home detection, "
            "pass a datetime-like column in traj_cols or as a keyword argument"
            )
    
    end_t_key = "end_datetime"

    # Ensure we can compute an end time
    end_col_present = loader._has_end_cols(stops.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stops.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        raise ValueError("stops_table must provide either an end time or a duration.")

    if not end_col_present:
        dur_col = traj_cols["duration"]
        stops[end_t_key] = stops[traj_cols[t_key]] + pd.to_timedelta(stops[dur_col], unit="m")
        
    # Nocturnal clipping
    stops_night = nocturnal_stops(
        stops,
        dusk_hour=dusk_hour,
        dawn_hour=dawn_hour,
        start_datetime=traj_cols[t_key],
        end_datetime=end_t_key,
    )

    # Dates and ISO weeks (convert timestamps if needed)
    dt = stops_night[traj_cols[t_key]]


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
    min_days,
    min_weeks,
    last_date=None,
    stops_table=None,
    traj_cols=None,
    **kwargs,
):
    """Select a single home location per user."""

    traj_cols = loader._parse_traj_cols(candidate_homes.columns, traj_cols, kwargs)

    # Last observation date
    if not last_date and stops_table:
        t_key, use_datetime = _fallback_time_cols(stops_table.columns, traj_cols, kwargs)
        dt_series = (
            stops_table[traj_cols[t_key]]
            if use_datetime
            else pd.to_datetime(stops_table[traj_cols[t_key]], unit="s", utc=True)
        )
        last_date = dt_series.dt.date.max()
    elif last_date:
        pass
    else:
        raise ValueError(
            "One of last_date or stops_table must be passed."
            )

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
