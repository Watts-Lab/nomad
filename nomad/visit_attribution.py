
import geopandas as gpd
import nomad.io.base as loader
from shapely.geometry import Point
import warnings
import pandas as pd
import nomad.io.base as loader
import numpy as np
from datetime import date, time, datetime, timedelta
from dateutil.parser import parse

def point_in_polygon(traj, labels, stop_table, poi_table, traj_cols, is_datetime, is_long_lat):
    # If either labels or stop_table is empty, there's nothing to do
    if labels.empty or stop_table.empty:
        stop_table['location'] = pd.Series(dtype='object')
        return stop_table

    # merge labels with data
    traj_with_labels = traj.copy()
    time_col = traj_cols['datetime'] if is_datetime else traj_cols['timestamp']
    traj_with_labels = traj_with_labels.merge(labels, left_on=time_col, right_index=True, how='left')

    # compute the location for each cluster
    space_cols = [traj_cols['longitude'], traj_cols['latitude']] if is_long_lat else [traj_cols['x'], traj_cols['y']]
    pings_df = traj_with_labels.groupby('cluster')[space_cols].mean()
    
    locations = poi_map(traj=pings_df,
                        poi_table=poi_table,
                        traj_cols=traj_cols)

    # Map the mode location back to the stop_table
    stop_table['location'] = locations

    return stop_table
    
def majority_poi(traj, labels, stop_table, poi_table, traj_cols, is_datetime, is_long_lat):
    # If either labels or stop_table is empty, there's nothing to do
    if labels.empty or stop_table.empty:
        stop_table['location'] = pd.Series(dtype='object')
        return stop_table
    
    # merge labels with data
    traj_with_labels = traj.copy()
    time_col = traj_cols['datetime'] if is_datetime else traj_cols['timestamp']
    traj_with_labels = traj_with_labels.merge(labels, left_on=time_col, right_index=True, how='left')

    # compute the location for each cluster
    space_cols = [traj_cols['longitude'], traj_cols['latitude']] if is_long_lat else [traj_cols['x'], traj_cols['y']]
    pings_df = traj_with_labels[space_cols].copy()
    
    traj_with_labels["building_id"] = poi_map(traj=pings_df,
                                              poi_table=poi_table,
                                              traj_cols=traj_cols)

    locations = traj_with_labels.groupby('cluster')['building_id'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    # Map the mode location back to the stop_table
    stop_table['location'] = locations

    return stop_table

    
def poi_map(traj, poi_table, traj_cols=None, max_distance=4, **kwargs):
    """
    Map elements in traj to closest polygon in poi_table with an allowed distance buffer.

    Parameters
    ----------
    traj : pd.DataFrame
        The trajectory DataFrame containing x and y coordinates.
    poi_table : gpd.GeoDataFrame
        The POI table containing building geometries and IDs.
    traj_cols : list
        The columns in the trajectory DataFrame to be used for mapping.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pd.Series
        A Series containing the building IDs corresponding to the pings in the trajectory.
    """
    # TO DO: warning if CRS is None. If is_long_lat inform that EPSG:4326 will be used.
    # TO DO: if not is_long_lat and CRS is None, Raise Error and inform of CRS of poi_table. 
    # TO DO: if POI_table has no CRS Raise Error. If poi_table has different CRS? ValueError suggest reprojection

    # Check if user wants long and lat
    is_long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    # Set initial schema
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)
    loader._has_spatial_cols(traj.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns and not is_long_lat:
        is_long_lat = False
        pings_df = traj[[traj_cols['x'], traj_cols['y']]].copy()
    else:
        is_long_lat = True
        pings_df = traj[[traj_cols['longitude'],  traj_cols['latitide']]].copy()

    # Build pings GeoDataFrame
    # use gpd.points_from_xy
    pings_df["pings_geometry"] = pings_df.apply(lambda row: Point(row[traj_cols['longitude']], row[traj_cols['latitude']]) if is_long_lat else Point(row[traj_cols['x']], row[traj_cols['y']]), axis=1)
    pings_df = gpd.GeoDataFrame(pings_df, geometry="pings_geometry", crs=poi_table.crs)
    
    # First spatial join (within)
    pings_df = gpd.sjoin(pings_df, poi_table, how="left", predicate="within")
   
    # Identify unmatched pings
    unmatched_mask = pings_df["building_id"].isna()
    unmatched_pings = pings_df[unmatched_mask].drop(columns=["building_id", "index_right"])

    if not unmatched_pings.empty:
        # Nearest spatial join for unmatched pings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
            nearest = gpd.sjoin_nearest(unmatched_pings, poi_table, how="left", max_distance=max_distance)

        # Keep only the first match for each original ping
        nearest = nearest.groupby(nearest.index).first()

        # Update original DataFrame with nearest matches
        pings_df.loc[unmatched_mask, "building_id"] = nearest["building_id"].values

    return pings_df["building_id"]

def oracle_map(traj, true_visits, traj_cols, **kwargs):
    """
    Map elements in traj to ground truth location based solely on the record's time.

    Parameters
    ----------
    traj : pd.DataFrame
        The trajectory DataFrame containing x and y coordinates.
    true_visits : pd.DataFrame
        A visitation table containing location IDs, start times, and durations/end times.
    traj_cols : list
        The columns in the trajectory DataFrame to be used for mapping.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pd.Series
        A Series containing the location IDs corresponding to the pings in the trajectory.
    """ 
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)

    loader._has_time_cols(traj.columns, traj_cols)
    loader._has_time_cols(true_visits.columns, traj_cols)

    end_col_present = _has_end_cols(true_visits.columns, traj_cols)
    duration_col_present = _has_duration_cols(true_visits.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        print("Missing required (end or duration) temporal columns for true_visits dataframe.")
        return False
    
    use_datetime = False
    if traj_cols['timestamp'] in traj.columns:
        time_col_in = traj_cols['timestamp']
        time_key = 'timestamp'
    elif traj_cols['start_timestamp'] in traj.columns:
        time_col_in = traj_cols['start_timestamp']
        time_key = 'start_timestamp'
    else:
        use_datetime = True 

    # TO DO: Check the same thing for true_visits start, and duration/end
    # TO DO: Conversion of everything to UTC
    # Loop through ground truth and numpy timestamps diff
    
    # Check whether to use timestamp or datetime columns
    is_long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in traj.columns and kwargs['longitude'] in traj.columns
    
    return location_ids


def dawn_time(day_part, dawn_hour=6):
    s,e = day_part
    return np.min([(e.hour*60 + e.minute),dawn_hour*60]) - np.min([(s.hour*60 + s.minute),dawn_hour*60]) 

def dusk_time(day_part, dusk_hour=19):
    s,e = day_part
    return np.max([(e.hour*60 + e.minute)-dusk_hour*60,0]) - np.max([(s.hour*60 + s.minute)-dusk_hour*60, 0])

def slice_datetimes_interval_fast(start, end):
    full_days = (datetime.combine(end, time.min) - datetime.combine(start, time.max)).days
    if full_days >= 0:
        day_parts = [(start.time(), time.max), (time.min, end.time())]
    else:
        full_days = 0
        day_parts = [(start.time(), end.time()), (start.time(), start.time())]
    return full_days, day_parts

def duration_at_night_fast(start, end, dawn_hour = 6, dusk_hour = 19):
    full_days, (part1, part2) = slice_datetimes_interval_fast(start, end)
    total_dawn_time = dawn_time(part1, dawn_hour)+dawn_time(part2, dawn_hour)
    total_dusk_time = dusk_time(part1, dusk_hour)+dusk_time(part2, dusk_hour)
    return int(total_dawn_time + total_dusk_time + full_days*(dawn_hour + (24-dusk_hour))*60)

def clip_stays_date(traj, dates, dawn_hour = 6, dusk_hour = 19):
    start = pd.to_datetime(traj['start_datetime'])
    duration = traj['duration']

    # Ensure timezone-aware clipping bounds
    tz = start.dt.tz
    date_0 = pd.Timestamp(parse(dates[0]), tz=tz)
    date_1 = pd.Timestamp(parse(dates[1]), tz=tz)

    end = start + pd.to_timedelta(duration, unit='m')

    # Clip to date range
    start_clipped = start.clip(lower=date_0, upper=date_1)
    end_clipped = end.clip(lower=date_0, upper=date_1)

    # Recompute durations
    duration_clipped = ((end_clipped - start_clipped).dt.total_seconds() // 60).astype(int)
    duration_night = [duration_at_night_fast(s, e, dawn_hour, dusk_hour) for s, e in zip(start_clipped, end_clipped)]

    return pd.DataFrame({
        'id': traj['id'].values,
        'start': start_clipped,
        'duration': duration_clipped,
        'duration_night': duration_night,
        'location': traj['location']
    })

def count_nights(usr_polygon, dawn_hour = 6, dusk_hour = 19, min_dwell = 10):   
    nights = set()
    weeks = set()

    for _, row in usr_polygon.iterrows():
        d = row['start']
        d = pd.to_datetime(d)
        full_days, (part1, part2) = slice_datetimes_interval_fast(d, d + pd.to_timedelta(row['duration'], unit='m'))

        dawn1 = dawn_time(part1, dawn_hour)
        dusk1 = dusk_time(part1, dusk_hour)
        dawn2 = dawn_time(part2, dawn_hour)
        dusk2 = dusk_time(part2, dusk_hour)

        if full_days == 0:
            if dawn1 >= min_dwell:
                night = d - timedelta(days=1)
                nights.add(night.date())
                weeks.add((night - timedelta(days=night.weekday())).date())

            if (dusk1 + dawn2) >= min_dwell:
                night = d
                nights.add(night.date())
                weeks.add((night - timedelta(days=night.weekday())).date())

            if dusk2 >= min_dwell:
                night = d + timedelta(days=1)
                nights.add(night.date())
                weeks.add((night - timedelta(days=night.weekday())).date())
        else:
            if dawn1 >= min_dwell:
                night = d - timedelta(days=1)
                nights.add(night.date())
                weeks.add((night - timedelta(days=night.weekday())).date())

            for t in range(full_days + 1):
                night = d + timedelta(days=t)
                nights.add(night.date())
                weeks.add((night - timedelta(days=night.weekday())).date())

            if dusk2 >= min_dwell:
                night = d + timedelta(days=full_days + 1)
                nights.add(night.date())
                weeks.add((night - timedelta(days=night.weekday())).date())

    identifier = usr_polygon['id'].iloc[0]
    location = usr_polygon['location'].iloc[0]

    return pd.DataFrame([{
        'id': identifier,
        'location': location,
        'night_count': len(nights),
        'week_count': len(weeks)
    }])


def night_stops(stop_table, user='user', dawn_hour = 6, dusk_hour = 19, min_dwell = 10):
    # Date range
    start_date = str(stop_table['start_datetime'].min().date())
    weeks = stop_table['start_datetime'].dt.strftime('%Y-%U')
    num_weeks = weeks.nunique()

    # turn dates to datetime
    stop_table['start_datetime'] = pd.to_datetime(stop_table['start_datetime'])

    if 'id' not in stop_table.columns:
        stop_table['id'] = user

    end_date = (parse(start_date) + timedelta(weeks=num_weeks)).date().isoformat()
    dates = (start_date, end_date)
    df_clipped = clip_stays_date(stop_table, dates, dawn_hour, dusk_hour)
    df_clipped = df_clipped[(df_clipped['duration'] > 0) & (df_clipped['duration_night'] >= 15)]
    
    return df_clipped.groupby(['id', 'location'], group_keys=False).apply(count_nights(dawn_hour, dusk_hour, min_dwell)).reset_index(drop=True)