import geopandas as gpd
import nomad.io.base as loader
import nomad.constants as constants
import warnings
import pandas as pd
import nomad.io.base as loader
from nomad.stop_detection.utils import _fallback_time_cols
import pyproj
import pdb

# TO DO: change to stops_to_poi
def point_in_polygon(data, poi_table, method='centroid', data_crs=None, max_distance=0,
                     cluster_label=None, location_id=None, traj_cols=None, **kwargs):
    """
    Assign each stop or cluster of pings in `data` to a polygon in `poi_table`, 
    either by the cluster’s centroid location or by the most frequent polygon hit.

    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        A table of pings (with optional stop/duration columns) or stops, 
        indexed by observation or cluster.
    poi_table : gpd.GeoDataFrame
        Polygons to match against, with CRS set and optional ID column.
    method : {'centroid', 'majority'}, default 'centroid'
        ‘centroid’ uses each cluster’s mean point; ‘majority’ picks the polygon 
        most often visited within each cluster (only for ping data).
    data_crs : str or pyproj.CRS, optional
        CRS for `data` when it is a plain DataFrame; ignored if `data` is a GeoDataFrame.
    max_distance : float, default 0
        Search radius for nearest‐neighbor fall-back; zero triggers strict 
        point-in-polygon matching.
    cluster_label : str, optional
        Column name holding cluster IDs in ping data; inferred from `data` if absent.
    location_id : str, optional
        Column in `poi_table` containing the output ID; uses the GeoDataFrame index if None.
    traj_cols : list of str, optional
        Names of the coordinate columns in `data` when it is a DataFrame.
    **kwargs
        Passed through to `poi_map` or the trajectory-column parser.

    Returns
    -------
    pd.Series
        Indexed like `data`, giving the matched polygon ID for each stop or ping.
        Points or clusters that fall outside every polygon or beyond `max_distance`
        are set to NaN.
    """
    # check if it is stop table
    traj_cols_w_deflts = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    end_col_present = loader._has_end_cols(data.columns, traj_cols_w_deflts)
    duration_col_present = loader._has_duration_cols(data.columns, traj_cols_w_deflts)
    is_stop_table = (end_col_present or duration_col_present)

    if is_stop_table:
        # is stop table
        if method=='majority':
            raise TypeError("Method `majority' requires ping data with cluster labels,\
                            but a stop table was provided")
        elif method=='centroid':
            stop_table = data.copy()
            location = poi_map(
                data=stop_table,
                poi_table=poi_table,
                max_distance=max_distance,
                data_crs=data_crs,
                location_id=location_id,
                traj_cols=traj_cols,
                **kwargs)

            return location
            
        else:
            raise ValueError(f"Method {method} not among implemented methods: `centroid' and `majority'")

    else:
        # is labeled pings
        if not cluster_label: #try defaults and raise
            if 'cluster_label' in data.columns:
                cluster_label = 'cluster_label'
            elif 'cluster' in data.columns:
                cluster_label = 'cluster'
            else:
                raise ValueError(f"Argument `cluster_label` is required for visit attribution of labeled pings.")

        clustered_pings = data.loc[data[cluster_label] != -1].copy()
        if method=='majority': 
            location = poi_map(
                data=clustered_pings,
                poi_table=poi_table,
                max_distance=max_distance,
                data_crs=data_crs,
                location_id=location_id,
                traj_cols=traj_cols,
                **kwargs                
            )
            loc_col = location.name
            clustered_pings = clustered_pings.join(location)
            
            location = clustered_pings.groupby(cluster_label)[loc_col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None) 
            
            return data[[cluster_label]].join(location, on=cluster_label)[loc_col]
            
        elif method=='centroid': # should be medoid?
            loader._has_spatial_cols(data.columns, traj_cols, exclusive=True)
            use_lon_lat = ('latitude' in traj_cols and 'longitude' in traj_cols)
            if use_lon_lat:
                warnings.warn("Spherical ('longitude', 'latitude') coordinates were passed. Centroids will not agree with geodetic distances")                
                centr_data = clustered_pings.groupby(cluster_label).agg({traj_cols['longitude']:'mean', traj_cols['latitude']:'mean'})
            else:
                centr_data = clustered_pings.groupby(cluster_label).agg({traj_cols['x']:'mean', traj_cols['y']:'mean'})

            location = poi_map(
                data=centr_data,
                poi_table=poi_table,
                max_distance=max_distance,
                data_crs=data_crs,
                location_id=location_id,
                traj_cols=traj_cols,
                **kwargs)
            loc_col = location.name
            
            return data[[cluster_label]].join(location, on=cluster_label)[loc_col]

        else:
            raise ValueError(f"Method {method} not among implemented methods: `centroid' and `majority'")

    return None
    
# change to point_in_polygon, move to filters.py
def poi_map(data, poi_table, max_distance=0, data_crs=None, location_id=None, traj_cols=None, **kwargs):
    """
    Assign each point in `data` to a polygon in `poi_table`, using containment when
    `max_distance==0` or the nearest neighbor within `max_distance` otherwise.

    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        Input points, either as a DataFrame with coordinate columns or a GeoDataFrame.
    poi_table : gpd.GeoDataFrame
        Polygons to match against, indexed or with `location_id` column.
    traj_cols : list of str, optional
        Names of the coordinate columns in `data` when it is a DataFrame.
    max_distance : float, default 0
        Maximum search radius for nearest‐neighbor matching; zero invokes a point‐in‐polygon test.
    data_crs : str or pyproj.CRS, optional
        CRS for `data` if it is a DataFrame; ignored for GeoDataFrames.
    location_id : str, optional
        Name of the geometry ID column in `poi_table`; uses the GeoDataFrame index if not provided.
    **kwargs
        Passed to trajectory‐column parsing helper.

    Returns
    -------
    pd.Series
        Indexed like `data`, with each entry set to the matching polygon’s ID (from
        `location_id` or `poi_table.index`). Points not contained or beyond `max_distance`
        yield NaN. When multiple polygons overlap a point, only the first match is kept.
    """
    # column name handling
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, defaults={})
        
    if poi_table.crs is None:
        raise ValueError(f"poi_table must have crs attribute for spatial join.")
   
    # Determine which geometry to use
    if isinstance(data, gpd.GeoDataFrame):
        pings_gdf = data.geometry
        # if geodataframe, data_crs is ignored but we Raise if conflicting crs because it is suspect
        if data_crs and not pyproj.CRS(pings_gdf.crs).equals(pyproj.CRS(data_crs)):
            raise ValueError(f"Provided CRS {data_crs} conflicts with traj CRS {data.crs}.")

    if isinstance(data, pd.DataFrame):
        # check that user specified x,y or lat, lon but not both
        loader._has_spatial_cols(data.columns, traj_cols, exclusive=True)
        use_lon_lat = ('latitude' in traj_cols and 'longitude' in traj_cols)

        if use_lon_lat:
            if data_crs:
                data_crs = pyproj.CRS(data_crs)
                if data_crs.is_projected:
                    warnings.warn(f"Provided CRS {data_crs.name} is a projected coordinate system, but "
                                 "spherical ('longitude', 'latitude') coordinates were passed. Did you mean to pass data_crs='EPSG:4326'?"
                                 )
            else: # we assume EPSG:4326
                warnings.warn("Argument `data_crs` not provided, assuming EPSG:4326 for ('longitude', 'latitude') coordinates")
                data_crs = pyproj.CRS("EPSG:4326")
            
            pings_gdf= gpd.points_from_xy(
                data[traj_cols['longitude']],
                data[traj_cols['latitude']],
                crs=data_crs) # order matters: lon first
        else:
            if not data_crs:
                raise ValueError(f"data_crs must be provided when using projected coordinates.")
            data_crs = pyproj.CRS(data_crs)
            if data_crs.is_geographic:
                warnings.warn(f"Provided CRS {data_crs.name} is a geographic coordinate system. "
                             "This will lead to errors if passed coordinates ('x', 'y') are projected."
                             f"Did you mean to use {poi_table.crs}?"
                             )
            pings_gdf= gpd.points_from_xy(
                data[traj_cols['x']],
                data[traj_cols['y']],
                crs=data_crs)
    else:
        raise TypeError("`data` must be a pandas DataFrame or a GeoDataFrame.")

    if data_crs != pyproj.CRS(poi_table.crs):
        raise ValueError("CRS for `data` does not match CRS for `poi_table`.")

    use_poi_idx = True
    if location_id is not None:
        loc_col = location_id
        if location_id in poi_table:
            use_poi_idx=False
        else:
            warnings.warn(f"{location_id} column not found in {poi_table.columns}, defaulting to poi_table.index for spatial join.")
    else:
        loc_col = 'location_id'
        warnings.warn(f"location_id column not provided, defaulting to poi_table.index for spatial join.")

        
    if max_distance>0:
        if data_crs.is_geographic:
            warnings.warn(f"Provided CRS {data_crs.name} is a geographic coordinate system. "
                             "This will lead to errors when computing euclidean distances."
                             f"Did you mean to use `max_distance=0'?"
                         )        
        
        p_idx, idx = poi_table.sindex.nearest(pings_gdf, max_distance=max_distance, return_all=False)
        if use_poi_idx:
            s = pd.Series(poi_table.iloc[idx].index, index=data.index[p_idx])
            s.name = loc_col
        else:
            s = pd.Series(poi_table.iloc[idx][loc_col].values, index=data.index[p_idx])
            s.name = loc_col
            
        return s.reindex(data.index)

    else: # default max_distance = 0
        p_idx, idx = poi_table.sindex.query(pings_gdf, predicate="within") # boundary counts; use "contains" to exclude it
        if use_poi_idx:
            s = pd.Series(poi_table.iloc[idx].index, index=data.index[p_idx]) # might have duplicates
            s = s.loc[~s.index.duplicated()]
            s.name = loc_col
        else:
            s = pd.Series(poi_table.iloc[idx][loc_col].values, index=data.index[p_idx])
            s = s.loc[~s.index.duplicated()]
            s.name = loc_col        
        return s.reindex(data.index)

def oracle_map(data, true_visits, traj_cols=None, **kwargs):
    """
    Map elements in traj to ground truth location based solely on time.

    Parameters
    ----------
    data : pd.DataFrame
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
    true_visits = true_visits.copy()
    data = data.copy()
       
    # determine temporal columns to use
    t_key_l, use_datetime_l = _fallback_time_cols(data.columns, traj_cols, kwargs)
    t_key_r, use_datetime_r = _fallback_time_cols(true_visits.columns, traj_cols, kwargs)

    
    traj_cols = loader._parse_traj_cols(true_visits.columns, traj_cols, kwargs) #load defaults
    if use_datetime_l != use_datetime_r:
        raise ValueError(f"Mismatch in temporal columns {traj_cols[t_key_l]} vs {traj_cols[t_key_r]}.")

    # check is diary table
    end_col_present = loader._has_end_cols(true_visits.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(true_visits.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        raise ValueError("Missing required (end or duration) temporal columns for true_visits dataframe.")

    if traj_cols['location_id'] not in true_visits.columns:
        raise ValueError(f"Missing {traj_cols[location_id]} column in {true_visits.columns}."
                        "pass `location_id` as keyword argument or in traj_cols."
                        )
    
    end_t_key = 'end_datetime' if use_datetime_r else 'end_timestamp'
    if not end_col_present:
        if use_datetime_r:
            true_visits[end_t_key] = true_visits[traj_cols[t_key_r]] + pd.to_timedelta(true_visits[traj_cols['duration']]*60, unit='s')
        else:
            true_visits[end_t_key] = true_visits[traj_cols[t_key_r]] + true_visits[traj_cols['duration']]*60

    
    # t_key_l and t_key_r match in type, and end_t_key exists
    data[traj_cols['location_id']] = pd.NA
    for idx, row in true_visits.loc[~true_visits[traj_cols['location_id']].isna()].iterrows():
        start, end, loc = row[traj_cols[t_key_r]], row[traj_cols[end_t_key]], row[traj_cols['location_id']]
        data.loc[(data[traj_cols[t_key_l]]>=start)&(data[traj_cols[t_key_l]]<end), traj_cols['location_id']] = loc
        
    return data[traj_cols['location_id']]


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