import geopandas as gpd
import warnings
import pandas as pd
import pyproj
import pdb
import numpy as np
from sklearn.cluster import DBSCAN
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils

# TO DO: change to stops_to_poi
def point_in_polygon(data, poi_table, method='centroid', data_crs=None, max_distance=0,
                     cluster_label=None, location_id=None, recompute_location=True, traj_cols=None, **kwargs):
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
    recompute_location : bool, default True
        For labeled ping data, ignored for stop data. If False and a location column
        (as determined by `location_id`) already exists in `data`, it will
        be reused instead of overwritten.
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
    coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(data.columns, traj_cols, kwargs)   
    traj_cols_outer = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    
    end_col_present = loader._has_end_cols(data.columns, traj_cols_outer)
    duration_col_present = loader._has_duration_cols(data.columns, traj_cols_outer)
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

        loc_col = location_id if location_id is not None else "location_id"
        clustered_pings = data.loc[data[cluster_label] != -1].copy()
        if method=='majority': 
            if recompute_location or (loc_col not in data.columns):
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
            
            major = clustered_pings.groupby(cluster_label)[loc_col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None) 
            
            return data[[cluster_label]].join(major, on=cluster_label)[loc_col]
            
        elif method=='centroid': # should be medoid?
            
            if use_lon_lat:
                warnings.warn("Spherical ('longitude', 'latitude') coordinates were passed. Centroids will not agree with geodetic distances")              
                centr_data = clustered_pings.groupby(cluster_label).agg({traj_cols_outer['longitude']:'mean', traj_cols_outer['latitude']:'mean'})
            else:
                centr_data = clustered_pings.groupby(cluster_label).agg({traj_cols_outer['x']:'mean', traj_cols_outer['y']:'mean'})

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
        coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(data.columns, traj_cols, kwargs) 

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

    if not data_crs.equals(pyproj.CRS(poi_table.crs)):
        poi_table = poi_table.to_crs(data_crs)
        warnings.warn("CRS for `poi_table` does not match crs for `data`. Reprojecting...")

    out_col = location_id if location_id is not None else "location_id"
    # choose where IDs come from: poi_table column (if it exists) else poi_table.index
    use_col = (location_id is not None) and (location_id in poi_table.columns)

    if location_id is None:
        warnings.warn("location_id not provided; using poi_table.index for spatial join.")
    elif not use_col:
        warnings.warn(f"{location_id} column not found in poi_table; using poi_table.index for spatial join.")
        
    if max_distance>0:
        if data_crs.is_geographic:
            warnings.warn(f"Provided CRS {data_crs.name} is a geographic coordinate system. "
                             "This will lead to errors when computing euclidean distances."
                             f"Did you mean to use `max_distance=0'?"
                         )        
        
        p_idx, idx = poi_table.sindex.nearest(pings_gdf, max_distance=max_distance, return_all=False)

        values = poi_table.iloc[idx][location_id] if use_col else pd.Series(poi_table.iloc[idx].index)
        return values.set_axis(data.index[p_idx]).rename(out_col).reindex(data.index)

    else: # default max_distance = 0
        p_idx, idx = poi_table.sindex.query(pings_gdf, predicate="within") # boundary counts; use "contains" to exclude it
        values = poi_table.iloc[idx][location_id] if use_col else pd.Series(poi_table.iloc[idx].index)
        
        s = values.set_axis(data.index[p_idx]).rename(out_col)
        s = s[~s.index.duplicated()]          # keep first match per ping
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
    t_key_l, use_datetime_l = loader._fallback_time_cols_dt(data.columns, traj_cols, kwargs)
    t_key_r, use_datetime_r = loader._fallback_time_cols_dt(true_visits.columns, traj_cols, kwargs)

    
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


def _get_location_center(coords, metric='euclidean'):
    """
    Calculate the center of a location cluster.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates (n_points, 2)
    metric : str
        'euclidean' for projected coordinates, 'haversine' for lat/lon

    Returns
    -------
    tuple
        (x, y) coordinates of the center
    """
    if metric == 'haversine':
        # For geographic coordinates, use angle-based mean
        # Convert to radians for calculation
        coords_rad = np.radians(coords)
        x = coords_rad[:, 0]
        y = coords_rad[:, 1]

        # Convert to 3D Cartesian coordinates
        cos_y = np.cos(y)
        cart_x = cos_y * np.cos(x)
        cart_y = cos_y * np.sin(x)
        cart_z = np.sin(y)

        # Average and convert back
        avg_x = np.mean(cart_x)
        avg_y = np.mean(cart_y)
        avg_z = np.mean(cart_z)

        lon = np.arctan2(avg_y, avg_x)
        hyp = np.sqrt(avg_x**2 + avg_y**2)
        lat = np.arctan2(avg_z, hyp)

        return np.degrees(lon), np.degrees(lat)
    else:
        # For projected coordinates, use simple mean
        return np.mean(coords[:, 0]), np.mean(coords[:, 1])


def cluster_locations_dbscan(
    stops,
    epsilon=100,
    num_samples=1,
    distance_metric='euclidean',
    agg_level='user',
    traj_cols=None,
    **kwargs
):
    """
    Cluster coordinate points or detected stops into locations using DBSCAN.

    Parameters
    ----------
    stops : pd.DataFrame or gpd.GeoDataFrame
        Coordinate points or stop rows. DataFrames require x/y or
        longitude/latitude columns. GeoDataFrames require point geometry.
    epsilon : float, default 100
        Maximum distance between rows in the same location. Units match projected
        coordinates or are meters for longitude/latitude coordinates.
    num_samples : int, default 1
        Minimum number of stops required to form a location
    distance_metric : str, default 'euclidean'
        Distance metric for projected coordinates. Geographic coordinates always
        use haversine distance.
    agg_level : str, default 'user'
        'user' = separate locations per user, 'dataset' = shared locations across users
    traj_cols : dict, optional
        Column name mappings for coordinates and user_id
    **kwargs
        Additional arguments passed to column detection

    Returns
    -------
    tuple of (pd.DataFrame, gpd.GeoDataFrame)
        - input rows with a nonnegative location ID for every row
        - locations GeoDataFrame with cluster centers and extents

    Notes
    -----
    Each DBSCAN noise row is retained as its own singleton location.
    """
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)
    location_col = traj_cols['location_id']
    crs = stops.crs if isinstance(stops, gpd.GeoDataFrame) else None

    # Return the expected output schema when there are no rows to cluster.
    if stops.empty:
        stops_out = stops.copy()
        stops_out[location_col] = pd.Series(index=stops.index, dtype='int64')
        locations = gpd.GeoDataFrame(
            columns=[location_col, 'center', 'extent', 'n_stops'],
            geometry='center',
            crs=crs,
        )
        return stops_out, locations

    # Extract coordinates from point geometry or configured DataFrame columns.
    if isinstance(stops, gpd.GeoDataFrame):
        coords = np.column_stack((stops.geometry.x, stops.geometry.y))
        use_lon_lat = crs is not None and pyproj.CRS(crs).is_geographic
    else:
        coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(
            stops.columns, traj_cols, kwargs
        )
        coords = stops[
            [traj_cols[coord_key1], traj_cols[coord_key2]]
        ].to_numpy(dtype='float64')
        if use_lon_lat:
            crs = 'EPSG:4326'

    # DBSCAN expects geographic coordinates in radians and latitude-first order.
    if use_lon_lat:
        cluster_coords = np.radians(coords[:, [1, 0]])
        cluster_epsilon = epsilon / constants.EARTH_RADIUS_METERS
        metric = 'haversine'
    else:
        cluster_coords = coords
        cluster_epsilon = epsilon
        metric = distance_metric

    # Cluster each user independently, or treat the entire dataset as one group.
    if agg_level == 'user':
        user_col = traj_cols['user_id']
        if user_col not in stops.columns:
            raise ValueError(
                "agg_level='user' requires a user_id column specified in traj_cols or kwargs"
            )
        group_positions = stops.groupby(
            user_col, sort=False, dropna=False
        ).indices.values()
    elif agg_level == 'dataset':
        user_col = None
        group_positions = [np.arange(len(stops))]
    else:
        raise ValueError("agg_level must be 'user' or 'dataset'")

    location_ids = np.empty(len(stops), dtype='int64')
    next_location_id = 0

    # Offset each group's labels and turn every noise row into a unique location.
    for positions in group_positions:
        labels = DBSCAN(
            eps=cluster_epsilon,
            min_samples=num_samples,
            metric=metric,
            algorithm='ball_tree',
        ).fit_predict(cluster_coords[positions])

        non_noise = labels >= 0
        cluster_codes = pd.factorize(labels[non_noise], sort=False)[0]
        n_clusters = cluster_codes.max() + 1 if cluster_codes.size else 0
        group_location_ids = np.empty(len(labels), dtype='int64')
        group_location_ids[non_noise] = next_location_id + cluster_codes
        group_location_ids[~non_noise] = np.arange(
            next_location_id + n_clusters,
            next_location_id + n_clusters + (~non_noise).sum(),
        )
        location_ids[positions] = group_location_ids
        next_location_id += n_clusters + (~non_noise).sum()

    stops_out = stops.copy()
    stops_out[location_col] = location_ids

    # Build one summary row for every assigned location.
    location_rows = pd.DataFrame(
        {
            location_col: location_ids,
            '_coord1': coords[:, 0],
            '_coord2': coords[:, 1],
        }
    )
    grouped_locations = location_rows.groupby(location_col, sort=True)
    center_values = grouped_locations[['_coord1', '_coord2']].apply(
        lambda group: _get_location_center(
            group.to_numpy(), metric='haversine' if use_lon_lat else 'euclidean'
        ),
        include_groups=False,
    )
    center_coords = np.vstack(center_values.to_numpy())
    locations = gpd.GeoDataFrame(
        {
            location_col: center_values.index.to_numpy(),
            'n_stops': grouped_locations.size().to_numpy(),
        },
        geometry=gpd.points_from_xy(
            center_coords[:, 0], center_coords[:, 1], crs=crs
        ),
        crs=crs,
    ).rename_geometry('center')

    # The convex hull records the observed spatial extent of each location.
    point_locations = gpd.GeoDataFrame(
        {location_col: location_ids},
        geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1], crs=crs),
        crs=crs,
    )
    extents = point_locations.dissolve(by=location_col).geometry.convex_hull
    locations['extent'] = gpd.GeoSeries(
        extents.reindex(locations[location_col]).to_numpy(),
        index=locations.index,
        crs=crs,
    )

    if user_col is not None:
        location_rows[user_col] = stops[user_col].to_numpy()
        location_users = location_rows.groupby(location_col, sort=True)[user_col].first()
        locations[user_col] = locations[location_col].map(location_users)

    return stops_out, locations


def cluster_locations_per_user(
    stops,
    epsilon=100,
    num_samples=1,
    distance_metric='euclidean',
    traj_cols=None,
    **kwargs
):
    """
    Convenience function to cluster locations per user.

    This is equivalent to calling cluster_locations_dbscan with agg_level='user'.

    Parameters
    ----------
    stops : pd.DataFrame or gpd.GeoDataFrame
        Stop data with user_id column
    epsilon : float, default 100
        Maximum distance between stops in same location (meters)
    num_samples : int, default 1
        Minimum stops required to form a location
    distance_metric : str, default 'euclidean'
        'euclidean' or 'haversine'
    traj_cols : dict, optional
        Column name mappings
    **kwargs
        Additional arguments

    Returns
    -------
    tuple of (pd.DataFrame, gpd.GeoDataFrame)
        Stops with location_id and locations table

    Raises
    ------
    ValueError
        If user_id column is not found
    """
    traj_cols_temp = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in stops.columns:
        raise ValueError(
            "cluster_locations_per_user requires a 'user_id' column "
            "specified in traj_cols or kwargs"
        )

    return cluster_locations_dbscan(
        stops,
        epsilon=epsilon,
        num_samples=num_samples,
        distance_metric=distance_metric,
        agg_level='user',
        traj_cols=traj_cols,
        **kwargs
    )


# Alias for convenience
generate_locations = cluster_locations_dbscan
