import pandas as pd

import nomad.io.base as loader
from nomad.stop_detection.postprocessing import merge_stops
from nomad.stop_detection.sequential_algs import lachesis
from nomad.visit_attribution.visit_attribution import detect_locations


def lachesis_visits(
    data,
    delta_roam,
    dt_max=60,
    dur_min=5,
    complete_output=False,
    passthrough_cols=[],
    keep_col_names=True,
    postprocessing='dbscan',
    postprocessing_kwargs=None,
    merge_kwargs=None,
    traj_cols=None,
    **kwargs
):
    """Detect and attribute visits for one user's trajectory."""
    traj_cols_temp = loader._parse_traj_cols(
        data.columns, traj_cols, kwargs, warn=False
    )
    user_col = traj_cols_temp['user_id']
    if user_col in data.columns and data[user_col].nunique(dropna=False) > 1:
        raise ValueError(
            "lachesis_visits expects one user per call; group the input by "
            "user_id and call lachesis_visits for each group."
        )

    stops = lachesis(
        data,
        delta_roam=delta_roam,
        dt_max=dt_max,
        dur_min=dur_min,
        complete_output=complete_output,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
        **kwargs,
    )

    if postprocessing in (None, 'none'):
        return stops
    if postprocessing == 'infomap':
        raise NotImplementedError("Lachesis postprocessing method 'infomap' is not implemented")
    if postprocessing != 'dbscan':
        raise ValueError("postprocessing must be one of: None, 'none', 'dbscan', 'infomap'")

    location_col = traj_cols_temp['location_id']
    if stops.empty:
        stops[location_col] = pd.Series(index=stops.index, dtype='int64')
        return stops

    location_options = dict(postprocessing_kwargs or {})
    location_options['return_locations'] = True
    location_ids, locations = detect_locations(
        stops,
        traj_cols=traj_cols,
        **kwargs,
        **location_options,
    )

    labeled_stops = stops.copy()
    labeled_stops[location_col] = location_ids.to_numpy()
    merge_options = dict(merge_kwargs or {})
    merge_options.setdefault('location_col', location_col)
    visits = merge_stops(
        labeled_stops,
        traj_cols=traj_cols,
        **kwargs,
        **merge_options,
    )

    coord_key1, coord_key2, _ = loader._fallback_spatial_cols(
        labeled_stops.columns, traj_cols_temp, kwargs
    )
    centers = locations.set_index(location_col).center
    visits[traj_cols_temp[coord_key1]] = visits[location_col].map(centers.x)
    visits[traj_cols_temp[coord_key2]] = visits[location_col].map(centers.y)
    return visits
