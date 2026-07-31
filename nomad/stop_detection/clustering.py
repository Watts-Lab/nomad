import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import nomad.io.base as loader


EARTH_RADIUS_METERS = 6_371_000


def dbscan_labels(stops, epsilon=100, num_samples=1, traj_cols=None, **kwargs):
    """
    Cluster stop locations with DBSCAN.

    Parameters
    ----------
    stops : pd.DataFrame
        Stop table containing x/y or longitude/latitude coordinates.
    epsilon : float, default 100
        Maximum distance between neighboring stops. Units match x/y coordinates
        or are meters for longitude/latitude coordinates.
    num_samples : int, default 1
        Minimum number of stops required to form a cluster.
    traj_cols : dict, optional
        Column mappings for x/y or longitude/latitude coordinates.
    **kwargs
        Additional coordinate-column mappings.

    Returns
    -------
    pd.Series
        Destination labels aligned with ``stops.index``. Noise is labeled -1.
    """
    if stops.empty:
        return pd.Series(index=stops.index, dtype="int64", name="location_id")

    coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(
        stops.columns, traj_cols, kwargs
    )
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)

    # parse x/y and lon/lat coordinates seperately
    if use_lon_lat:
        coords = np.radians(
            stops[[traj_cols[coord_key2], traj_cols[coord_key1]]].to_numpy(
                dtype="float64"
            )
        )
        epsilon /= EARTH_RADIUS_METERS
        metric = "haversine"
    else:
        coords = stops[
            [traj_cols[coord_key1], traj_cols[coord_key2]]
        ].to_numpy(dtype="float64")
        metric = "euclidean"

    # run DBSCAN to assign destination labels to each stop
    labels = DBSCAN(
        eps=epsilon,
        min_samples=num_samples,
        metric=metric,
        algorithm="ball_tree",
    ).fit_predict(coords)

    return pd.Series(labels, index=stops.index, name="location_id")


__all__ = ["dbscan_labels"]
