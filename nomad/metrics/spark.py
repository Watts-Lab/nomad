import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from nomad.constants import SEC_PER_UNIT
from nomad.stop_detection import utils
import nomad.io.base as loader

def rog_spark(stops, agg_freq='D', weighted=True, traj_cols=None, **kwargs):
    """
    Compute radius of gyration per period per user (Spark DataFrame version).

    Parameters
    ----------
    stops : pyspark.sql.DataFrame
    agg_freq : str, e.g. 'd', 'w', 'm'
    weighted : bool
        If False, each stop gets equal weight.
    traj_cols : dict, optional

    Returns
    -------
    pyspark.sql.DataFrame
        Columns: period, user_id, rog
    """
    t_key, coord_x, coord_y, use_datetime, use_lon_lat = utils._fallback_st_cols(
        stops.columns, traj_cols, kwargs
    )
    if use_lon_lat:
        raise ValueError("rog_spark: only x/y coordinates supported for Spark implementation.")
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)

    uid_key = traj_cols['user_id']
    if uid_key not in stops.columns:
        raise ValueError("rog_spark: requires user_id column.")
    dur_key = traj_cols['duration']
    if dur_key not in stops.columns:
        raise ValueError("rog_spark: requires stop duration column.")

    # Assign period column
    if use_datetime:
        unit_map = {'d': 'day', 'w': 'week', 'm': 'month'}
        stops = stops.withColumn(
            'period', F.date_trunc(unit_map[agg_freq.lower()], F.col(traj_cols[t_key]))
        )
        schema = "period timestamp, {} string, rog double".format(uid_key)
    else:
        step = SEC_PER_UNIT[agg_freq.lower()]
        stops = stops.withColumn(
            'period', (F.col(traj_cols[t_key]) // F.lit(step) * F.lit(step)).cast('long')
        )
        schema = "period long, {} string, rog double".format(uid_key)

    keys = ['period', uid_key]

    def _rog_udf(pdf: pd.DataFrame) -> pd.DataFrame:
        x = pdf[coord_x].values
        y = pdf[coord_y].values
        w = pdf[dur_key].values if weighted else np.ones(len(pdf)) / len(pdf)
        cx = np.sum(x * w) / w.sum()
        cy = np.sum(y * w) / w.sum()
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        rog_val = np.sqrt(np.sum(d2 * w) / w.sum())
        out = pdf.iloc[[0]][keys].copy()
        out['rog'] = rog_val
        return out

    rog_df = (
        stops.groupBy(keys)
        .applyInPandas(_rog_udf, schema=schema)
    )
    return rog_df
