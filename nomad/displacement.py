from nomad.agg.privacy import _laplace_pcts
import numpy as np
import pandas as pd

def normalize_od(data, origin_col, dest_col, weight_col,
                 diff_privacy_eps=None, seed=None):
    """
    Convert an origin–destination table to percentages
    (share of total trips) with optional Laplace DP noise.
    Parameters
    ----------
    data            : pandas DataFrame, OD table.
    origin_col      : str, column with origin geography.
    dest_col        : str, column with destination geography.
    weight_col      : str, column with trip counts (non-negative ints).
    diff_privacy_eps: float or None.  If set, add Laplace noise with
                      privacy budget ε; if None, return exact percentages.
    seed            : optional int for reproducibility when ε is given.
    Returns
    -------
    pandas DataFrame with columns
        [origin_col, dest_col, 'percentage']
    where 'percentage' sums to 100.
    """
    for col in [origin_col, dest_col, weight_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")

    N = data[weight_col].sum()
    out = data.copy()
    out['_fraction'] = out[weight_col] / N        # in [0,1]

    if diff_privacy_eps is not None:
        out['_fraction'] = _laplace_pcts(
            out['_fraction'].to_numpy(),
            total_trips=N,
            epsilon=diff_privacy_eps,
            seed=seed
        )

    out['percentage'] = 100 * out['_fraction']
    return out[[origin_col, dest_col, 'percentage']]


def normalized_remained(data, origin_col, dest_col, weight_col,
                         diff_privacy_eps=None, seed=None):
    """
    Percentage of trips that *remain* in the origin geography,
    with optional per-origin Laplace DP noise.
    Each origin contributes a single row; percentages are computed
    relative to that origin’s total outgoing trips.
    Parameters
    ----------
    data, origin_col, dest_col, weight_col : see `normalize_od`.
    diff_privacy_eps : float or None.  ε for DP; None for exact values.
    seed             : optional int.
    Returns
    -------
    pandas DataFrame with columns
        [origin_col, 'percentage'].
    """
    for col in [origin_col, dest_col, weight_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")

    origin_totals = (
        data.groupby(origin_col)[weight_col]
            .sum()
            .rename('origin_total')
    )

    stayed = data[data[origin_col] == data[dest_col]].copy()
    stayed = stayed.merge(origin_totals, left_on=origin_col, right_index=True)
    stayed['_fraction'] = stayed[weight_col] / stayed['origin_total']

    if diff_privacy_eps is not None:
         stayed['_fraction'] = _laplace_pcts(
             stayed['_fraction'].to_numpy(),
             total_trips=stayed['origin_total'],
             epsilon=diff_privacy_eps,
             seed=seed)
        
    stayed['percentage'] = 100 * stayed['_fraction']
    return stayed[[origin_col, 'percentage']]


def normalized_moved(data, origin_col, dest_col, weight_col,
                     diff_privacy_eps=None, seed=None):
    """
    Normalised list of *moving* OD flows
    (origin ≠ destination) with optional DP noise.
    Parameters
    ----------
    data, origin_col, dest_col, weight_col : see `normalize_od`.
    diff_privacy_eps : float or None.  ε for DP; None for exact values.
    seed             : optional int.
    Returns
    -------
    pandas DataFrame with columns
        [origin_col, dest_col, 'percentage']  whose percentages sum to 100.
    """
    for col in [origin_col, dest_col, weight_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")

    moved = data[data[origin_col] != data[dest_col]].copy()
    moved_total = moved[weight_col].sum()
    moved['_fraction'] = moved[weight_col] / moved_total

    if diff_privacy_eps is not None:
         moved['_fraction'] = _laplace_pcts(
             moved['_fraction'].to_numpy(),
             total_trips=moved['origin_total'],
             epsilon=diff_privacy_eps,
             seed=seed)

    moved['percentage'] = 100 * moved['_fraction']
    return moved[[origin_col, dest_col, 'percentage']]