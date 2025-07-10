import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# quick-and-dirty DP helper
# ---------------------------------------------------------------------
def _noisy_fractions(fractions, total_trips, epsilon, seed=None):
    """
    Add Laplace(0, 1/(N·ε)) noise to a 1-D array of fractions that sum to 1,
    then re-normalise so the noisy values still sum to 1.

    Parameters
    ----------
    fractions    : 1-D numpy array, non-negative and summing to 1
    total_trips  : int, denominator N used to form the fractions
    epsilon      : float, privacy budget
    seed         : optional int for reproducibility

    Returns
    -------
    1-D numpy array of the same length whose entries sum to 1.
    """
    rng = np.random.default_rng(seed)
    sensitivity = 1 / total_trips          # Δ = 1/N
    scale = sensitivity / epsilon          # b = Δ/ε
    noisy = fractions + rng.laplace(0.0, scale, size=len(fractions))
    noisy = np.clip(noisy, 0.0, None)      # keep non-negative
    if noisy.sum() == 0:                   # pathological noise
        noisy[:] = 1 / len(noisy)
    else:
        noisy /= noisy.sum()               # re-normalise
    return noisy


# ---------------------------------------------------------------------
# normalisation utilities with optional differential privacy
# ---------------------------------------------------------------------
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
        out['_fraction'] = _noisy_fractions(
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
        rng = np.random.default_rng(seed)
        # noise scale varies by origin_total
        scales = (1 / stayed['origin_total']) / diff_privacy_eps
        noise = rng.laplace(0.0, scales.to_numpy())
        stayed['_fraction'] = np.clip(stayed['_fraction'] + noise, 0.0, 1.0)

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
        moved['_fraction'] = _noisy_fractions(
            moved['_fraction'].to_numpy(),
            total_trips=moved_total,
            epsilon=diff_privacy_eps,
            seed=seed
        )

    moved['percentage'] = 100 * moved['_fraction']
    return moved[[origin_col, dest_col, 'percentage']]


# ---------------------------------------------------------------------
# top-k utilities (unchanged, but kept here for completeness)
# ---------------------------------------------------------------------
def top_k_destinations(data, dest_col, weight_col, k=4):
    if dest_col not in data.columns or weight_col not in data.columns:
        raise ValueError("Required columns not found.")
    dest_counts = data.groupby(dest_col)[weight_col].sum()
    percentages = 100 * dest_counts / dest_counts.sum()
    top_k = (
        percentages.sort_values(ascending=False)
        .head(k)
        .reset_index()
        .rename(columns={0: 'percentage'})
    )
    top_k.columns = [dest_col, 'percentage']
    return top_k


def top_k_origins(data, origin_col, weight_col, k=4):
    if origin_col not in data.columns or weight_col not in data.columns:
        raise ValueError("Required columns not found.")
    origin_counts = data.groupby(origin_col)[weight_col].sum()
    percentages = 100 * origin_counts / origin_counts.sum()
    top_k = (
        percentages.sort_values(ascending=False)
        .head(k)
        .reset_index()
        .rename(columns={0: 'percentage'})
    )
    top_k.columns = [origin_col, 'percentage']
    return top_k