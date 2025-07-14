import numpy.random as npr
import numpy as np

# ---------------------------------------------------------------------
# quick-and-dirty DP helper
# ---------------------------------------------------------------------
def _laplace_pcts(pcts, total_trips, epsilon, seed=None):
    """
    Add Laplace(0, 1/(N·ε)) noise to a 1-D array of percentages adding to 1,
    then re-normalise so the noisy values still sum to 1.
    Parameters
    ----------
    pcts    : 1-D numpy array, non-negative and summing to 1
    total_trips  : int, denominator N used to form the pcts
    epsilon      : float, privacy budget
    seed         : optional int for reproducibility
    Returns
    -------
    1-D numpy array of the same length whose entries sum to 1.
    """
    
    scale = 1 / (total_trips * epsilon)
    nz_pcts = pcts.loc[pcts>0]

    rng = npr.default_rng(seed)
    noisy_nz_pcts = np.clip(pcts + rng.laplace(0.0, scale, size=len(nz_pcts)), 0.0, 1.0)

    attempts = 0
    # avoid error from totally destructive noise    
    while noisy.sum() == 0:
        if attempts > 5:
            raise TimeoutError(f"Could not generate non-zero output in {attempts} attempts")
        attempts +=1
        noisy_nz_pcts = np.clip(pcts + rng.laplace(0.0, scale, size=len(nz_pcts)), 0.0, 1.0)
        
    noisy_pcts = pcts.copy() # re-normalise
    noisy_pcts.loc[noisy_pcts>0] = noisy_nz_pcts
    return noisy_pcts

