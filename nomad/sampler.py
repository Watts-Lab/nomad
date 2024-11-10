import pandas as pd
import numpy as np
import numpy.random as npr
from numpy.linalg import norm
from datetime import datetime


def sample_hier_nhpp(traj, beta_start, beta_durations, beta_ping, dt=1, nu=1/4, seed=0):
    """
    Sample from simulated trajectory, drawn using hierarchical Poisson processes.

    Parameters
    ----------
    traj: numpy array
        simulated trajectory from simulate_traj
    beta_start: float
        mean of Poisson controlling burst start times
    beta_durations: float
        mean of Exponential controlling burst durations
    beta_ping: float
        mean of Poisson controlling ping sampling within a burst
    dt: float
        time step between pings
    nu: float
        sampling noise. Pings are sampled as (true x + eps_x, true y + eps_y)
        where (eps_x, eps_y) ~ N(0, nu/1.96).
    seed : int0
        The seed for random number generation.
    """

    npr.seed(seed)

    # Adjust beta's to account for time step
    beta_start = beta_start / dt
    beta_durations = beta_durations / dt
    beta_ping = beta_ping / dt

    # Sample starting points of bursts
    inter_arrival_times = npr.exponential(scale=beta_start, size=len(traj))
    burst_start_points = np.cumsum(inter_arrival_times).astype(int)
    burst_start_points = burst_start_points[burst_start_points < len(traj)]

    # Sample durations of each burst
    burst_durations = np.random.exponential(scale=beta_durations, size=len(burst_start_points)).astype(int)

    # Create start_points and end_points
    burst_end_points = burst_start_points + burst_durations
    burst_end_points = np.minimum(burst_end_points, len(traj) - 1)

    # Adjust end_points to handle overlaps
    for i in range(len(burst_start_points) - 1):
        if burst_end_points[i] > burst_start_points[i + 1]:
            burst_end_points[i] = burst_start_points[i + 1]

    # Sample pings within each burst
    sampled_trajectories = []
    for start, end in zip(burst_start_points, burst_end_points):
        burst_indices = np.arange(start, end)

        if len(burst_indices) == 0:
            continue

        ping_intervals = np.random.exponential(scale=beta_ping, size=len(burst_indices))
        ping_times = np.unique(np.cumsum(ping_intervals).astype(int))
        ping_times = ping_times[ping_times < (end - start)] + start

        if len(ping_times) == 0:
            continue

        burst_data = traj.iloc[ping_times].copy()

        sampled_trajectories.append(burst_data)

    if sampled_trajectories:
        sampled_traj = pd.concat(sampled_trajectories).sort_values(by='unix_timestamp')
    else:  # empty
        sampled_traj = pd.DataFrame(columns=list(traj.columns))

    sampled_traj = sampled_traj.drop_duplicates('local_timestamp')

    # Add sampling noise
    noise = npr.normal(loc=0, scale=nu/1.96, size=(sampled_traj.shape[0], 2))
    sampled_traj[['x', 'y']] = sampled_traj[['x', 'y']] + noise
    sampled_traj['hor_accuracy'] = np.linalg.norm(noise, axis=1)

    return sampled_traj