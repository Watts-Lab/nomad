import pandas as pd
import numpy as np
import numpy.random as npr
from numpy.linalg import norm
from datetime import datetime
#from tick.hawkes import SimuHawkes, HawkesKernelExp


def brownian(n, dt, sigma, radius, p):
    """
    NOT RELEVANT FOR GARDEN CITY

    Generate an instance of two-dimensional Brownian motion constrained within
    a circle with some probability of staying still.

    The Weiner process is given by X(t) = N(0, sigma^2 * t),
    or iteratively as X(t + dt) = X(t) + N(0, sigma^2 * dt).

    Arguments
    ---------
    n: int
        number of points to generate (initial position + n-1 steps)
    dt: float
        time step.
    sigma: float
        "speed" of the Brownian motion.  The random variable X(t) of the
        position at  time t has a normal distribution with mean 0 and
        variance sigma^2 * t.
    radius: float 
        radius of the circle that bounds the simulated motion.
    p: float, 0 <= p <= 1
        probability of staying still

    Returns
    -------
    A numpy array of floats with shape (n,2) with initial position (0,0).
    """

    out = np.empty((n, 2))
    point = (0, 0)
    out[0, :] = point
    stay_probs = np.random.binomial(1, p, n)

    for t in range(n-1):

        # move with probability 1-p
        if stay_probs[t] == 0:

            r = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=2)
            point = out[t, :] + r

            # redraw if new point falls outside circle
            while norm(point) >= radius:
                r = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=2)
                point = out[t, :] + r

        out[t+1, :] = point

    return out


def simulate_traj(stays, moves, seed=0):
    """
    NOT RELEVANT FOR GARDEN CITY

    Simulate a trajectory using Brownian motion.

    Parameters
    ----------
    stays : list of tuples
        A list of tuples, each representing a stay. Each tuple consists of three elements:
        - c (numpy array of shape (2,)): The center coordinates of the stay.
        - r (int): The radius of the stay.
        - t (int): The duration of the stay in time units.
        - p (float): The probability of staying still (0 <= p <= 1)
    moves : list of int
        A list of integers specifying the duration of each move between stays. The length of this list
        must be one less than the length of `stays`.
    seed : int
        The seed for random number generation.

    Returns
    -------
    A numpy array with columns 'x', 'y', 'local_timestamp', 'unix_timestamp', 'identifier'
    """

    npr.seed(seed)

    traj = np.empty((0, 2))
    n_stays = len(stays)

    for i in range(n_stays):

        (center, radius, time, p_still) = stays[i]

        stay_traj = brownian(time, 1, sigma=radius/1.96, radius=radius, p=p_still) + center.reshape(1, -1)

        if (i+1 < n_stays):
            travel_traj = np.linspace(stay_traj[-1, :], stays[i+1][0], moves[i], axis=0)
            #noise_scale = 10
            #travel_traj = travel_traj + np.random.normal(0, noise_scale, travel_traj.shape)
            traj = np.concatenate((traj, stay_traj, travel_traj), axis=0)
        else:
            traj = np.concatenate((traj, stay_traj), axis=0)

    df = pd.DataFrame(traj, columns=['x', 'y'])
    df['local_timestamp'] = pd.to_datetime([datetime(2022, 1, 1, int(t//60), int(t%60)).isoformat() for t in range(len(traj))])
    df['unix_timestamp'] = (df['local_timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df['identifier'] = 'User X'

    return df


def sample_hier_nhpp(traj, beta_start, beta_durations, beta_ping, nu=1/4, seed=0):
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
    nu: float
        sampling noise. Pings are sampled as (true x + eps_x, true y + eps_y)
        where (eps_x, eps_y) ~ N(0, nu/1.96).
    seed : int0
        The seed for random number generation.
    """

    npr.seed(seed)

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


# def sample_traj(traj, baseline, alpha, decay, nu=20, seed=None):
#     """
#     Sample from simulated trajectory, drawn using a self-exciting Hawkes process.

#     Parameters
#     ----------
#     traj: numpy array
#         simulated trajectory from simulate_traj
#     freq : 0 or 1
#         0 = low frequency (average of 3 pings/hour)
#         1 = high frequency (average of 12 pings/hour)
#     nu: float
#         sampling noise. Pings are sampled as (true x + eps_x, true y + eps_y)
#         where (eps_x, eps_y) ~ N(0, nu/1.96).
#     seed : int0
#         The seed for random number generation.
#     """

#     if seed:
#         npr.seed(seed)
#     else:
#         seed = npr.randint(0, 1000, 1)[0]
#         npr.seed(seed)
#         print("Seed:", seed)

#     #baseline = [2, 9][freq]    # baseline intensity
#     #alpha = [1/3, 1/4][freq]   # intensity of the kernel
#     #decay = 4.6                # decay of the kernel

#     kernel = HawkesKernelExp(alpha, decay)
#     hawkes = SimuHawkes(n_nodes=1, end_time=len(traj)/60, verbose=False, seed=seed)
#     hawkes.set_kernel(0, 0, kernel)
#     hawkes.set_baseline(0, baseline)
#     hawkes.simulate()
#     timestamps = hawkes.timestamps

#     samples = [int(t) for t in timestamps[0] * 60]
#     df = traj.iloc[samples]
#     df = df.drop_duplicates('local_timestamp')

#     # Add sampling noise
#     noise = np.random.normal(loc=0, scale=nu/1.96, size=(df.shape[0], 2))
#     df[['x','y']] = df[['x','y']] + noise

#     return df