import pandas as pd
import numpy as np
import numpy.random as npr
from numpy.linalg import norm
from datetime import datetime
from math import sqrt
from tick.hawkes import SimuHawkes, HawkesKernelExp

def brownian(n, dt, sigma, radius):
    """
    Generate an instance of two-dimensional Brownian motion.
    
    Arguments
    ---------
    n: int 
        number of points to generate (initial position + n-1 steps)
    dt: float
        time step.
    sigma: float 
        "speed" of the Brownian motion.  The random variable X(t) of the position at 
        time t has a normal distribution with mean 0 and variance sigma^2 * t.
    radius: float 
        radius of the circle that bounds the simulated motion.

    Returns
    -------
    A numpy array of floats with shape (n,2) with initial position (0,0).
    """
    
    out = np.empty((n,2))
    out[0,:] = (0,0)
    
    for t in range(n-1):
            
        r = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=2)
        p = out[t,:] + r
        
        # redraw if new point falls outside circle
        while norm(p) >= radius:
            r = np.random.normal(loc=0, scale=sigma * np.sqrt(dt), size=2)
            p = out[t,:] + r
            
        out[t+1,:] = p
        
    return out

def simulate_traj(stays, moves, seed=None):
    """
    Simulate a trajectory using Brownian motion.

    Parameters
    ----------
    stays : list of tuples
        A list of tuples, each representing a stay. Each tuple consists of three elements:
        - c (numpy array of shape (2,)): The center coordinates of the stay.
        - r (int): The radius of the stay.
        - t (int): The duration of the stay in time units.
    moves : list of int
        A list of integers specifying the duration of each move between stays. The length of this list
        must be one less than the length of `stays`.
    seed : int
        The seed for random number generation.
        
    Returns
    -------
    A numpy array with columns 'x', 'y', 'local_timestamp', 'unix_timestamp', 'identifier'
    """
    
    if seed:
        npr.seed(seed)
    else:
        seed = npr.randint(0,1000,1)[0]
        npr.seed(seed)
        print("Seed:", seed)
        
    traj = np.empty((0,2))
    n_stays = len(stays)
    
    for i in range(n_stays):
    
        (center, radius, time) = stays[i]
        
        stay_traj = brownian(time, 1, sigma=0.5*radius, radius=radius) + center.reshape(1, -1)
        
        if (i+1 < n_stays):
            travel_traj = np.linspace(stay_traj[-1, :], stays[i+1][0], moves[i], axis=0)
            #noise_scale = 10
            #travel_traj = travel_traj + np.random.normal(0, noise_scale, travel_traj.shape)
            traj = np.concatenate((traj, stay_traj, travel_traj), axis=0)
        else:
            traj = np.concatenate((traj, stay_traj), axis=0)
    
    df = pd.DataFrame(traj, columns = ['x', 'y'])
    df['local_timestamp'] = pd.to_datetime([datetime(2022, 1, 1, int(t//60), int(t%60)).isoformat() for t in range(len(traj))])
    df['unix_timestamp'] = (df['local_timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df['identifier'] = 'User X'
    
    return df

def sample_traj(traj, freq, seed=None):
    """
    Sample from simulated trajectory, drawn using a self-exciting Hawkes process.
    
    Parameters
    ----------
    traj: numpy array
        simulated trajectory from simulate_traj
    freq : 0 or 1
        0 = low frequency (average of 3 pings/hour)
        1 = high frequency (average of 12 pings/hour)
    seed : int
        The seed for random number generation.
    """
    
    if seed:
        npr.seed(seed)
    else:
        seed = npr.randint(0,1000,1)[0]
        npr.seed(seed)
        print("Seed:", seed)

    baseline = [2, 9][freq]    # baseline intensity
    alpha = [1/3, 1/4][freq]   # intensity of the kernel
    decay = 4.6                # decay of the kernel
    
    kernel = HawkesKernelExp(alpha, decay)
    hawkes = SimuHawkes(n_nodes=1, end_time=len(traj)/60, verbose=False, seed=int(seed))
    hawkes.set_kernel(0, 0, kernel)
    hawkes.set_baseline(0, baseline)
    hawkes.simulate()
    timestamps = hawkes.timestamps
    
    samples = [int(t) for t in timestamps[0] * 60]
    df = traj.iloc[samples]
    df = df.drop_duplicates('local_timestamp')
    
    return df