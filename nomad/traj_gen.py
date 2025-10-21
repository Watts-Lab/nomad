from tracemalloc import start
import pandas as pd
import numpy as np
import numpy.random as npr
from matplotlib import cm
from shapely.geometry import box, Point, MultiLineString
from shapely.ops import unary_union
from shapely import distance as shp_distance
from datetime import timedelta
from zoneinfo import ZoneInfo
import warnings
import funkybob
from functools import lru_cache
import s3fs
import pyarrow as pa
import pyarrow.dataset as ds

from nomad.io.base import from_df, to_file

from nomad.city_gen import *
from nomad.constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS
from nomad.constants import FAST_STILL_PROBS, SLOW_STILL_PROBS, ALLOWED_BUILDINGS, DEFAULT_STAY_PROBS

def _xy_or_loc_col(col_names, verbose=False):
    if ('x' in col_names and 'y' in col_names):
        return "xy"
    elif 'location' in col_names:
        return "location"
    else:
        if verbose:
            warnings.warn("No trajectory data was found or spatial columns ('x','y', 'location') in keyword arguments.\
                          Agent's home will be used as trajectory starting point.")
        return "missing"

def _datetime_or_ts_col(col_names, verbose=False):
    if 'datetime' in col_names:
        return "datetime"
    elif 'timestamp' in col_names:
        return 'timestamp'
    else:
        if verbose:
            warnings.warn("No trajectory data was found or time columns ('datetime', 'timestamp')\
                          in keyword arguments. '2025-01-01 00:00Z' will be used for starting trajectory time.")
        return "missing"

def parse_agent_attr(attr, N, name):
    """
    Parse agent attribute (homes/workplaces) into a callable that returns the i-th value.
    
    Parameters
    ----------
    attr : str, list, or None
        The attribute value. Can be:
        - None: returns None for all indices
        - str: returns the same string for all indices
        - list: must have length N, returns the i-th element
    N : int
        Expected number of agents
    name : str
        Name of the attribute for error messages
        
    Returns
    -------
    callable
        A function that takes an index i and returns the corresponding attribute value
    """
    if attr is None:
        return lambda i: None
    elif isinstance(attr, str):
        return lambda i: attr
    elif isinstance(attr, list):
        if len(attr) != N:
            raise ValueError(f"{name} must be a list of length {N}, got {len(attr)}")
        return lambda i: attr[i]
    else:
        raise ValueError(f"{name} must be either a string, a list of length {N}, or None")

def sample_hier_nhpp(traj,
                     beta_start=None,
                     beta_durations=None,
                     beta_ping=5,
                     ha=3/4,
                     seed=None,
                     output_bursts=False,
                     deduplicate=True):
    """
    Sample from simulated trajectory, drawn using hierarchical Poisson processes.

    Parameters
    ----------
    traj: numpy array
        simulated trajectory from simulate_traj
    beta_start: float
        scale parameter (mean) of Exponential distribution modeling burst inter-arrival times
        where 1/beta_start is the rate of events (bursts) per minute.
    beta_durations: float
        scale parameter (mean) of Exponential distribution modeling burst durations.
        if beta_start and beta_durations are None, a single burst covering the whole trajectory is used.
    beta_ping: float
        scale parameter (mean) of Exponential distribution modeling ping inter-arrival times
        within a burst, where 1/beta_ping is the rate of events (pings) per minute.
    ha: float
        Mean horizontal-accuracy radius *in 15 m blocks*. The actual per-ping accuracy is random: ha ≥ 8 m/15 m and follows a
        Pareto distribution with that mean.  For each ping the positional error (ε_x, ε_y) is drawn i.i.d. N(0, σ²) with σ = HA / 1.515 so that
        |ε| ≤ HA with 68 % probability.
    seed : int0
        The seed for random number generation.
    output_bursts : bool
        If True, outputs the latent variables on when bursts start and end.
    deduplicate : bool
        If True, sampled times are also discretized to be in ticks
    """
    rng = npr.default_rng(seed)

    # convert minutes→seconds
    beta_ping   = beta_ping   * 60
    if beta_start    is not None: beta_start    *= 60
    if beta_durations is not None: beta_durations *= 60

    # absolute window
    t0   = int(traj['timestamp'].iloc[0])
    t_end = int(traj['timestamp'].iloc[-1])

    # 1) bursts in continuous seconds
    if beta_start is None and beta_durations is None:
        burst_start_points = np.array([0.0])
        burst_end_points   = np.array([t_end - t0], dtype=float)
    else:
        # draw at least 3× the mean number of bursts + 10
        est_n = int(3 * (t_end - t0) / beta_start) + 10
        inter_arrival_times = rng.exponential(scale=beta_start, size=est_n)
        burst_start_points = np.cumsum(inter_arrival_times)
        # keep only those inside the window
        burst_start_points = burst_start_points[burst_start_points < (t_end - t0)]

        # durations
        burst_durations = rng.exponential(scale=beta_durations,
                                          size=burst_start_points.size)
        burst_end_points = burst_start_points + burst_durations

        # handle cases where no bursts are generated
        if burst_end_points.size == 0:
            empty_traj = pd.DataFrame(columns=traj.columns)
            empty_burst_info = pd.DataFrame(columns=['start_time','end_time'])
            if output_bursts:
                return empty_traj, empty_burst_info
            return empty_traj

        # forbid overlap: each burst_end ≤ next burst_start
        burst_end_points[:-1] = np.minimum(burst_end_points[:-1], burst_start_points[1:])
        # clip last end point
        burst_end_points[-1] = min(burst_end_points[-1], t_end - t0)

    # 2) pings continuously
    ping_times = []
    burst_info = []
    tz = traj['datetime'].dt.tz

    for start, end in zip(burst_start_points, burst_end_points):
        if output_bursts:
            burst_info.append([
                pd.to_datetime(t0 + start, unit='s', utc=True)
                  .tz_convert(tz),
                pd.to_datetime(t0 + end, unit='s', utc=True)
                  .tz_convert(tz)
            ])

        dur = end - start
        if dur <= 0:
            continue

        # oversample ping intervals, then clip
        est_pings = int(3 * dur / beta_ping) + 10
        ping_intervals = rng.exponential(scale=beta_ping, size=est_pings)
        times_rel = np.cumsum(ping_intervals)
        times_rel = times_rel[times_rel < dur]

        ping_times.append(t0 + start + times_rel)

    if not ping_times:
        empty = pd.DataFrame(columns=traj.columns)
        if output_bursts:
            return empty, pd.DataFrame(burst_info, columns=['start_time','end_time'])
        return empty

    ping_times = np.concatenate(ping_times).astype(int)

    # 3) map to last tick via two-index searchsorted
    traj_ts = traj['timestamp'].to_numpy()
    idx = np.searchsorted(traj_ts, ping_times, side='right') - 1
    valid = idx >= 0
    idx = idx[valid]
    ping_times = ping_times[valid]

    if deduplicate:
        _, keep = np.unique(idx, return_index=True)
        idx = idx[keep]
        ping_times = ping_times[keep]

    sampled_traj = traj.iloc[idx].copy()
    sampled_traj['timestamp'] = ping_times
    sampled_traj['datetime'] = (
        pd.to_datetime(ping_times, unit='s', utc=True)
          .tz_convert(tz)
    )

    # realized horizontal accuracy

    x_m = 8/15
    if ha <= x_m:
        raise ValueError("ha must exceed 8 m / 15 m ≈ 0.533 blocks")
    alpha = ha / (ha - x_m)
    n = len(sampled_traj)
    ha_realized = (rng.pareto(alpha, size=n) + 1) * x_m
    ha_realized = np.minimum(ha_realized, 20, out=ha_realized) # no unrealistic ha (in blocks)
    sampled_traj['ha'] = ha_realized    
    sigma = ha_realized / 1.515
    # spatial noise
    noise = rng.standard_normal((n, 2)) * sigma[:, None]
    np.clip(noise, -250, 250, out=noise)
    sampled_traj[['x', 'y']] += noise

    if output_bursts:
        burst_info = pd.DataFrame(burst_info, columns=['start_time','end_time'])
        return sampled_traj, burst_info

    return sampled_traj


# =============================================================================
# AGENT CLASS
# =============================================================================


class Agent:
    """
    Represents an agent in the city simulation.

    Attributes
    ----------
    still_probs : dict
        Dictionary containing probabilities of the agent staying still.
    speeds : dict
        Dictionary containing possible speeds of the agent.
    dt : float
        Time step duration.

    Methods
    -------
    plot_traj
        Plots the trajectory of the agent on the given axis.
    sample_traj_hier_nhpp
        Samples a sparse trajectory using a hierarchical non-homogeneous Poisson process.
    """

    def __init__(self, 
                 identifier: str, 
                 city: City,
                 home: str = None,
                 workplace: str = None,
                 still_probs: dict = DEFAULT_STILL_PROBS, 
                 speeds: dict = DEFAULT_SPEEDS,
                 destination_diary: pd.DataFrame = None,
                 trajectory: pd.DataFrame = None,
                 diary: pd.DataFrame = None,
                 seed: int = 0):
        """
        Initializes an agent in the city simulation with a trajectory and diary.
        If `trajectory` is not provided, the agent initialize with a ping at their home.

        Parameters
        ----------
        identifier : str
            Name of the agent.
        home : str
            Building ID representing the home location.
        workplace : str
            Building ID representing the workplace location.
        city : City
            The city object containing relevant information about the city's layout and properties.
        still_probs : dict, optional (default=DEFAULT_STILL_PROBS)
            Dictionary containing probabilities of the agent staying still.
        speeds : dict, optional (default=DEFAULT_SPEEDS)
            Dictionary containing possible speeds of the agent.
        destination_diary : pandas.DataFrame, optional (default=None)
            DataFrame containing the following columns: 'timestamp', 'datetime', 'duration', 'location'.
        trajectory : pandas.DataFrame, optional (default=None)
            DataFrame containing the following columns: 'x', 'y', 'datetime', 'timestamp', 'identifier'.
        diary : pandas.DataFrame,  optional (default=None)
            DataFrame containing the following columns: 'timestamp', 'datetime', 'duration', 'location'.
        dt : float, optional (default=1)
            Time step duration.
        """

        rng = npr.default_rng(seed)
        
        self.identifier = identifier
        self.city = city

        if home is None:
            home = city.building_types[city.building_types['type'] == 'home'].sample(n=1, random_state=rng)['id'].iloc[0]
        if workplace is None:
            workplace = city.building_types[city.building_types['type'] == 'work'].sample(n=1, random_state=rng)['id'].iloc[0]

        self.home = home
        self.workplace = workplace

        self.still_probs = still_probs
        self.speeds = speeds
        self.visit_freqs = None

        self.destination_diary = destination_diary if destination_diary is not None else pd.DataFrame(
            columns=['datetime', 'timestamp', 'duration', 'location'])
        self.trajectory = trajectory
        self.dt = None
        self.diary = diary if diary is not None else pd.DataFrame(
            columns=['datetime', 'timestamp', 'duration', 'location', 'identifier'])
        self.last_ping = trajectory.iloc[-1] if (trajectory is not None) else None
        self.sparse_traj = None


    def reset_trajectory(self, trajectory = True, sparse = True, last_ping = True, diary = True):
        """
        Resets the agent's trajectories and diaries to the initial state. 
        Keeps the agent's identifier, home, and workplace.
        This method is useful for reinitializing the agent after a simulation run.
        """
        self.destination_diary = pd.DataFrame(columns=self.destination_diary.columns)
        self.dt = None
        
        if trajectory:
            self.trajectory = None
        if diary:
            self.diary = pd.DataFrame(columns=self.diary.columns)
        if last_ping:
            self.last_ping = None
        if sparse:
            self.sparse_traj = None

    def plot_traj(self,
                  ax,
                  color='black',
                  alpha=1,
                  doors=True,
                  address=True,
                  heatmap=False):
        """
        Plots the trajectory of the agent on the given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the trajectory.
        color : str, optional
            The color of the trajectory.
        alpha : float, optional
            The transparency of the trajectory.
        doors : bool, optional
            Whether to plot doors of buildings.
        address : bool, optional
            Whether to plot the address of buildings.
        heatmap : bool, optional
            Whether to plot a heatmap of time spent in each building.
        """

        if heatmap:
            self.city.plot_city(ax, doors=doors, address=address, zorder=1, heatmap_agent=self)
        else:
            ax.scatter(self.trajectory.x, self.trajectory.y, s=6, color=color, alpha=alpha, zorder=2)
            self.city.plot_city(ax, doors=doors, address=address, zorder=1)

    # def _sample_step(self, start_point, dest_building, dt, rng):
    #     """
    #     From a destination diary, generates (x, y) pings.

    #     Parameters
    #     ----------
    #     start_point : tuple
    #         The coordinates of the current position as a tuple (x, y).
    #     dest_building : Building
    #         The destination building of the agent.
    #     dt : float
    #         The time step duration.
    #     rng : numpy.random.generator
    #         random number generator for reproducibility.

    #     Returns
    #     -------
    #     coord : numpy.ndarray
    #         A numpy array of floats with shape (1, 2) representing the new coordinates.
    #     location : str or None
    #         The building ID if the step is a stay, or `None` if the step is a move.
    #     """
    #     city = self.city

    #     start_block = np.floor(start_point)
    #     start_geometry = city.get_block(tuple(start_block))

    #     curr = np.array(start_point)

    #     if start_geometry == dest_building or start_point == dest_building.door_centroid:
    #         location = dest_building.id
    #         p = self.still_probs[dest_building.building_type]
    #         sigma = self.speeds[dest_building.building_type]

    #         if rng.uniform() < p:
    #             coord = curr
    #         else: # Draw until coord falls inside building
    #             while True:
    #                 coord = rng.normal(loc=curr, scale=sigma*np.sqrt(dt), size=2)
    #                 if dest_building.geometry.contains(Point(coord)):
    #                     break
    #     else: # Agent travels to building along the streets
    #         location = None
    #         dest_point = dest_building.door

    #         if start_geometry in city.buildings.values():
    #             start_segment = [start_point, start_geometry.door_centroid]
    #             start = start_geometry.door
    #         else:
    #             start_segment = []
    #             start = tuple(start_block.astype(int))

    #         street_path = city.shortest_paths[start][dest_point]
    #         path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
    #         path = start_segment + path + [dest_building.door_centroid]
    #         path_ml = MultiLineString([path])
    #         path_length = path_ml.length

    #         # Bounding polygon: needs to stay in street
    #         street_poly = unary_union([city.get_block(block).geometry for block in street_path])
    #         bound_poly = unary_union([start_geometry.geometry, street_poly])
            
    #         # Transformed coordinates of current position
    #         path_coord = _path_coords(path_ml, start_point)

    #         heading_drift = 3.33 * dt
    #         sigma = 0.5 * dt / 1.96

    #         while True:
    #             # Step in transformed (path-based) space
    #             step = rng.normal(loc=[heading_drift, 0], scale=sigma * np.sqrt(dt), size=2)
    #             path_coord = (path_coord[0] + step[0], 0.7 * path_coord[1] + step[1])

    #             if path_coord[0] > path_length:
    #                 coord = np.array(dest_building.geometry.centroid.coords[0])
    #                 break

    #             coord = _cartesian_coords(path_ml, *path_coord)

    #             if bound_poly.contains(Point(coord)):
    #                 break

    #     return coord, location

    def _sample_step(self, start_point, dest_building, dt, rng):
        """Lazy version that computes paths on demand."""
        city = self.city
        start_block = np.floor(start_point)
        start_geometry = city.get_block(tuple(start_block))
        curr = np.array(start_point)
    
        if start_geometry == dest_building or start_point == dest_building.door_centroid:
            # Same building logic - unchanged
            location = dest_building.id
            p = self.still_probs[dest_building.building_type]
            sigma = self.speeds[dest_building.building_type]
    
            if rng.uniform() < p:
                coord = curr
            else:
                while True:
                    coord = rng.normal(loc=curr, scale=sigma*np.sqrt(dt), size=2)
                    if dest_building.geometry.contains(Point(coord)):
                        break
        else:
            # Agent travels to building
            location = None
            dest_point = dest_building.door

            if start_geometry in city.buildings.values():
                start_segment = [start_point, start_geometry.door_centroid]
                start = start_geometry.door
            else:
                start_segment = []
                start = tuple(start_block.astype(int))

            street_path = self._manhattan_street_path(start, dest_point)

            path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
            path = start_segment + path + [dest_building.door_centroid]
            path_ml = MultiLineString([path])
            path_length = path_ml.length
    
            street_poly = unary_union([city.get_block(block).geometry for block in street_path])
            bound_poly = unary_union([start_geometry.geometry, street_poly])
            
            path_coord = _path_coords(path_ml, start_point)
            heading_drift = 3.33 * dt
            sigma = 0.5 * dt / 1.96
    
            while True:
                step = rng.normal(loc=[heading_drift, 0], scale=sigma * np.sqrt(dt), size=2)
                path_coord = (path_coord[0] + step[0], 0.7 * path_coord[1] + step[1])
    
                if path_coord[0] > path_length:
                    coord = np.array(dest_building.geometry.centroid.coords[0])
                    break
    
                coord = _cartesian_coords(path_ml, *path_coord)
    
                if bound_poly.contains(Point(coord)):
                    break
    
        return coord, location
    
    def _manhattan_street_path(self, start, dest):
        """Generate Manhattan distance path between two street coordinates."""
        x1, y1 = start
        x2, y2 = dest
        
        path = [(x1, y1)]
        
        # Move horizontally first
        while x1 != x2:
            x1 += 1 if x2 > x1 else -1
            path.append((x1, y1))
        
        # Then move vertically
        while y1 != y2:
            y1 += 1 if y2 > y1 else -1
            path.append((x1, y1))
        
        return path

    def _traj_from_dest_diary(self, dt, seed=0):
        """
        Simulate a trajectory and give agent true travel diary attribute.

        Parameters
        ----------
        dt : float
            The time step duration.

        Returns
        -------
        None (updates self.trajectory, self.diary)
        """

        rng = np.random.default_rng(seed) # random generator for steps
        city = self.city
        destination_diary = self.destination_diary

        trajectory_update = []

        if self.diary.empty:
            current_entry = None
        else:
            current_entry = self.diary.iloc[-1].to_dict()
            self.diary = self.diary.iloc[:-1]

        tick_secs = int(60*dt)

        entry_update = []
        for i in range(destination_diary.shape[0]):
            destination_info = destination_diary.iloc[i]
            duration = int(destination_info['duration'] * 1/dt)
            building_id = destination_info['location']

            duration_in_ticks = int(destination_info['duration'] / dt)
            for _ in range(duration_in_ticks):
                prev_ping = self.last_ping
                start_point = (prev_ping['x'], prev_ping['y'])
                dest_building = city.buildings[building_id]
                unix_timestamp = prev_ping['timestamp'] + tick_secs
                datetime = prev_ping['datetime'] + timedelta(seconds=tick_secs)               
                coord, location = self._sample_step(start_point, dest_building, dt, rng)
                ping = {'x': coord[0], 
                        'y': coord[1],
                        'datetime': datetime,
                        'timestamp': unix_timestamp,
                        'user_id': self.identifier}

                trajectory_update.append(ping)
                self.last_ping = ping
                if current_entry == None:
                    current_entry = {'datetime': datetime,
                                     'timestamp': unix_timestamp,
                                     'duration': dt,
                                     'location': location,
                                     'user_id': self.identifier}
                elif (current_entry['location'] != location):
                    entry_update.append(current_entry)
                    current_entry = {'datetime': datetime,
                                     'timestamp': unix_timestamp,
                                     'duration': dt,
                                     'location': location,
                                     'user_id': self.identifier}
                else:
                    current_entry['duration'] += 1*dt #add one tick to the duration

        if self.trajectory is None:
            self.trajectory = pd.DataFrame(trajectory_update)
        else:
            self.trajectory = pd.concat([self.trajectory, pd.DataFrame(trajectory_update)],
                                    ignore_index=True)

        entry_update.append(current_entry)
        if (self.diary.empty):
            self.diary = pd.DataFrame(entry_update)
        else:
            self.diary = pd.concat([self.diary, pd.DataFrame(entry_update)], ignore_index=True)
        self.destination_diary = destination_diary.drop(destination_diary.index)

    def _generate_dest_diary(self, 
                             end_time, 
                             epr_time_res = 15,
                             stay_probs = DEFAULT_STAY_PROBS,
                             rho = 0.4, 
                             gamma = 0.3, 
                             seed = 0):
        """
        Exploration and preferential return.

        Parameters
        ----------
        end_time : pd.Timestamp
            The end time to generate the destination diary until.
        epr_time_res : int
            The granularity of destination durations in epr generation.
        stay_probs : dict
            Dictionary containing the probability of staying in the same building.
            This is modeled as a geometric distribution with `p = 1 - ((1/avg_duration_hrs)/timesteps_in_1_hr)`.
        rho : float
            Parameter for exploring, influencing the probability of exploration.
        gamma : float
            Parameter for exploring, influencing the probability of preferential return.
        seed : int
            Random seed for reproducibility.
        """
        rng = npr.default_rng(seed)

        id2door = pd.DataFrame([[s, b.door] for s, b in self.city.buildings.items()],
                               columns=['id', 'door']).set_index('door')  # could this be a field of city?

        if end_time.tz is None:
            tz = getattr(self.last_ping['datetime'], 'tz', None)
            if tz is not None:
                end_time.tz_localize(tz)
                warnings.warn(
                    f"The end_time input is timezone-naive. Assuming it is in {tz}.")

        if isinstance(end_time, pd.Timestamp):
            end_time = int(end_time.timestamp())  # Convert to unix

        # Create visit frequency table is user does not already have one
        visit_freqs = self.visit_freqs
        if visit_freqs is None:
            visit_freqs = pd.DataFrame({
                'id': list(self.city.buildings.keys()),
                'type': [b.building_type for b in self.city.buildings.values()],
                'freq': 0,
                'p': 0
            }).set_index('id')

            # Initializes past counts randomly
            visit_freqs.loc[self.home, 'freq'] = 20
            visit_freqs.loc[self.workplace, 'freq'] = 20
            visit_freqs.loc[visit_freqs.type == 'park', 'freq'] = 5

            initial_locs = []
            initial_locs += list(rng.choice(visit_freqs.loc[visit_freqs.type == 'retail'].index, size=rng.poisson(4)))
            initial_locs += list(rng.choice(visit_freqs.loc[visit_freqs.type == 'work'].index, size=rng.poisson(2)))
            initial_locs += list(rng.choice(visit_freqs.loc[visit_freqs.type == 'home'].index, size=rng.poisson(2)))
            visit_freqs.loc[initial_locs, 'freq'] += 2

        if self.destination_diary.empty:
            start_time_local = self.last_ping['datetime']
            start_time = self.last_ping['timestamp']
            curr = self.city.get_block((self.last_ping['x'], self.last_ping['y'])).id  # Always a building?? Could be street
        else:
            last_entry = self.destination_diary.iloc[-1]
            start_time_local = last_entry.datetime + timedelta(minutes=int(last_entry.duration))
            start_time = last_entry.unix_timestamp + last_entry.duration*60
            curr = last_entry.location

        dest_update = []
        while start_time < end_time:
            curr_type = visit_freqs.loc[curr, 'type']
            allowed = allowed_buildings(start_time_local)
            x = visit_freqs.loc[(visit_freqs['type'].isin(allowed)) & (visit_freqs.freq > 0)]

            S = len(x) # Fix depending on whether "explore" should depend only on allowed buildings

            #probability of exploring
            p_exp = rho*(S**(-gamma))

            # Stay
            if (curr_type in allowed) & (rng.uniform() < stay_probs[curr_type]):
                pass

            # Exploration
            elif rng.uniform() < p_exp:
                visit_freqs['p'] = self.city.gravity.xs(
                    self.city.buildings[curr].door, level=0).join(id2door, how='right').set_index('id')
                y = visit_freqs.loc[(visit_freqs['type'].isin(allowed)) & (visit_freqs.freq == 0)]

                if not y.empty and y['p'].sum() > 0:
                    curr = rng.choice(y.index, p=y['p']/y['p'].sum())
                else:
                    # If there are no more buildings to explore, then preferential return
                    curr = rng.choice(x.index, p=x['freq']/x['freq'].sum())

                visit_freqs.loc[curr, 'freq'] += 1

            # Preferential return
            else:
                curr = rng.choice(x.index, p=x['freq']/x['freq'].sum())
                visit_freqs.loc[curr, 'freq'] += 1

            # Update destination diary
            entry = {'datetime': start_time_local,
                     'timestamp': start_time,
                     'duration': epr_time_res,
                     'location': curr}
            dest_update.append(entry)

            start_time_local = start_time_local + timedelta(minutes=int(epr_time_res))
            start_time = start_time + epr_time_res*60 # because start_time in seconds

        if self.destination_diary.empty:
            self.destination_diary = pd.DataFrame(dest_update)
        else:
            self.destination_diary = pd.concat(
                [self.destination_diary, pd.DataFrame(dest_update)], ignore_index=True)
        self.destination_diary = condense_destinations(self.destination_diary)

        self.visit_freqs = visit_freqs

        return None

    def generate_trajectory(self,
                            destination_diary= None,
                            end_time=None, 
                            epr_time_res=15,
                            dt=1,
                            seed=0,
                            step_seed=None,
                            verbose=False,
                            **kwargs):
        """
        Generate a trajectory for an agent.

        Parameters
        ----------
        destination_diary : pandas.DataFrame, optional (default=None)
            DataFrame containing the following columns: 'timestamp', 'datetime', 'duration', 'location'.
        end_time : pd.Timestamp, optional
            The end time to generate the trajectory until.
        epr_time_res : int, optional
            The granularity of destination durations in epr generation.
        seed : int, optional
            Random seed for reproducibility.
        kwargs : dict, optional
            Additional keyword arguments for trajectory generation. 
            Can include 'x', 'y', 'datetime', 'timestamp', 'tz'
            These are used to set the initial position of the agent.
        
        Returns
        -------
        None (updates self.trajectory)
        """

        if self.dt is None:
            self.dt = dt
        if self.dt != dt:
            raise ValueError(f"dt ({dt}) does not match the agent's dt ({self.dt}).")

        # handle destination diary
        if destination_diary is not None:
            self.destination_diary = destination_diary
            # warning for overwriting agent's destination diary if it exists?

            loc = destination_diary.iloc[0]['location']
            loc_centroid = self.city.buildings[loc].geometry.centroid
            x_coord, y_coord = loc_centroid.x, loc_centroid.y
            datetime = destination_diary.iloc[0]['datetime']
            unix_timestamp = int(datetime.timestamp())
            self.last_ping = pd.Series({
                'x': x_coord,
                'y': y_coord,
                'datetime': datetime,
                'timestamp': unix_timestamp,
                'user_id': self.identifier
                })
            self.trajectory = pd.DataFrame([self.last_ping])

        # ensure last ping
        if self.trajectory is None:
            if _xy_or_loc_col(kwargs.keys(), verbose) == "location":
                loc_centroid = self.city.buildings[kwargs['location']].geometry.centroid
                x_coord, y_coord = loc_centroid.x, loc_centroid.y
            elif _xy_or_loc_col(kwargs.keys(), verbose) == "xy":
                x_coord, y_coord = kwargs['x'], kwargs['y']
            else:
                loc_centroid = self.city.buildings[self.home].geometry.centroid
                x_coord, y_coord = loc_centroid.x, loc_centroid.y

            if _datetime_or_ts_col(kwargs.keys(), verbose) == "datetime":
                datetime = kwargs['datetime']
                if not isinstance(datetime, pd.Timestamp):
                    try:
                        datetime = pd.to_datetime(datetime)
                    except Exception as e:
                        raise ValueError(f"datetime is not of a convertible type: {e}")
                if datetime.tz is None and 'tz' in kwargs:
                    datetime = datetime.tz_localize(kwargs['tz'])
                unix_timestamp = int(datetime.timestamp())
            elif _datetime_or_ts_col(kwargs.keys(), verbose) == 'timestamp':
                unix_timestamp = kwargs['timestamp']
                if 'tz' in kwargs:
                    datetime = pd.to_datetime(unix_timestamp, unit='s', utc=True).tz_convert(kwargs['tz'])
                datetime = pd.to_datetime(unix_timestamp, unit='s')
            else:
                datetime = pd.to_datetime('2025-01-01 00:00Z')
                unix_timestamp = int(datetime.timestamp())

            self.last_ping = pd.Series({
                'x': x_coord,
                'y': y_coord,
                'datetime': datetime,
                'timestamp': unix_timestamp,
                'user_id': self.identifier
                })

        else:
            if ('x' in kwargs)or('y' in kwargs)or('datetime'in kwargs)or('timestamp' in kwargs):
                raise ValueError(
                    "Keywords arguments conflict with existing trajectory or destination diary,\
                    use Agent.reset_trajectory() or do not provide keyword arguments"
                )
            self.last_ping = self.trajectory.iloc[-1]

        if self.destination_diary.empty:
            if end_time is None:
                raise ValueError(
                    "Destination diary is empty. Provide an end_time to generate a trajectory."
                )
            self._generate_dest_diary(end_time=end_time,
                                      epr_time_res=epr_time_res,
                                      seed=seed)

        if step_seed:
            self._traj_from_dest_diary(dt=dt, seed=step_seed)
        else:
            self._traj_from_dest_diary(dt=dt, seed=seed)

        return None

    def sample_trajectory(self,
                          beta_start=None,
                          beta_durations=None,
                          beta_ping=5,
                          seed=0,
                          ha=3/4,
                          dt=None,
                          output_bursts=False,
                          deduplicate=True,
                          replace_sparse_traj=False,
                          cache_traj=False):
        """
        Samples a sparse trajectory using a hierarchical non-homogeneous Poisson process.

        Parameters
        ----------
        beta_start : float
            The rate parameter governing the Poisson Process controlling 
            the start of the trajectory.
        beta_durations : float
            The rate parameter governing the Exponential controlling 
            for the durations of the trajectory.
        beta_ping : float
            The rate parameter governing the Poisson Process controlling
            which pings are sampled.
        seed : int
            Random seed for reproducibility.
        ha : float
            Horizontal accuracy
        output_bursts : bool
            If True, outputs the latent variables on when bursts start and end.
        replace_sparse_traj : bool
            if True, replaces existing sparse_traj field with the new sparsified trajectory
            rather than appending.
        cache_traj : bool
            if True, empties the Agent's trajectory DataFrame.
        """
        if not self.trajectory.timestamp.is_monotonic_increasing:
            raise ValueError("The input trajectory is not sorted chronologically.")
            
        result = sample_hier_nhpp(
            self.trajectory, 
            beta_start, 
            beta_durations, 
            beta_ping, 
            ha=ha,
            seed=seed, 
            output_bursts=output_bursts,
            deduplicate=deduplicate
        )

        if output_bursts:
            sparse_traj, burst_info = result
        else:
            sparse_traj = result

        if not sparse_traj.timestamp.is_monotonic_increasing:
            raise ValueError("The sampled trajectory is not sorted chronologically.")
            
        sparse_traj = sparse_traj.set_index('timestamp', drop=False)

        if self.sparse_traj is None or replace_sparse_traj:
            self.sparse_traj = sparse_traj
        else:
            self.sparse_traj = pd.concat([self.sparse_traj, sparse_traj], ignore_index=False)
            if not self.sparse_traj.timestamp.is_monotonic_increasing:
                raise ValueError("The aggregated sampled trajectory is not sorted chronologically.")

        if cache_traj:
            self.last_ping = self.trajectory.iloc[-1]
            self.trajectory = self.trajectory.loc[[]]  # empty df

        if output_bursts:
            return burst_info



def _cartesian_coords(multilines, distance, offset, eps=0.001):
    """
    Converts path-based coordinates (distance along path, signed perpendicular offset)
    into cartesian coordinates on the plane.

    Parameters
    ----------
    multilines : shapely.geometry.MultiLineString
        MultiLineString representing the street path.
    distance : float
        Distance along the path.
    offset : float
        Signed perpendicular offset from the path (positive to the left, negative to the right).
    eps : float, optional
        Small delta used to estimate the path's tangent direction.

    Returns
    -------
    tuple
        Cartesian coordinates (x, y) corresponding to the input path-based coordinates.
    """
    point_on_path = multilines.interpolate(distance)
    offset_point = multilines.interpolate(distance - eps)

    p = np.array([point_on_path.x, point_on_path.y])
    q = np.array([offset_point.x, offset_point.y])
    direction = p - q
    unit_direction = direction / np.linalg.norm(direction)

    # Rotate 90° counter-clockwise to get the normal vector
    normal = np.flip(unit_direction) * np.array([-1, 1])

    return tuple(p + offset * normal)

def _path_coords(multilines, point, eps=0.001):
    """
    Given a MultiLineString and a cartesian point, returns the transformed coordinates:
    distance along the path and signed perpendicular offset.

    Parameters
    ----------
    multilines : shapely.geometry.MultiLineString
        MultiLineString representing the street path.
    point : shapely.geometry.Point or tuple
        The cartesian point to transform.
    eps : float, optional
        Small delta used to estimate the path's tangent direction.

    Returns
    -------
    tuple
        (distance_along_path, orthogonal_offset)
    """
    if not isinstance(point, Point):
        point = Point(point)

    distance = multilines.project(point)
    point_on_path = multilines.interpolate(distance)
    offset_point = multilines.interpolate(distance - eps)

    p = np.array([point_on_path.x, point_on_path.y])
    q = np.array([offset_point.x, offset_point.y])
    direction = p - q
    unit_direction = direction / np.linalg.norm(direction)

    # Rotate 90° counter-clockwise to get the normal vector
    normal = np.flip(unit_direction) * np.array([-1, 1])

    delta = np.array([point.x - p[0], point.y - p[1]])
    offset = np.dot(delta, normal)

    return distance, offset

def condense_destinations(destination_diary):
    """
    Modifies a sequence of timestamped destinations, joining consecutive 
    destinations in the same location into a single entry with the aggregated duration.

    Parameters
    ----------
    destination_diary : pandas.DataFrame
        DataFrame containing timestamped locations the user is heading towards.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with condensed destination entries.
    """

    if destination_diary.empty:
        return pd.DataFrame()

    # Detect changes in location
    destination_diary['new_segment'] = destination_diary['location'].ne(destination_diary['location'].shift())

    # Create segment identifiers for grouping
    destination_diary['segment_id'] = destination_diary['new_segment'].cumsum()
    # Aggregate data by segment
    condensed_df = destination_diary.groupby('segment_id').agg({
        'datetime': 'first',
        'timestamp': 'first',
        'duration': 'sum',
        'location': 'first'
    }).reset_index(drop=True)

    return condensed_df

# =============================================================================
# POPULATION
# =============================================================================

class Population:
    """
    A class to represent a population of agents within a city.
    Contains methods to initialize agents and randomize their attributes and trajectories.

    Attributes
    ----------
    roster : dict
        A dictionary to store agents with their identifiers as keys.
    city : City
        The city in which the population resides.
    dt : float
        The time step duration for the agents.

    Methods
    -------
    add_agent:
        Adds an agent to the population.
    generate_agents:
        Generates N agents with randomized attributes.
    save_pop:
        Saves trajectories, homes, and diaries as Parquet files to S3.
    sample_step:
        Generates (x, y) pings from a destination diary.
    traj_from_dest_diary:
        Simulates a trajectory and updates the agent's travel diary.
    generate_dest_diary:
        Generates a destination diary using exploration and preferential return.
    generate_trajectory:
        Generates a trajectory for an agent.
    plot_population:
        Plots the population on a given axis.
    """

    def __init__(self, 
                 city: City,
                 dt: float = 1):
        self.roster = {}
        self.city = city
        self.dt = dt

    def add_agent(self,
                  agent: Agent,
                  verbose: bool=True):
        """
        Adds an agent to the population. 
        If the agent identifier already exists in the population, it will be replaced.

        Parameters
        ----------
        agent : Agent
            The agent to be added to the population.
        verbose : bool, optional
            If True, prints a message if the agent identifier already exists in the population.
        """

        if verbose and agent.identifier in self.roster:
            print("Agent identifier already exists in population. Replacing corresponding agent.")
        self.roster[agent.identifier] = agent

    def generate_agents(self, N, seed=0, name_count=2, agent_homes=None, agent_workplaces=None):
        """
        Generates N agents, with randomized attributes.
        """
        master_rng = np.random.default_rng(seed)
        
        name_seed = int(master_rng.integers(0, 2**32))
        generator = funkybob.UniqueRandomNameGenerator(members=name_count, seed=seed)

        # Create efficient accessors for agent homes and workplaces
        get_home = parse_agent_attr(agent_homes, N, "agent_homes")
        get_workplace = parse_agent_attr(agent_workplaces, N, "agent_workplaces")
            
        for i in range(N):
            agent_seed = int(master_rng.integers(0, 2**32))
            identifier = generator[i]              
            agent = Agent(identifier=identifier,
                          city=self.city,
                          home=get_home(i),
                          workplace=get_workplace(i),
                          seed=agent_seed)
            self.add_agent(agent)

    def save_pop(self,
                 traj_cols=None,
                 sparse_path=None,
                 full_path=None,
                 homes_path=None,
                 diaries_path=None,
                 partition_cols=None,
                 mixed_timezone_behavior="naive",
                 filesystem=None,
                 fmt='parquet',
                 **kwargs):
        """
        Save trajectories, homes, and diaries to local or S3 destinations.
    
        Parameters
        ----------
        sparse_path : str or Path, optional
            Destination path for sparse trajectories.
        full_path : str or Path, optional
            Destination path for full (ground truth) trajectories.
        homes_path : str or Path, optional
            Destination path for the homes table.
        diaries_path : str or Path, optional
            Destination path for diaries.
        partition_cols : list of partition column names.
        filesystem : pyarrow.fs.FileSystem or None
            Optional filesystem object (e.g., s3fs.S3FileSystem). If None, inferred automatically.
        **kwargs : dict, optional
            Additional static columns to include in the homes table. Each key-value pair
            represents a column name and its values. Values must be a list/array of length N
            (number of agents) or a single value to be repeated for all agents.
        """
        if full_path:
            full_df = pd.concat([agent.trajectory for agent in self.roster.values()], ignore_index=True)
            full_df = from_df(full_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(full_df,
                    path=full_path,
                    format=fmt,
                    partition_by=partition_cols,
                    filesystem=filesystem,
                    existing_data_behavior='delete_matching')
    
        if sparse_path:
            sparse_df = pd.concat([agent.sparse_traj for agent in self.roster.values()], ignore_index=True)
            sparse_df = from_df(sparse_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(sparse_df,
                    path=sparse_path,
                    format=fmt,
                    partition_by=partition_cols,
                    filesystem=filesystem,
                    existing_data_behavior='delete_matching',
                    traj_cols=traj_cols)
    
        if diaries_path:
            diaries_df = pd.concat([agent.diary for agent in self.roster.values()], ignore_index=True)
            diaries_df = from_df(diaries_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(diaries_df,
                    path=diaries_path,
                    format=fmt,
                    partition_by=partition_cols,
                    filesystem=filesystem,
                    existing_data_behavior='delete_matching',
                    traj_cols=traj_cols)
    
        if homes_path:
            homes_df = self._build_agent_static_data(**kwargs)
            
            table = pa.Table.from_pandas(homes_df, preserve_index=False)
            ds.write_dataset(table,
                             base_dir=str(homes_path),
                             format=fmt,
                             partitioning_flavor='hive',
                             filesystem=filesystem,
                             existing_data_behavior='delete_matching')

    def _build_agent_static_data(self, **static_columns):
        """Build DataFrame with agent static data (user_id, homes, workplaces, user-level attributes)."""
        N = len(self.roster)
        
        # Process static columns
        processed_static = self._process_static_columns(static_columns, N)
        
        # Build base data
        base_data = []
        for agent_id, agent in self.roster.items():
            ts = agent.last_ping['datetime']
            iso_date = ts.date().isoformat()
            base_data.append({
                'user_id': agent_id,
                'home': agent.home,
                'workplace': agent.workplace,
                'date': iso_date
            })
        
        # Create base DataFrame
        homes_df = pd.DataFrame(base_data)
        
        # Add static columns
        for col_name, col_values in processed_static.items():
            homes_df[col_name] = col_values
            
        return homes_df
    
    def _process_static_columns(self, static_columns, N):
        """Process static columns, validating lengths and handling single values."""
        processed = {}
        
        for col_name, col_values in static_columns.items():
            if isinstance(col_values, (list, tuple, np.ndarray)):
                if len(col_values) != N:
                    raise ValueError(f"Static column '{col_name}' has length {len(col_values)}, "
                                   f"but expected length {N} (number of agents)")
                processed[col_name] = col_values
            else:
                # Single value - repeat for all agents
                processed[col_name] = [col_values] * N
                
        return processed

    def reproject_to_mercator(self, sparse_traj=True, full_traj=False, diaries=False,
                             block_size=15, false_easting=-4265699, false_northing=4392976,
                             poi_data=None):
        """
        Reproject all agent trajectories from Garden City coordinates to Web Mercator.
        
        Parameters
        ----------
        sparse_traj : bool, default True
            Whether to reproject sparse trajectories
        full_traj : bool, default False
            Whether to reproject full trajectories  
        diaries : bool, default False
            Whether to reproject diaries (must have x, y columns)
        block_size : float, default 15
            Size of one city block in meters
        false_easting : float, default -4265699
            False easting offset for Garden City
        false_northing : float, default 4392976
            False northing offset for Garden City
        poi_data : pd.DataFrame, optional
            DataFrame with building coordinates (building_id, x, y) to join with diaries
        """
        for agent in self.roster.values():
            if sparse_traj and agent.sparse_traj is not None:
                agent.sparse_traj = garden_city_to_mercator(
                    agent.sparse_traj, block_size=block_size, 
                    false_easting=false_easting, false_northing=false_northing)
            
            if full_traj and agent.trajectory is not None:
                agent.trajectory = garden_city_to_mercator(
                    agent.trajectory, block_size=block_size,
                    false_easting=false_easting, false_northing=false_northing)
            
            if diaries and agent.diary is not None:
                # Join with poi_data if provided
                if poi_data is not None:
                    agent.diary = agent.diary.merge(
                        poi_data, left_on='location', right_on='building_id', how='left')
                    # Drop the building_id column added by merge
                    agent.diary = agent.diary.drop(columns=['building_id'])
                
                agent.diary = garden_city_to_mercator(
                    agent.diary, block_size=block_size,
                    false_easting=false_easting, false_northing=false_northing)

# =============================================================================
# AUXILIARY METHODS
# =============================================================================

def garden_city_to_mercator(data, block_size=15, false_easting=-4265699, false_northing=4392976):
    """
    Convert Garden City block coordinates to Web Mercator coordinates.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'x', 'y' columns in Garden City block coordinates
    block_size : float, default 15
        Size of one city block in meters
    false_easting : float, default -4265699
        False easting offset for Garden City
    false_northing : float, default 4392976
        False northing offset for Garden City
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'x', 'y' columns updated to Web Mercator coordinates
    """
    # Validate required columns
    if 'x' not in data.columns or 'y' not in data.columns:
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    
    # Create a copy to avoid modifying original
    result = data.copy()
    
    # Apply Garden City transformation to Web Mercator
    result['x'] = block_size * result['x'] + false_easting
    result['y'] = block_size * result['y'] + false_northing
    
    # Scale horizontal accuracy if present
    if 'ha' in result.columns:
        result['ha'] = block_size * result['ha']
    
    return result


def allowed_buildings(local_ts):
    """
    Finds allowed buildings for the timestamp
    """
    hour = local_ts.hour
    return ALLOWED_BUILDINGS[hour]
