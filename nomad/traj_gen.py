import pandas as pd
import numpy as np
import numpy.random as npr
from matplotlib import cm
from shapely.geometry import box, Point, MultiLineString
from shapely.ops import unary_union
from datetime import timedelta
from zoneinfo import ZoneInfo
import warnings
import funkybob
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
    elif "unix_timestamp" in col_names:
        return "unix_timestamp"
    else:
        if verbose:
            warnings.warn("No trajectory data was found or time columns ('datetime', 'unix_timestamp')\
                          in keyword arguments. '2025-01-01 00:00Z' will be used for starting trajectory time.")
        return "missing"

def sample_hier_nhpp(traj,
                     beta_start,
                     beta_durations,
                     beta_ping,
                     dt=1,
                     ha=3/4,
                     seed=None,
                     output_bursts=False):
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
        scale parameter (mean) of Exponential distribution modeling burst durations
    beta_ping: float
        scale parameter (mean) of Exponential distribution modeling ping inter-arrival times
        within a burst, where 1/beta_ping is the rate of events (pings) per minute.
    dt: float
        the time resolution of the output sequence. Must match that of the input trajectory.  
    ha: float
        horizontal accuracy controlling the measurement error. Noisy locations are sampled as
        (true x + eps_x, true y + eps_y) where (eps_x, eps_y) ~ N(0, nu/1.96).
    seed : int0
        The seed for random number generation.
    output_bursts : bool
        If True, outputs the latent variables on when bursts start and end.
    """
    if seed:
        npr.seed(seed)

    # Adjust betas to dt time resolution
    # should match the resolution of traj?
    beta_start = beta_start / dt
    beta_durations = beta_durations / dt
    beta_ping = beta_ping / dt

    # Sample starting points of bursts using at most max_burst_samples times
    max_burst_samples = len(traj)
    
    inter_arrival_times = npr.exponential(scale=beta_start, size=max_burst_samples)
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
    burst_info = []
        
    for start, end in zip(burst_start_points, burst_end_points):
        if output_bursts:
            burst_info += [traj.loc[[start, end], 'datetime'].tolist()]       
            
        burst_indices = np.arange(start, end)

        if len(burst_indices) == 0:
            continue

        # max_ping_samples = min(int(5*len(traj)/beta_start), len(burst_indices))
        max_ping_samples = len(burst_indices)
        
        ping_intervals = np.random.exponential(scale=beta_ping, size=max_ping_samples)
        ping_times = np.unique(np.cumsum(ping_intervals).astype(int))
        ping_times = ping_times[ping_times < (end - start)] + start

        if len(ping_times) == 0:
            continue

        burst_data = traj.iloc[ping_times].copy()

        sampled_trajectories.append(burst_data)

    if sampled_trajectories:
        sampled_traj = pd.concat(sampled_trajectories).sort_values(by='unix_timestamp') #why wouldn't they be sorted already?
    else:  # empty
        sampled_traj = pd.DataFrame(columns=list(traj.columns))

    sampled_traj = sampled_traj.drop_duplicates('datetime')

    # Add sampling noise
    noise = npr.normal(loc=0, scale=ha/1.96, size=(sampled_traj.shape[0], 2))
    sampled_traj[['x', 'y']] = sampled_traj[['x', 'y']] + noise
    sampled_traj['ha'] = ha

    if output_bursts:
        burst_info = pd.DataFrame(burst_info, columns = ['start_time', 'end_time'])
        return sampled_traj, burst_info
    else:
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
            DataFrame containing the following columns: 'unix_timestamp', 'datetime', 'duration', 'location'.
        trajectory : pandas.DataFrame, optional (default=None)
            DataFrame containing the following columns: 'x', 'y', 'datetime', 'unix_timestamp', 'identifier'.
        diary : pandas.DataFrame,  optional (default=None)
            DataFrame containing the following columns: 'unix_timestamp', 'datetime', 'duration', 'location'.
        dt : float, optional (default=1)
            Time step duration.
        """

        npr.seed(seed)
        
        self.identifier = identifier
        self.city = city

        if home is None:
            home = city.building_types[city.building_types['type'] == 'home'].sample(n=1)['id'].iloc[0]
        if workplace is None:
            workplace = city.building_types[city.building_types['type'] == 'work'].sample(n=1)['id'].iloc[0]

        self.home = home
        self.workplace = workplace

        self.still_probs = still_probs
        self.speeds = speeds
        self.visit_freqs = None

        self.destination_diary = destination_diary if destination_diary is not None else pd.DataFrame(
            columns=['datetime', 'unix_timestamp', 'duration', 'location'])
        self.trajectory = trajectory
        self.dt = None
        self.diary = diary if diary is not None else pd.DataFrame(
            columns=['datetime', 'unix_timestamp', 'duration', 'location', 'identifier'])
        self.last_ping = trajectory.iloc[-1] if (trajectory is not None) else None
        self.sparse_traj = None


    def reset_trajectory(self):
        """
        Resets the agent's trajectories and diaries to the initial state. 
        Keeps the agent's identifier, home, and workplace.
        This method is useful for reinitializing the agent after a simulation run.
        """
        self.destination_diary = pd.DataFrame(columns=self.destination_diary.columns)
        self.trajectory = None
        self.dt = None
        self.diary = pd.DataFrame(columns=self.diary.columns)
        self.last_ping = None
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

    def _sample_step(self, start_point, dest_building, dt):
        """
        From a destination diary, generates (x, y) pings.

        Parameters
        ----------
        start_point : tuple
            The coordinates of the current position as a tuple (x, y).
        dest_building : Building
            The destination building of the agent.
        dt : float
            The time step duration.

        Returns
        -------
        coord : numpy.ndarray
            A numpy array of floats with shape (1, 2) representing the new coordinates.
        location : str or None
            The building ID if the step is a stay, or `None` if the step is a move.
        """
        city = self.city
    
        start_block = np.floor(start_point)
        start_geometry = city.get_block(tuple(start_block))
    
        curr = np.array(start_point)
    
        if start_geometry == dest_building or start_point == dest_building.door_centroid:
            location = dest_building.id
            p = self.still_probs[dest_building.building_type]
            sigma = self.speeds[dest_building.building_type]
    
            if npr.uniform() < p:
                coord = curr
            else: # Draw until coord falls inside building
                while True:
                    coord = np.random.normal(loc=curr, scale=sigma*np.sqrt(dt), size=2)
                    if dest_building.geometry.contains(Point(coord)):
                        break
        else: # Agent travels to building along the streets
            location = None
            dest_point = dest_building.door

            if start_geometry in city.buildings.values():
                start_segment = [start_point, start_geometry.door_centroid]
                start = start_geometry.door
            else:
                start_segment = []
                start = tuple(start_block.astype(int))

            street_path = city.shortest_paths[start][dest_point]
            path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
            path = start_segment + path + [dest_building.door_centroid]
            path_ml = MultiLineString([path])
            path_length = path_ml.length

            # Bounding polygon
            street_poly = unary_union([city.get_block(block).geometry for block in street_path])
            bound_poly = unary_union([start_geometry.geometry, street_poly])
            # Snap to path
            snap_point_dist = path_ml.project(Point(start_point))
    
            delta = 3.33 * dt
            sigma = 0.5 * dt / 1.96
    
            while True:
                transformed_step = np.random.normal(loc=[delta, 0], scale=sigma*np.sqrt(dt), size=2)
                distance = snap_point_dist + transformed_step[0]
    
                if distance > path_length:
                    coord = np.array(dest_building.geometry.centroid.coords[0])
                    break
                coord = _ortho_coord(path_ml, distance, transformed_step[1])
                if bound_poly.contains(Point(coord)):
                    break
                    
        return coord, location


    def _traj_from_dest_diary(self, dt):
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

        city = self.city
        destination_diary = self.destination_diary

        trajectory_update = []

        if self.diary.empty:
            current_entry = None
        else:
            current_entry = self.diary.iloc[-1].to_dict()
            self.diary = self.diary.iloc[:-1]

        entry_update = []
        for i in range(destination_diary.shape[0]):
            destination_info = destination_diary.iloc[i]
            duration = int(destination_info['duration'] * 1/dt)
            building_id = destination_info['location']

            for t in range(int(duration//dt)):
                prev_ping = self.last_ping
                start_point = (prev_ping['x'], prev_ping['y'])
                dest_building = city.buildings[building_id]
                unix_timestamp = prev_ping['unix_timestamp'] + 60*dt
                datetime = prev_ping['datetime'] + timedelta(minutes=dt)               
                coord, location = self._sample_step(start_point, dest_building, dt)
                ping = {'x': coord[0], 
                        'y': coord[1],
                        'datetime': datetime,
                        'unix_timestamp': unix_timestamp,
                        'identifier': self.identifier}

                trajectory_update.append(ping)
                self.last_ping = ping
                if current_entry == None:
                    current_entry = {'datetime': datetime,
                                     'unix_timestamp': unix_timestamp,
                                     'duration': dt,
                                     'location': location,
                                     'identifier': self.identifier}
                elif (current_entry['location'] != location):
                    entry_update.append(current_entry)
                    current_entry = {'datetime': datetime,
                                     'unix_timestamp': unix_timestamp,
                                     'duration': dt,
                                     'location': location,
                                     'identifier': self.identifier}
                else:
                    current_entry['duration'] += 1*dt

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
                             end_time: pd.Timestamp, 
                             epr_time_res: int = 15,
                             stay_probs: dict = DEFAULT_STAY_PROBS,
                             rho: float = 0.6, 
                             gamma: float = 0.2, 
                             seed: int = 0):
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
        npr.seed(seed)

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
            visit_freqs.loc[self.home, 'freq'] = 35
            visit_freqs.loc[self.workplace, 'freq'] = 35
            visit_freqs.loc[visit_freqs.type == 'park', 'freq'] = 3

            initial_locs = []
            initial_locs += list(npr.choice(visit_freqs.loc[visit_freqs.type == 'retail'].index, size=npr.poisson(6)))
            initial_locs += list(npr.choice(visit_freqs.loc[visit_freqs.type == 'work'].index, size=npr.poisson(3)))
            initial_locs += list(npr.choice(visit_freqs.loc[visit_freqs.type == 'home'].index, size=npr.poisson(3)))
            visit_freqs.loc[initial_locs, 'freq'] += 2

        if self.destination_diary.empty:
            start_time_local = self.last_ping['datetime']
            start_time = self.last_ping['unix_timestamp']
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
            if (curr_type in allowed) & (npr.uniform() < stay_probs[curr_type]):
                pass

            # Exploration
            elif npr.uniform() < p_exp:
                visit_freqs['p'] = self.city.gravity.xs(
                    self.city.buildings[curr].door, level=0).join(id2door, how='right').set_index('id')
                y = visit_freqs.loc[(visit_freqs['type'].isin(allowed)) & (visit_freqs.freq == 0)]

                if not y.empty and y['p'].sum() > 0:
                    curr = npr.choice(y.index, p=y['p']/y['p'].sum())
                else:
                    # If there are no more buildings to explore, then preferential return
                    curr = npr.choice(x.index, p=x['freq']/x['freq'].sum())

                visit_freqs.loc[curr, 'freq'] += 1

            # Preferential return
            else:
                curr = npr.choice(x.index, p=x['freq']/x['freq'].sum())
                visit_freqs.loc[curr, 'freq'] += 1

            # Update destination diary
            entry = {'datetime': start_time_local,
                     'unix_timestamp': start_time,
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
                            destination_diary: pd.DataFrame = None,
                            end_time: pd.Timestamp=None, 
                            epr_time_res: int=15,
                            dt: float=1,
                            seed: int=0,
                            verbose=False,
                            **kwargs):
        """
        Generate a trajectory for an agent.

        Parameters
        ----------
        destination_diary : pandas.DataFrame, optional (default=None)
            DataFrame containing the following columns: 'unix_timestamp', 'datetime', 'duration', 'location'.
        end_time : pd.Timestamp, optional
            The end time to generate the trajectory until.
        epr_time_res : int, optional
            The granularity of destination durations in epr generation.
        seed : int, optional
            Random seed for reproducibility.
        kwargs : dict, optional
            Additional keyword arguments for trajectory generation. 
            Can include 'x', 'y', 'datetime', 'unix_timestamp', 'tz'
            These are used to set the initial position of the agent.
        
        Returns
        -------
        None (updates self.trajectory)
        """

        npr.seed(seed)
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
                'unix_timestamp': unix_timestamp,
                'identifier': self.identifier
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
            elif _datetime_or_ts_col(kwargs.keys(), verbose) == "unix_timestamp":
                unix_timestamp = kwargs['unix_timestamp']
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
                'unix_timestamp': unix_timestamp,
                'identifier': self.identifier
                })
        
        else:
            if ('x' in kwargs)or('y' in kwargs)or('datetime'in kwargs)or('unix_timestamp' in kwargs):
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

        self._traj_from_dest_diary(dt=dt)

        return None

    def sample_trajectory(self,
                          beta_start,
                          beta_durations,
                          beta_ping,
                          seed=0,
                          ha=3/4,
                          dt=None,
                          output_bursts=False,
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

        # Compute the empirical dt as the mode of the empirical delta time between pings
        empirical_dt = self.trajectory['datetime'].diff()
        empirical_dt = empirical_dt.mode().iloc[0].total_seconds() / 60
        if dt is None:
            dt = empirical_dt
        if dt is not None and dt != empirical_dt:
            warnings.warn(f"dt ({dt}) does not match the empirical dt ({empirical_dt}).\
                          The trajectory may not be sampled correctly.")

        result = sample_hier_nhpp(
            self.trajectory, 
            beta_start, 
            beta_durations, 
            beta_ping, 
            dt=dt, 
            ha=ha,
            seed=seed, 
            output_bursts=output_bursts
        )

        if output_bursts:
            sparse_traj, burst_info = result
        else:
            sparse_traj = result
            
        sparse_traj = sparse_traj.set_index('unix_timestamp', drop=False)

        if self.sparse_traj is None or replace_sparse_traj:
            self.sparse_traj = sparse_traj
        else:
            self.sparse_traj = pd.concat([self.sparse_traj, sparse_traj], ignore_index=False)

        if cache_traj:
            self.last_ping = self.trajectory.iloc[-1]
            self.trajectory = self.trajectory.loc[[]]  # empty df

        if output_bursts:
            return burst_info


def _ortho_coord(multilines, distance, offset, eps=0.001):
    """
    Given a MultiLineString, a distance along it, and an orthogonal offset,
    returns the coordinates of a point offset from the path at that distance.

    Parameters
    ----------
    multilines : shapely.geometry.MultiLineString
        MultiLineString representing the street path.
    distance : float
        Distance along the path to project from.
    offset : float
        Perpendicular offset from the path (positive to the left, negative to the right).
    eps : float, optional
        Small delta used to estimate the path's tangent direction.

    Returns
    -------
    tuple
        Coordinates of the offset point (x, y).
    """
    point = multilines.interpolate(distance)
    offset_point = multilines.interpolate(distance - eps)

    p = np.array([point.x, point.y])
    q = np.array([offset_point.x, offset_point.y])
    direction = p - q
    unit_direction = direction / np.linalg.norm(direction)

    # Rotate 90° counter-clockwise to get the normal vector
    normal = np.flip(unit_direction) * np.array([-1, 1])

    return tuple(p + offset * normal)

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
        'unix_timestamp': 'first',
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

    def generate_agents(self,
                        N: int,
                        seed: int = 0,
                        name_count: int = 2):
        """
        Generates N agents, with randomized attributes.
        """

        generator = funkybob.UniqueRandomNameGenerator(members=name_count, seed=seed)
        for i in range(N):
            identifier = generator[i]
            agent = Agent(identifier=identifier,
                          city=self.city,
                          seed=seed+i)
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
        partition_cols : dict, optional
            Dict with keys in {'full_traj', 'sparse_traj', 'diaries'} and values as lists of partition column names.
        filesystem : pyarrow.fs.FileSystem or None
            Optional filesystem object (e.g., s3fs.S3FileSystem). If None, inferred automatically.
        """
        if full_path:
            full_df = pd.concat([agent.trajectory for agent in self.roster.values()], ignore_index=True)
            full_df = from_df(full_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(full_df,
                    path=full_path,
                    format="parquet",
                    partition_by=partition_cols.get('full_traj') if partition_cols else None,
                    filesystem=filesystem, 
                    existing_data_behavior='delete_matching')
    
        if sparse_path:
            sparse_df = pd.concat([agent.sparse_traj for agent in self.roster.values()], ignore_index=True)
            sparse_df = from_df(sparse_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(sparse_df,
                    path=sparse_path,
                    format="parquet",
                    partition_by=partition_cols.get('sparse_traj') if partition_cols else None,
                    filesystem=filesystem, 
                    existing_data_behavior='delete_matching')
    
        if diaries_path:
            diaries_df = pd.concat([agent.diary for agent in self.roster.values()], ignore_index=True)
            diaries_df = from_df(diaries_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(diaries_df,
                    path=diaries_path,
                    format="parquet",
                    partition_by=partition_cols.get('diaries') if partition_cols else None,
                    filesystem=filesystem, 
                    existing_data_behavior='delete_matching')
    
        if homes_path:
            homes_data = []
            for agent_id, agent in self.roster.items():
                ts = agent.last_ping['datetime']
                iso_date = ts.date().isoformat()
                homes_data.append((agent_id, agent.home, agent.workplace, iso_date))
            homes_df = pd.DataFrame(homes_data, columns=["uid", "home", "workplace", "date"])
    
            table = pa.Table.from_pandas(homes_df, preserve_index=False)
            ds.write_dataset(table,
                             base_dir=str(homes_path),
                             format="parquet",
                             partitioning_flavor='hive',
                             filesystem=filesystem,
                             existing_data_behavior='delete_matching')


# =============================================================================
# AUXILIARY METHODS
# =============================================================================


def allowed_buildings(local_ts):
    """
    Finds allowed buildings for the timestamp
    """
    hour = local_ts.hour
    return ALLOWED_BUILDINGS[hour]
