from shapely.geometry import box, Point, LineString, MultiLineString
from shapely.ops import unary_union
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import numpy.random as npr
from matplotlib import cm
import funkybob
import s3fs

from nomad.city_gen import *
from nomad.constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS
from nomad.constants import FAST_STILL_PROBS, SLOW_STILL_PROBS, ALLOWED_BUILDINGS, DEFAULT_STAY_PROBS

import pdb


# =============================================================================
# NHPP SAMPLER
# =============================================================================

def sample_hier_nhpp(traj, beta_start, beta_durations, beta_ping, dt=1, ha=3/4, seed=None, output_bursts=False):
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
    # max_burst_samples = min(int(5*len(traj)/beta_start), len(traj))
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
            burst_info += [traj.loc[[start, end], 'local_timestamp'].tolist()]       
            
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

    sampled_traj = sampled_traj.drop_duplicates('local_timestamp')

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
                 home: str, 
                 workplace: str, 
                 city: City,
                 still_probs: dict = DEFAULT_STILL_PROBS, 
                 speeds: dict = DEFAULT_SPEEDS,
                 destination_diary: pd.DataFrame = None,
                 trajectory: pd.DataFrame = None,
                 diary: pd.DataFrame = None,
                 start_time: datetime=datetime(2024, 1, 1, hour=8, minute=0), 
                 dt: float = 1):
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
            DataFrame containing the following columns: 'unix_timestamp', 'local_timestamp', 'duration', 'location'.
        trajectory : pandas.DataFrame, optional (default=None)
            DataFrame containing the following columns: 'x', 'y', 'local_timestamp', 'unix_timestamp', 'identifier'.
        diary : pandas.DataFrame,  optional (default=None)
            DataFrame containing the following columns: 'unix_timestamp', 'local_timestamp', 'duration', 'location'.
        start_time : datetime, optional (default=datetime(2024, 1, 1, hour=8, minute=0))
            If `trajectory` is None, the first ping will occur at `start_time`.
        dt : float, optional (default=1)
            Time step duration.
        """

        self.identifier = identifier
        self.home = home
        self.workplace = workplace
        self.city = city

        if destination_diary is not None:
            start_time = destination_diary['local_timestamp'][0]
            self.destination_diary = destination_diary
        else:
            self.destination_diary = pd.DataFrame(
                columns=['unix_timestamp', 'local_timestamp', 'duration', 'location'])

        self.diary = diary if diary is not None else pd.DataFrame(
            columns=['unix_timestamp', 'local_timestamp', 'duration', 'location'])

        self.still_probs = still_probs
        self.speeds = speeds
        self.dt = dt
        self.visit_freqs = None

        # If trajectory is not provided, then the first ping is at the home centroid at start_time
        if trajectory is None:
            home_centroid = self.city.buildings[home].geometry.centroid
            x_coord, y_coord = home_centroid.x, home_centroid.y
            local_timestamp = start_time
            unix_timestamp = int(local_timestamp.timestamp())
            trajectory = pd.DataFrame([{
                'x': x_coord,
                'y': y_coord,
                'local_timestamp': local_timestamp,
                'unix_timestamp': unix_timestamp,
                'identifier': self.identifier
                }])

            diary = pd.DataFrame([{
                'unix_timestamp': unix_timestamp,
                'local_timestamp': local_timestamp,
                'duration': self.dt,
                'location': home
                }])

        self.trajectory = trajectory
        self.diary = diary

    def plot_traj(self, ax, color='black', alpha=1, doors=True, address=True, heatmap=False):
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


    def sample_traj_hier_nhpp(self,
                              beta_start,
                              beta_durations,
                              beta_ping,
                              seed=0,
                              ha=3/4,
                              output_bursts=False,
                              reset_traj=False,
                              save_to=None):
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
        reset_traj : bool
            if True, removes all but the last row of the Agent's trajectory DataFrame.
        save_to : DataFrame
            If provided, appends sparsified trajectory to the dataframe. Must have column
            ['x', 'y', 'local_timestamp', 'unix_timestamp', 'identifier']
        """

        result = sample_hier_nhpp(
            self.trajectory, 
            beta_start, 
            beta_durations, 
            beta_ping, 
            dt=self.dt, 
            ha=ha,
            seed=seed, 
            output_bursts=output_bursts
        )

        if output_bursts:
            sparse_traj, burst_info = result
        else:
            sparse_traj = result

        self.sparse_traj = sparse_traj.set_index('unix_timestamp', drop=False)

        if reset_traj:
            self.trajectory = self.trajectory.tail(1)

        if save_to is not None:
            # Append sparse_traj to save_to in place (indices will be messed up)
            save_to.loc[len(save_to):] = self.sparse_traj

        if output_bursts:
            return burst_info


def _ortho_coord(multilines, distance, offset, eps=0.001):  # Calculus approach. Probably super slow.
    """
    Given a MultiLineString, a distance along it, an offset distance, and a small epsilon,
    returns the coordinates of a point that is distance along the MultiLineString and offset
    from it.

    Parameters
    ----------
    multilines : shapely.geometry.multilinestring.MultiLineString
        MultiLineString object representing the path.
    distance : float
        Distance along the MultiLineString.
    offset : float
        Offset distance from the MultiLineString.
    eps : float, optional
        Small epsilon for numerical stability.

    Returns
    -------
    tuple
        A tuple with the (x, y) coordinates of the point.
    """

    point = multilines.interpolate(distance)
    offset_point = multilines.interpolate(distance - eps)
    p = np.array([point.x, point.y])
    x = p - np.array([offset_point.x, offset_point.y])
    x = np.flip(x/np.linalg.norm(x))*np.array([-1,1])*offset
    return tuple(x+p)


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
        'unix_timestamp': 'first',
        'local_timestamp': 'first',
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
                 city: City):
        self.roster = {}
        self.city = city
        self.all_sparse_trajs = pd.DataFrame(
            columns=['x', 'y', 'local_timestamp', 'unix_timestamp', 'identifier']
        )

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
                        start_time: datetime = datetime(2024, 1, 1, hour=8, minute=0),
                        dt: float = 1, 
                        seed: int = 0):
        """
        Generates N agents, with randomized attributes.
        """

        npr.seed(seed)

        b_types = pd.DataFrame({
            'id': list(self.city.buildings.keys()),
            'type': [b.building_type for b in self.city.buildings.values()]
        }).set_index('id')  # Maybe this should be an attribute of city since we end up using it a lot

        homes = b_types[b_types['type'] == 'home'].sample(n=N, replace=True)
        workplaces = b_types[b_types['type'] == 'work'].sample(n=N, replace=True)

        generator = funkybob.UniqueRandomNameGenerator(members=2, seed=seed)
        for i in range(N):
            identifier = generator[i]
            agent = Agent(identifier=identifier,
                          home=homes.index[i],
                          workplace=workplaces.index[i],
                          city=self.city,
                          start_time=start_time,
                          dt=dt)  # how do we add other args?
            self.add_agent(agent)

    def save_pop(self, bucket, prefix):
        """
        Save trajectories, homes, and diaries as Parquet files to S3.

        Parameters
        ----------
        bucket : str
            The name of the S3 bucket.
        prefix : str
            The path prefix within the bucket (e.g., 'folder/subfolder/').
        """

        fs = s3fs.S3FileSystem()

        # raw trajectories
        trajs = pd.concat([agent.trajectory for agent_id, agent in self.roster.items()])
        trajs.to_parquet(f's3://{bucket}/{prefix}trajectories.parquet', engine='pyarrow', filesystem=fs)

        # sparse trajectories
        sparse_trajs = pd.concat([agent.sparse_traj for agent_id, agent in self.roster.items()])
        sparse_trajs.to_parquet(f's3://{bucket}/{prefix}sparse_trajectories.parquet', engine='pyarrow', filesystem=fs)

        # home table
        homes = pd.DataFrame([(agent_id, agent.home, agent.workplace) for agent_id, agent in self.roster.items()],
                             columns=['id', 'home', 'workplace'])
        homes.to_parquet(f's3://{bucket}/{prefix}homes.parquet', engine='pyarrow', filesystem=fs)

        # diary
        diaries = []
        for agent_id, agent in self.roster.items():
            ag_d = agent.diary
            ag_d['id'] = agent_id
            diaries += [ag_d]
        pd.concat(diaries).to_parquet(f's3://{bucket}/{prefix}diaries.parquet', engine='pyarrow', filesystem=fs)

#     TODO: allow for parallelization

    def sample_step(self, agent, start_point, dest_building, dt):
        """
        From a destination diary, generates (x, y) pings.

        Parameters
        ----------
        agent : Agent
            The agent for whom a step will be sampled.
        start_point : tuple
            The coordinates of the current position as a tuple (x, y).
        dest_building : Building
            The destination building of the agent.
        dt : float
            Time step (i.e., number of minutes per ping).

        Returns
        -------
        coord : numpy.ndarray
            A numpy array of floats with shape (1, 2) representing the new coordinates.
        location : str or None
            The building ID if the step is a stay, or `None` if the step is a move.
        """
        city = self.city

        # Find current geometry
        start_block = np.floor(start_point)  # blocks are indexed by bottom left
        start_geometry = city.get_block(tuple(start_block))

        curr = np.array(start_point)

        # Agent moves within the building
        if start_geometry == dest_building or start_point == dest_building.door_centroid:
            location = dest_building.id
            p = agent.still_probs[dest_building.building_type]
            sigma = agent.speeds[dest_building.building_type]

            if npr.uniform() < p:
                coord = curr
            else:

                # Draw until coord falls inside building
                while True:
                    coord = np.random.normal(loc=curr, scale=sigma*np.sqrt(dt), size=2)
                    if dest_building.geometry.contains(Point(coord)):
                        break

        # Agent travels to building along the streets
        else:
            location = None
            dest_point = dest_building.door

            if start_geometry in city.buildings.values():
                start_segment = [start_point, start_geometry.door_centroid]
                start = start_geometry.door
            else:
                start_segment = []
                start = tuple(start_block.astype(int))

            street_path = city.shortest_paths[start][dest_point]
            path = [(x+0.5, y+0.5) for x, y in street_path]
            path = start_segment + path + [dest_building.geometry.centroid]
            path_ml = MultiLineString([path])

            # Bounding polygon
            street_poly = unary_union([city.get_block(block).geometry for block in street_path])

            bound_poly = unary_union([start_geometry.geometry, street_poly])
            # Snap to path
            snap_point_dist = path_ml.project(Point(start_point))

            #TODO: SHOULD THESE ALSO BE CONSTANT?
            delta = 3.33*dt      # 50m/min; blocks are 15m x 15m
            sigma = 0.5*dt/1.96  # 95% prob of moving 0.5

            # Draw until coord falls inside bound_poly
            while True:
                # consider a "path" coordinate and "orthogonal coordinate"
                transformed_step = np.random.normal(loc=[delta, 0], scale=sigma*np.sqrt(dt), size=2)

                if snap_point_dist + transformed_step[0] > path_ml.length:
                    coord = np.array(dest_building.geometry.centroid.coords[0])
                    break
                else:
                    coord = _ortho_coord(path_ml, snap_point_dist+transformed_step[0], transformed_step[1])
                    if bound_poly.contains(Point(coord)):
                        break

        return coord, location

    def traj_from_dest_diary(self, agent):
        """
        Simulate a trajectory and give agent true travel diary attribute.

        Parameters
        ----------
        agent: Agent
            The agent for whom to simulate the trajectory.

        Returns
        -------
        None (updates agent.trajectory, agent.diary)
        """

        city = self.city
        dt = agent.dt

        destination_diary = agent.destination_diary

        current_loc = agent.trajectory.iloc[-1]
        trajectory_update = []

        if agent.diary.empty:
            current_entry = None
        else:
            current_entry = agent.diary.iloc[-1].to_dict()
            agent.diary = agent.diary.iloc[:-1]
        entry_update = []
        for i in range(destination_diary.shape[0]):
            destination_info = destination_diary.iloc[i]
            duration = int(destination_info['duration'] * 1/dt)
            building_id = destination_info['location']
            for t in range(int(duration//dt)):
                prev_ping = current_loc
                start_point = (prev_ping['x'], prev_ping['y'])
                dest_building = city.buildings[building_id]
                unix_timestamp = prev_ping['unix_timestamp'] + 60*dt
                local_timestamp = pd.to_datetime(unix_timestamp, unit='s')

                # You should be asleep between 0:00 and 5:59!
                # if local_timestamp.hour <= 5:
                #     coord = start_point  # stay in place
                #     location = building_id
                # else:
                coord, location = self.sample_step(agent, start_point, dest_building, dt)
                ping = {'x': coord[0], 'y': coord[1],
                        'local_timestamp': local_timestamp,
                        'unix_timestamp': unix_timestamp,
                        'identifier': agent.identifier}

                current_loc = ping
                trajectory_update.append(ping)
                if(current_entry == None or current_entry['location'] != location):
                    entry_update.append(current_entry)
                    current_entry = {'unix_timestamp': unix_timestamp,
                             'local_timestamp': local_timestamp,
                             'duration': dt,
                             'location': location}
                else:
                    current_entry['duration'] += 1*dt

        agent.trajectory = pd.concat([agent.trajectory, pd.DataFrame(trajectory_update)],
                                ignore_index=True)

        entry_update.append(current_entry)
        if(agent.diary.empty):
            agent.diary = pd.DataFrame(entry_update)
        else:
            pd.concat([agent.diary, pd.DataFrame(entry_update)], ignore_index=True)
        agent.destination_diary = destination_diary.drop(destination_diary.index)

    def generate_dest_diary(self, 
                            agent: Agent, 
                            T: datetime, 
                            duration: int = 15,
                            stay_probs: dict = DEFAULT_STAY_PROBS,
                            rho: float = 0.6, 
                            gamma: float = 0.2, 
                            seed: int = 0):
        """
        Exploration and preferential return.

        Parameters
        ----------
        agent : Agent
            The agent for whom to generate the destination diary.
        T : datetime
            The end time to generate the destination diary until.
        duration : int
            The duration of each destination entry in minutes.
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

        if isinstance(T, datetime):
            T = int(T.timestamp())  # Convert to unix

        # Create visit frequency table is user does not already have one
        visit_freqs = agent.visit_freqs
        if not visit_freqs:
            visit_freqs = pd.DataFrame({
                'id': list(self.city.buildings.keys()),
                'type': [b.building_type for b in self.city.buildings.values()],
                'freq': 0,
                'p': 0
            }).set_index('id')

            # Initializes past counts randomly
            visit_freqs.loc[agent.home, 'freq'] = 25
            visit_freqs.loc[agent.workplace, 'freq'] = 25
            visit_freqs.loc[visit_freqs.type == 'park', 'freq'] = 3  # Agents love to comeback to park
            # ALTERNATIVELY: start with 1 at home and 1 at work, and do a burnout period of 2 weeks. 

            initial_locs = []
            initial_locs += list(npr.choice(visit_freqs.loc[visit_freqs.type == 'retail'].index, size=npr.poisson(8)))
            initial_locs += list(npr.choice(visit_freqs.loc[visit_freqs.type == 'work'].index, size=npr.poisson(4)))
            initial_locs += list(npr.choice(visit_freqs.loc[visit_freqs.type == 'home'].index, size=npr.poisson(4)))
            visit_freqs.loc[initial_locs, 'freq'] += 1

        if agent.destination_diary.empty:
            last_ping = agent.trajectory.iloc[-1]
            start_time_local = last_ping.local_timestamp
            start_time = last_ping.unix_timestamp
            curr = self.city.get_block((last_ping.x, last_ping.y)).id  # Always a building?? Could be street
        else:
            last_entry = agent.destination_diary.iloc[-1]
            start_time_local = last_entry.local_timestamp + timedelta(minutes=int(last_entry.duration))
            start_time = last_entry.unix_timestamp + last_entry.duration*60
            curr = last_entry.location

        dest_update = []
        while start_time < T:
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
            entry = {'unix_timestamp': start_time,
                     'local_timestamp': start_time_local,
                     'duration': duration,
                     'location': curr}
            dest_update.append(entry)
            # entry = pd.DataFrame([entry])
            # if agent.destination_diary.empty:
            #     agent.destination_diary = entry
            # else:
            #     agent.destination_diary = pd.concat([agent.destination_diary, entry], ignore_index=True)

            start_time_local = start_time_local + timedelta(minutes=int(duration))
            start_time = start_time + duration*60

        if agent.destination_diary.empty:
            agent.destination_diary = pd.DataFrame(dest_update)
        else:
            pd.concat([agent.destination_diary, pd.DataFrame(dest_update)], ignore_index=True)
        agent.destination_diary = condense_destinations(agent.destination_diary)

        agent.visit_freqs = visit_freqs

        return None

    def generate_trajectory(self, 
                            agent: Agent, 
                            T: datetime=None, 
                            duration: int=15, 
                            seed: int=0):
        """
        Generate a trajectory for an agent.

        Parameters
        ----------
        agent : Agent
            The agent for whom to generate a trajectory.
        T : datetime, optional
            The end time to generate the trajectory until.
        duration : int, optional
            The duration of each destination entry in minutes.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        None (updates agent.trajectory)
        """

        npr.seed(seed)
        dt = agent.dt

        if agent.destination_diary.empty:
            if T is None:
                raise ValueError(
                    "Destination diary is empty. Provide an argument T to generate destination diary using EPR."
                )
            self.generate_dest_diary(agent, T, duration=duration, seed=seed)

        self.traj_from_dest_diary(agent)

        return None


    def plot_population(self, ax, doors=True, address=True):
        for i, agent_id in enumerate(self.roster):
            agent = self.roster[agent_id]
            col = cm.tab20c(i/len(self.roster))
            ax.scatter(agent.trajectory.x, agent.trajectory.y, s=6, color=col, alpha=1, zorder=2)
        self.city.plot_city(ax, doors=doors, address=address, zorder=1)


# =============================================================================
# AUXILIARY METHODS
# =============================================================================


def allowed_buildings(local_ts):
    """
    Finds allowed buildings for the timestamp
    """
    hour = local_ts.hour
    return ALLOWED_BUILDINGS[hour]
