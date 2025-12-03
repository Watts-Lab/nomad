import pandas as pd
import numpy as np
import numpy.random as npr
from shapely.geometry import box, Point, MultiLineString
from shapely.ops import unary_union, linemerge
from shapely import distance as shp_distance
from datetime import timedelta
import warnings
import funkybob
import pyarrow as pa
import pyarrow.dataset as ds

from nomad.io.base import from_df, to_file

from nomad.city_gen import *
from nomad.filters import to_timestamp
from nomad.constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS
from nomad.constants import FAST_STILL_PROBS, SLOW_STILL_PROBS, ALLOWED_BUILDINGS, DEFAULT_STAY_PROBS

def parse_agent_attr(attr, N, name):
    """
    Parse agent attribute (homes/workplaces/datetimes) into a callable that returns the i-th value.
    
    Parameters
    ----------
    attr : str, list, pd.Timestamp, or None
        The attribute value. Can be:
        - None: returns None for all indices
        - str or pd.Timestamp: returns the same value for all indices
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
    elif isinstance(attr, (str, pd.Timestamp)):
        return lambda i: attr
    elif isinstance(attr, list):
        if len(attr) != N:
            raise ValueError(f"{name} must be a list of length {N}, got {len(attr)}")
        return lambda i: attr[i]
    else:
        raise ValueError(f"{name} must be a string, pd.Timestamp, list of length {N}, or None")

def sample_bursts_gaps(traj,
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
        Mean horizontal-accuracy radius *in 15 m blocks*. The actual per-ping accuracy is random: ha ≥ 8 m/15m
        and follows a Pareto distribution with that mean. For each ping the spatial noise (ε_x, ε_y) is drawn
        i.i.d. N(0, σ²) with σ = HA / 1.515 so that |ε| ≤ HA with 68 % probability.
    seed : int0
        The seed for random number generation.
    output_bursts : bool
        If True, outputs the latent variables on when bursts start and end.
    deduplicate : bool
        If True, sampled times are also discretized to be in ticks
    """
    rng = npr.default_rng(seed)

    # absolute window
    t0   = int(traj['timestamp'].iloc[0])
    t_end = int(traj['timestamp'].iloc[-1])
    # Step 1: generate ping_times (and bursts if requested)
    tz = traj['datetime'].dt.tz
    if output_bursts:
        ping_times, bursts = generate_ping_times(
            t0, t_end,
            beta_start=beta_start,
            beta_durations=beta_durations,
            beta_ping=beta_ping,
            seed=seed,
            return_bursts=True,
            tz=tz,
        )
    else:
        ping_times = generate_ping_times(
            t0, t_end,
            beta_start=beta_start,
            beta_durations=beta_durations,
            beta_ping=beta_ping,
            seed=seed,
        )

    # Step 2: thin trajectory
    sampled_traj = thin_traj_by_times(traj, ping_times, deduplicate=deduplicate)
    if sampled_traj.empty:
        empty = sampled_traj
        if output_bursts:
            return empty, pd.DataFrame(columns=['start_time','end_time'])
        return empty

    # Step 3: add horizontal noise
    rng = npr.default_rng(seed)
    n = len(sampled_traj)
    ha_realized, noise = _sample_horizontal_noise(n, ha=ha, rng=rng)
    sampled_traj['ha'] = ha_realized
    sampled_traj[['x', 'y']] += noise

    if output_bursts:
        burst_df = pd.DataFrame(bursts, columns=['start_time','end_time'])
        return sampled_traj, burst_df

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
                 identifier, 
                 city,
                 home=None,
                 workplace=None,
                 still_probs=DEFAULT_STILL_PROBS, 
                 speeds=DEFAULT_SPEEDS,
                 destination_diary=None,
                 trajectory=None,
                 diary=None,
                 seed=0,
                 x=None,
                 y=None,
                 location=None,
                 datetime=None,
                 timestamp=None):
        """
        Initialize an agent in the city simulation, with optional existing data.

        Parameters
        ----------
        identifier : str
            Agent identifier.
        city : City
            City object with buildings and topology.
        home : str, optional
            Building ID for the agent's home. If None, sampled from `city.buildings_gdf`.
        workplace : str, optional
            Building ID for the agent's workplace. If None, sampled from `city.buildings_gdf`.
        still_probs : dict, optional
            Per-building-type probabilities of staying still (default: DEFAULT_STILL_PROBS).
        speeds : dict, optional
            Per-building-type movement scalars (default: DEFAULT_SPEEDS).
        destination_diary : pandas.DataFrame, optional
            If provided, a DataFrame with columns ['datetime','timestamp','duration','location'].
        trajectory : pandas.DataFrame, optional
            If provided, a DataFrame with columns ['x','y','datetime','timestamp','identifier'].
        diary : pandas.DataFrame, optional
            If provided, a DataFrame with columns ['datetime','timestamp','duration','location'].
        seed : int, optional
            RNG seed used for sampling fallback home/work locations.
        """

        rng = npr.default_rng(seed)
        
        self.identifier = identifier
        self.city = city

        if home is None:
            home = city.buildings_gdf[city.buildings_gdf['building_type'] == 'home'].sample(n=1, random_state=rng)['id'].iloc[0]
        if workplace is None:
            workplace = city.buildings_gdf[city.buildings_gdf['building_type'] == 'workplace'].sample(n=1, random_state=rng)['id'].iloc[0]

        home_building = city.get_building(identifier=home)
        workplace_building = city.get_building(identifier=workplace)
        if home_building is None or workplace_building is None:
            raise ValueError(f"Home {home} or workplace {workplace} not found in city buildings.")
        self.home = home
        self.workplace = workplace
        self.home_centroid = home_building['geometry'].iloc[0].centroid
        self.workplace_centroid = workplace_building['geometry'].iloc[0].centroid

        self.still_probs = still_probs
        self.speeds = speeds
        self.visit_freqs = None

        # Initialize last_ping
        if trajectory is not None:
            if x is not None or y is not None or location is not None or datetime is not None or timestamp is not None:
                warnings.warn("Both trajectory and position kwargs provided. Trajectory takes precedence.")
            self.last_ping = trajectory.iloc[-1]
        else:
            # Determine initial position
            if x is not None and y is not None:
                x_coord, y_coord = x, y
            elif location is not None:
                loc_centroid = self.city.buildings_gdf.loc[location, 'geometry'].centroid
                x_coord, y_coord = loc_centroid.x, loc_centroid.y
            else:
                x_coord, y_coord = self.home_centroid.x, self.home_centroid.y
            
            # Determine initial time
            if datetime is not None:
                init_time = pd.to_datetime(datetime) if not isinstance(datetime, pd.Timestamp) else datetime
            else:
                init_time = pd.Timestamp.now(tz='America/New_York')
            
            if timestamp is not None:
                init_timestamp = timestamp
            else:
                init_timestamp = to_timestamp(init_time)
            
            self.last_ping = pd.Series({
                'x': x_coord,
                'y': y_coord,
                'datetime': init_time,
                'timestamp': init_timestamp,
                'user_id': identifier
            })

        self.destination_diary = destination_diary if destination_diary is not None else pd.DataFrame(
            columns=['datetime', 'timestamp', 'duration', 'location'])
        self.trajectory = trajectory
        self.dt = None
        self.diary = diary if diary is not None else pd.DataFrame(
            columns=['datetime', 'timestamp', 'duration', 'location', 'identifier'])
        self.sparse_traj = None
        
        # Trajectory simulation parameters (caching for performance)
        self._cached_bound_poly = None
        self._cached_path_ml = None
        self._previous_dest_building_row = None
        self._current_dest_building_row = None


    def reset_trajectory(self, trajectory = True, sparse = True, last_ping = True, diary = True):
        """
        Resets the agent's trajectories and diaries to the initial state. 
        Keeps the agent's identifier, home, and workplace.
        This method is useful for reinitializing the agent after a simulation run.
        """
        self.destination_diary = pd.DataFrame(columns=self.destination_diary.columns)
        self.dt = None
        # null cache for trajectory generation
        self._cached_path_ml = None
        self._cached_bound_poly = None
        self._cached_dest_id = None
        self._cached_bound_poly_blocks_set = None
        self._previous_dest_building_row = None
        self._current_dest_building_row = None
        
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

    def _sample_step(self, start_point, dt, rng):
        """
        From a destination diary, generates a single (x, y) ping step towards the current destination building.

        Parameters
        ----------
        start_point : tuple
            The coordinates of the current position as a tuple (x, y).
        dt : float
            The time step duration (minutes).
        rng : numpy.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        coord : numpy.ndarray
            A numpy array of floats with shape (1, 2) representing the new coordinates.
        location : str or None
            The building ID if the step is a stay, or `None` if the step is a move.
        """
        city = self.city

        # Resolve destination building geometry and attributes from cache
        brow = self._current_dest_building_row
        dest_type = brow['building_type']

        # Determine start block and geometry
        start_block = tuple(np.floor(start_point).astype(int))
        start_point_arr = np.asarray(start_point, dtype=float)
        
        # Check if agent is in building using integer truncation
        in_current_dest = _point_in_blocks(start_point_arr, self._current_dest_building_row.get('blocks_set')) if self._current_dest_building_row is not None else False
        in_previous_dest = _point_in_blocks(start_point_arr, self._previous_dest_building_row.get('blocks_set')) if self._previous_dest_building_row is not None else False

        # If already at destination building area, stay-within-building dynamics
        if in_current_dest:
            # Clear cache when arriving at destination (bound_poly only for inter-building movement)
            self._cached_path_ml = None
            self._cached_bound_poly = None
            self._cached_dest_id = None
            location = brow['id']
            p = self.still_probs.get(dest_type, 0.5)
            sigma = self.speeds.get(dest_type, 0.5)

            if rng.uniform() < p:
                coord = start_point_arr
            else:
                # Draw until coord falls inside building
                while True:
                    coord = rng.normal(loc=start_point_arr, scale=sigma*np.sqrt(dt), size=2)
                    if _point_in_blocks(coord, brow.get('blocks_set')):
                        break

            return coord, location

        # Otherwise, move along streets toward destination door cell
        location = None
        start_segment = []
        # If currently inside previous destination building, first move towards its door
        if in_previous_dest:
            prev_door_point = self._previous_dest_building_row['door_point']
            start_segment = [tuple(start_point), prev_door_point]
            start_node = (int(self._previous_dest_building_row['door_cell_x']), int(self._previous_dest_building_row['door_cell_y']))
        else:
            start_node = start_block

        # Resolve destination door coordinates for path computation
        dest_cell = (int(brow['door_cell_x']), int(brow['door_cell_y']))

        # Check if cached geometry is valid for current destination
        use_cache = False
        if self._cached_path_ml is not None and self._cached_bound_poly is not None and self._cached_dest_id == brow['id']:
            use_cache = True

        if use_cache:
            path_ml = self._cached_path_ml
            bound_poly = self._cached_bound_poly
            bound_poly_blocks_set = self._cached_bound_poly_blocks_set
        else:
            # Shortest path between street blocks (door cells)
            street_path = city.get_shortest_path(start_node, dest_cell)
            street_blocks = city.blocks_gdf.loc[street_path]

            # Build continuous path through block centers, include start/end segments
            centroids = street_blocks['geometry'].centroid
            path_segments = [start_segment + [(pt.x, pt.y) for pt in centroids] + [brow['door_point']]]
            path_ml = MultiLineString([linemerge(MultiLineString(path_segments))])
            street_geom = unary_union(street_blocks['geometry'])
            # Use previous destination building geometry if agent is departing from it, otherwise use start block geometry
            if in_previous_dest and self._previous_dest_building_row is not None:
                start_geom = self._previous_dest_building_row['geometry']
            else:
                start_geom = city.blocks_gdf.loc[start_block]['geometry']
            bound_poly = unary_union([start_geom, street_geom])
            
            # Build bound_poly_blocks_set from components
            if in_previous_dest and self._previous_dest_building_row is not None:
                start_blocks = self._previous_dest_building_row.get('blocks_set', set())
            else:
                start_blocks = {start_block}
            bound_poly_blocks_set = start_blocks | set(street_path)
            
            # Cache the results
            self._cached_path_ml = path_ml
            self._cached_bound_poly = bound_poly
            self._cached_dest_id = brow['id']
            self._cached_bound_poly_blocks_set = bound_poly_blocks_set

        # Transformed coordinates of current position along the path
        path_coord = _path_coords(path_ml, start_point_arr)

        heading_drift = 3.33 * dt
        sigma = 0.5 * dt / 1.96

        while True:
            # Step in transformed (path-based) space
            step = rng.normal(loc=[heading_drift, 0], scale=sigma * np.sqrt(dt), size=2)
            path_coord = (path_coord[0] + step[0], 0.7 * path_coord[1] + step[1])

            if path_coord[0] > path_ml.length:
                coord = np.array(brow['door_point'])
                break

            coord = _cartesian_coords(path_ml, *path_coord)

            if _point_in_blocks(coord, bound_poly_blocks_set):
                break

        return coord, location

    def _traj_from_dest_diary(self, dt, seed=0):
        """
        Simulate a trajectory and update agent diary from destination_diary.
        """
        rng = np.random.default_rng(seed)  # random generator for steps
        city = self.city
        destination_diary = self.destination_diary

        trajectory_update = []

        if self.diary.empty:
            current_entry = None
        else:
            current_entry = self.diary.iloc[-1].to_dict()
            self.diary = self.diary.iloc[:-1]

        tick_secs = int(60*dt)

        # Initialize previous destination building to building containing start ping, if any
        prev_ping = self.last_ping
        start_point = (prev_ping['x'], prev_ping['y'])
        start_block = tuple(np.floor(start_point).astype(int))
        start_info = city.get_block(start_block)

        if start_info['building_type'] is not None and start_info['building_type'] != 'street' and start_info['building_id'] is not None:
            building_dict = city.buildings_gdf.loc[start_info['building_id']].to_dict()
            building_dict['blocks_set'] = set(building_dict.get('blocks', []))
            self._previous_dest_building_row = building_dict
        else:
            self._previous_dest_building_row = None

        # Initialize current destination building to first entry
        first_building_id = destination_diary.iloc[0]['location']
        building_dict = city.buildings_gdf.loc[first_building_id].to_dict()
        building_dict['blocks_set'] = set(building_dict.get('blocks', []))
        self._current_dest_building_row = building_dict
        entry_update = []
        for i in range(destination_diary.shape[0]):
            building_id = destination_diary.iloc[i]['location']
            
            # Shift: previous = current, current = new destination (skip shift on first iteration)
            if i > 0:
                self._previous_dest_building_row = self._current_dest_building_row
            if building_id in city.buildings_gdf.index:
                building_dict = city.buildings_gdf.loc[building_id].to_dict()
                building_dict['blocks_set'] = set(building_dict.get('blocks', []))
                self._current_dest_building_row = building_dict
            else:
                self._current_dest_building_row = None

            duration_in_ticks = int(destination_diary.iloc[i]['duration'] / dt)
            for _ in range(duration_in_ticks):
                prev_ping = self.last_ping
                # define point                   
                start_point = (prev_ping['x'], prev_ping['y'])
                unix_timestamp = prev_ping['timestamp'] + tick_secs
                datetime = prev_ping['datetime'] + timedelta(seconds=tick_secs)

                coord, location = self._sample_step(start_point, dt, rng)
                ping = {'x': coord[0], 
                        'y': coord[1],
                        'datetime': datetime,
                        'timestamp': unix_timestamp,
                        'user_id': self.identifier}

                trajectory_update.append(ping)
                self.last_ping = ping
                if current_entry is None:
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
                    current_entry['duration'] += 1*dt  # add one tick to the duration

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
        return None

    def _initialize_visits_unif(self,
                                rng=None,
                                home_work_freq=20,
                                initial_k=None,
                                other_locs_freq=2):
        """Seed initial visit frequencies uniformly per building type

        Builds a fresh visit_freqs DataFrame from self.city.buildings_gdf and
        returns it. Does not mutate self.visit_freqs. Warns if an existing
        self.visit_freqs appears to already contain positive frequencies.
        """
        if rng is None:
            rng = npr.default_rng()
        if initial_k is None:
            initial_k = {'retail': 4, 'workplace': 2, 'home': 2, 'park': 2}

        # Warn if existing frequencies are already set
        if getattr(self, 'visit_freqs', None) is not None:
            if (self.visit_freqs['freq'] > 0).any():
                warnings.warn("Existing visit frequencies are non-zero; re-initializing will overwrite them.")

        bdf = self.city.buildings_gdf
        visit_freqs = pd.DataFrame({
            'id': bdf['id'].values,
            'building_type': bdf['building_type'].values,
            'freq': 0,
        }).set_index('id')

        # Strong prior for agent's home/work
        visit_freqs.loc[self.home, 'freq'] = int(home_work_freq)
        visit_freqs.loc[self.workplace, 'freq'] = int(home_work_freq)

        # Seed additional locations per type
        for btype, k in initial_k.items():
            k = int(k)
            ids = visit_freqs.index[visit_freqs.building_type == btype]
            # Exclude agent's own home/work
            excl = [self.home, self.workplace]
            ids = ids.difference(excl)

            size = min(k, len(ids))
            chosen = rng.choice(ids.to_numpy(), size=size, replace=False)
            visit_freqs.loc[chosen, 'freq'] = visit_freqs.loc[chosen, 'freq'] + int(other_locs_freq)

        return visit_freqs

    def generate_dest_diary(self, 
                             end_time, 
                             epr_time_res = 15,
                             stay_probs = DEFAULT_STAY_PROBS,
                             rho = 0.4, 
                             gamma = 0.3, 
                             seed = 0,
                             verbose = False):
        """Generate the destination diary (exploration + preferential return).

        Parameters
        ----------
        end_time : pd.Timestamp
            Generate until this timestamp (inclusive).
        epr_time_res : int
            Time-step in minutes for each diary entry.
        stay_probs : dict
            Probability of staying put, keyed by building type.
        rho : float
            Exploration parameter; lower values bias toward exploration.
        gamma : float
            Preferential return parameter; controls decay by visit count.
        seed : int
            RNG seed.

        Notes
        -----
        - Requires `city.grav` to be precomputed via `city.compute_gravity(...)`
        """
        rng = npr.default_rng(seed)

        # Early check: gravity matrix required for EPR generation
        if self.city.grav is None:
            raise RuntimeError("city.grav is not available. Call city.compute_gravity() before trajectory generation.")

        # Validate last_ping exists
        if self.last_ping is None:
            raise RuntimeError(
                "Agent has no last_ping. This should not happen unless reset_trajectory(last_ping=True) was called."
            )

        if end_time.tz is None:
            tz = getattr(self.last_ping['datetime'], 'tz', None)
            if tz is not None:
                end_time = end_time.tz_localize(tz)

        if isinstance(end_time, pd.Timestamp):
            end_time = to_timestamp(end_time)

        visit_freqs = self.visit_freqs
        if (visit_freqs is None) or (not (visit_freqs['freq'] > 0).any()):
            visit_freqs = self._initialize_visits_unif(
                rng=rng,
                home_work_freq=20,
                initial_k={'retail': 4, 'workplace': 2, 'home': 2, 'park': 2},
                other_locs_freq=2,
            )

        if self.destination_diary.empty:
            start_time_local = self.last_ping['datetime']
            start_time = self.last_ping['timestamp']
            curr_info = self.city.get_block((int(np.floor(self.last_ping['x'])), int(np.floor(self.last_ping['y']))))
            curr = curr_info['building_id'] if curr_info['building_type'] is not None and curr_info['building_type'] != 'street' and curr_info['building_id'] is not None else self.home
        else:
            last_entry = self.destination_diary.iloc[-1]
            last_datetime = pd.to_datetime(last_entry.datetime) if not isinstance(last_entry.datetime, pd.Timestamp) else last_entry.datetime
            start_time_local = last_datetime + timedelta(minutes=int(last_entry.duration))
            # Derive timestamp from datetime if not present
            if 'timestamp' in self.destination_diary.columns:
                start_time = int(last_entry.timestamp + last_entry.duration*60)
            else:
                start_time = to_timestamp(last_entry.datetime) + int(last_entry.duration*60)
            curr = last_entry.location

        # Check if start_time exceeds end_time
        if start_time > end_time:
            raise ValueError(
                f"Agent {self.identifier}: last_ping timestamp ({start_time}) is at or beyond end_time ({end_time}). "
                "No destinations will be generated. Consider providing an earlier last_ping or later end_time."
            )
            return
        
        dest_update = []
        # verbosity
        if verbose:
            print(f"Generating destination diary via EPR (rho={rho}, gamma={gamma}, epr_time_res={epr_time_res} min, seed={seed})")
        while start_time < end_time:
            curr_type = visit_freqs.loc[curr, 'building_type'] if curr in visit_freqs.index else 'home'
            allowed = allowed_buildings(start_time_local)
            x = visit_freqs.loc[(visit_freqs['building_type'].isin(allowed)) & (visit_freqs.freq > 0)]

            S = len(x) if len(x) > 0 else 1

            # probability of exploring
            p_exp = rho*(S**(-gamma))

            # Stay
            if (curr_type in allowed) & (rng.uniform() < stay_probs.get(curr_type, 0.5)):
                pass

            # Exploration
            elif rng.uniform() < p_exp:
                # Compute gravity probs from current door cell to unexplored candidates
                y = visit_freqs.loc[(visit_freqs['building_type'].isin(allowed)) & (visit_freqs.freq == 0)]
                if not y.empty:
                    if callable(self.city.grav):
                        probs = self.city.grav(curr).loc[y.index].values
                    else:
                        probs = self.city.grav.loc[curr, y.index].values
                    
                    probs = probs / probs.sum()
                    curr = rng.choice(y.index, p=probs)
                else:
                    # Preferential return
                    curr = _choose_destination(visit_freqs, x, rng)

                visit_freqs.loc[curr, 'freq'] += 1

            # Preferential return
            else:
                curr = _choose_destination(visit_freqs, x, rng)
                visit_freqs.loc[curr, 'freq'] += 1

            # Update destination diary
            entry = {'datetime': start_time_local,
                     'timestamp': start_time,
                     'duration': epr_time_res,
                     'location': curr}
            dest_update.append(entry)

            start_time_local = start_time_local + timedelta(minutes=int(epr_time_res))
            start_time = start_time + epr_time_res*60  # because start_time in seconds

        if self.destination_diary.empty:
            self.destination_diary = pd.DataFrame(dest_update)
        else:
            self.destination_diary = pd.concat(
                [self.destination_diary, pd.DataFrame(dest_update)], ignore_index=True)
        self.destination_diary = condense_destinations(self.destination_diary)

        self.visit_freqs = visit_freqs

        return None

    def generate_trajectory(self,
                            destination_diary=None,
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
            DataFrame containing 'location' and 'datetime' columns (required),
            and optionally 'timestamp' and 'duration'. If 'timestamp' is missing,
            it will be derived from 'datetime'.
        end_time : pd.Timestamp, optional
            The end time to generate the trajectory until. Required if
            destination_diary is empty.
        epr_time_res : int, optional
            The granularity of destination durations in epr generation (minutes).
        dt : float, optional
            Time step duration for trajectory simulation (minutes).
        seed : int, optional
            Random seed for reproducibility.
        step_seed : int, optional
            Random seed for trajectory steps. If None, uses seed.
        verbose : bool, optional
            Whether to print verbose warnings.
        kwargs : dict, optional
            Additional keyword arguments for setting initial position.
            Can include 'x', 'y', 'location', 'datetime', 'timestamp'.
            If 'x' and 'y' are provided, used directly. Otherwise, if 'location'
            is provided, uses that building's centroid. If neither, uses agent's home.

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
            if not self.destination_diary.empty:
                warnings.warn("Overwriting existing destination_diary with new one.")
            self.destination_diary = destination_diary

            row = destination_diary.iloc[0] #first destination 
            loc_centroid = self.city.buildings_gdf.loc[row['location'], 'geometry'].centroid
            x_coord, y_coord = loc_centroid.x, loc_centroid.y
            
            datetime = row['datetime']
            timestamp = int(row.get('timestamp', to_timestamp(datetime)))
            
            self.last_ping = pd.Series({
                'x': x_coord,
                'y': y_coord,
                'datetime': datetime,
                'timestamp': timestamp,
                'user_id': self.identifier
                })
            self.trajectory = pd.DataFrame([self.last_ping])

        # Handle trajectory initialization and overrides
        if self.trajectory is None:
            # Allow overriding last_ping fields with kwargs
            if 'x' in kwargs or 'y' in kwargs or 'location' in kwargs or 'datetime' in kwargs or 'timestamp' in kwargs:
                if 'x' in kwargs and 'y' in kwargs:
                    self.last_ping['x'] = kwargs['x']
                    self.last_ping['y'] = kwargs['y']
                elif 'location' in kwargs:
                    loc_centroid = self.city.buildings_gdf.loc[kwargs['location'], 'geometry'].centroid
                    self.last_ping['x'] = loc_centroid.x
                    self.last_ping['y'] = loc_centroid.y
                
                if 'datetime' in kwargs:
                    self.last_ping['datetime'] = kwargs['datetime']
                    self.last_ping['timestamp'] = to_timestamp(self.last_ping['datetime'])
                if 'timestamp' in kwargs:
                    self.last_ping['timestamp'] = kwargs['timestamp']
                    self.last_ping['datetime'] = pd.to_datetime(self.last_ping['timestamp'], unit='s')
            
            self.trajectory = pd.DataFrame([self.last_ping])

        else:
            if 'x' in kwargs or 'y' in kwargs or 'location' in kwargs or 'datetime' in kwargs or 'timestamp' in kwargs:
                raise ValueError(
                    "Keyword arguments conflict with existing trajectory. "
                    "Use Agent.reset_trajectory() or do not provide keyword arguments."
                )
            self.last_ping = self.trajectory.iloc[-1]

        if self.destination_diary.empty:
            if end_time is None:
                raise ValueError(
                    "Destination diary is empty. Provide an end_time to generate a trajectory."
                )
            self.generate_dest_diary(end_time=end_time,
                                      epr_time_res=epr_time_res,
                                      seed=seed)

        s = step_seed if step_seed else seed
        self._traj_from_dest_diary(dt=dt, seed=s)

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
            
        result = sample_bursts_gaps(
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


def condense_destinations(destination_diary, *, time_cols=None):
    """
    Modifies a destination diary, joining consecutive entries for the same location
    into a single entry with the total duration.

    Parameters
    ----------
    destination_diary : pandas.DataFrame
        Diary containing the destinations of the user.
    time_cols : dict, optional (keyword-only)
        Optional mapping for non-canonical column names. Expected keys:
        {'datetime': <col_name>, 'timestamp': <col_name>}.
        Defaults are 'datetime' and 'timestamp'.

    Returns
    -------
    pandas.DataFrame
        Updated destination diary with canonical columns 'datetime' and 'timestamp'.
    """

    if destination_diary.empty:
        return pd.DataFrame(columns=['datetime','timestamp','duration','location'])

    # Resolve column names
    dt_col = 'datetime'
    ts_col = 'timestamp'
    if time_cols:
        dt_col = time_cols.get('datetime', dt_col)
        ts_col = time_cols.get('timestamp', ts_col)

    # If inputs use local/unix names, allow mapping through kwargs only
    required = {'location', 'duration', dt_col, ts_col}
    missing = required - set(destination_diary.columns)
    if missing:
        raise KeyError(f"condense_destinations expected columns {required}, missing {missing}")

    df = destination_diary.copy()

    # Detect changes in location
    df['new_segment'] = df['location'].ne(df['location'].shift())
    # Create segment identifiers for grouping
    df['segment_id'] = df['new_segment'].cumsum()

    # Aggregate data by segment with provided column names, then rename canonically
    condensed_df = df.groupby('segment_id').agg({
        dt_col: 'first',
        ts_col: 'first',
        'duration': 'sum',
        'location': 'first'
    }).reset_index(drop=True)

    # Canonical column names in the output
    if dt_col != 'datetime':
        condensed_df = condensed_df.rename(columns={dt_col: 'datetime'})
    if ts_col != 'timestamp':
        condensed_df = condensed_df.rename(columns={ts_col: 'timestamp'})

    return condensed_df


def generate_ping_times(t0,
                        t_end,
                        *,
                        beta_start=None,
                        beta_durations=None,
                        beta_ping=5,
                        seed=None,
                        return_bursts=False,
                        tz=None):
    """Generate absolute ping timestamps (seconds) within [t0, t_end].

    If return_bursts is True, also returns a list of (start_time, end_time)
    for bursts that produced at least one ping. If tz is provided, start/end
    are timezone-aware pandas Timestamps; otherwise they are Unix seconds (int).
    """
    rng = npr.default_rng(seed)

    # convert minutes→seconds
    beta_ping_s = beta_ping * 60
    beta_start_s = beta_start * 60 if beta_start is not None else None
    beta_dur_s = beta_durations * 60 if beta_durations is not None else None

    if beta_start_s is None and beta_dur_s is None:
        burst_start_points = np.array([0.0])
        burst_end_points = np.array([t_end - t0], dtype=float)
    else:
        est_n = int(3 * (t_end - t0) / beta_start_s) + 10
        inter_arrival_times = rng.exponential(scale=beta_start_s, size=est_n)
        burst_start_points = np.cumsum(inter_arrival_times)
        burst_start_points = burst_start_points[burst_start_points < (t_end - t0)]
        burst_durations = rng.exponential(scale=beta_dur_s, size=burst_start_points.size)
        burst_end_points = burst_start_points + burst_durations
        if burst_end_points.size > 0:
            burst_end_points[:-1] = np.minimum(burst_end_points[:-1], burst_start_points[1:])
            burst_end_points[-1] = min(burst_end_points[-1], t_end - t0)

    ping_times_chunks: list[np.ndarray] = []
    bursts_out = [] if return_bursts else None
    for start, end in zip(burst_start_points, burst_end_points):
        dur = end - start
        if dur <= 0:
            continue
        est_pings = int(3 * dur / beta_ping_s) + 10
        ping_intervals = rng.exponential(scale=beta_ping_s, size=est_pings)
        times_rel = np.cumsum(ping_intervals)
        times_rel = times_rel[times_rel < dur]
        if times_rel.size:
            ping_times_chunks.append(t0 + start + times_rel)
            if return_bursts:
                if tz is not None:
                    sdt = pd.to_datetime(t0 + start, unit='s', utc=True).tz_convert(tz)
                    edt = pd.to_datetime(t0 + end, unit='s', utc=True).tz_convert(tz)
                else:
                    sdt = int(t0 + start)
                    edt = int(t0 + end)
                bursts_out.append([sdt, edt])

    if not ping_times_chunks:
        if return_bursts:
            return np.array([], dtype=int), []
        return np.array([], dtype=int)
    ping = np.concatenate(ping_times_chunks).astype(int)
    if return_bursts:
        return ping, bursts_out
    return ping


def thin_traj_by_times(traj,
                       ping_times,
                       *,
                       deduplicate=True):
    """Apply ping_times to a dense traj via searchsorted thinning."""
    if ping_times.size == 0:
        return pd.DataFrame(columns=traj.columns)

    traj_ts = traj['timestamp'].to_numpy()
    tz = traj['datetime'].dt.tz

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
    return sampled_traj


def _sample_horizontal_noise(n,
                             *,
                             ha=3/4,
                             rng=None):
    """Sample per-ping horizontal accuracy and Gaussian noise (internal)."""
    if ha is None or ha==0:
        return np.zeros(n), np.zeros((n, 2))

    if rng is None:
        rng = npr.default_rng()
    x_m = 8/15
    if ha <= x_m:
        raise ValueError("ha must exceed 8 m / 15 m ≈ 0.533 blocks")
    alpha = ha / (ha - x_m)
    ha_realized = (rng.pareto(alpha, size=n) + 1) * x_m
    ha_realized = np.minimum(ha_realized, 20, out=ha_realized)
    sigma = ha_realized / 1.515
    noise = rng.standard_normal((n, 2)) * sigma[:, None]
    np.clip(noise, -250, 250, out=noise)
    return ha_realized, noise


def _point_in_blocks(point_arr, blocks_set):
    """Check if point is in any block using integer truncation."""
    if blocks_set is None:
        return False
    block_idx = (int(np.floor(point_arr[0])), int(np.floor(point_arr[1])))
    return block_idx in blocks_set


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
                 city,
                 dt=1):
        self.roster = {}
        self.city = city
        self.dt = dt

    def add_agent(self,
                  agent,
                  verbose=True):
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

    def generate_agents(self, N, seed=0, name_count=2, agent_homes=None, agent_workplaces=None, datetimes=None):
        """
        Generates N agents, with randomized attributes.
        """
        master_rng = np.random.default_rng(seed)
        generator = funkybob.UniqueRandomNameGenerator(members=name_count, seed=seed)

        # Create efficient accessors for agent homes and workplaces
        get_home = parse_agent_attr(agent_homes, N, "agent_homes")
        get_workplace = parse_agent_attr(agent_workplaces, N, "agent_workplaces")
        get_datetime = parse_agent_attr(datetimes, N, "datetimes")

        for i in range(N):
            agent_seed = int(master_rng.integers(0, 2**32))
            identifier = generator[i]              
            agent = Agent(identifier=identifier,
                          city=self.city,
                          home=get_home(i),
                          workplace=get_workplace(i),
                          datetime=get_datetime(i),
                          seed=agent_seed)
            self.add_agent(agent)

    def save_pop(self,
                 traj_cols=None,
                 sparse_path=None,
                 full_path=None,
                 homes_path=None,
                 diaries_path=None,
                 dest_diaries_path=None,
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
        dest_diaries_path : str or Path, optional
            Destination path for destination diaries.
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
            if partition_cols and 'date' in partition_cols and 'date' not in full_df.columns:
                full_df['date'] = pd.to_datetime(full_df['timestamp'], unit='s').dt.date.astype(str)

            full_df = from_df(full_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(full_df,
                    path=full_path,
                    format=fmt,
                    partition_by=partition_cols,
                    filesystem=filesystem,
                    existing_data_behavior='delete_matching')
    
        if sparse_path:
            sparse_df = pd.concat([agent.sparse_traj for agent in self.roster.values()], ignore_index=True)
            if partition_cols and 'date' in partition_cols and 'date' not in sparse_df.columns:
                sparse_df['date'] = pd.to_datetime(sparse_df['timestamp'], unit='s').dt.date.astype(str)

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
            if partition_cols and 'date' in partition_cols and 'date' not in diaries_df.columns:
                diaries_df['date'] = pd.to_datetime(diaries_df['timestamp'], unit='s').dt.date.astype(str)

            diaries_df = from_df(diaries_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
            to_file(diaries_df,
                    path=diaries_path,
                    format=fmt,
                    partition_by=partition_cols,
                    filesystem=filesystem,
                    existing_data_behavior='delete_matching',
                    traj_cols=traj_cols)
    
        if dest_diaries_path:
            # TODO: from_df should be made compatible with destination diaries
            dest_diaries_list = []
            for agent in self.roster.values():
                if agent.destination_diary is not None and not agent.destination_diary.empty:
                    df = agent.destination_diary.copy()
                    df['identifier'] = agent.identifier
                    dest_diaries_list.append(df)
            
            if dest_diaries_list:
                dest_diaries_df = pd.concat(dest_diaries_list, ignore_index=True)
                dest_diaries_df['date'] = pd.to_datetime(dest_diaries_df['datetime'], unit='s').dt.date.astype(str)
                dest_diaries_df = from_df(dest_diaries_df, traj_cols=traj_cols, mixed_timezone_behavior=mixed_timezone_behavior)
                to_file(dest_diaries_df,
                        path=dest_diaries_path,
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

    def reproject_to_mercator(self, sparse_traj=True, full_traj=False, diaries=False, poi_data=None):
        """
        Reproject all agent trajectories from city block coordinates to Web Mercator.
        Uses the city's stored transformation parameters (block_side_length, web_mercator_origin_x/y).
        
        Parameters
        ----------
        sparse_traj : bool, default True
            Whether to reproject sparse trajectories
        full_traj : bool, default False
            Whether to reproject full trajectories  
        diaries : bool, default False
            Whether to reproject diaries (must have x, y columns)
        poi_data : pd.DataFrame, optional
            DataFrame with building coordinates (building_id, x, y) to join with diaries.
            If not provided, derived from city's buildings_gdf using door coordinates.
        """
        for agent in self.roster.values():
            if sparse_traj and agent.sparse_traj is not None:
                agent.sparse_traj = self.city.to_mercator(agent.sparse_traj)
            
            if full_traj and agent.trajectory is not None:
                agent.trajectory = self.city.to_mercator(agent.trajectory)
            
            if diaries and agent.diary is not None:
                # Derive poi_data from city's buildings_gdf if not provided
                if poi_data is None:
                    bdf = self.city.buildings_gdf
                    poi_data = pd.DataFrame({
                        'building_id': bdf['id'].values,
                        'x': (bdf['door_cell_x'].astype(float) + 0.5).values,
                        'y': (bdf['door_cell_y'].astype(float) + 0.5).values
                    })

                agent.diary = agent.diary.merge(poi_data, left_on='location', right_on='building_id', how='left')
                agent.diary = agent.diary.drop(columns=['building_id'])
                agent.diary = self.city.to_mercator(agent.diary)

# =============================================================================
# AUXILIARY METHODS
# =============================================================================

def _choose_destination(visit_freqs, x, rng):
    """
    Select destination using preferential return from allowed, visited buildings.
    Falls back to uniform random selection if no visited buildings are available.
    
    Parameters
    ----------
    visit_freqs : pandas.DataFrame
        DataFrame with building IDs as index and 'freq' column
    x : pandas.DataFrame
        Subset of visit_freqs with allowed, visited buildings (freq > 0)
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    str
        Building ID of selected destination
    """
    if not x.empty and x['freq'].sum() > 0:
        return rng.choice(x.index, p=x['freq']/x['freq'].sum())
    else:
        return rng.choice(visit_freqs.index)


def allowed_buildings(local_ts):
    """
    Finds allowed buildings for the timestamp
    """
    hour = local_ts.hour
    return ALLOWED_BUILDINGS[hour]


## moved into Agent as _initialize_visits_unif
