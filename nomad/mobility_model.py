from shapely.geometry import box, Point, Polygon, LineString, MultiLineString
from shapely.affinity import scale, rotate
from shapely.ops import unary_union
import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import funkybob
import networkx as nx
import nx_parallel as nxp
import s3fs
import pyarrow

import sampler
import stop_detection as sd
from constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS
from constants import FAST_STILL_PROBS, SLOW_STILL_PROBS, ALLOWED_BUILDINGS, DEFAULT_STAY_PROBS

import pdb


# =============================================================================
# STREET CLASS
# =============================================================================


class Street:
    """
    Class for street block in the city through which individuals move from building to building.

    Attributes
    ----------
    coordinates : tuple
        A tuple representing the (x, y) coordinates of the street block.
    neighbors_streets : list
        A list of neighboring Street objects.
    neighbors_buildings : list
        A list of neighboring Building objects.
    geometry : shapely.geometry.polygon.Polygon
        A polygon representing the geometry of the street block.
    id : str
        A unique identifier for the street block, formatted as 's-x{coordinates[0]}-y{coordinates[1]}'.

    Methods
    -------
    add_neighbor
        Adds a neighboring street or building to the street block.
    """

    def __init__(self, 
                 coordinates: tuple):

        self.coordinates = coordinates
        self.neighbors_streets = []
        self.neighbors_buildings = []
        self.geometry = box(coordinates[0], coordinates[1],
                            coordinates[0]+1, coordinates[1]+1)

        self.id = f's-x{coordinates[0]}-y{coordinates[1]}'

    def add_neighbor(self, neighbor):
        """
        Parameters
        ----------
        neighbor : object
            The neighbor object to be added. It can be of type Street or Building.

        Returns
        -------
        bool
            True if the neighbor was successfully added, False otherwise.
        """
        
        if neighbor is None or not check_adjacent(self.geometry, neighbor.geometry):
            return False
        if isinstance(neighbor, Street):
            self.neighbors_streets.append(neighbor)
            return True
        if isinstance(neighbor, Building) and neighbor.door == self.coordinates:
            self.neighbors_buildings.append(neighbor)
            return True
        return False


# =============================================================================
# BUILDING CLASS
# =============================================================================


class Building:
    """
    Class for buildings in the city where individuals dwell with different parameters.

    Attributes
    ----------
    building_type : str
        The type of building, e.g., 'home', 'work', 'retail', 'park'.
    door : tuple
        The coordinates of the door of the building.
    city : City
        The city object containing the building.
    still_prob : float
        The probability of an individual staying still in the building.
    sigma : float
        The standard deviation of the Brownian motion for an individual in the building.
    id : str
        A unique identifier for the building, formatted as '{building_type[0]}-x{door[0]}-y{door[1]}'.
    blocks : list
        A list of blocks that the building spans.
    geometry : shapely.geometry.polygon.Polygon
        A polygon representing the geometry of the building.
    door_centroid : tuple
        The coordinates of the door centroid of the building.

    Methods
    -------
    None
    """

    def __init__(self,
                 building_type: str, 
                 door: tuple, 
                 city,
                 still_prob: float, 
                 sigma: float, 
                 blocks: list = None, 
                 bbox: Polygon = None):

        self.building_type = building_type
        self.door = door
        self.city = city

        self.still_prob = still_prob
        self.sigma = sigma

        self.id = f'{building_type[0]}-x{door[0]}-y{door[1]}'

        # Calculate the bounding box of the building
        if blocks:
            min_x = min([block[0] for block in blocks])
            min_y = min([block[1] for block in blocks])
            max_x = max([block[0]+1 for block in blocks])
            max_y = max([block[1]+1 for block in blocks])
            bbox = box(min_x, min_y, max_x, max_y)
        elif bbox:
            blocks = []
            for x in range(int(bbox.bounds[0]), int(bbox.bounds[2])):
                for y in range(int(bbox.bounds[1]), int(bbox.bounds[3])):
                    blocks += [(x, y)]
        else:
            raise ValueError(
                "Either blocks spanned or bounding box must be provided."
            )

        self.blocks = blocks
        self.geometry = bbox

        # Compute door centroid
        door = self.geometry.intersection(self.city.streets[self.door].geometry)
        self.door_centroid = ((door.coords[0][0] + door.coords[1][0]) / 2,
                              (door.coords[0][1] + door.coords[1][1]) / 2)


# =============================================================================
# CITY CLASS
# =============================================================================


class City:
    """
    Class for representing a city containing buildings, streets, and methods for city management.

    Attributes
    ----------
    buildings : dict
        A dictionary of Building objects with their IDs as keys.
    streets : dict
        A dictionary of Street objects with their coordinates as keys.
    buildings_outline : shapely.geometry.polygon.Polygon
        A polygon representing the combined geometry of all buildings in the city.
    address_book : dict
        A dictionary mapping coordinates to Building objects.
    still_probs : dict
        A dictionary containing the probability of an individual staying still in each type of building.
    sigma : dict
        A dictionary containing the standard deviation of the Brownian motion for each type of building.
    city_boundary : shapely.geometry.polygon.Polygon
        A polygon representing the boundary of the city.
    dimensions : tuple
        A tuple representing the dimensions of the city (width, height).
    street_graph : dict
        A dictionary representing the graph of streets with their neighbors.
    shortest_paths : dict
        A dictionary containing the shortest paths between all pairs of streets.
    gravity : pandas.DataFrame
        A DataFrame containing the gravity values between all pairs of streets.

    Methods
    -------
    add_building
        Adds a building to the city.
    get_block
        Returns the block (Street or Building) at the given coordinates.
    get_street_graph
        Constructs the graph of streets and calculates the shortest paths between all pairs of streets.
    save
        Saves the city object to a file.
    plot_city
        Plots the city on a given matplotlib axis.
    """

    def __init__(self,
                 dimensions: tuple = (0,0),
                 still_probs: dict = DEFAULT_STILL_PROBS,
                 speeds=DEFAULT_SPEEDS):

        self.buildings = {}
        self.streets = {}
        self.buildings_outline = Polygon()
        self.address_book = {}

        self.still_probs = still_probs

        # controls "speed" of Brownian motion when simulating stay trajectory
        # The random variable X(t) of the position at time t has a normal
        # distribution with mean 0 and variance sigma^2 * t.
        # x/1.96 = 95% probability of moving x standard deviations
        self.sigma = speeds

        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all(isinstance(d, int) for d in dimensions)):
            raise ValueError("Dimensions must be a tuple of two integers.")
        self.city_boundary = box(0, 0, dimensions[0], dimensions[1])

        for x in range(0, dimensions[0]):
            for y in range(0, dimensions[1]):
                self.streets[(x, y)] = Street((x, y))
        self.dimensions = dimensions

    def add_building(self, building_type, door, blocks=None, bbox=None):
        """
        Adds a building to the city.

        Parameters
        ----------
        building_type : str
            The type of building, e.g., 'home', 'work', 'retail', 'park'.
        door : tuple
            The coordinates of the door of the building.
        blocks : list
            A list of blocks that the building spans.
        bbox : shapely.geometry.polygon.Polygon
            A polygon representing the bounding box of the building.
        """

        building = Building(building_type=building_type,
                            door=door,
                            city=self,
                            still_prob=self.still_probs[building_type],
                            sigma=self.sigma[building_type],
                            blocks=blocks, 
                            bbox=bbox)

        combined_plot = unary_union([building.geometry, self.streets[door].geometry])
        if self.buildings_outline.contains(combined_plot) or self.buildings_outline.overlaps(combined_plot):
            raise ValueError(
                "New building or its door overlap with existing buildings."
            )

        if not check_adjacent(building.geometry, self.streets[door].geometry):
            raise ValueError(f"Door {door} must be adjacent to new building.")

        # add building
        self.buildings[building.id] = building
        self.buildings_outline = unary_union([self.buildings_outline, building.geometry])

        # blocks are no longer streets
        for block in building.blocks:
            self.address_book[block] = building
            del self.streets[block]

        # expand city boundary if necessary
        buffered_building_geom = building.geometry.buffer(1)
        if not self.city_boundary.contains(buffered_building_geom):
            new_boundary = self.city_boundary.union(buffered_building_geom).envelope
            self.city_boundary = new_boundary
            self.dimensions = (int(new_boundary.bounds[2]), int(new_boundary.bounds[3]))
            # Update the streets for any new blocks within the expanded boundary
            minx, miny, maxx, maxy = map(int, new_boundary.bounds)
            for x in range(minx, maxx+1):
                for y in range(miny, maxy+1):
                    if (x, y) not in self.streets:
                        # Initialize new Street objects for the expanded city area
                        self.streets[(x, y)] = Street((x, y))

    def get_block(self, coordinates):
        """
        Returns the block (Street or Building) at the given coordinates.

        Parameters
        ----------
        coordinates : tuple
            A tuple representing the (x, y) coordinates of the block.

        Returns
        -------
        object
            The block object at the given coordinates 
            or None if the coordinates are outside the city.
        """

        x, y = coordinates
        bx, by = self.dimensions

        if (x < 0 or x >= bx or y < 0 or y >= bx):
            return None

        new_coords = (int(x), int(y))
        if new_coords in self.address_book:
            return self.address_book[new_coords]
        else:
            return self.streets[new_coords]

    def get_street_graph(self):
        """
        Generates a street graph from the streets data, calculates the shortest paths between all pairs of nodes,
        and computes a gravity DataFrame based on the inverse square of the shortest path lengths.

        Add the following attributes to the City object:
            self.street_graph (dict): A dictionary representing the street graph with coordinates as keys and lists of neighboring coordinates as values.
            self.shortest_paths (dict): A dictionary containing the shortest paths between all pairs of nodes.
            self.gravity (pd.DataFrame): A DataFrame indexed by origin and destination coordinates, containing gravity values based on the shortest path lengths.
        """

        self.street_graph = {}
        for coords, _ in self.streets.items():
            x, y = coords
            neighbors = [
                (x, y + 1),
                (x, y - 1),
                (x + 1, y),
                (x - 1, y)
            ]

            self.street_graph[coords] = [neighbor for neighbor in neighbors if neighbor in self.streets]

        G = nx.from_dict_of_lists(self.street_graph)
        sp = dict(nxp.all_pairs_shortest_path(G))
        self.shortest_paths = {node: paths for node, paths in sp.items()}

        data = [
            {'origin': origin, 'dest': dest, 'gravity': (1 / (len(path) - 1) ** 2 if len(path) > 1 else 0)}
            for origin, paths in sp.items()
            for dest, path in paths.items()
        ]
        self.gravity = pd.DataFrame(data, columns=['origin', 'dest', 'gravity'])
        self.gravity = self.gravity.set_index(['origin', 'dest'])

    def save(self, filename):
        """
        Saves the city object to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the city object to.
        """

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def plot_city(self, ax, doors=True, address=True, zorder=1, heatmap_agent=None):
        """
        Plots the city on a given matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the city.
        doors : bool
            Whether to plot doors of buildings.
        address : bool
            Whether to plot the address of buildings.
        zorder : int
            The z-order of the plot.
        heatmap_agent : Agent
            The agent for which to plot a heatmap of time spent in each building.
        """

        # Draw city boundary
        x, y = self.city_boundary.exterior.xy
        ax.plot(np.array(x), np.array(y), linewidth=2, color='black')  # Dashed line for the boundary

        # Define colors for different building types
        colors = {
            'home': 'skyblue',
            'work': '#C9A0DC',
            'retail': 'lightgrey',
            'park': 'lightgreen'
        }

        if heatmap_agent is not None:
            weights = heatmap_agent.diary.groupby('location').duration.sum()
            norm = Normalize(vmin=0, vmax=max(weights)/2)
            base_color = np.array([1, 0, 0])  # RGB for red

            for building in self.buildings.values():
                x, y = building.geometry.exterior.xy
                weight = weights.get(building.id, 0)
                alpha = norm(weight) if weight > 0 else 0

                ax.fill(x, y, facecolor=base_color, alpha=alpha,
                        edgecolor='black', linewidth=0.5,
                        label=building.building_type.capitalize(), zorder=zorder)
                ax.plot(x, y, color='black', alpha=1, linewidth=0.5, zorder=zorder + 1)

                if doors:
                    door_line = building.geometry.intersection(self.streets[building.door].geometry)
                    scaled_door_line = scale(door_line, xfact=0.25, yfact=0.25, origin=door_line.centroid)
                    dx, dy = scaled_door_line.xy
                    ax.plot(dx, dy, linewidth=2, color='white', zorder=zorder)

                if address:
                    door_coord = building.door
                    bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                    axes_width_in_inches = bbox.width
                    axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                    fontsize = (axes_width_in_inches / axes_data_range) * 13  # Example scaling factor

                    ax.text(door_coord[0] + 0.15, door_coord[1] + 0.15,
                            f"{door_coord[0]}, {door_coord[1]}",
                            ha='left', va='bottom',
                            fontsize=fontsize, color='black')

            sm = ScalarMappable(cmap=cm.Reds, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.02)
            cbar.set_label('Minutes Spent')

        else:
            for building in self.buildings.values():
                x, y = building.geometry.exterior.xy
                ax.fill(x, y, facecolor=colors[building.building_type],
                        edgecolor='black', linewidth=0.5,
                        label=building.building_type.capitalize(), zorder=zorder)

                # Plot doors
                if doors:
                    door_line = building.geometry.intersection(self.streets[building.door].geometry)
                    scaled_door_line = scale(door_line, xfact=0.25, yfact=0.25, origin=door_line.centroid)
                    dx, dy = scaled_door_line.xy
                    ax.plot(dx, dy, linewidth=2, color='white', zorder=zorder)

                if address:
                    door_coord = building.door
                    bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                    axes_width_in_inches = bbox.width
                    axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                    fontsize = (axes_width_in_inches / axes_data_range) * 13  # Example scaling factor

                    ax.text(door_coord[0] + 0.15, door_coord[1] + 0.15,
                            f"{door_coord[0]}, {door_coord[1]}",
                            ha='left', va='bottom',
                            fontsize=fontsize, color='black')

        ax.set_aspect('equal')

        # Set integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


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

            diary_entry = {'unix_timestamp': unix_timestamp,
                           'local_timestamp': local_timestamp,
                           'duration': dt,
                           'location': home}
            self.diary = pd.DataFrame([diary_entry])

        self.trajectory = trajectory

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

    def sample_traj_hier_nhpp(self, beta_start, beta_durations, beta_ping, seed=0):
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
        """

        sparse_traj = sampler.sample_hier_nhpp(self.trajectory, beta_start, beta_durations, beta_ping, seed=seed)
        sparse_traj = sparse_traj.set_index('unix_timestamp', drop=False)
        self.sparse_traj = sparse_traj


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
# Container for all the agents and responsible for their initialization and
# randomizing their attributes and trajectories
# =============================================================================


class Population:
    """
    A class to represent a population of agents within a city.

    Attributes
    ----------
    roster : dict
        A dictionary to store agents with their identifiers as keys.
    city : City
        The city in which the population resides.
    global_execution : int
        A counter for global execution.

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
        self.global_execution = 0

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
            Whether to print a message if the agent identifier already exists in the population.
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
#     @staticmethod
#     def process_agent(row, load_trajectories, load_diaries, city):
#         agent_id = row['id']
#         traj = load_trajectories[load_trajectories['identifier'] == agent_id].reset_index(drop=True)
#         diary = load_diaries[load_diaries['id'] == agent_id].reset_index(drop=True)
#         agent = Agent(agent_id, 
#                       row['home'], 
#                       row['workplace'], 
#                       city,
#                       trajectory=traj,
#                       diary=diary)
#         return agent

#     def load_pop(self, path, parallelize=False):
#         """
#         gotta fix this -- it's too slow! maybe parquet/s3
#         """
#         load_trajectories = pd.read_csv(path+'trajectories.csv', 
#                                         usecols=lambda column: column != 'Unnamed: 0')
#         load_diaries = pd.read_csv(path+'diaries.csv', 
#                                    usecols=lambda column: column != 'Unnamed: 0')
#         load_homes = pd.read_csv(path+'homes.csv', 
#                                  usecols=lambda column: column != 'Unnamed: 0')

#         if parallelize:
#             with ProcessPoolExecutor() as executor:
#                 agents = list(executor.map(self.process_agent, [row for _, row in load_homes.iterrows()],
#                                            [load_trajectories] * len(load_homes),
#                                            [load_diaries] * len(load_homes),
#                                            [self.city] * len(load_homes)))
#             for agent in agents:
#                 self.add_agent(agent)
#         else:
#             for _, row in load_homes.iterrows():
#                 agent = self.process_agent(row, load_trajectories, load_diaries, self.city)
#                 self.add_agent(agent)

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


            #PACO: SHOULD THESE ALSO BE CONSTANT?
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

    def traj_from_dest_diary(self, agent, dt):
        """
        Simulate a trajectory and give agent true travel diary attribute.

        Parameters
        ----------
        agent: Agent

        destination_diary: Pandas Dataframe
            with "unix_timestamp", "local_timestamp", "duration", "location"

        Returns
        -------
        None (updates agent.trajectory, agent.diary)
        """

        city = self.city

        destination_diary = agent.destination_diary
        #print(destination_diary)

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
        #print(trajectory_update)
        agent.trajectory = pd.concat([agent.trajectory, pd.DataFrame(trajectory_update)],
                                ignore_index=True)

        entry_update.append(current_entry)
        if(agent.diary.empty):
            agent.diary = pd.DataFrame(entry_update)
        else:
            pd.concat([agent.diary, pd.DataFrame(entry_update)], ignore_index=True)
        agent.destination_diary = destination_diary.drop(destination_diary.index)

    def generate_dest_diary(self, agent, T, duration=15,
                            stay_probs=DEFAULT_STAY_PROBS,
                            rho=0.6, gamma=0.2, seed=0):
        """
        Exploration and preferential return.

        Parameters
        ----------
        T : int
            Timestamp until which to generate.
        rho : float
            Parameter for exploring, influencing the probability of exploration.
        gamma : float
            Parameter for exploring, influencing the probability of preferential return.
        stay_probs : dict
            Dictionary containing the probability of staying in the same building.
            This is modeled as a geometric distribution with `p = 1 - ((1/avg_duration_hrs)/timesteps_in_1_hr)`.
        """
        npr.seed(seed)

        id2door = pd.DataFrame([[s, b.door] for s, b in self.city.buildings.items()],
                               columns=['id', 'door']).set_index('door')  # could this be a field of city?

        if isinstance(T, datetime):
            T = int(T.timestamp())  # Convert to unix

        probs = pd.DataFrame({
            'id': list(self.city.buildings.keys()),
            'type': [b.building_type for b in self.city.buildings.values()],
            'freq': 0,
            'p': 0
        }).set_index('id')

        # Initializes past counts randomly
        probs.loc[agent.home, 'freq'] = 25
        probs.loc[agent.workplace, 'freq'] = 25
        probs.loc[probs.type == 'park', 'freq'] = 3  # Agents love to comeback to park
        # ALTERNATIVELY: start with 1 at home and 1 at work, and do a burnout period of 2 weeks. 

        initial_locs = []
        initial_locs += list(npr.choice(probs.loc[probs.type == 'retail'].index, size=npr.poisson(8)))
        initial_locs += list(npr.choice(probs.loc[probs.type == 'work'].index, size=npr.poisson(4)))
        initial_locs += list(npr.choice(probs.loc[probs.type == 'home'].index, size=npr.poisson(4)))
        probs.loc[initial_locs, 'freq'] += 1

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
            curr_type = probs.loc[curr, 'type']
            allowed = allowed_buildings(start_time_local)
            x = probs.loc[(probs['type'].isin(allowed)) & (probs.freq > 0)]

            S = len(x) # Fix depending on whether "explore" should depend only on allowed buildings

            #probability of exploring
            p_exp = rho*(S**(-gamma))

            # You should be asleep! I don't think we're doing this anymore
            # if (start_time_local.hour >= 22) | (start_time_local.hour <= 5):
            #     curr = agent.home  # this doesn't permit the possibility of sleepovers
            #     probs.loc[curr, 'freq'] += 1
            #     pass

            # Stay
            if (curr_type in allowed) & (npr.uniform() < stay_probs[curr_type]):
                pass

            # Exploration
            elif npr.uniform() < p_exp:
                probs['p'] = self.city.gravity.xs(
                    self.city.buildings[curr].door, level=0).join(id2door, how='right').set_index('id')
                y = probs.loc[(probs['type'].isin(allowed)) & (probs.freq == 0)]

                if not y.empty and y['p'].sum() > 0:
                    curr = npr.choice(y.index, p=y['p']/y['p'].sum())
                else:
                    # If there are no more buildings to explore, then preferential return
                    curr = npr.choice(x.index, p=x['freq']/x['freq'].sum())

                probs.loc[curr, 'freq'] += 1

            # Preferential return
            else:
                curr = npr.choice(x.index, p=x['freq']/x['freq'].sum())
                probs.loc[curr, 'freq'] += 1

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

        return None

    def generate_trajectory(self, agent, T=None, duration=15, seed=0, dt=1):
        """
        Generate a trajectory for an agent.

        Parameters
        ----------
        agent : Agent
            The agent for whom to generate a trajectory.
        T : int, optional
            Timestamp until which to generate.
        duration : int, optional
            Duration of each stay in minutes.
        seed : int, optional
            Random seed for reproducibility.
        dt : float, optional
            Time step duration.
        
        Returns
        -------
        None (updates agent.trajectory)
        """

        npr.seed(seed)

        if agent.destination_diary.empty:
            if T is None:
                raise ValueError(
                    "Destination diary is empty. Provide a parameter T to generate destination diary from transition matrix."
                )
            self.generate_dest_diary(agent, T, duration=duration, seed=seed)
        #print(agent.destination_diary)
        self.traj_from_dest_diary(agent, dt)

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


def load(filename):
    """Load a city object from a file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save(city, filename):
    """Save the city object to a file."""
    with open(filename, 'wb') as file:
        pickle.dump(city, file)


def check_adjacent(geom1, geom2):
    intersection = geom1.intersection(geom2)
    return isinstance(intersection, (LineString, MultiLineString))


def allowed_buildings(local_ts):
    """
    Finds allowed buildings for the timestamp
    """
    hour = local_ts.hour
    return ALLOWED_BUILDINGS[hour]