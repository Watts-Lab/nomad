from shapely.geometry import box, Point, Polygon, LineString, MultiLineString
from shapely.affinity import scale, rotate
from shapely.ops import unary_union
import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import numpy.random as npr
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import funkybob

import mobility_model as mmod
import stop_detection as sd
from constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS, FAST_STILL_PROBS, SLOW_STILL_PROBS


# =============================================================================
# STREETS
# Blocks through which individuals move from building to building.
# =============================================================================

class Street:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.neighbors_streets = []
        self.neighbors_buildings = []
        self.geometry = box(coordinates[0], coordinates[1],
                            coordinates[0]+1, coordinates[1]+1)

    def add_neighbor(self, neighbor):
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
# BUILDING
# Units of the city in which individuals dwell at different speeds.
# =============================================================================


class Building:
    def __init__(self, building_type, door, city, p_still, sigma, blocks=None, bbox=None):
        self.building_type = building_type
        self.door = door
        self.city = city

        self.p_still = p_still
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
# CITY
# Container for buildings, streets, and generators of cities
# =============================================================================


class City:
    def __init__(self, dimensions=(0, 0)):
        self.buildings = {}
        self.streets = {}
        self.buildings_outline = Polygon()
        self.address_book = {}

        # maybe at some point, p_still and sigma are inputs rather than hard coded
        self.p_still = {'park': 0.1,
                        'home': 0.6,
                        'work': 0.9,
                        'retail': 0.2}

        # controls "speed" of Brownian motion when simulating stay trajectory
        # The random variable X(t) of the position at time t has a normal
        # distribution with mean 0 and variance sigma^2 * t.
        # x/1.96 = 95% probability of moving x standard deviations
        self.sigma = {'park': 2.5/1.96,
                      'home': 1/1.96,
                      'work': 1/1.96,
                      'retail': 2/1.96}

        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all(isinstance(d, int) for d in dimensions)):
            raise ValueError("Dimensions must be a tuple of two integers.")
        self.city_boundary = box(0, 0, dimensions[0], dimensions[1])

        for x in range(0, dimensions[0]):
            for y in range(0, dimensions[1]):
                self.streets[(x, y)] = Street((x, y))
        self.dimensions = dimensions

    def add_building(self, building_type, door, blocks=None, bbox=None):
        building = Building(building_type, door, self,
                            self.p_still[building_type],
                            self.sigma[building_type],
                            blocks, bbox)

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

        # expand city boundary?
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
        # Return None if coordinates are outside the city
        x, y = coordinates
        bx, by = self.dimensions
        if (x < 0 or x >= bx or y < 0 or y >= bx):
            return None

        new_coords = (int(x), int(y))
        if new_coords in self.address_book:
            return self.address_book[new_coords]
        else:
            return self.streets[new_coords]

    # Determine adjacent streets and buildings for each street block and construct graph
    def get_street_graph(self):
        for x, y in self.streets.keys():
            street = self.streets[(x, y)]
            neighbors = [
                (x, y + 1),
                (x, y - 1),
                (x + 1, y),
                (x - 1, y)
            ]

            for neighbor in neighbors:
                block = self.get_block(neighbor)
                street.add_neighbor(block)

        # Construct graph of streets
        self.street_graph = {block: [] for block in self.streets.keys()}
        for street in self.street_graph.keys():
            self.street_graph[street] = [neighbor.coordinates for neighbor in
                                         self.get_block(street).neighbors_streets]

        # Compute shortest path between every pair of street coordinates
        self.shortest_paths = {street: {} for street in self.streets.keys()}
        for s_from in self.shortest_paths.keys():
            self.shortest_paths[s_from] = {street: [] for street in self.streets.keys()}
            for s_to in self.shortest_paths[s_from].keys():
                path = BFS_shortest_path(graph=self.street_graph, start=s_from, end=s_to)
                self.shortest_paths[s_from][s_to] = path

    def save(self, filename):
        """Save the city object to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def plot_city(self, ax, doors=True, address=True, zorder=1):
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

        # Draw buildings
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

#             if address:
#                 door_coord = building.door

#                 bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
#                 axes_width_in_inches = bbox.width
#                 axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
#                 fontsize = (axes_width_in_inches / axes_data_range) * 13  # Example scaling factor

#                 ax.text(door_coord[0] + 0.15, door_coord[1] + 0.15,
#                         f"{door_coord[0]}, {door_coord[1]}",
#                         ha='left', va='bottom',
#                         fontsize=fontsize, color='black')

        ax.set_aspect('equal')
        # Set integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# =============================================================================
# AGENT
# =============================================================================


class Agent:
    def __init__(self, identifier, home, workplace, city,
                 still_probs=DEFAULT_STILL_PROBS, speeds=DEFAULT_SPEEDS,
                 destination_diary=None, trajectory=None, diary=None, transitions=None,
                 start_time=datetime(2024, 1, 1, hour=8, minute=0)):
        """
        Parameters
        ---------
        identifier: string
            name of agent
        home: string building_id
        workplace: string building_id
        city: City
        destination_diary: pandas.DataFrame
            dataframe with columns 'unix_timestamp', 'local_timestamp', 'duration', 'location'
        trajectory: pandas.DataFrame
            dataframe with columns 'x', 'y', 'local_timestamp', 'unix_timestamp', 'identifier'
        diary: pandas.DataFrame
            dataframe with columns 'unix_timestamp', 'local_timestamp', 'duration', 'location'
        start_time: datetime
            if trajectory is None, the first ping will be at start_time.
        """

        self.identifier = identifier
        self.home = home
        self.workplace = workplace
        self.city = city
        self.transitions = transitions

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

        # If trajectory is not provided, then it is at the front door at start_time
        if trajectory is None:
            x_coord, y_coord = self.city.buildings[home].door_centroid
            local_timestamp = start_time
            unix_timestamp = int(local_timestamp.timestamp())
            trajectory = pd.DataFrame([{
                'x': x_coord,
                'y': y_coord,
                'local_timestamp': local_timestamp,
                'unix_timestamp': unix_timestamp,
                'identifier': self.identifier
                }])

            # TODO: HOW TO CORRECT INITIALIZE DIARY?
            diary_entry = {'unix_timestamp': unix_timestamp,
                           'local_timestamp': local_timestamp,
                           'duration': 1,  # should be dt
                           'location': home}
            diary_entry = pd.DataFrame([diary_entry])
            self.diary = pd.concat([self.diary, diary_entry],
                                   ignore_index=True)

        self.trajectory = trajectory

    def plot_traj(self, ax, color='black', alpha=1, doors=True, address=True):
        ax.scatter(self.trajectory.x, self.trajectory.y, s=6, color=color, alpha=alpha, zorder=2)
        self.city.plot_city(ax, doors=doors, address=address, zorder=1)

    def sample_traj_hier_nhpp(self, beta_start, beta_durations, beta_ping, seed=None):
        sparse_traj = mmod.sample_hier_nhpp(self.trajectory, beta_start, beta_durations, beta_ping, seed=seed)
        sparse_traj = sparse_traj.set_index('unix_timestamp', drop=False)
        self.sparse_traj = sparse_traj


def ortho_coord(multilines, distance, offset, eps=0.001):  # Calculus approach. Probably super slow.
    point = multilines.interpolate(distance)
    offset_point = multilines.interpolate(distance - eps)
    angle = 90 if offset < 0 else -90
    ortho_direction = rotate(offset_point, angle, origin=point)
    ortho_segment = LineString([point, ortho_direction])
    scaled_segment = scale(
        ortho_segment, xfact=offset / ortho_segment.length,
        yfact=offset / ortho_segment.length, origin=point)
    return scaled_segment.coords[1]


def condense_destinations(destination_diary):  # This might be a more general clustering algorithm
    """
    Modifies a sequence of timestamped destinations, joining consecutive 
    destinations in the same location into a single entry with the aggregated duration.

    Parameters
    ----------
    destination_diary : DataFrame
        DataFrame containing timestamped locations the user is heading towards.

    Returns
    -------
    DataFrame
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
    def __init__(self, city):
        self.roster = {}
        self.city = city

    def add_agent(self, agent):
        if agent.identifier in self.roster:
            print("Agent identifier already exists in population. Replacing corresponding agent.")
        self.roster[agent.identifier] = agent

    def generate_agents(self, N, seed=None):
        """
        Generates N agents, with randomized attributes.
        """
        if seed:
            npr.seed(seed)
        else:
            seed = npr.randint(0, 1000, 1)[0]
            npr.seed(seed)

        b_types = pd.DataFrame({
            'id': list(self.city.buildings.keys()),
            'type': [b.building_type for b in self.city.buildings.values()]
        }).set_index('id') # Maybe this should be an attribute of city since we end up using it a lot

        homes = b_types[b_types['type'] == 'home'].sample(n=N, replace=True)
        workplaces = b_types[b_types['type'] == 'work'].sample(n=N, replace=True)

        generator = funkybob.UniqueRandomNameGenerator(members=2, seed=seed)
        for i in range(N):
            identifier = generator[i]
            agent = Agent(identifier=identifier,
                          home=homes.index[i],
                          workplace=workplaces.index[i],
                          city=self.city)  # how do we add other args?
            self.add_agent(agent)

    def sample_step(self, agent, start_point, dest_building, dt):
        """
        TO DO

        Parameters
        ---------
        agent: Agent
            the agent for whom a step will be sampled
        start_point: tuple
            the coordinates of the current position
        dest_building: Building
            the destination building of the agent
        dt: float
            time step (i.e., number of minutes per ping)

        Returns
        -------
        coord: numpy array of floats with shape (1,2).
        location: building id if step is a stay or None is step is a move
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

            if npr.uniform()<p:
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
            path = [(x+0.5, y+0.5) for x,y in street_path]
            path = start_segment + path + [dest_building.door_centroid]
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
                    coord = np.array(dest_building.door_centroid)
                    break
                else:
                    coord = ortho_coord(path_ml, snap_point_dist+transformed_step[0], transformed_step[1])
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
        for i in range(destination_diary.shape[0]):
            destination_info = destination_diary.iloc[i]
            duration = int(destination_info['duration'] * 1/dt)
            building_id = destination_info['location']
            for t in range(duration):
                prev_ping = agent.trajectory.iloc[-1]

                start_point = (prev_ping['x'], prev_ping['y'])
                dest_building = city.buildings[building_id]
                coord, location = self.sample_step(agent, start_point, dest_building, dt)

                unix_timestamp = prev_ping['unix_timestamp'] + 60*dt
                local_timestamp = pd.to_datetime(unix_timestamp, unit='s')
                ping = {'x': coord[0], 'y': coord[1],
                        'local_timestamp': local_timestamp,
                        'unix_timestamp': unix_timestamp,
                        'identifier': agent.identifier}
                ping = pd.DataFrame([ping])
                agent.trajectory = pd.concat([agent.trajectory, ping],
                                             ignore_index=True)

                if agent.diary.empty or agent.diary.iloc[-1]['location'] != location:
                    entry = {'unix_timestamp': unix_timestamp,
                             'local_timestamp': local_timestamp,
                             'duration': dt,
                             'location': location}
                    entry = pd.DataFrame([entry])
                    agent.diary = pd.concat([agent.diary, entry],
                                            ignore_index=True)
                else:
                    agent.diary.loc[agent.diary.shape[0]-1, 'duration'] += 1*dt

        # empty the destination diary
        agent.destination_diary = destination_diary.drop(destination_diary.index)

    def generate_transition_probs(self, agent, location, building_choices):
        N = len(building_choices)
        probs = [1] * N
        idx = building_choices.index[building_choices.id == location][0]
        probs[idx] = N-1
        probs = [p / sum(probs) for p in probs]  # normalize probabilities
        return probs

    def dest_diary_from_trans(self, agent, T, duration=15, dt=1, trans=None):
        """
        Simulate a destination diary from transition matrices

        Parameters
        ----------
        agent: Agent
        T: int
            how many entries to generate
        duration: int
            how long each entry is (total duration is duration * T)

        Returns
        -------
        updates destination diary of agent
        """
        city = self.city
        pr_weight = 4

        # Ensure building IDs are the index
        b_types = pd.DataFrame({
            'id': list(city.buildings.keys()),
            'type': [b.building_type for b in city.buildings.values()]
        }).set_index('id')

        N = len(b_types)
        trans = pd.DataFrame(np.ones((N, N)), index=b_types.index, columns=b_types.index)

        trans[agent.home] += pr_weight
        trans[agent.workplace] += pr_weight

        np.fill_diagonal(trans.values, N + pr_weight)

        # Correct allowed_buildings function
        def allowed_buildings(ts, b_types):
            hour = ts.hour
            if 0 <= hour < 8 or 20 <= hour < 24:
                return b_types[b_types['type'] == 'home'].index
            elif 8 <= hour < 9:
                return b_types[b_types['type'] != 'home'].index
            elif 9 <= hour < 12 or 13.5 <= hour < 17.5:
                return b_types[b_types['type'] == 'work'].index
            elif 12 <= hour < 13.5:
                return b_types[b_types['type'] == 'retail'].index
            elif 17.5 <= hour < 20:
                return b_types[b_types['type'] != 'work'].index
            return b_types[b_types['type'] == 'home'].index

        last_entry = (agent.destination_diary.iloc[-1] if not agent.destination_diary.empty else agent.diary.iloc[-1])
        cur_loc = last_entry.location
        next_ts = last_entry.local_timestamp + timedelta(minutes=int(last_entry.duration))

        new_entries = []

        for _ in range(T):
            unix_ts = int(next_ts.timestamp())

            allowed_indices = allowed_buildings(next_ts, b_types)
            trans_probs = trans.loc[cur_loc, allowed_indices]
            trans_probs = trans_probs / trans_probs.sum()

            cur_loc = npr.choice(trans_probs.index, p=trans_probs.values)

            entry = {'unix_timestamp': unix_ts,
                     'local_timestamp': next_ts,
                     'duration': duration,
                     'location': cur_loc}

            new_entries.append(entry)
            next_ts += timedelta(minutes=duration)

        new_entries_df = pd.DataFrame(new_entries)
        agent.destination_diary = pd.concat([agent.destination_diary, new_entries_df], ignore_index=True)

        agent.destination_diary = condense_destinations(agent.destination_diary)

    def generate_trajectory(self, agent, T=None, duration=15, seed=None, dt=1):

        if seed:
            npr.seed(seed)
        else:
            seed = npr.randint(0, 1000, 1)[0]
            npr.seed(seed)
            print("Seed:", seed)

        if agent.destination_diary.empty:
            if T is None:
                raise ValueError(
                    "Destination diary is empty. Provide a parameter T to generate destination diary from transition matrix."
                )
            self.dest_diary_from_trans(agent, T, duration=duration, dt=dt, trans=agent.transitions)

        self.traj_from_dest_diary(agent, dt)

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


def BFS_shortest_path(graph, start, end):
    """
    Computes the shortest path between a start and end street block.

    Parameters
    ---------
    graph: dict
        undirected graph with streets as vectices and edges between adjacent streets
    start: tuple
        coordinates of starting street (i.e., door of start building)
    end: tuple
        coordinates of ending street (i.e., door of end building)

    Returns
    -------
    A list denoting the shortest path of street blocks going from the start to the end.
    """

    # is this the most efficient way to do this? maybe Dijkstra's algorithm

    explored = []
    queue = [[start]]

    if start == end:
        return [start]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in explored:
            neighbors = graph[node]

            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

                if neighbor == end:
                    return new_path
            explored.append(node)

    print("Path does not exist.")
    return None