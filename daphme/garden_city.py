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
import networkx as nx
import nx_parallel as nxp

import mobility_model as mmod
import stop_detection as sd
from constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS
from constants import FAST_STILL_PROBS, SLOW_STILL_PROBS, ALLOWED_BUILDINGS, DEFAULT_STAY_PROBS

import pdb



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

        self.id = f's-x{coordinates[0]}-y{coordinates[1]}'

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
    def __init__(self,
                 dimensions=(0, 0),
                 still_probs=DEFAULT_STILL_PROBS,
                 speeds=DEFAULT_SPEEDS):
        self.buildings = {}
        self.streets = {}
        self.buildings_outline = Polygon()
        self.address_book = {}

        # maybe at some point, p_still and sigma are inputs rather than hard coded
        self.p_still = still_probs

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
        """Save the city object to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def plot_city(self, ax, doors=True, address=True, zorder=1, heatmap_agent=None):
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

#         if address:
#             door_coord = building.door

#             bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
#             axes_width_in_inches = bbox.width
#             axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
#             fontsize = (axes_width_in_inches / axes_data_range) * 13  # Example scaling factor

#             ax.text(door_coord[0] + 0.15, door_coord[1] + 0.15,
#                     f"{door_coord[0]}, {door_coord[1]}",
#                     ha='left', va='bottom',
#                     fontsize=fontsize, color='black')

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
                 destination_diary=None, trajectory=None, diary=None,
                 start_time=datetime(2024, 1, 1, hour=8, minute=0), dt=1):
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
        if heatmap:
            self.city.plot_city(ax, doors=doors, address=address, zorder=1, heatmap_agent=self)
        else:
            ax.scatter(self.trajectory.x, self.trajectory.y, s=6, color=color, alpha=alpha, zorder=2)
            self.city.plot_city(ax, doors=doors, address=address, zorder=1)

    def sample_traj_hier_nhpp(self, beta_start, beta_durations, beta_ping, seed=0):
        sparse_traj = mmod.sample_hier_nhpp(self.trajectory, beta_start, beta_durations, beta_ping, seed=seed)
        sparse_traj = sparse_traj.set_index('unix_timestamp', drop=False)
        self.sparse_traj = sparse_traj


def ortho_coord(multilines, distance, offset, eps=0.001):  # Calculus approach. Probably super slow.
    point = multilines.interpolate(distance)
    offset_point = multilines.interpolate(distance - eps)
    p = np.array([point.x, point.y])
    x = p - np.array([offset_point.x, offset_point.y])
    x = np.flip(x/np.linalg.norm(x))*np.array([-1,1])*offset
    return tuple(x+p)


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
        self.global_execution = 0

    def add_agent(self, agent, verbose=True):
        if verbose and agent.identifier in self.roster:
            print("Agent identifier already exists in population. Replacing corresponding agent.")
        self.roster[agent.identifier] = agent

    def generate_agents(self, N, seed=0):
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
                          city=self.city)  # how do we add other args?
            self.add_agent(agent)

    def save_pop(self, path):
        """
        TODO: CHANGE SO SAVES TO S3 AS PARQUET
        """

        # raw trajectories
        trajs = pd.concat([agent.trajectory for agent_id, agent in self.roster.items()])
        trajs.to_csv(path+'trajectories.csv')

        # home table
        homes = pd.DataFrame([(agent_id, agent.home, agent.workplace) for agent_id, agent in self.roster.items()],
                             columns=['id', 'home', 'workplace'])
        homes.to_csv(path+'homes.csv')

        # diary
        diaries = []
        for agent_id, agent in self.roster.items():
            ag_d = agent.diary
            ag_d['id'] = agent_id
            diaries += [ag_d]
        pd.concat(diaries).to_csv(path+'diaries.csv')

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
        From a destination diary, generates (x,y) pings.

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
            for t in range(duration//dt):
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

        T = timestamp of when to generate til
        rho, gamma = Parameters for exploring
        stay_probs = dictionary
            Probability of staying in same building
            geometric with p = 1-((1/avg_duration_hrs)/timesteps_in_1_hr)
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