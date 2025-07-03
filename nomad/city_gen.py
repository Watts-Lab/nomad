from shapely.geometry import box, Polygon, LineString, MultiLineString
from shapely.affinity import scale
from shapely.ops import unary_union
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import networkx as nx


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
                 blocks: list = None, 
                 bbox: Polygon = None):

        self.building_type = building_type
        self.door = door
        self.city = city

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
                 dimensions: tuple = (0,0)):

        self.buildings = {}
        self.streets = {}
        self.buildings_outline = Polygon()
        self.address_book = {}
        self.building_types = pd.DataFrame(columns=["id", "type"])

        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all(isinstance(d, int) for d in dimensions)):
            raise ValueError("Dimensions must be a tuple of two integers.")
        self.city_boundary = box(0, 0, dimensions[0], dimensions[1])

        for x in range(0, dimensions[0]):
            for y in range(0, dimensions[1]):
                self.streets[(x, y)] = Street((x, y))
        self.dimensions = dimensions

    def add_building(self,
                     building_type,
                     door,
                     blocks=None,
                     bbox=None):
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

        if blocks is None and bbox is None:
            raise ValueError(
                "Either blocks spanned or bounding box must be provided."
            )

        building = Building(building_type=building_type,
                            door=door,
                            city=self,
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
        self.building_types = pd.concat(
            [self.building_types, pd.DataFrame([{"id": building.id, "type": building_type}])],
            ignore_index=True
        )

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
            x = max(0, min(x, bx-1))
            y = max(0, min(y, by-1))

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
        sp = dict(nx.all_pairs_shortest_path(G))
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

    def plot_city(self, ax, doors=True, address=True, zorder=1, heatmap_agent=None, colors=None):
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
        colors : dict
            A dictionary mapping building types to colors for plotting.
        """

        # Draw city boundary
        x, y = self.city_boundary.exterior.xy
        ax.plot(np.array(x), np.array(y), linewidth=2, color='black')  # Dashed line for the boundary

        # Define colors for different building types
        if colors is None:
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
