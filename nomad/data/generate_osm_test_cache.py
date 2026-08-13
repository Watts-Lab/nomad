"""Cache raw OSMnx calls used by map tests.

Usage:
  python -m nomad.data.generate_osm_test_cache
"""

from pathlib import Path

import nomad.data as data_folder
import osmnx as ox

from nomad.constants import (
    PARK_TAGS,
    STREET_EXCLUDE_COVERED,
    STREET_EXCLUDE_TUNNELS,
    STREET_EXCLUDED_SERVICE_TYPES,
    STREET_EXCLUDED_SURFACES,
    STREET_HIGHWAY_TYPES,
)


DATA_DIR = Path(data_folder.__file__).parent
OUTPUT_DIR = DATA_DIR.parent / "tests" / "tmp_osm_cache"
SMALL_TOWN = "Riverton, Burlington County, New Jersey, USA"

TAG_CASES = {
    "buildings": {"building": True},
    "parking": {"amenity": ["parking"]},
    "parks": {
        key: values
        for key, values in PARK_TAGS.items()
        if key not in ["landuse", "amenity"]
    },
}

BBOX_CASES = {
    "philly_small": (-75.1590, 39.9470, -75.1530, 39.9500),
    "philly_large": (-75.1662060, 39.9411582, -75.1456557, 39.9557201),
    "schuylkill_water": (
        -75.18832868207252,
        39.94457101861711,
        -75.17632868207252,
        39.95657101861711,
    ),
    "washington_square_park": (
        -75.1561894880407,
        39.9432501797551,
        -75.1501894880407,
        39.9492501797551,
    ),
    "parking_lot": (
        -75.1555204236118,
        39.94369232826087,
        -75.1545204236118,
        39.94469232826087,
    ),
    "speculative_classification": (
        -75.15849966125,
        39.946618968625,
        -75.15336208875,
        39.950259458375,
    ),
}


def street_filter():
    highway_types = "|".join(STREET_HIGHWAY_TYPES)
    parts = [f'["highway"~"^({highway_types})$"]']
    if STREET_EXCLUDED_SERVICE_TYPES:
        excluded_services = "|".join(STREET_EXCLUDED_SERVICE_TYPES)
        parts.append(f'["service"!~"{excluded_services}"]')
    if STREET_EXCLUDE_TUNNELS:
        parts.append('["tunnel"!="yes"]')
    if STREET_EXCLUDE_COVERED:
        parts.append('["covered"!="yes"]')
    if STREET_EXCLUDED_SURFACES:
        excluded_surfaces = "|".join(STREET_EXCLUDED_SURFACES)
        parts.append(f'["surface"!~"{excluded_surfaces}"]')
    return "".join(parts)


def save_gdf(gdf, path):
    if len(gdf) == 0:
        raise AssertionError(f"OSM cache fixture is empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    print(f"written: {path}")


def save_graph(graph, path):
    if graph.number_of_nodes() == 0:
        raise AssertionError(f"OSM cache fixture is empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=str(path))
    print(f"written: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    custom_filter = street_filter()

    for case, bbox in BBOX_CASES.items():
        for tag_case, tags in TAG_CASES.items():
            path = OUTPUT_DIR / "features_from_bbox" / f"{case}__{tag_case}.parquet"
            save_gdf(ox.features_from_bbox(bbox=bbox, tags=tags), path)

    town_gdf = ox.geocode_to_gdf(SMALL_TOWN)
    save_gdf(town_gdf, OUTPUT_DIR / "geocode_to_gdf" / "small_town.parquet")

    town_polygon = town_gdf.geometry.iloc[0].simplify(tolerance=0.001)
    for tag_case, tags in TAG_CASES.items():
        path = OUTPUT_DIR / "features_from_polygon" / f"small_town__{tag_case}.parquet"
        save_gdf(ox.features_from_polygon(town_polygon, tags), path)

    graph = ox.graph_from_bbox(
        bbox=BBOX_CASES["philly_small"],
        custom_filter=custom_filter,
        truncate_by_edge=True,
        simplify=True,
    )
    save_graph(graph, OUTPUT_DIR / "graph_from_bbox" / "philly_small.graphml")

    graph = ox.graph_from_polygon(town_polygon, custom_filter=custom_filter, simplify=True)
    save_graph(graph, OUTPUT_DIR / "graph_from_polygon" / "small_town.graphml")


if __name__ == "__main__":
    main()
