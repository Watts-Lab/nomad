from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
import nomad.map_utils as nm

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "virtual_philadelphia" / "sandbox"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_GPKG_PATH = OUTPUT_DIR.parent / "philadelphia_osm_raw.gpkg"

print("Loading full Philadelphia data...")
buildings = gpd.read_file(RAW_GPKG_PATH, layer="buildings_rotated")
streets = gpd.read_file(RAW_GPKG_PATH, layer="streets_rotated")
boundary = gpd.read_file(RAW_GPKG_PATH, layer="city_boundary_rotated")

print(f"Loaded {len(buildings):,} buildings, {len(streets):,} streets")

OLD_CITY_BBOX = box(-75.1662060, 39.9411582, -75.1456557, 39.9557201)
old_city_4326 = gpd.GeoDataFrame(geometry=[OLD_CITY_BBOX], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[0]

boundary_centroid = boundary.geometry.iloc[0].centroid
rotation_deg = 37.64

from shapely.affinity import rotate
old_city_rotated = rotate(old_city_4326, rotation_deg, origin=(boundary_centroid.x, boundary_centroid.y))

print("Clipping to Old City bbox...")
sandbox_buildings = gpd.clip(buildings, gpd.GeoDataFrame(geometry=[old_city_rotated], crs=buildings.crs))
sandbox_streets = gpd.clip(streets, gpd.GeoDataFrame(geometry=[old_city_rotated], crs=streets.crs))

print(f"Sandbox: {len(sandbox_buildings):,} buildings, {len(sandbox_streets):,} streets")

SANDBOX_GPKG = OUTPUT_DIR / "sandbox_data.gpkg"
if SANDBOX_GPKG.exists():
    SANDBOX_GPKG.unlink()

sandbox_buildings.to_file(SANDBOX_GPKG, layer="buildings", driver="GPKG")
sandbox_streets.to_file(SANDBOX_GPKG, layer="streets", driver="GPKG", mode="a")
gpd.GeoDataFrame(geometry=[old_city_rotated], crs=buildings.crs).to_file(SANDBOX_GPKG, layer="boundary", driver="GPKG", mode="a")

print(f"Saved sandbox data to {SANDBOX_GPKG}")

