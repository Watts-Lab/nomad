# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import time
import matplotlib.pyplot as plt
from shapely.geometry import box
from nomad.city_gen import City

# %%
# Initialize city and time the build
t0 = time.perf_counter()
city = City(dimensions=(22, 22))
# add a park
city.add_building(building_type='park', door=(13, 11), geom=box(9, 9, 13, 13))
city.add_building(building_type='home', door=(8, 8), blocks=[(7, 7), (7, 8)])
city.add_building('home', (9, 8), blocks=[(8, 7), (9, 7)])
city.add_building('home', (10, 8), blocks=[(10, 7)])
city.add_building('home', (11, 8), blocks=[(11, 7)])
city.add_building('home', (13, 6), blocks=[(13, 7)])
city.add_building('home', (14, 6), blocks=[(14, 7)])
city.add_building('home', (13, 8), blocks=[(14, 8)])
city.add_building('home', (13, 9), blocks=[(14, 9)])
city.add_building('home', (13, 11), blocks=[(14, 11)])
city.add_building('home', (13, 12), blocks=[(14, 12)])
city.add_building('home', (15, 13), blocks=[(14, 13)])
city.add_building('home', (13, 13), blocks=[(13, 14), (14, 14)])
city.add_building('home', (12, 13), blocks=[(12, 14)])
city.add_building('home', (11, 13), blocks=[(11, 14)])
city.add_building('home', (9, 13), blocks=[(9, 14)])
city.add_building('home', (8, 13), blocks=[(8, 14)])
city.add_building('home', (7, 15), blocks=[(7, 14)])
city.add_building('home', (6, 13), blocks=[(7, 13)])
city.add_building('home', (8, 12), blocks=[(7, 12)])
city.add_building('home', (8, 10), blocks=[(7, 10), (7, 9)])

# add workplaces
city.add_building('workplace', (3, 4), blocks=[(4, 4), (4, 5)])
city.add_building('workplace', (5, 3), blocks=[(5, 4), (5, 5)])
city.add_building('workplace', (6, 6), geom=box(6, 4, 8, 6))
city.add_building('workplace', (8, 6), geom=box(8, 4, 10, 6))
city.add_building('workplace', (12, 6), geom=box(11, 5, 14, 6))
city.add_building('workplace', (12, 3), geom=box(11, 4, 14, 5))
city.add_building('workplace', (15, 3), geom=box(14, 4, 17, 6))
city.add_building('workplace', (18, 4), geom=box(17, 4, 18, 6))
city.add_building('workplace', (18, 6), geom=box(16, 6, 18, 8))
city.add_building('workplace', (15, 9), geom=box(16, 8, 17, 10))
city.add_building('workplace', (18, 8), geom=box(17, 8, 18, 10))
city.add_building('workplace', (18, 10), geom=box(16, 10, 18, 12))
city.add_building('workplace', (18, 13), geom=box(16, 13, 18, 15))
city.add_building('workplace', (18, 15), geom=box(16, 15, 18, 16))
city.add_building('workplace', (15, 15), geom=box(15, 16, 18, 17))
city.add_building('workplace', (14, 15), blocks=[(14, 16)])
city.add_building('workplace', (16, 18), geom=box(16, 17, 18, 18))
city.add_building('workplace', (15, 18), geom=box(14, 17, 16, 18))
city.add_building('workplace', (13, 18), geom=box(12, 16, 14, 18))
city.add_building('workplace', (11, 18), geom=box(10, 17, 12, 18))
city.add_building('workplace', (11, 15), geom=box(10, 16, 12, 17))
city.add_building('workplace', (8, 18), geom=box(7, 16, 9, 18))
city.add_building('workplace', (6, 18), geom=box(5, 17, 7, 18))
city.add_building('workplace', (6, 15), geom=box(5, 16, 7, 17))
city.add_building('workplace', (3, 16), blocks=[(4, 16), (4, 17)])
city.add_building('workplace', (3, 13), geom=box(4, 13, 6, 16))
city.add_building('workplace', (6, 12), geom=box(4, 12, 6, 13))
city.add_building('workplace', (3, 10), blocks=[(4, 9), (4, 10)])
city.add_building('workplace', (6, 9), blocks=[(5, 9), (5, 10)])
city.add_building('workplace', (6, 8), blocks=[(4, 8), (5, 8)])
city.add_building('workplace', (3, 6), geom=box(4, 6, 6, 8))

# add retail places
city.add_building('retail', (0, 1), geom=box(1, 1, 3, 3))
city.add_building('retail', (3, 0), geom=box(3, 1, 5, 3))
city.add_building('retail', (5, 0), blocks=[(5, 1)])
city.add_building('retail', (5, 3), blocks=[(5, 2)])
city.add_building('retail', (6, 0), geom=box(6, 1, 8, 2))
city.add_building('retail', (6, 3), geom=box(6, 2, 8, 3))
city.add_building('retail', (9, 3), geom=box(9, 1, 10, 3))
city.add_building('retail', (12, 3), geom=box(10, 1, 13, 3))
city.add_building('retail', (14, 3), geom=box(13, 1, 15, 3))
city.add_building('retail', (15, 3), blocks=[(15, 2)])
city.add_building('retail', (16, 3), blocks=[(16, 2)])
city.add_building('retail', (15, 0), blocks=[(15, 1)])
city.add_building('retail', (16, 0), blocks=[(16, 1)])
city.add_building('retail', (17, 3), geom=box(17, 2, 19, 3))
city.add_building('retail', (18, 0), geom=box(17, 1, 19, 2))
city.add_building('retail', (19, 0), geom=box(19, 1, 21, 2))
city.add_building('retail', (18, 3), geom=box(19, 2, 21, 4))
city.add_building('retail', (18, 5), geom=box(19, 4, 21, 6))
city.add_building('retail', (18, 7), geom=box(19, 6, 20, 8))
city.add_building('retail', (21, 7), geom=box(20, 6, 21, 8))
city.add_building('retail', (18, 10), geom=box(19, 9, 21, 11))
city.add_building('retail', (18, 11), geom=box(19, 11, 21, 13))
city.add_building('retail', (18, 13), geom=box(19, 13, 20, 15))
city.add_building('retail', (21, 13), geom=box(20, 13, 21, 15))
city.add_building('retail', (21, 16), geom=box(19, 15, 21, 17))
city.add_building('retail', (21, 18), geom=box(19, 17, 21, 19))

city.add_building('retail', (21, 19), geom=box(19, 19, 21, 20))
city.add_building('retail', (20, 21), geom=box(19, 20, 21, 21))
city.add_building('retail', (17, 18), geom=box(17, 19, 18, 21))
city.add_building('retail', (16, 18), geom=box(16, 19, 17, 21))
city.add_building('retail', (14, 18), geom=box(13, 19, 16, 20))
city.add_building('retail', (15, 21), geom=box(14, 20, 16, 21))
city.add_building('retail', (13, 21), geom=box(12, 20, 14, 21))
city.add_building('retail', (12, 18), geom=box(12, 19, 13, 20))
city.add_building('retail', (11, 18), geom=box(10, 19, 12, 21))
city.add_building('retail', (9, 18), geom=box(8, 19, 10, 20))
city.add_building('retail', (9, 21), geom=box(8, 20, 10, 21))
city.add_building('retail', (6, 21), geom=box(5, 19, 7, 21))
city.add_building('retail', (4, 21), geom=box(3, 20, 5, 21))
city.add_building('retail', (4, 18), geom=box(3, 19, 5, 20))
city.add_building('retail', (2, 18), geom=box(2, 19, 3, 21))
city.add_building('retail', (1, 18), geom=box(1, 19, 2, 21))
city.add_building('retail', (3, 17), geom=box(1, 16, 3, 18))
city.add_building('retail', (3, 15), geom=box(1, 15, 3, 16))
city.add_building('retail', (3, 14), geom=box(1, 14, 3, 15))
city.add_building('retail', (3, 12), geom=box(1, 12, 3, 14))
city.add_building('retail', (3, 11), geom=box(1, 11, 3, 12))
city.add_building('retail', (3, 10), geom=box(1, 10, 3, 11))
city.add_building('retail', (3, 8), geom=box(1, 8, 3, 10))
city.add_building('retail', (3, 7), geom=box(1, 7, 3, 8))
city.add_building('retail', (0, 5), geom=box(1, 4, 2, 7))
city.add_building('retail', (3, 6), blocks=[(2, 6)])
city.add_building('retail', (3, 5), blocks=[(2, 5)])
city.add_building('retail', (3, 4), blocks=[(2, 4)])

city.get_street_graph()
elapsed = time.perf_counter() - t0
print(f"City built and street graph computed in {elapsed:.3f}s; buildings={len(city.buildings_gdf)} streets={len(city.streets_gdf)}")

# Persist as GeoPackage
city.save_geopackage('garden-city.gpkg')

# Plot a city
fig, ax = plt.subplots(figsize=(6, 6))
plt.box(on=False)
city.plot_city(ax, doors=True, address=False)
plt.show(block=False)
plt.close(fig)
