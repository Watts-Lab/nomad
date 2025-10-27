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
#     display_name: Python (nomad repo venv)
#     language: python
#     name: nomad-repo-venv
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
city.add_building(building_type='park', door=(13, 11), geometry=box(9, 9, 13, 13))
city.add_building(building_type='home', door=(8, 8), blocks=[(7, 7), (7, 8)])
city.add_building('home', (9, 8), [(8, 7), (9, 7)])
city.add_building('home', (10, 8), [(10, 7)])
city.add_building('home', (11, 8), [(11, 7)])
city.add_building('home', (13, 6), [(13, 7)])
city.add_building('home', (14, 6), [(14, 7)])
city.add_building('home', (13, 8), [(14, 8)])
city.add_building('home', (13, 9), [(14, 9)])
city.add_building('home', (13, 11), [(14, 11)])
city.add_building('home', (13, 12), [(14, 12)])
city.add_building('home', (15, 13), [(14, 13)])
city.add_building('home', (13, 13), [(13, 14), (14, 14)])
city.add_building('home', (12, 13), [(12, 14)])
city.add_building('home', (11, 13), [(11, 14)])
city.add_building('home', (9, 13), [(9, 14)])
city.add_building('home', (8, 13), [(8, 14)])
city.add_building('home', (7, 15), [(7, 14)])
city.add_building('home', (6, 13), [(7, 13)])
city.add_building('home', (8, 12), [(7, 12)])
city.add_building('home', (8, 10), [(7, 10), (7, 9)])

# add workplaces
city.add_building('work', (3, 4), [(4, 4), (4, 5)])
city.add_building('work', (5, 3), [(5, 4), (5, 5)])
city.add_building('work', (6, 6), geometry=box(6, 4, 8, 6))
city.add_building('work', (8, 6), geometry=box(8, 4, 10, 6))
city.add_building('work', (12, 6), geometry=box(11, 5, 14, 6))
city.add_building('work', (12, 3), geometry=box(11, 4, 14, 5))
city.add_building('work', (15, 3), geometry=box(14, 4, 17, 6))
city.add_building('work', (18, 4), geometry=box(17, 4, 18, 6))
city.add_building('work', (18, 6), geometry=box(16, 6, 18, 8))
city.add_building('work', (15, 9), geometry=box(16, 8, 17, 10))
city.add_building('work', (18, 8), geometry=box(17, 8, 18, 10))
city.add_building('work', (18, 10), geometry=box(16, 10, 18, 12))
city.add_building('work', (18, 13), geometry=box(16, 13, 18, 15))
city.add_building('work', (18, 15), geometry=box(16, 15, 18, 16))
city.add_building('work', (15, 15), geometry=box(15, 16, 18, 17))
city.add_building('work', (14, 15), [(14, 16)])
city.add_building('work', (16, 18), geometry=box(16, 17, 18, 18))
city.add_building('work', (15, 18), geometry=box(14, 17, 16, 18))
city.add_building('work', (13, 18), geometry=box(12, 16, 14, 18))
city.add_building('work', (11, 18), geometry=box(10, 17, 12, 18))
city.add_building('work', (11, 15), geometry=box(10, 16, 12, 17))
city.add_building('work', (8, 18), geometry=box(7, 16, 9, 18))
city.add_building('work', (6, 18), geometry=box(5, 17, 7, 18))
city.add_building('work', (6, 15), geometry=box(5, 16, 7, 17))
city.add_building('work', (3, 16), [(4, 16), (4, 17)])
city.add_building('work', (3, 13), geometry=box(4, 13, 6, 16))
city.add_building('work', (6, 12), geometry=box(4, 12, 6, 13))
city.add_building('work', (3, 10), [(4, 9), (4, 10)])
city.add_building('work', (6, 9), [(5, 9), (5, 10)])
city.add_building('work', (6, 8), [(4, 8), (5, 8)])
city.add_building('work', (3, 6), geometry=box(4, 6, 6, 8))

# add retail places
city.add_building('retail', (0, 1), geometry=box(1, 1, 3, 3))
city.add_building('retail', (3, 0), geometry=box(3, 1, 5, 3))
city.add_building('retail', (5, 0), [(5, 1)])
city.add_building('retail', (5, 3), [(5, 2)])
city.add_building('retail', (6, 0), geometry=box(6, 1, 8, 2))
city.add_building('retail', (6, 3), geometry=box(6, 2, 8, 3))
city.add_building('retail', (9, 3), geometry=box(9, 1, 10, 3))
city.add_building('retail', (12, 3), geometry=box(10, 1, 13, 3))
city.add_building('retail', (14, 3), geometry=box(13, 1, 15, 3))
city.add_building('retail', (15, 3), [(15, 2)])
city.add_building('retail', (16, 3), [(16, 2)])
city.add_building('retail', (15, 0), [(15, 1)])
city.add_building('retail', (16, 0), [(16, 1)])
city.add_building('retail', (17, 3), geometry=box(17, 2, 19, 3))
city.add_building('retail', (18, 0), geometry=box(17, 1, 19, 2))
city.add_building('retail', (19, 0), geometry=box(19, 1, 21, 2))
city.add_building('retail', (18, 3), geometry=box(19, 2, 21, 4))
city.add_building('retail', (18, 5), geometry=box(19, 4, 21, 6))
city.add_building('retail', (18, 7), geometry=box(19, 6, 20, 8))
city.add_building('retail', (21, 7), geometry=box(20, 6, 21, 8))
city.add_building('retail', (18, 10), geometry=box(19, 9, 21, 11))
city.add_building('retail', (18, 11), geometry=box(19, 11, 21, 13))
city.add_building('retail', (18, 13), geometry=box(19, 13, 20, 15))
city.add_building('retail', (21, 13), geometry=box(20, 13, 21, 15))
city.add_building('retail', (21, 16), geometry=box(19, 15, 21, 17))
city.add_building('retail', (21, 18), geometry=box(19, 17, 21, 19))

city.add_building('retail', (21, 19), geometry=box(19, 19, 21, 20))
city.add_building('retail', (20, 21), geometry=box(19, 20, 21, 21))
city.add_building('retail', (17, 18), geometry=box(17, 19, 18, 21))
city.add_building('retail', (16, 18), geometry=box(16, 19, 17, 21))
city.add_building('retail', (14, 18), geometry=box(13, 19, 16, 20))
city.add_building('retail', (15, 21), geometry=box(14, 20, 16, 21))
city.add_building('retail', (13, 21), geometry=box(12, 20, 14, 21))
city.add_building('retail', (12, 18), geometry=box(12, 19, 13, 20))
city.add_building('retail', (11, 18), geometry=box(10, 19, 12, 21))
city.add_building('retail', (9, 18), geometry=box(8, 19, 10, 20))
city.add_building('retail', (9, 21), geometry=box(8, 20, 10, 21))
city.add_building('retail', (6, 21), geometry=box(5, 19, 7, 21))
city.add_building('retail', (4, 21), geometry=box(3, 20, 5, 21))
city.add_building('retail', (4, 18), geometry=box(3, 19, 5, 20))
city.add_building('retail', (2, 18), geometry=box(2, 19, 3, 21))
city.add_building('retail', (1, 18), geometry=box(1, 19, 2, 21))
city.add_building('retail', (3, 17), geometry=box(1, 16, 3, 18))
city.add_building('retail', (3, 15), geometry=box(1, 15, 3, 16))
city.add_building('retail', (3, 14), geometry=box(1, 14, 3, 15))
city.add_building('retail', (3, 12), geometry=box(1, 12, 3, 14))
city.add_building('retail', (3, 11), geometry=box(1, 11, 3, 12))
city.add_building('retail', (3, 10), geometry=box(1, 10, 3, 11))
city.add_building('retail', (3, 8), geometry=box(1, 8, 3, 10))
city.add_building('retail', (3, 7), geometry=box(1, 7, 3, 8))
city.add_building('retail', (0, 5), geometry=box(1, 4, 2, 7))
city.add_building('retail', (3, 6), [(2, 6)])
city.add_building('retail', (3, 5), [(2, 5)])
city.add_building('retail', (3, 4), [(2, 4)])

city.get_street_graph()
elapsed = time.perf_counter() - t0
print(f"City built and street graph computed in {elapsed:.3f}s; buildings={len(city.buildings_gdf)} streets={len(city.streets_df)}")

# Persist as GeoPackage (optional)

fig, ax = plt.subplots(figsize=(6, 6))
plt.box(on=False)
city.plot_city(ax, doors=True, address=False)
plt.show(block=False)
plt.close(fig)

# %%
city.__dict__.keys()

# %%
city.save_geopackage('synthetic_pois.gpkg')
