# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic Philadelphia
#
# This notebook will use the functions in `map_utils.py` to create a synthetic rasterized version of Philadelphia. It starts by downloading and classifying buildings from OSM in web mercator coordinates, and reporting on the building counts for each subtype and each of the _garden city building types_ which are:
#   - park
#   - home
#   - work
#   - retail
#
# It also identifies which rotation best aligns a random sample of streets with a N-S, E-W grid. 

# %%
import nomad.map_utils as nm
import geopandas as gpd
import contextily as ctx # for base maps
