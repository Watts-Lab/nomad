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

# %% [markdown] id="460ff464-7812-41fb-bc5b-bc4f24e16499"
# # **Loading and Sampling Trajectory Data**
#
# ## Getting started
#
# Real-world mobility files vary widely in structure and formatting:
# - e.g. **Timestamps** may be **UNIX** integers or **ISO-formatted strings**
# - May have **timezones**, e.g. -05:00, Z, (GMT+01), -3600
# - Coordinates might be **projected** or **geographical**
# - Files may be a flat **CSV**, or **partitioned Parquets**, local or **in S3**.
#
# `nomad.io` is here to help.

# %% executionInfo={"elapsed": 3404, "status": "ok", "timestamp": 1753083319439, "user": {"displayName": "Thomas Li", "userId": "03526318197962168317"}, "user_tz": -120} id="ca448248-3077-4e67-ad81-6d1ba1b170db"
from nomad.io import base as loader
import pandas as pd
import geopandas as gpd

# %% [markdown] id="c78e81f2-bcf3-4cc5-8c26-b6111484df73"
# ## Typical data ingestion ( `pandas`, `geopandas`) vs `nomad` `io` utilities

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 849, "status": "ok", "timestamp": 1753083322765, "user": {"displayName": "Thomas Li", "userId": "03526318197962168317"}, "user_tz": -120} id="904bf840-4253-41e3-a1d3-d54874072613" outputId="9dd16f0f-7e96-4aaa-ade3-c431698bafbc"
df = pd.read_csv("../../tutorials/IC2S2-2025/IC2S2-2025/gc_data.csv")
city = gpd.read_file("../../tutorials/IC2S2-2025/IC2S2-2025/garden_city.geojson")

df.head()

# %% [markdown]
# ## `nomad.io` — facilitates type casting and default names
#
# `nomad.io.base.from_file` is basically a `pandas` / `pyarrow` wrapper, trying to simplify the formatting of canonical variables
#
# - dates and datetimes in **ISO format** are cast to `pandas.datetime64`
# - **unix timestamps** are cast to integers and **reformatted to seconds**.
# - **user identifiers** are cast to strings
# - **partition folders** can be read as columns (Hive)
# - **timezone handling** parses ISO datetime strings (with or without timezones)

# %% [markdown] id="03b7bf33-48a1-4d75-bd95-ae05fb7f9357"
# Don't read partitioned data with a for loop! `nomad`'s `from_file` wraps `PyArrow`'s file readers maintaning the same signature.

# %% id="b33de9d2-ee49-46ac-96a1-56784674d40c"
# For the partitioned dataset
traj_cols = {"user_id": "user_id",
             "timestamp": "timestamp",
             "latitude": "latitude",
             "longitude": "longitude",
             "datetime": "datetime",
             "date": "date"}

file_path = "../../tutorials/IC2S2-2025/IC2S2-2025/gc_data/" # partitioned


df = loader.from_file(file_path, format="csv", traj_cols=traj_cols, parse_dates=True)
print(df.dtypes)

# %%
from nomad.constants import DEFAULT_SCHEMA
print("Canonical column names in nomad")
DEFAULT_SCHEMA

# %% [markdown]
# ```from_file``` automatically detects and reads Parquet files (single or partitioned directories) using ```PyArrow```'s dataset API, applying the same validation, type casting, and timezone handling as for CSV inputs.

# %%
traj_cols = {"user_id": "uid",
             "timestamp": "timestamp",
             "latitude": "latitude",
             "longitude": "longitude",
             "date": "date"}

file_path = "../../nomad/data/partitioned_parquet/" # partitioned

df = loader.from_file(file_path, format="parquet", traj_cols=traj_cols, parse_dates=True)
print(df.dtypes)
