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

# %% [markdown] id="460ff464-7812-41fb-bc5b-bc4f24e16499"
# # Loading Trajectory Data
#
# Mobility data comes in many formats: timestamps as unix integers or ISO strings (with timezones), 
# coordinates in lat/lon or projected, files as single CSVs or partitioned directories.
#
# `nomad.io.from_file` handles these cases with a single function call.

# %%
import glob
import pandas as pd
import nomad.io.base as loader
import nomad.data as data_folder
from pathlib import Path

data_dir = Path(data_folder.__file__).parent

# %% [markdown]
# ## Pandas vs nomad.io for partitioned data
#
# Partitioned directories (e.g., `date=2024-01-01/`, `date=2024-01-02/`, ...) require a loop with pandas:

# %%
csv_files = glob.glob(str(data_dir / "partitioned_csv" / "*" / "*.csv"))
df_list = []
for f in csv_files:
    df_list.append(pd.read_csv(f))
df_pandas = pd.concat(df_list, ignore_index=True)

print(f"Pandas: {len(df_pandas)} rows")
print(df_pandas.dtypes)
print("\nFirst few rows:")
print(df_pandas.head(3))

# %% [markdown]
# `nomad.io.from_file` handles partitioned directories in one line, plus automatic type casting and column mapping:

# %%
traj_cols = {"user_id": "user_id",
             "latitude": "dev_lat",
             "longitude": "dev_lon",
             "datetime": "local_datetime"}

df = loader.from_file(data_dir / "partitioned_csv", format="csv", traj_cols=traj_cols, parse_dates=True)
print(f"nomad.io: {len(df)} rows")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head(3))
print("\nNote: 'local_datetime' is now datetime64[ns], not object!")

# %% [markdown]
# The same pattern works for Parquet files, with the type casting and processing relying on passing to the functions which columns correspond to the default "typical" spatio-temporal column names

# %%
traj_cols = {"user_id": "uid", "timestamp": "timestamp", 
             "latitude": "latitude", "longitude": "longitude", "date": "date"}

df = loader.from_file(data_dir / "partitioned_parquet", format="parquet", traj_cols=traj_cols, parse_dates=True)
print(f"Loaded {len(df)} rows")
print(df.dtypes)

# %%
# These are the default canonical columnn names
from nomad.constants import DEFAULT_SCHEMA
print(DEFAULT_SCHEMA.keys())
