# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3.10 (daphme)
#     language: python
#     name: daphme
# ---

# %% [markdown] jupyter={"source_hidden": true}
# ### Outline
# explain how nomad can address different challenges in the data ingestion process:
# 1) the data can be partitioned, complicating things for users most familiar with pandas, a simple wrapper function from_file simplifies things, same function for simple csvs or partitioned data
# 2) column names and formats might vary from dataset to dataset, but spatial analysis functions require spatial and temporal columns, sometimes the time is expected as a unix timestamp, sometimes as a datetime with the local timezone. Similarly, an algorithm might require the latitude and longitude. Users always have the alternative of renaming the data so that those column names match the defaults of those functions, or they can input the right column names (or pass the relevant columns) on functions that have such flexibility. Nothing wrong with that. However, it could be preferrable to not alter the data permanently for analysis, specially if one will perform some filtering or produce a derivative dataset that is expected to be joined later on with the original data. Passing the correct column names every time to processing functions can be burdensome and verbose, and makes code less reusable when applied to a dataset with different column names. nomad addresses this by using an auxiliary dictionary of column names which other processing methods can use to find the appropriate columns. This is somewhat equivalent to passing the column names as keyword arguments, but functions also have a fallback to default column names for certain expected columns (like latitude, longitude, user_id, timestamp, etc).
# 3) We can demonstrate the flexibiilty that this auxiliary dictionary offers, by loading some device-level data fond in `gc-data.csv`. Beyond being a wrapper for the pandas or pyarrow reader functions, the `io` reader method, `from_file`, also ensures the trajectory columns (coordinates and time columns) are cast to the correct data types, issues warnings when unix timestamps are not in seconds, and raises errors when the data seems to lack spatial or temporal columns that will likely be required in downstream processing. This can be evidenced by comparing the output of simply using `pandas.read_csv` with that of `from_file`, where we see that the right columns have been cast to the right data types:
#
# 4) Of particular importance is the standardized handling of datetime strings in iso8601 formats. These can be timezone naive, have timezone information, and even have mixed timezones. For instance, when a trajectory spans multiple regions, or when there are daylight savings changes. nomad tries to simplify the parsing of dates in such cases, with three cases: [code explaining]
#
# 5) This last case is important because distributed algorithms relying on Spark do not store timezone information in the timestamp format. This presents a challenge in which analysis related to local datetime is required, but this information is lost. Switching to utc time is always an option which makes naive datetimes comparable, but it makes analysis of day-time, night-time behaviors more complicated when there are mixed timezones. A standard way to deal with timezone data is to strip the timezone information from timestamps and represent it in a separate column as the offset from UTC time in seconds. Thus, for compatibility with Spark workflows, setting `mixed_timezone_behavior = "naive"` will create a `tz_offset` column (when one does not already exist).
#
# 6) The flexibility provided by nomad to easily switch between small analyses using a small example of data, which could be stored in a .csv file, for testing code, and then using the same (or similar functions) to scale up in a distributed environment, facilitates a common (and recommended) workflow in which users can easily read data from some users from a large dataset and use standard pandas functionalities, benchmark their code, etc, and then scale up using more resources once they are certain their code is in good shape. This can easily be done with io methods like `sample_users`, `sample_from_file` (which may optionally take a sample of users drawn from somewhere else). This is shown as follows:
#
# 7) Finally, a user might want to persist such a sample with care for the data types and, perhaps, recovering the date string format with timezone, which is possible even when this information was saved in the tz_offset column. Notice that this writer function can also seamlessly switch between csv and parquet formats, leveraging pyarrow and pandas. FOr example: 

# %% [markdown]
# # Tutorial 1: Loading and Sampling Trajectory Data
#
# Real-world mobility files vary widely in structure and formatting. Timestamps may be recorded as UNIX integers or ISO-formatted strings, with or without timezone offsets. Coordinate columns may follow different naming conventions, and files may be stored either as flat CSVs or as partitioned Parquet directories. This notebook demonstrates how `nomad.io.base` standardizes data loading across these variations using two example datasets: a CSV file (`gc-data.csv`) and a partitioned Parquet directory (`gc-data/`).
#
# ## Inspecting schemas
# Let's start by inspecting the schemas of the datasets we will use with the nomad helper function `table_columns` from the `io` module. This method reports column names for both flat files and partitioned datasets without reading the full content into memory.

# %%
from nomad.io import base as loader

print(loader.table_columns("gc-data.csv", format="csv"))
print(loader.table_columns("gc-data/", format="parquet")) # <<< SHOULD BE A PARTITIONED CSV

# %% [markdown]
# ## Loading data 
#
# Reading data with `pandas` or Parquet readers does not enforce any particular schema, but spatiotemporal data often contains columns that must follow specific formats. The `from_file` function applies consistent type casting, converting temporal fields to `datetime` objects, ensuring coordinates are floats, unix timestamps are integers, and optionally creating a `tz_offset` column to store timezone offsets when parsing datetime strings. This enables compatibility with engines like Spark, in which `Timestamp` objects cannot store timezone information.
#
# To make reproducing the code as easy as possible, we want to abstract away the different possible column names, understanding that in most cases we get the same columns. While renaming is an option in some cases, `nomad` can handle different column names in most methods by simply storing a a `traj_cols` dictionary mapping default column names to the actual column names in the dataset.This also allows downstream functions to know where to find required spatial, temporal, or even tessellation columns without excessive argument-passing.
#

# %%
traj_cols = {"user_id": "identifier", "timestamp": "unix_timestamp", "latitude": "device_lat", "longitude": "device_lon", "datetime": "local_datetime", "date": "date"}
df_mapped = loader.from_file("gc-data.csv", format="csv", traj_cols=traj_cols)
df_mapped.head()

# %% [markdown]
# This mapping makes the dataset compatible with nomad tools without modifying its original structure. However, in the case in which a dataset has the default names, possibly due to the columns being renamed, many `nomad` methods will work without passing any mappings or excessive arguments. After inspecting the default column names, we see that the second dataset uses those, and thus the casting of column types and parsing of dates identifies (and is applied) on the appropriate columns. 

# %%
from nomad.constants import DEFAULT_SCHEMA
DEFAULT_SCHEMA

# %%
# This dataset has default column names, so no traj_cols argument is necessary
df_pq = loader.from_file("gc-data/", format="parquet", parse_dates=True)
df_pq.head()

# %% [markdown]
# Even when GPS data is stored in partitioned directories (e.g. date=2024-01-01/), `from_file` can handle it using PyArrow's file reader.

# %% [markdown]
# ## Working on smaller samples and persistence
#
# Large mobility datasets should typically not be fully loaded into the memory of a machine during interactive analysis, so subsampling by user is a common step in early analyses. nomad's `sample_users` selects a reproducible subset of user IDs, and `sample_from_file` filters the input dataset to include only those records. The resulting sample can be written to disk using `to_file`, partitioned by date in `hive` format to preserve compatibility with distributed engines. Reading the output back with `from_file` confirms that the sample was saved correctly and remains compatible with the same loading functions.

# %%
users = loader.sample_users("gc-data/", format="parquet", size=10, seed=300)
sample_df = loader.sample_from_file("gc-data/", users=users, format="parquet", frac_records=0.25)

loader.to_file(sample_df, "/tmp/nomad_sample", format="parquet", partition_by=["date"], existing_data_behavior='delete_matching')

round_trip = loader.from_file("/tmp/nomad_sample", format="parquet")

# %%
# The amount of data in the working sample is much smaller, and sufficient for prototyping

# %%
print("- Value counts for sample of data:\n")
print(round_trip.user_id.value_counts())
print("\n---------------------------------\n")
print("- Value counts for original data:\n")
print(df_pq.user_id.value_counts())

# %%
