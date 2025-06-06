# Tutorial 1: Loading and Sampling Trajectory Data

This notebook shows how to load, validate, and optionally sample trajectory data using nomad's I/O module. The goal of this module is to reduce friction when working with real-world mobility data: inconsistent column names, different file formats, partitioned directories, mixed timezones, and massive datasets that must be sampled efficiently.

These are common challenges when dealing with commercial or open GPS data. Files may come in different formats (CSV or Parquet), or in partitioned directories organized by date. Timestamps may be stored as integers or ISO strings, sometimes with timezones, sometimes without, and sometimes mixed. Latitude and longitude might be called `lat`/`lon` in one dataset, and `latitude`/`longitude` in another, or the data might have other coordinate projections altogether. 

Nomad's I/O functions give users a consistent interface for reading this data, without having to worry about these low-level details. Let's start by verifying the schemas of the data we will use, which come in two different formats—CSV and Parquet with hive-style partitioning. This can be achieved with nomad helper function `table_columns` from the `io` module.



```python
from nomad.io import base as loader

print(loader.table_columns("gc-data.csv", format="csv"))
print(loader.table_columns("gc-data/", format="parquet"))
```

    Index(['identifier', 'unix_timestamp', 'device_lat', 'device_lon', 'date',
           'offset_seconds', 'local_datetime'],
          dtype='object')
    Index(['user_id', 'timestamp', 'latitude', 'longitude', 'tz_offset',
           'datetime', 'date'],
          dtype='object')
    

explain how nomad can address different challenges in the data ingestion process:
1) the data can be partitioned, complicating things for users most familiar with pandas, a simple wrapper function from_file simplifies things, same function for simple csvs or partitioned data
2) column names and formats might vary from dataset to dataset, but spatial analysis functions require spatial and temporal columns, sometimes the time is expected as a unix timestamp, sometimes as a datetime with the local timezone. Similarly, an algorithm might require the latitude and longitude. Users always have the alternative of renaming the data so that those column names match the defaults of those functions, or they can input the right column names (or pass the relevant columns) on functions that have such flexibility. Nothing wrong with that. However, it could be preferrable to not alter the data permanently for analysis, specially if one will perform some filtering or produce a derivative dataset that is expected to be joined later on with the original data. Passing the correct column names every time to processing functions can be burdensome and verbose, and makes code less reusable when applied to a dataset with different column names. nomad addresses this by using an auxiliary dictionary of column names which other processing methods can use to find the appropriate columns. This is somewhat equivalent to passing the column names as keyword arguments, but functions also have a fallback to default column names for certain expected columns (like latitude, longitude, user_id, timestamp, etc).
3) We can demonstrate the flexibiilty that this auxiliary dictionary offers, by loading some device-level data fond in `gc-data.csv`. Beyond being a wrapper for the pandas or pyarrow reader functions, the `io` reader method, `from_file`, also ensures the trajectory columns (coordinates and time columns) are cast to the correct data types, issues warnings when unix timestamps are not in seconds, and raises errors when the data seems to lack spatial or temporal columns that will likely be required in downstream processing. This can be evidenced by comparing the output of simply using `pandas.read_csv` with that of `from_file`, where we see that the right columns have been cast to the right data types:

4) Of particular importance is the standardized handling of datetime strings in iso8601 formats. These can be timezone naive, have timezone information, and even have mixed timezones. For instance, when a trajectory spans multiple regions, or when there are daylight savings changes. nomad tries to simplify the parsing of dates in such cases, with three cases: [code explaining]

5) This last case is important because distributed algorithms relying on Spark do not store timezone information in the timestamp format. This presents a challenge in which analysis related to local datetime is required, but this information is lost. Switching to utc time is always an option which makes naive datetimes comparable, but it makes analysis of day-time, night-time behaviors more complicated when there are mixed timezones. A standard way to deal with timezone data is to strip the timezone information from timestamps and represent it in a separate column as the offset from UTC time in seconds. Thus, for compatibility with Spark workflows, setting `mixed_timezone_behavior = "naive"` will create a `tz_offset` column (when one does not already exist).

6) The flexibility provided by nomad to easily switch between small analyses using a small example of data, which could be stored in a .csv file, for testing code, and then using the same (or similar functions) to scale up in a distributed environment, facilitates a common (and recommended) workflow in which users can easily read data from some users from a large dataset and use standard pandas functionalities, benchmark their code, etc, and then scale up using more resources once they are certain their code is in good shape. This can easily be done with io methods like `sample_users`, `sample_from_file` (which may optionally take a sample of users drawn from somewhere else). This is shown as follows:

7) Finally, a user might want to persist such a sample with care for the data types and, perhaps, recovering the date string format with timezone, which is possible even when this information was saved in the tz_offset column. Notice that this writer function can also seamlessly switch between csv and parquet formats, leveraging pyarrow and pandas. FOr example: 


```python
from nomad.io import base as loader
import matplotlib.pyplot as plt
```

Nomad's data loader combines convenient loader functions from ```Pandas``` and ```Pyarrow``` that parse through partitioned folder structures to return a pandas dataframe. 


```python
df = loader.from_file('gc-sample-data/', format='parquet')
df
```


```python
# Plot the trajectory of a single user in a day
user_traj = df.loc[(df.uid == "youthful_galileo")&(df.date=='2024-01-15')]

user_traj.plot('longitude',
               'latitude',
               marker='o',
               xlabel='Longitude',
               ylabel='Latitude',
               xticks=[],
               yticks=[],
               title= 'Sample Trajectory for a random user on "2024-01-15"')
```


```python
part_path = "s3://synthetic-raw-data/agents-*/sparse_trajectories.parquet/"

traj_cols =  {"user_id":"identifier",
              "x":"x",
              "y":"y",
              "datetime":"local_timestamp"}
```

### Get a sample of users


```python
u_sample = loader.sample_users("s3://synthetic-raw-data/agents-*/sparse_trajectories.parquet/", format='parquet', frac_users=0.2, user_id='identifier')
```


```python
u_sample
```

### Load data for users in u_sample for 3 days


```python
filepath = ['s3://phl-pings/gravy_clean/date=2019-11-01/',
            's3://phl-pings/gravy_clean/date=2019-11-02/',
            's3://phl-pings/gravy_clean/date=2019-11-03/',
            's3://phl-pings/gravy_clean/date=2019-11-04/',
            's3://phl-pings/gravy_clean/date=2019-11-05/',
            's3://phl-pings/gravy_clean/date=2019-11-06/']

data = loader.sample_from_file(part_path, users=u_sample, format='parquet', traj_cols=traj_cols, user_id = 'identifier')
```


```python
data
```


```python
data['timestamp'] = data[traj_cols['datetime']].astype(int) // 10**9
```
