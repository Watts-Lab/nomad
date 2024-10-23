import pyarrow.parquet as pq
import pandas as pd
from functools import partial
import multiprocessing
from multiprocessing import Pool
import re
from pyspark.sql import SparkSession
# from . import constants
import constants

def get_pq_users(path: str, id_string: str):
    return pq.read_table(path, columns=[id_string]).column(id_string).unique().to_pandas()

def get_pq_user_data(path: str, users: list[str], id_string: str):
    return pq.read_table(path, filters=[(id_string, 'in', users)]).to_pandas()

def _read_partitioned_pq(path):
    return None

def _update_schema(original, new_labels):
    updated_schema = dict(original)
    for label in new_labels:
        if label in constants.SCHEMA_NAMES:
            updated_schema[label] = new_labels[label]
    return updated_schema

def _has_traj_cols(df, traj_cols):
    
    # Check for sufficient spatial columns
    spatial_exists = (
        ('latitude' in traj_cols and 'longitude' in traj_cols and 
         traj_cols['latitude'] in df and traj_cols['longitude'] in df) or
        ('x' in traj_cols and 'y' in traj_cols and 
         traj_cols['x'] in df and traj_cols['y'] in df) or
        ('geohash' in traj_cols and traj_cols['geohash'] in df)
    )
    
    # Check for sufficient temporal columns
    temporal_exists = (
        ('datetime' in traj_cols and traj_cols['datetime'] in df) or
        ('timestamp' in traj_cols and traj_cols['timestamp'] in df)
    )
    
    if not spatial_exists:
        raise ValueError(
            "Could not find required spatial columns in {}. The dataframe columns must contain or map to at least one of the following sets: "
            "('latitude', 'longitude'), ('x', 'y'), or 'geohash'.".format(df.columns.tolist())
        )
        
    if not temporal_exists:
        raise ValueError(
            "Could not find required temporal columns in {}. The dataframe columns must contain or map to either 'datetime' or 'timestamp'.".format(df.columns.tolist())
        )
    
    return spatial_exists and temporal_exists


def _cast_traj_cols(df, traj_cols):
    if 'datetime' in traj_cols and traj_cols['datetime'] in df:
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[traj_cols['datetime']].dtype):
            df[traj_cols['datetime']] = pd.to_datetime(df[traj_cols['datetime']])
    if 'timestamp' in traj_cols and traj_cols['timestamp'] in df:
        # Coerce to integer if it's not already
        if not pd.core.dtypes.common.is_integer_dtype(df[traj_cols['timestamp']].dtype):
            df[traj_cols['timestamp']] = df[traj_cols['timestamp']].astype(int)

    float_cols = ['latitude', 'longitude', 'x', 'y']
    for col in float_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_float_dtype(df[traj_cols[col]].dtype):
                df[traj_cols[col]] = df[traj_cols[col]].astype("float")

    string_cols = ['user_id', 'geohash']
    for col in string_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_string_dtype(df[traj_cols[col]].dtype):
                df[traj_cols[col]] = df[traj_cols[col]].astype("str")

    return df

def _is_traj_df(df, traj_cols = None, **kwargs):
    
    if not (isinstance(df, pd.DataFrame) or isinstance(df, gpd.GeoDataFrame)):
        return False
    
    if not traj_cols:
        traj_cols = {}
        traj_cols = _update_schema(traj_cols, kwargs) #kwargs ignored if traj_cols
        
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    
    if not _has_traj_cols(df, traj_cols):
        return False
    
    if 'datetime' in traj_cols and traj_cols['datetime'] in df:
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[traj_cols['datetime']].dtype):
            return False
    elif 'timestamp' in traj_cols and traj_cols['timestamp'] in df:
        if not pd.core.dtypes.common.is_integer_dtype(df[traj_cols['timestamp']].dtype):
            return False

    float_cols = ['latitude', 'longitude', 'x', 'y']
    for col in float_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_float_dtype(df[traj_cols[col]].dtype):
                return False

    string_cols = ['user_id', 'geohash']
    for col in string_cols:
        if col in traj_cols and traj_cols[col] in df:
            if not pd.core.dtypes.common.is_string_dtype(df[traj_cols[col]].dtype):
                return False

    return True


def from_object(df, traj_cols = None, spark_enabled=False, **kwargs):

    if not (isinstance(df, pd.DataFrame) or isinstance(df, gpd.GeoDataFrame)):
        raise TypeError(
            "Expected the data argument to be either a pandas DataFrame or a GeoPandas GeoDataFrame."
        )
    
    # valid trajectory column names passed to **kwargs collected
    if not traj_cols:
        traj_cols = {}
        traj_cols = _update_schema(traj_cols, kwargs) #kwargs ignored if traj_cols
            
    for key, value in traj_cols.items():
        if value not in df:
            warnings.warn(f"Trajectory column '{value}' specified for '{key}' not found in df.")
            
    # include defaults when missing
    traj_cols = _update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    
    if _has_traj_cols(df, traj_cols):
        return _cast_traj_cols(df, traj_cols)

def from_file():
    return None

    


