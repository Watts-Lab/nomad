import pytest
import warnings
import pandas as pd
from .. import io.base as loader
from .. import constants


# many columns, 2 users, different types for timestamp, 
@pytest.fixture
def simple_df_multi_user():
    df = pd.DataFrame(
        [[1, 39.984094, 116.319236, '2023-01-01 13:53:05'],
         [1, 39.984198, 116.319322, '2023-01-02 13:53:06'],
         [1, 39.984224, 116.319402, '2023-01-02 13:53:11'],
         [1, 39.984211, 116.319389, '2023-01-07 13:53:16'],
         [2, 39.984100, 116.319500, '2023-01-01 13:53:20'],
         [2, 39.984300, 116.319600, '2023-01-01 13:53:25'],
         [2, 39.984400, 116.319700, '2023-01-01 13:53:30'],
         [3, 20.984000, 116.319800, '2023-01-04 13:53:35'],
         [3, 20.984500, 116.319900, '2023-01-05 13:53:40'],
         [4, 39.984100, 116.319500, '2023-01-03 13:53:20'],
         [4, 39.984300, 116.319600, '2023-01-04 13:53:25'],
         [4, 39.984400, 116.319700, '2023-01-04 13:53:30']],
        columns=['user_id', 'latitude', 'longitude', 'timestamp']
    )
    return df


@pytest.fixture
def path_1()

# from_object works on simple_df, i.e. _is_traj_df

# from_object works with different names

# from_object has correct values in some entries

# from object fails if no spatial or temporal columns are provided (and no defaults)

# from_file works on data samples, i.e. _is_traj_df for pandas/pyarrow reads

# from_file works with pyspark/pyarrow (without instantiating) until _is_traj_df on data sample 3
