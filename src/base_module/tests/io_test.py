import pytest
import warnings
import pandas as pd
from .. import daphmeIO as loader
from .. import constants

@pytest.fixture
def simple_df():
    df = pd.DataFrame([[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
                       [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
                       [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
                       [1, 39.984211, 116.319389, '2008-10-23 13:53:16']],
                      columns = ['uid', 'latitude', 'longitude', 'time'])
    return df

