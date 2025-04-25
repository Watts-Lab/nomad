import pytest
import pandas as pd
from nomad.traj_gen import Agent, Population, condense_destinations
import nomad.city_gen as cg
import pdb
import nomad.io.base as loader


@pytest.fixture
def base_city():
    test_dir = Path(__file__).resolve().parent
    city_path = test_dir.parent / "data" / "garden-city.pkl"
    city = cg.load(city_path)
    population = Population(city)
    return city, population

@pytest.fixture
def basic_dest_diary():
    tz = ZoneInfo("America/New_York")
    start_time = pd.date_range(start='2024-06-01 00:00', periods=4, freq='60min', tz=tz)
    tz_offset = loader._offset_seconds_from_ts(start_time[0])
    unix_timestamp = [int(t.timestamp()) for t in start_time]
    duration = [60]*4  # in minutes
    location = ['h-x13-y11'] * 1 + ['h-x13-y9'] * 1 + ['w-x18-y10'] * 1 + ['w-x18-y8'] * 1
    
    destination = pd.DataFrame(
        {"local_timestamp":start_time,
         "unix_timestamp":unix_timestamp,
         "tz_offset":tz_offset,
         "duration":duration,
         "location":location}
         )
    destination = condense_destinations(destination)
    return destination

def test_reset_agent():
    # assert _is_traj_df(Charlie.traj)
    # Charlie.reset_agent( )
    # assert Charlie.traj = None # or len = 1
    # assert Charlie.sparse_traj = None
    # assert Charlie.diary = None 
    return None