import pytest
import pandas as pd
from pathlib import Path
from nomad.traj_gen import Agent, Population, condense_destinations
import nomad.city_gen as cg
import pdb
import nomad.io.base as loader
from zoneinfo import ZoneInfo


@pytest.fixture
def base_city_pop():
    test_dir = Path(__file__).resolve().parent
    # repo root is parent of nomad/tests → tests/.. → nomad
    repo_root = test_dir.parents[1]
    city_path = repo_root / "examples" / "garden-city.gpkg"
    city = cg.City.from_geopackage(city_path)
    population = Population(city)
    return city, population

@pytest.fixture
def base_dest_diary():
    tz = ZoneInfo("America/New_York")
    start_time = pd.date_range(start='2024-06-01 00:00', periods=4, freq='60min', tz=tz)
    unix_timestamp = [int(t.timestamp()) for t in start_time]
    duration = [60]*4  # in minutes
    location = ['h-x13-y11'] * 1 + ['h-x13-y9'] * 1 + ['w-x18-y10'] * 1 + ['w-x18-y8'] * 1
    
    destination = pd.DataFrame(
        {"local_timestamp":start_time,
         "unix_timestamp":unix_timestamp,
         "duration":duration,
         "location":location}
         )
    destination = condense_destinations(destination)
    return destination


def test_reset_agent(base_city_pop, base_dest_diary):
    city, population = base_city_pop
    Charlie = Agent(identifier="Charlie",
              home='h-x8-y13',
              workplace='w-x15-y15',
              city=city,
              destination_diary=base_dest_diary)
    
    population.add_agent(agent=Charlie)
    Charlie.generate_trajectory(agent=Charlie, seed=100)
    traj_cols = {"datetime":"local_timestamp",
         "timestamp":"unix_timestamp",
         "x":"x",
         "y":"y"}
    assert loader._is_traj_df(df=Charlie.trajectory, traj_cols=traj_cols)
    
    Charlie.reset_trajectory()
    assert Charlie.trajectory is None
    assert Charlie.sparse_traj is None
    assert Charlie.diary.empty