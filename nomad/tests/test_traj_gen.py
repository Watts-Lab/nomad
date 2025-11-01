import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from zoneinfo import ZoneInfo
import tempfile
import shutil

from nomad.traj_gen import (
    Agent, 
    Population, 
    condense_destinations,
    sample_hier_nhpp,
    parse_agent_attr,
)
import nomad.city_gen as cg


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def garden_city():
    """Load the Garden City from the data directory."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    city_path = data_dir / "garden-city.gpkg"
    # Garden-city fixture (legacy) uses 'type' instead of 'building_type'
    city = cg.City.from_geopackage(city_path, poi_cols={'building_type':'type'})
    # Normalize legacy labels in fixture to current categories
    if 'building_type' in city.buildings_gdf.columns:
        city.buildings_gdf['building_type'] = city.buildings_gdf['building_type'].replace({'work': 'workplace', 'residential': 'home'})
    return city


@pytest.fixture
def simple_dest_diary():
    """Create a simple destination diary for testing."""
    tz = ZoneInfo("America/New_York")
    start_time = pd.date_range(start='2024-06-01 00:00', periods=4, freq='60min', tz=tz)
    ts = [int(t.timestamp()) for t in start_time]
    duration = [60] * 4  # in minutes
    location = ['h-x1-y15', 'h-x6-y5', 'w-x24-y15', 'w-x14-y15']
    
    return pd.DataFrame({
        "datetime": start_time,
         "timestamp": ts,
         "duration": duration,
        "location": location
    })


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# =============================================================================
# TESTS
# =============================================================================

def test_complete_workflow(garden_city, simple_dest_diary, temp_output_dir):
    """
    Integration test for the complete trajectory generation workflow.
    Tests: Population creation, agent generation, trajectory generation,
    sampling, reprojection, and saving to disk.
    """
    # Create population with multiple agents
    pop = Population(garden_city)
    pop.generate_agents(
        N=3,
        seed=42,
        name_count=2,
        agent_homes='h-x1-y15',
        agent_workplaces='w-x24-y15'
    )
    
    assert len(pop.roster) == 3
    
    # Generate trajectories for each agent
    beta_pings = [5, 10, 15]
    for i, agent in enumerate(pop.roster.values()):
        agent.generate_trajectory(
            destination_diary=simple_dest_diary,
            dt=0.5,
            seed=42 + i,
            step_seed=42 + i
        )
        
        assert agent.trajectory is not None
        assert len(agent.trajectory) > 0
        assert agent.trajectory['timestamp'].is_monotonic_increasing
        
        agent.sample_trajectory(
            beta_ping=beta_pings[i],
            seed=42 + i,
            ha=0.75,
            replace_sparse_traj=True
        )
        
        assert agent.sparse_traj is not None
        assert len(agent.sparse_traj) <= len(agent.trajectory)
    
    # Reproject to Web Mercator
    cent = garden_city.buildings_gdf['door_point'] if 'door_point' in garden_city.buildings_gdf.columns else garden_city.buildings_gdf.geometry.centroid
    poi_data = pd.DataFrame({
        'building_id': garden_city.buildings_gdf['id'].values,
        'x': (garden_city.buildings_gdf['door_cell_x'].astype(float) + 0.5).values if 'door_cell_x' in garden_city.buildings_gdf.columns else cent.x.values,
        'y': (garden_city.buildings_gdf['door_cell_y'].astype(float) + 0.5).values if 'door_cell_y' in garden_city.buildings_gdf.columns else cent.y.values
    })
    
    pop.reproject_to_mercator(sparse_traj=True, full_traj=True, diaries=True, poi_data=poi_data)
    
    # Verify reprojection changed coordinates
    first_agent = list(pop.roster.values())[0]
    assert abs(first_agent.sparse_traj['x'].iloc[0]) > 1000  # Should be in Mercator range
    
    # Save output files
    sparse_path = Path(temp_output_dir) / "sparse_traj"
    diaries_path = Path(temp_output_dir) / "diaries"
    homes_path = Path(temp_output_dir) / "homes"
    full_path = Path(temp_output_dir) / "full_traj"
    
    pop.save_pop(
        sparse_path=str(sparse_path),
        diaries_path=str(diaries_path),
        homes_path=str(homes_path),
        full_path=str(full_path),
        beta_ping=beta_pings,
        ha=[0.75] * 3
    )
    
    # Verify output files exist
    assert sparse_path.exists()
    assert diaries_path.exists()
    assert homes_path.exists()
    assert full_path.exists()
    assert len(list(sparse_path.glob("*.parquet"))) > 0


def test_invalid_building_ids(garden_city, simple_dest_diary):
    """
    Test that invalid building IDs raise appropriate errors.
    Edge case: Non-existent building IDs should be caught.
    """
    # Test invalid home
    with pytest.raises(ValueError, match="not found in city buildings"):
        Agent(
            identifier="test_agent",
            city=garden_city,
            home='invalid-building-id',
            workplace='w-x24-y15'
        )
    
    # Test invalid workplace
    with pytest.raises(ValueError, match="not found in city buildings"):
        Agent(
            identifier="test_agent",
            city=garden_city,
            home='h-x1-y15',
            workplace='invalid-workplace-id'
        )
    
    # Test invalid location in destination diary
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home='h-x1-y15',
        workplace='w-x24-y15'
    )
    
    bad_diary = simple_dest_diary.copy()
    bad_diary.loc[0, 'location'] = 'nonexistent-building'
    
    with pytest.raises(ValueError, match="not found in city buildings"):
        agent.generate_trajectory(destination_diary=bad_diary, dt=1, seed=42)


def test_empty_and_edge_case_trajectories(garden_city):
    """
    Test handling of empty data and edge cases.
    Edge cases: Empty destination diaries, very short trajectories, no pings sampled.
    """
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home='h-x1-y15',
        workplace='w-x24-y15'
    )
    
    # Test with EPR generation (needs end_time when destination_diary is empty)
    tz = ZoneInfo("America/New_York")
    start_time = pd.Timestamp('2024-06-01 00:00', tz=tz)
    agent.last_ping = pd.Series({
        'x': 1.5,
        'y': 15.5,
        'datetime': start_time,
        'timestamp': int(start_time.timestamp()),
        'user_id': 'test_agent'
    })
    agent.trajectory = pd.DataFrame([agent.last_ping])
    
    # Should raise error without end_time
    with pytest.raises(ValueError, match="Provide an end_time"):
        agent.generate_trajectory(dt=1, seed=42)
    
    # Test empty condense_destinations
    empty_df = pd.DataFrame()
    result = condense_destinations(empty_df)
    assert result.empty
    assert list(result.columns) == ['datetime', 'timestamp', 'duration', 'location']
    
    # Test sampling with very restrictive parameters (may result in empty)
    short_traj = pd.DataFrame({
        'x': [1.0, 2.0],
        'y': [1.0, 2.0],
        'datetime': pd.date_range(start='2024-06-01', periods=2, freq='1s', tz=tz),
        'timestamp': [1717214400, 1717214401]
    })
    
    result = sample_hier_nhpp(
        short_traj,
        beta_start=1000,  # Very low probability
        beta_durations=0.1,
        beta_ping=1000,
        seed=42,
        ha=0.75
    )
    
    # Should return DataFrame with correct columns even if empty
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in short_traj.columns)


def test_trajectory_monotonicity_and_data_quality(garden_city, simple_dest_diary):
    """
    Test that generated and sampled trajectories maintain data quality.
    Tests: Monotonic timestamps, proper deduplication, timestamp integrity.
    """
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home='h-x1-y15',
        workplace='w-x24-y15'
    )
    
    # Generate trajectory
    agent.generate_trajectory(
        destination_diary=simple_dest_diary,
        dt=1,
        seed=42
    )
    
    # Check monotonicity
    assert agent.trajectory['timestamp'].is_monotonic_increasing
    
    # Sample trajectory with deduplication
    agent.sample_trajectory(
        beta_ping=2,
        seed=42,
        ha=0.75,
        deduplicate=True
    )
    
    # Check sampled trajectory quality
    assert agent.sparse_traj['timestamp'].is_monotonic_increasing
    
    # Check no duplicate timestamps
    assert len(agent.sparse_traj['timestamp'].unique()) == len(agent.sparse_traj)
    
    # Check horizontal accuracy is present and reasonable
    assert 'ha' in agent.sparse_traj.columns
    assert (agent.sparse_traj['ha'] >= 8/15).all()  # Above minimum
    assert (agent.sparse_traj['ha'] <= 20).all()  # Below cap


def test_agent_state_management(garden_city, simple_dest_diary):
    """
    Test agent state management: reset, trajectory replacement, caching.
    Tests: reset_trajectory, replace_sparse_traj, cache_traj.
    """
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home='h-x1-y15',
        workplace='w-x24-y15'
    )
    
    # Generate initial trajectory
    agent.generate_trajectory(destination_diary=simple_dest_diary, dt=1, seed=42)
    initial_traj_len = len(agent.trajectory)
    
    # Sample trajectory
    agent.sample_trajectory(beta_ping=5, seed=42, ha=0.75)
    initial_sparse_len = len(agent.sparse_traj)
    
    # Test partial reset (keep trajectory, reset sparse)
    agent.reset_trajectory(trajectory=False, sparse=True, last_ping=False, diary=False)
    assert agent.trajectory is not None
    assert len(agent.trajectory) == initial_traj_len
    assert agent.sparse_traj is None
    
    # Regenerate sparse
    agent.sample_trajectory(beta_ping=5, seed=43, ha=0.75, replace_sparse_traj=True)
    assert agent.sparse_traj is not None
    
    # Test cache_traj (empties full trajectory but keeps last_ping)
    agent.sample_trajectory(beta_ping=5, seed=44, ha=0.75, cache_traj=True, replace_sparse_traj=True)
    assert agent.trajectory.empty
    assert agent.last_ping is not None
    assert agent.sparse_traj is not None
    
    # Test full reset
    agent.reset_trajectory()
    assert agent.trajectory is None
    assert agent.sparse_traj is None
    assert agent.last_ping is None


def test_parse_agent_attr_validation():
    """
    Test parse_agent_attr function with various inputs.
    Tests: Input validation, list length checking, type checking.
    """
    # Valid cases
    get_none = parse_agent_attr(None, 5, "test")
    assert get_none(0) is None
    
    get_str = parse_agent_attr("h-x1-y1", 5, "test")
    assert get_str(0) == "h-x1-y1"
    assert get_str(4) == "h-x1-y1"
    
    get_list = parse_agent_attr(["a", "b", "c"], 3, "test")
    assert get_list(0) == "a"
    assert get_list(2) == "c"
    
    # Invalid cases
    with pytest.raises(ValueError, match="must be a list of length 5"):
        parse_agent_attr(["a", "b"], 5, "test")
    
    with pytest.raises(ValueError, match="must be either a string"):
        parse_agent_attr(123, 5, "test")


def test_population_agent_generation(garden_city):
    """
    Test population agent generation with various configurations.
    Tests: Fixed homes/workplaces, list-based assignments, random assignments.
    """
    pop = Population(garden_city)
    
    # Test basic generation with fixed home/workplace
    pop.generate_agents(N=5, seed=42, name_count=2, agent_homes='h-x1-y15', agent_workplaces='w-x24-y15')
    assert len(pop.roster) == 5
    for agent in pop.roster.values():
        assert agent.home == 'h-x1-y15'
        assert agent.workplace == 'w-x24-y15'
    
    # Test with list of homes/workplaces
    pop2 = Population(garden_city)
    homes = ['h-x1-y15', 'h-x6-y5', 'h-x1-y25']
    workplaces = ['w-x24-y15', 'w-x14-y15', 'w-x1-y0']
    
    pop2.generate_agents(N=3, seed=42, name_count=2, agent_homes=homes, agent_workplaces=workplaces)
    
    home_list = [agent.home for agent in pop2.roster.values()]
    work_list = [agent.workplace for agent in pop2.roster.values()]
    
    assert set(home_list) == set(homes)
    assert set(work_list) == set(workplaces)
    
    # Test random assignment
    pop3 = Population(garden_city)
    pop3.generate_agents(N=3, seed=42, name_count=2)
    for agent in pop3.roster.values():
        assert agent.home is not None
        assert agent.workplace is not None


def test_sample_hier_nhpp_edge_cases():
    """
    Test sample_hier_nhpp function with edge cases and parameters.
    Tests: Burst sampling, deduplication, ha validation, empty results.
    """
    tz = ZoneInfo("America/New_York")
    times = pd.date_range(start='2024-06-01 00:00', periods=100, freq='1min', tz=tz)
    traj = pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y': np.linspace(0, 10, 100),
        'datetime': times,
        'timestamp': [int(t.timestamp()) for t in times]
    })
    
    # Test basic sampling
    result = sample_hier_nhpp(traj, beta_ping=5, seed=42, ha=0.75)
    assert len(result) > 0
    assert len(result) <= len(traj)
    assert 'ha' in result.columns
    
    # Test with bursts
    sampled, bursts = sample_hier_nhpp(
        traj,
        beta_start=30,
        beta_durations=20,
        beta_ping=5,
        seed=42,
        ha=0.75,
        output_bursts=True
    )
    assert 'start_time' in bursts.columns
    assert 'end_time' in bursts.columns
    
    # Test deduplication
    result_dedup = sample_hier_nhpp(traj, beta_ping=1, seed=42, ha=0.75, deduplicate=True)
    assert len(result_dedup['timestamp'].unique()) == len(result_dedup)
    
    # Test ha validation
    with pytest.raises(ValueError, match="ha must exceed"):
        sample_hier_nhpp(traj, beta_ping=5, ha=0.4, seed=42)


def test_condense_destinations_logic(simple_dest_diary):
    """
    Test condense_destinations function with various diary patterns.
    Tests: Consecutive location condensing, no-op when already condensed.
    """
    # Test condensing consecutive locations
    tz = ZoneInfo("America/New_York")
    times = pd.date_range(start='2024-06-01 00:00', periods=6, freq='15min', tz=tz)
    diary = pd.DataFrame({
        'datetime': times,
        'timestamp': [int(t.timestamp()) for t in times],
        'duration': [15] * 6,
        'location': ['A', 'A', 'A', 'B', 'B', 'A']
    })
    
    result = condense_destinations(diary)
    
    assert len(result) == 3
    assert result.iloc[0]['location'] == 'A'
    assert result.iloc[0]['duration'] == 45  # 3 * 15
    assert result.iloc[1]['location'] == 'B'
    assert result.iloc[1]['duration'] == 30  # 2 * 15
    assert result.iloc[2]['location'] == 'A'
    assert result.iloc[2]['duration'] == 15
    
    # Test when already condensed (no-op)
    result2 = condense_destinations(result)
    assert len(result2) == len(result)
    pd.testing.assert_frame_equal(result, result2)


def test_coordinate_reprojection(garden_city, simple_dest_diary):
    """
    Test coordinate reprojection from Garden City to Web Mercator.
    Tests: Transformation accuracy, ha scaling, preservation of other columns.
    """
    pop = Population(garden_city)
    pop.generate_agents(N=1, seed=42, name_count=2, agent_homes='h-x1-y15', agent_workplaces='w-x24-y15')
    
    agent = list(pop.roster.values())[0]
    agent.generate_trajectory(destination_diary=simple_dest_diary, dt=0.5, seed=42)
    agent.sample_trajectory(beta_ping=5, seed=42, ha=0.75)
    
    # Record original coordinates
    original_x = agent.sparse_traj['x'].iloc[0]
    original_y = agent.sparse_traj['y'].iloc[0]
    original_ha = agent.sparse_traj['ha'].iloc[0]
    
    # Reproject
    pop.reproject_to_mercator(sparse_traj=True, full_traj=True, diaries=False)
    
    # Verify transformation
    new_x = agent.sparse_traj['x'].iloc[0]
    new_y = agent.sparse_traj['y'].iloc[0]
    new_ha = agent.sparse_traj['ha'].iloc[0]
    
    # Check coordinates transformed (block_size=15, offsets applied)
    assert new_x != original_x
    assert new_y != original_y
    assert abs(new_x - (-4265699 + 15 * original_x)) < 1e-6
    assert abs(new_y - (4392976 + 15 * original_y)) < 1e-6
    
    # Check ha scaled
    assert abs(new_ha - (15 * original_ha)) < 1e-6
    
    # Check full trajectory also transformed
    assert abs(agent.trajectory['x'].iloc[0]) > 1000  # In Mercator range
