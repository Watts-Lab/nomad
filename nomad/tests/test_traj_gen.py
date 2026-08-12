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
    parse_agent_attr,
    generate_ping_times,
    thin_traj_by_times,
    _sample_horizontal_noise,
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
    city = cg.City.from_geopackage(city_path)
    city.compute_shortest_paths(callable_only=False)
    return city


@pytest.fixture
def default_ids(garden_city):
    b = garden_city.buildings_gdf
    homes = b[b['building_type'] == 'home']['id'].tolist()
    workplaces = b[b['building_type'] == 'workplace']['id'].tolist()
    print("DEBUG home ids (first 10):", homes[:10])
    print("DEBUG workplace ids (first 10):", workplaces[:10])
    assert homes and workplaces, "Regenerated garden city must contain home and workplace buildings"
    return {
        'home': homes[0],
        'home2': homes[1] if len(homes) > 1 else homes[0],
        'work': workplaces[0],
        'work2': workplaces[1] if len(workplaces) > 1 else workplaces[0],
    }


@pytest.fixture
def simple_dest_diary(default_ids):
    """Create a simple destination diary for testing."""
    tz = ZoneInfo("America/New_York")
    start_time = pd.date_range(start='2024-06-01 00:00', periods=4, freq='60min', tz=tz)
    ts = [int(t.timestamp()) for t in start_time]
    duration = [60] * 4  # in minutes
    location = [default_ids['home'], default_ids['home2'], default_ids['work'], default_ids['work2']]
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

def test_complete_workflow(garden_city, simple_dest_diary, temp_output_dir, default_ids):
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
        agent_homes=default_ids['home'],
        agent_workplaces=default_ids['work']
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

        agent.set_beta_params(
            beta_start=None,
            beta_durations=None,
            beta_ping=beta_pings[i]
        )
        agent.sample_trajectory(
            seed=42 + i,
            ha=0.75,
            replace_sparse_traj=True
        )
        
        assert agent.sparse_traj is not None
        assert len(agent.sparse_traj) <= len(agent.trajectory)
    
    # Reproject to Web Mercator
    poi_data = pd.DataFrame({
        'building_id': garden_city.buildings_gdf['id'].values,
        'x': garden_city.buildings_gdf['door_point'].apply(lambda p: p[0]).values,
        'y': garden_city.buildings_gdf['door_point'].apply(lambda p: p[1]).values
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


def test_invalid_building_ids(garden_city, simple_dest_diary, default_ids):
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
                workplace=default_ids['work']
        )
    
    # Test invalid workplace
    with pytest.raises(ValueError, match="not found in city buildings"):
            Agent(
            identifier="test_agent",
            city=garden_city,
                home=default_ids['home'],
            workplace='invalid-workplace-id'
        )
    
    # Test invalid location in destination diary
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
    )
    
    bad_diary = simple_dest_diary.copy()
    bad_diary.loc[0, 'location'] = 'nonexistent-building'
    
    with pytest.raises(KeyError):
        agent.generate_trajectory(destination_diary=bad_diary, dt=1, seed=42)


def test_empty_and_edge_case_trajectories(garden_city, default_ids):
    """
    Test handling of empty data and edge cases.
    Edge cases: Empty destination diaries, very short trajectories, no pings sampled.
    """
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
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
    
    agent.trajectory = short_traj
    agent.set_beta_params(
        beta_start=1000,  # Very low probability
        beta_durations=0.1,
        beta_ping=1000
    )
    agent.sample_trajectory(seed=42, ha=0.75, replace_sparse_traj=True)
    
    # Should return DataFrame with correct columns even if empty
    assert isinstance(agent.sparse_traj, pd.DataFrame)
    assert all(col in agent.sparse_traj.columns for col in short_traj.columns)


def test_trajectory_monotonicity_and_data_quality(garden_city, simple_dest_diary, default_ids):
    """
    Test that generated and sampled trajectories maintain data quality.
    Tests: Monotonic timestamps, proper deduplication, timestamp integrity.
    """
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
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
    agent.set_beta_params(beta_start=None, beta_durations=None, beta_ping=2)
    agent.sample_trajectory(
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


def test_agent_state_management(garden_city, simple_dest_diary, default_ids):
    """
    Test agent state management: reset, trajectory replacement, caching.
    Tests: reset_trajectory, replace_sparse_traj, flush_traj_cache.
    """
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
    )
    
    # Generate initial trajectory
    agent.generate_trajectory(destination_diary=simple_dest_diary, dt=1, seed=42)
    initial_traj_len = len(agent.trajectory)

    # Sample trajectory
    agent.set_beta_params(beta_start=None, beta_durations=None, beta_ping=5)
    agent.sample_trajectory(seed=42, ha=0.75)
    initial_sparse_len = len(agent.sparse_traj)
    
    # Test partial reset (keep trajectory, reset sparse)
    agent.reset_trajectory(trajectory=False, sparse=True, last_ping=False, diary=False)
    assert agent.trajectory is not None
    assert len(agent.trajectory) == initial_traj_len
    assert agent.sparse_traj is None
    
    # Regenerate sparse
    agent.sample_trajectory(seed=43, ha=0.75, replace_sparse_traj=True)
    assert agent.sparse_traj is not None
    
    # Test flush_traj_cache (keeps only the final ping for chunked generation)
    agent.sample_trajectory(seed=44, ha=0.75, flush_traj_cache=True, replace_sparse_traj=True)
    assert len(agent.trajectory) == 1
    assert agent.trajectory.iloc[-1].to_dict() == agent.last_ping
    assert agent.sparse_traj is not None

    flushed_timestamp = agent.trajectory.timestamp.iloc[-1]
    agent.destination_diary = simple_dest_diary.iloc[[-1]].assign(duration=1)
    agent.generate_trajectory(dt=1, seed=45)
    assert len(agent.trajectory) == 2
    assert agent.trajectory.timestamp.iloc[-1] > flushed_timestamp
    
    # Test full reset
    agent.reset_trajectory()
    assert agent.trajectory is None
    assert agent.sparse_traj is None
    assert agent.last_ping is None


def test_agent_set_beta_params(garden_city, default_ids):
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
    )

    agent.set_beta_params(beta_start=30, beta_durations=10, beta_ping=5)
    assert agent.sparsity_params == {
        'beta_start': 30,
        'beta_durations': 10,
        'beta_ping': 5
    }

    agent.set_beta_params(beta_start=0, beta_durations=np.inf, beta_ping=5)
    assert agent.sparsity_params == {
        'beta_start': None,
        'beta_durations': None,
        'beta_ping': 5
    }

    params = Population.gen_params_target_q(
        q=0.4,
        beta_start=100,
        beta_ping=5,
        seed=42
    )
    agent.set_beta_params(params, beta_start=30, beta_durations=10, beta_ping=5)
    assert agent.sparsity_params == {
        'beta_start': 100,
        'beta_durations': 40,
        'beta_ping': 5,
        'q': 0.4
    }

    with pytest.raises(ValueError, match="beta_ping must be provided"):
        agent.set_beta_params(beta_start=None, beta_durations=None)

    with pytest.raises(ValueError, match="beta_start and beta_durations must either both be provided"):
        agent.set_beta_params(beta_start=30, beta_durations=None, beta_ping=5)

    with pytest.raises(ValueError, match="beta_start and beta_durations must either both be provided"):
        agent.set_beta_params(beta_start=0, beta_durations=10, beta_ping=5)

    with pytest.raises(ValueError, match="beta_start and beta_durations must either both be provided"):
        agent.set_beta_params(beta_start=30, beta_durations=np.inf, beta_ping=5)

    with pytest.raises(ValueError, match="missing required keys"):
        agent.set_beta_params({'beta_start': 30, 'beta_durations': 10, 'q': 0.4})


def test_agent_sample_trajectory_uses_sparsity_params(garden_city, simple_dest_diary, default_ids):
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work'],
        sparsity_params={
            'beta_start': None,
            'beta_durations': None,
            'beta_ping': 7,
            'q': 0.5,
            'f': 0.02
        }
    )

    agent.generate_trajectory(destination_diary=simple_dest_diary, dt=1, seed=42)
    burst_info = agent.sample_trajectory(seed=42, ha=0.75)

    assert agent.sparse_traj is not None
    assert list(burst_info.columns) == ['start_time', 'end_time']
    assert agent.sparsity_params['beta_ping'] == 7
    assert agent.sparsity_params['q'] == 0.5
    assert agent.sparsity_params['f'] == 0.02

    agent_without_params = Agent(
        identifier="no_params",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
    )
    agent_without_params.generate_trajectory(destination_diary=simple_dest_diary, dt=1, seed=42)
    with pytest.raises(ValueError, match="Set them with Agent.set_beta_params"):
        agent_without_params.sample_trajectory(seed=42, ha=0.75)

    agent_without_params.sparsity_params = {'beta_ping': 5}
    with pytest.raises(ValueError, match="Set them with Agent.set_beta_params"):
        agent_without_params.sample_trajectory(seed=42, ha=0.75)

    agent_without_params.sparsity_params = {
        'beta_start': None,
        'beta_durations': None,
        'beta_ping': None
    }
    with pytest.raises(ValueError, match="beta_ping must be provided"):
        agent_without_params.sample_trajectory(seed=42, ha=0.75)


def test_agent_sample_trajectory_clears_stored_burst_params(garden_city, simple_dest_diary, default_ids):
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
    )

    agent.generate_trajectory(destination_diary=simple_dest_diary, dt=1, seed=42)
    agent.set_beta_params(beta_start=0, beta_durations=np.inf, beta_ping=5)
    agent.sample_trajectory(seed=42, ha=0.75)

    assert agent.sparse_traj is not None
    assert agent.sparsity_params['beta_start'] is None
    assert agent.sparsity_params['beta_durations'] is None
    assert agent.sparsity_params['beta_ping'] == 5


def test_agent_sample_trajectory_debug_mode(garden_city, simple_dest_diary, default_ids):
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work']
    )
    agent.generate_trajectory(destination_diary=simple_dest_diary, dt=1, seed=42)
    agent.set_beta_params(beta_start=None, beta_durations=None, beta_ping=5)

    agent.trajectory = agent.trajectory.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="input trajectory is not sorted"):
        agent.sample_trajectory(seed=42, ha=0.75, debug_mode=True)

    agent.trajectory = agent.trajectory.sort_values('timestamp').reset_index(drop=True)
    agent.sparse_traj = agent.trajectory.iloc[[-1]].set_index('timestamp', drop=False)
    with pytest.raises(ValueError, match="sparse trajectory is not sorted"):
        agent.sample_trajectory(seed=42, ha=0.75, debug_mode=True)


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
    
    with pytest.raises(ValueError, match="must be a string"):
        parse_agent_attr(123, 5, "test")


def test_population_agent_generation(garden_city, default_ids):
    """
    Test population agent generation with various configurations.
    Tests: Fixed homes/workplaces, list-based assignments, random assignments.
    """
    pop = Population(garden_city)
    
    # Test basic generation with fixed home/workplace
    pop.generate_agents(N=5, seed=42, name_count=2, agent_homes=default_ids['home'], agent_workplaces=default_ids['work'])
    assert len(pop.roster) == 5
    for agent in pop.roster.values():
        assert agent.home == default_ids['home']
        assert agent.workplace == default_ids['work']
    
    # Test with list of homes/workplaces
    pop2 = Population(garden_city)
    b = garden_city.buildings_gdf
    homes = b[b['building_type']=='home']['id'].head(3).tolist()
    workplaces = b[b['building_type']=='workplace']['id'].head(3).tolist()
    
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


def test_population_gen_params_target_q_derives_missing_beta():
    params = Population.gen_params_target_q(
        q=0.4,
        beta_start=(100, 100),
        beta_ping=[3, 5],
        beta_ping_probs=[0.0, 1.0],
        seed=42
    )

    assert params["q"] == 0.4
    assert params["beta_start"] == 100
    assert params["beta_durations"] == 40
    assert params["beta_ping"] == 5

    params = Population.gen_params_target_q(
        q=0.25,
        beta_durations=30,
        beta_ping=(2, 8),
        seed=42
    )

    assert params["beta_start"] == 120
    assert 2 <= params["beta_ping"] <= 8


def test_population_gen_params_target_f_derives_missing_beta():
    params = Population.gen_params_target_f(
        f=0.02,
        beta_start=100,
        beta_ping=10,
        beta_durations=None,
        seed=42
    )

    assert params["f"] == 0.02
    assert params["beta_durations"] == 20
    assert params["beta_start"] == 100
    assert params["beta_ping"] == 10

    params = Population.gen_params_target_f(
        f=0.02,
        beta_start=None,
        beta_ping=5,
        beta_durations=20,
        seed=42
    )

    assert params["beta_start"] == 200

    params = Population.gen_params_target_f(
        f=0.001,
        beta_start=100,
        beta_ping=None,
        beta_durations=20,
        seed=42
    )

    assert params["beta_ping"] == 200
    assert params["beta_durations"] / (params["beta_start"] * params["beta_ping"]) == params["f"]


def test_population_param_sampling_validates_inputs():
    params = Population.sample_from_intervals(
        beta_start={"values": [100, 200], "probs": [0.0, 1.0]},
        beta_ping=(5, 5),
        beta_durations=[20],
        seed=42
    )

    assert params == {
        "beta_durations": 20,
        "beta_start": 200,
        "beta_ping": 5
    }

    single_value_params = Population.sample_from_intervals(
        beta_start=200,
        beta_ping=5,
        beta_durations=20,
        seed=42
    )

    assert single_value_params == {
        "beta_durations": 20,
        "beta_start": 200,
        "beta_ping": 5
    }

    with pytest.raises(ValueError, match="Provide exactly one"):
        Population.gen_params_target_q(q=0.5, beta_start=100, beta_ping=5, beta_durations=40)

    with pytest.raises(ValueError, match="beta_ping must be provided"):
        Population.gen_params_target_q(q=0.5, beta_start=100, beta_durations=40)

    with pytest.raises(ValueError, match="Provide exactly one"):
        Population.gen_params_target_q(q=0.5, beta_ping=5)

    with pytest.raises(ValueError, match="Provide exactly two"):
        Population.gen_params_target_f(f=0.02, beta_start=100, beta_ping=5, beta_durations=20)

    with pytest.raises(ValueError, match="q must be provided"):
        Population.gen_params_target_q(q=None, beta_start=100, beta_ping=5)


def test_agent_sample_trajectory_edge_cases(garden_city, default_ids):
    """
    Test sample_trajectory with edge cases and parameters.
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
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work'],
        trajectory=traj
    )
    
    # Test basic sampling
    agent.set_beta_params(beta_start=None, beta_durations=None, beta_ping=5)
    agent.sample_trajectory(seed=42, ha=0.75)
    assert len(agent.sparse_traj) > 0
    assert len(agent.sparse_traj) <= len(traj)
    assert 'ha' in agent.sparse_traj.columns
    assert agent.sparse_traj.index.name == 'timestamp'
    
    # Test with bursts
    agent.set_beta_params(
        beta_start=30,
        beta_durations=20,
        beta_ping=5
    )
    bursts = agent.sample_trajectory(seed=42, ha=0.75, replace_sparse_traj=True)
    assert 'start_time' in bursts.columns
    assert 'end_time' in bursts.columns
    
    # Test deduplication
    agent.set_beta_params(beta_start=None, beta_durations=None, beta_ping=1)
    agent.sample_trajectory(seed=42, ha=0.75, deduplicate=True, replace_sparse_traj=True)
    assert len(agent.sparse_traj['timestamp'].unique()) == len(agent.sparse_traj)
    
    # Test ha validation
    with pytest.raises(ValueError, match="ha must exceed"):
        agent.sample_trajectory(ha=0.4, seed=42, replace_sparse_traj=True)


def test_generate_ping_times_and_thinning():
    tz = ZoneInfo("America/New_York")
    times = pd.date_range(start='2024-06-01 00:00', periods=120, freq='1s', tz=tz)
    traj = pd.DataFrame({
        'x': np.linspace(0, 1, 120),
        'y': np.linspace(0, 1, 120),
        'datetime': times,
        'timestamp': [int(t.timestamp()) for t in times]
    })

    t0 = int(traj['timestamp'].iloc[0])
    t_end = int(traj['timestamp'].iloc[-1])

    pts, burst_info = generate_ping_times(t0, t_end, beta_ping=0.1, seed=123)
    assert isinstance(pts, np.ndarray)
    assert list(burst_info.columns) == ['start_time', 'end_time']
    assert np.all((pts >= t0) & (pts <= t_end)) or pts.size == 0
    if pts.size:
        assert np.all(np.diff(pts) >= 0)

    thinned = thin_traj_by_times(traj, pts, deduplicate=True)
    assert set(['x','y','datetime','timestamp']).issubset(set(thinned.columns))
    if pts.size:
        # After thinning with dedup, timestamps align one-to-one with rows
        assert len(thinned) == len(np.unique(np.searchsorted(traj['timestamp'].to_numpy(), pts, side='right') - 1))


def test_sample_horizontal_noise_basic():
    n = 100
    ha_realized, noise = _sample_horizontal_noise(n, ha=0.75, rng=np.random.default_rng(123))
    assert len(ha_realized) == n
    assert noise.shape == (n, 2)
    assert np.isfinite(ha_realized).all()
    assert np.isfinite(noise).all()
    assert (noise >= -250).all() and (noise <= 250).all()


def test_bursts_info_nonempty_and_tz(garden_city, default_ids):
    tz = ZoneInfo("America/New_York")
    times = pd.date_range(start='2024-06-01 00:00', periods=600, freq='1s', tz=tz)
    traj = pd.DataFrame({
        'x': np.linspace(0, 1, 600),
        'y': np.linspace(0, 1, 600),
        'datetime': times,
        'timestamp': [int(t.timestamp()) for t in times]
    })
    agent = Agent(
        identifier="test_agent",
        city=garden_city,
        home=default_ids['home'],
        workplace=default_ids['work'],
        trajectory=traj
    )
    agent.set_beta_params(
        beta_start=0.2,      # bursts every ~12s on average
        beta_durations=0.1,  # durations ~6s
        beta_ping=0.1        # ping mean ~6s
    )
    bursts = agent.sample_trajectory(seed=123, ha=0.75)

    assert 'start_time' in bursts.columns and 'end_time' in bursts.columns
    assert not bursts.empty
    # tz-aware
    assert getattr(bursts['start_time'].dt.tz, 'key', None) == tz.key
    assert getattr(bursts['end_time'].dt.tz, 'key', None) == tz.key
    # within window
    t0 = traj['datetime'].iloc[0]
    t1 = traj['datetime'].iloc[-1]
    assert bursts['start_time'].min() >= t0
    assert bursts['end_time'].max() <= t1


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


def test_coordinate_reprojection(garden_city, simple_dest_diary, default_ids):
    """
    Test coordinate reprojection from Garden City to Web Mercator.
    Tests: Transformation accuracy, ha scaling, preservation of other columns.
    """
    pop = Population(garden_city)
    pop.generate_agents(N=1, seed=42, name_count=2, agent_homes=default_ids['home'], agent_workplaces=default_ids['work'])
    
    agent = list(pop.roster.values())[0]
    agent.generate_trajectory(destination_diary=simple_dest_diary, dt=0.5, seed=42)
    agent.set_beta_params(beta_start=None, beta_durations=None, beta_ping=5)
    agent.sample_trajectory(seed=42, ha=0.75)
    
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


def test_agent_last_ping_initialization(garden_city, default_ids):
    """
    Test various ways to initialize Agent.last_ping.
    Covers: trajectory, x/y coords, location, datetime (string vs Timestamp), timestamp.
    """
    home_id = default_ids['home']
    work_id = default_ids['work']
    tz = ZoneInfo("America/New_York")
    
    # Case 1: Default initialization (no kwargs)
    agent1 = Agent('agent1', garden_city, home=home_id, workplace=work_id)
    assert agent1.last_ping is not None
    assert 'x' in agent1.last_ping
    assert 'y' in agent1.last_ping
    assert 'datetime' in agent1.last_ping
    assert 'timestamp' in agent1.last_ping
    assert isinstance(agent1.last_ping['datetime'], pd.Timestamp)
    assert hasattr(agent1.last_ping['datetime'], 'hour')
    
    # Case 2: Initialize with x, y coordinates
    agent2 = Agent('agent2', garden_city, home=home_id, workplace=work_id, x=100.5, y=200.5)
    assert agent2.last_ping['x'] == 100.5
    assert agent2.last_ping['y'] == 200.5
    assert isinstance(agent2.last_ping['datetime'], pd.Timestamp)
    
    # Case 3: Initialize with location (building ID)
    agent3 = Agent('agent3', garden_city, home=home_id, workplace=work_id, location=work_id)
    work_centroid = garden_city.buildings_gdf[garden_city.buildings_gdf['id'] == work_id].iloc[0]['geometry'].centroid
    assert abs(agent3.last_ping['x'] - work_centroid.x) < 1e-6
    assert abs(agent3.last_ping['y'] - work_centroid.y) < 1e-6
    
    # Case 4: Initialize with datetime as string
    agent4 = Agent('agent4', garden_city, home=home_id, workplace=work_id, 
                   datetime='2024-01-15 14:30:00')
    assert isinstance(agent4.last_ping['datetime'], pd.Timestamp)
    assert agent4.last_ping['datetime'].hour == 14
    assert agent4.last_ping['datetime'].minute == 30
    
    # Case 5: Initialize with datetime as pd.Timestamp
    dt = pd.Timestamp('2024-02-20 09:15:00', tz=tz)
    agent5 = Agent('agent5', garden_city, home=home_id, workplace=work_id, datetime=dt)
    assert isinstance(agent5.last_ping['datetime'], pd.Timestamp)
    assert agent5.last_ping['datetime'].hour == 9
    assert agent5.last_ping['datetime'].minute == 15
    
    # Case 6: Initialize with timestamp
    ts = 1704085200  # 2024-01-01 00:00:00 EST
    agent6 = Agent('agent6', garden_city, home=home_id, workplace=work_id, timestamp=ts)
    assert agent6.last_ping['timestamp'] == ts
    
    # Case 7: Initialize with trajectory (takes precedence)
    traj = pd.DataFrame({
        'x': [10.0, 20.0, 30.0],
        'y': [15.0, 25.0, 35.0],
        'datetime': pd.date_range('2024-03-01', periods=3, freq='1h', tz=tz),
        'timestamp': [1709251200, 1709254800, 1709258400],
        'user_id': ['agent7'] * 3
    })
    agent7 = Agent('agent7', garden_city, home=home_id, workplace=work_id, trajectory=traj)
    assert agent7.last_ping['x'] == 30.0
    assert agent7.last_ping['y'] == 35.0
    assert agent7.last_ping['timestamp'] == 1709258400
    
    # Case 8: Trajectory takes precedence over kwargs (should warn)
    with pytest.warns(UserWarning, match="trajectory and position kwargs provided"):
        agent8 = Agent('agent8', garden_city, home=home_id, workplace=work_id, 
                      trajectory=traj, x=999.0, datetime='2025-01-01')
    assert agent8.last_ping['x'] == 30.0  # From trajectory, not 999.0
    
    # Case 9: Combined x, y, and datetime (string with timezone)
    agent9 = Agent('agent9', garden_city, home=home_id, workplace=work_id,
                   x=50.0, y=60.0, datetime='2024-04-10 16:45:00-04:00')
    assert agent9.last_ping['x'] == 50.0
    assert agent9.last_ping['y'] == 60.0
    assert isinstance(agent9.last_ping['datetime'], pd.Timestamp)
    assert agent9.last_ping['datetime'].hour == 16
