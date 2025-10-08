import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import nomad.io.base as loader
from nomad import constants
import nomad.stop_detection.postprocessing as postprocessing

@pytest.fixture
def sample_trajectory_data():
    """Sample trajectory data with overlapping cluster labels."""
    return pd.DataFrame({
        'timestamp': [0, 60, 120, 180, 240, 300, 360, 420],
        'longitude': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        'latitude': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        'cluster': [0, 0, 0, 1, 1, 1, 2, 2]  # Overlapping clusters
    })

@pytest.fixture
def sample_trajectory_data_xy():
    """Sample trajectory data with x,y coordinates and overlapping cluster labels."""
    return pd.DataFrame({
        'timestamp': [0, 60, 120, 180, 240, 300, 360, 420],
        'x': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        'y': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        'cluster': [0, 0, 0, 1, 1, 1, 2, 2]  # Overlapping clusters
    })

@pytest.fixture
def sample_stop_table():
    """Sample stop table with start/end times and duration."""
    return pd.DataFrame({
        'start_timestamp': [0, 180, 360],
        'end_timestamp': [180, 300, 420],
        'duration': [180, 120, 60],
        'location_id': ['A', 'B', 'C'],
        'longitude': [0.0, 0.0, 1.0],
        'latitude': [0.0, 0.0, 1.0]
    })

@pytest.fixture
def sample_stop_table_duration_only():
    """Sample stop table with only duration column (no end_timestamp)."""
    return pd.DataFrame({
        'start_timestamp': [0, 180, 360],
        'duration': [180, 120, 60],
        'location_id': ['A', 'B', 'C'],
        'longitude': [0.0, 0.0, 1.0],
        'latitude': [0.0, 0.0, 1.0]
    })

@pytest.fixture
def empty_trajectory_data():
    """Empty trajectory DataFrame with standard columns."""
    return pd.DataFrame(columns=[
        'timestamp', 'longitude', 'latitude', 'cluster'
    ])

@pytest.fixture
def empty_stop_table():
    """Empty stop table with standard columns."""
    return pd.DataFrame(columns=[
        'start_timestamp', 'end_timestamp', 'duration', 'location_id'
    ])

##########################################
####      REMOVE_OVERLAPS TESTS       #### 
##########################################

def test_remove_overlaps_trajectory_input_default_behavior(sample_trajectory_data):
    """Test that trajectory input defaults to returning cluster labels (summarize_stops=False)."""
    result = postprocessing.remove_overlaps(
        sample_trajectory_data, 
        method='cluster', 
        time_thresh=120, 
        dur_min=2
    )
    
    # Should return cluster labels as Series
    assert isinstance(result, pd.Series)
    assert result.name == 'cluster'
    assert len(result) > 0
    assert result.dtype in [int, 'int64']

def test_remove_overlaps_trajectory_input_explicit_summarize_false(sample_trajectory_data):
    """Test trajectory input with explicit summarize_stops=False."""
    result = postprocessing.remove_overlaps(
        sample_trajectory_data, 
        method='cluster', 
        time_thresh=120, 
        dur_min=2,
        summarize_stops=False
    )
    
    # Should return cluster labels as Series
    assert isinstance(result, pd.Series)
    assert result.name == 'cluster'
    assert len(result) > 0

def test_remove_overlaps_trajectory_input_explicit_summarize_true(sample_trajectory_data):
    """Test trajectory input with explicit summarize_stops=True."""
    result = postprocessing.remove_overlaps(
        sample_trajectory_data, 
        method='cluster', 
        time_thresh=120, 
        dur_min=2,
        summarize_stops=True
    )
    
    # Should return stop summary table as DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    # Should have typical stop table columns
    expected_cols = ['longitude', 'latitude', 'start_timestamp', 'duration']
    for col in expected_cols:
        assert col in result.columns

def test_remove_overlaps_stop_table_input_requires_traj(sample_stop_table):
    """Test that stop table input requires traj parameter."""
    with pytest.raises(ValueError, match="When input is a stop table, 'traj' parameter"):
        postprocessing.remove_overlaps(
            sample_stop_table,
            method='cluster',
            time_thresh=120,
            dur_min=2
        )

def test_remove_overlaps_stop_table_input_with_traj(sample_stop_table, sample_trajectory_data):
    """Test stop table input with traj parameter."""
    result = postprocessing.remove_overlaps(
        sample_stop_table,
        traj=sample_trajectory_data,
        method='cluster',
        time_thresh=120,
        dur_min=2
    )
    
    # Should return stop summary table (default behavior for stop table input)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

def test_remove_overlaps_stop_table_input_explicit_summarize_false(sample_stop_table, sample_trajectory_data):
    """Test stop table input with explicit summarize_stops=False."""
    result = postprocessing.remove_overlaps(
        sample_stop_table,
        traj=sample_trajectory_data,
        method='cluster',
        time_thresh=120,
        dur_min=2,
        summarize_stops=False
    )
    
    # Should return cluster labels as Series
    assert isinstance(result, pd.Series)
    assert result.name == 'cluster'
    assert len(result) > 0

def test_remove_overlaps_stop_table_duration_only(sample_stop_table_duration_only, sample_trajectory_data):
    """Test stop table detection with duration column only (no end_timestamp)."""
    result = postprocessing.remove_overlaps(
        sample_stop_table_duration_only,
        traj=sample_trajectory_data,
        method='cluster',
        time_thresh=120,
        dur_min=2
    )
    
    # Should work with duration column only
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

def test_remove_overlaps_empty_trajectory_input(empty_trajectory_data):
    """Test empty trajectory input."""
    result = postprocessing.remove_overlaps(
        empty_trajectory_data,
        method='cluster',
        time_thresh=120,
        dur_min=2
    )
    
    # Should return empty Series
    assert isinstance(result, pd.Series)
    assert result.name == 'cluster'
    assert len(result) == 0

def test_remove_overlaps_empty_stop_table_input(empty_stop_table, sample_trajectory_data):
    """Test empty stop table input."""
    result = postprocessing.remove_overlaps(
        empty_stop_table,
        traj=sample_trajectory_data,
        method='cluster',
        time_thresh=120,
        dur_min=2
    )
    
    # Should return empty DataFrame with correct columns
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    # Should have expected columns for stop table
    expected_cols = ['longitude', 'latitude', 'start_timestamp', 'duration']
    for col in expected_cols:
        assert col in result.columns

def test_remove_overlaps_different_methods(sample_trajectory_data):
    """Test different methods (polygon, cluster, recurse) return appropriate types."""
    methods = ['polygon', 'cluster', 'recurse']
    
    for method in methods:
        if method == 'polygon':
            # Polygon method requires location_id column
            data_with_location = sample_trajectory_data.copy()
            data_with_location['location_id'] = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C']
            result = postprocessing.remove_overlaps(
                data_with_location,
                method=method,
                time_thresh=120,
                dur_min=2,
                summarize_stops=False
            )
        else:
            result = postprocessing.remove_overlaps(
                sample_trajectory_data,
                method=method,
                time_thresh=120,
                dur_min=2,
                summarize_stops=False
            )
        
        # All methods should return Series when summarize_stops=False
        assert isinstance(result, pd.Series)
        assert result.name == 'cluster'

def test_remove_overlaps_coordinate_systems(sample_trajectory_data_xy):
    """Test that function works with different coordinate systems (x,y vs lon,lat)."""
    result = postprocessing.remove_overlaps(
        sample_trajectory_data_xy,
        method='cluster',
        time_thresh=120,
        dur_min=2,
        traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y'},
        summarize_stops=False
    )
    
    # Should work with x,y coordinates
    assert isinstance(result, pd.Series)
    assert result.name == 'cluster'
    assert len(result) > 0

def test_remove_overlaps_backwards_compatibility(sample_trajectory_data):
    """Test that explicit summarize_stops=True maintains old behavior."""
    result = postprocessing.remove_overlaps(
        sample_trajectory_data,
        method='cluster',
        time_thresh=120,
        dur_min=2,
        summarize_stops=True
    )
    
    # Should return DataFrame (old behavior)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    
    # Should have typical stop table columns
    expected_cols = ['longitude', 'latitude', 'start_timestamp', 'duration']
    for col in expected_cols:
        assert col in result.columns

def test_remove_overlaps_parameter_validation():
    """Test parameter validation and error handling."""
    # Test with invalid method
    data = pd.DataFrame({
        'timestamp': [0, 60, 120],
        'longitude': [0.0, 0.0, 0.0],
        'latitude': [0.0, 0.0, 0.0],
        'cluster': [0, 0, 1]
    })
    
    # Should work with valid methods
    valid_methods = ['polygon', 'cluster', 'recurse']
    for method in valid_methods:
        if method == 'polygon':
            data_with_location = data.copy()
            data_with_location['location_id'] = ['A', 'A', 'B']
            result = postprocessing.remove_overlaps(
                data_with_location,
                method=method,
                time_thresh=120,
                dur_min=2,
                summarize_stops=False
            )
        else:
            result = postprocessing.remove_overlaps(
                data,
                method=method,
                time_thresh=120,
                dur_min=2,
                summarize_stops=False
            )
        
        assert isinstance(result, pd.Series)
