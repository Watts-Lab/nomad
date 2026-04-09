import pandas as pd
import pytest

# Tests for _get_empty_stop_columns function
def test_get_empty_stop_columns_basic():
    """Test _get_empty_stop_columns with basic parameters."""
    import nomad.stop_detection.utils as utils
    
    # Test basic case - should match summarize_stop output
    input_columns = ['timestamp', 'longitude', 'latitude']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols={'longitude': 'longitude', 'latitude': 'latitude', 'timestamp': 'timestamp'},
        keep_col_names=True,
        is_grid_based=False
    )
    
    # Should have basic columns: longitude, latitude, timestamp (original name), duration
    expected = ['longitude', 'latitude', 'timestamp', 'duration']
    assert columns == expected, f"Expected {expected}, got {columns}"


def test_get_empty_stop_columns_complete_output():
    """Test _get_empty_stop_columns with complete_output=True."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'longitude', 'latitude']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=True,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=False
    )
    
    # Should have additional columns for complete output
    expected = ['longitude', 'latitude', 'timestamp', 'duration', 'end_timestamp', 'diameter', 'n_pings', 'max_gap']
    assert columns == expected, f"Expected {expected}, got {columns}"


def test_get_empty_stop_columns_with_passthrough():
    """Test _get_empty_stop_columns with passthrough columns."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'longitude', 'latitude', 'user_id', 'location_id']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=False,
        passthrough_cols=['user_id', 'location_id'],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=False
    )
    
    # Should include passthrough columns
    expected = ['longitude', 'latitude', 'timestamp', 'duration', 'user_id', 'location_id']
    assert columns == expected, f"Expected {expected}, got {columns}"


def test_get_empty_stop_columns_xy_coordinates():
    """Test _get_empty_stop_columns with x,y coordinates."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'x', 'y']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=False,
        x='x',
        y='y'
    )
    
    # Should use x,y instead of longitude,latitude
    expected = ['x', 'y', 'timestamp', 'duration']
    assert columns == expected, f"Expected {expected}, got {columns}"


def test_get_empty_stop_columns_custom_traj_cols():
    """Test _get_empty_stop_columns with custom traj_cols."""
    import nomad.stop_detection.utils as utils
    
    traj_cols = {
        'timestamp': 'unix_timestamp',
        'longitude': 'lon',
        'latitude': 'lat'
    }
    
    input_columns = ['unix_timestamp', 'lon', 'lat']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=traj_cols,
        keep_col_names=True,
        is_grid_based=False
    )
    
    # Should use custom column names
    expected = ['lon', 'lat', 'unix_timestamp', 'duration']
    assert columns == expected, f"Expected {expected}, got {columns}"


def test_get_empty_stop_columns_grid_based():
    """Test _get_empty_stop_columns for grid-based summarization."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'location_id']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=True
    )
    
    # Should have grid-based columns: timestamp (original name), duration, location_id
    expected = {'timestamp', 'duration', 'location_id'}
    assert set(columns) == expected, f"Expected {expected}, got {set(columns)}"


def test_get_empty_stop_columns_grid_based_complete():
    """Test _get_empty_stop_columns for grid-based with complete output."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'location_id']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=True,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=True
    )
    
    # Should have complete grid-based columns
    expected = {'timestamp', 'end_timestamp', 'n_pings', 'max_gap', 'duration', 'location_id'}
    assert set(columns) == expected, f"Expected {expected}, got {set(columns)}"


def test_get_empty_stop_columns_grid_based_with_geometry():
    """Test _get_empty_stop_columns for grid-based with geometry."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'location_id', 'geometry']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=True
    )
    
    # Should include geometry column
    expected = {'timestamp', 'duration', 'location_id', 'geometry'}
    assert set(columns) == expected, f"Expected {expected}, got {set(columns)}"


def test_get_empty_stop_columns_keep_col_names_false():
    """Test _get_empty_stop_columns with keep_col_names=False."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'longitude', 'latitude']
    columns = utils._get_empty_stop_columns(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=False,
        is_grid_based=False
    )
    
    # Should use default schema column names
    expected = ['longitude', 'latitude', 'start_timestamp', 'duration']
    assert columns == expected, f"Expected {expected}, got {columns}"


def test_get_empty_stop_columns_matches_actual_summarize():
    """Test that _get_empty_stop_columns produces the same columns as actual summarize functions."""
    import nomad.stop_detection.utils as utils
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'timestamp': [1609459200],
        'longitude': [0.0],
        'latitude': [0.0],
        'user_id': ['test_user']
    })
    
    # Get columns from actual summarize_stop
    actual_result = utils.summarize_stop(
        test_data,
        complete_output=True,
        keep_col_names=True,
        passthrough_cols=['user_id'],
        traj_cols={'longitude': 'longitude', 'latitude': 'latitude', 'timestamp': 'timestamp'}
    )
    actual_columns = list(actual_result.index)
    
    # Get columns from _get_empty_stop_columns
    predicted_columns = utils._get_empty_stop_columns(
        test_data.columns,
        complete_output=True,
        passthrough_cols=['user_id'],
        traj_cols={'longitude': 'longitude', 'latitude': 'latitude', 'timestamp': 'timestamp'},
        keep_col_names=True,
        is_grid_based=False
    )
    
    # Should match (order might differ, so sort both)
    assert sorted(actual_columns) == sorted(predicted_columns), \
        f"Actual columns {sorted(actual_columns)} don't match predicted {sorted(predicted_columns)}"


def test_has_overlapping_stops_timestamp_detects_overlap():
    import nomad.stop_detection.utils as utils

    stops = pd.DataFrame(
        {
            "start_timestamp": [0, 100, 220],
            "duration": [2, 2, 1],
        }
    )

    assert utils.has_overlapping_stops(stops) is True


def test_has_overlapping_stops_timestamp_no_overlap_at_boundary():
    import nomad.stop_detection.utils as utils

    stops = pd.DataFrame(
        {
            "start_timestamp": [0, 120, 180],
            "duration": [2, 1, 1],
        }
    )

    assert utils.has_overlapping_stops(stops) is False


def test_has_overlapping_stops_datetime_with_end_columns():
    import nomad.stop_detection.utils as utils

    starts = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:03:00"])
    ends = pd.to_datetime(["2025-01-01 00:04:00", "2025-01-01 00:06:00"])
    stops = pd.DataFrame({"start_datetime": starts, "end_datetime": ends})

    assert utils.has_overlapping_stops(stops) is True


def test_has_overlapping_stops_raises_without_end_or_duration():
    import nomad.stop_detection.utils as utils

    stops = pd.DataFrame({"start_timestamp": [0, 60, 120]})

    with pytest.raises(ValueError, match=r"Missing required \(end or duration\)"):
        utils.has_overlapping_stops(stops)
