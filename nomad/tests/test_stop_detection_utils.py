import pandas as pd
import pytest

def _assert_empty_stop_df(empty_df, expected_columns, expected_dtypes):
    assert empty_df.empty
    assert list(empty_df.columns) == expected_columns
    assert {col: str(dtype) for col, dtype in empty_df.dtypes.items()} == expected_dtypes


def _summarize_stop_clusters(utils, data, complete_output, passthrough_cols, traj_cols, keep_col_names=True):
    merged = data[data["cluster"] != -1]
    if merged.empty:
        return utils._get_empty_stop_df(
            data.columns,
            complete_output=complete_output,
            passthrough_cols=passthrough_cols,
            traj_cols=traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
        )

    return merged.groupby("cluster", sort=False).apply(
        lambda grp: utils.summarize_stop(
            grp,
            complete_output=complete_output,
            keep_col_names=keep_col_names,
            passthrough_cols=passthrough_cols,
            traj_cols=traj_cols,
        ),
        include_groups=False,
    ).reset_index(drop=True)


# Tests for _get_empty_stop_df function
def test_get_empty_stop_df_basic():
    """Test _get_empty_stop_df with basic parameters."""
    import nomad.stop_detection.utils as utils
    
    # Test basic case - should match summarize_stop output
    input_columns = ['timestamp', 'longitude', 'latitude']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols={'longitude': 'longitude', 'latitude': 'latitude', 'timestamp': 'timestamp'},
        keep_col_names=True,
        is_grid_based=False
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['longitude', 'latitude', 'timestamp', 'duration'],
        {
            'longitude': 'Float64',
            'latitude': 'Float64',
            'timestamp': 'Int64',
            'duration': 'Int64',
        },
    )


def test_get_empty_stop_df_complete_output():
    """Test _get_empty_stop_df with complete_output=True."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'longitude', 'latitude']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=True,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=False
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['longitude', 'latitude', 'timestamp', 'diameter', 'n_pings', 'end_timestamp', 'duration', 'max_gap'],
        {
            'longitude': 'Float64',
            'latitude': 'Float64',
            'timestamp': 'Int64',
            'diameter': 'Float64',
            'n_pings': 'Int64',
            'end_timestamp': 'Int64',
            'duration': 'Int64',
            'max_gap': 'Int64',
        },
    )


def test_get_empty_stop_df_with_passthrough():
    """Test _get_empty_stop_df with passthrough columns."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'longitude', 'latitude', 'user_id', 'location_id']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=False,
        passthrough_cols=['user_id', 'location_id'],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=False
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['longitude', 'latitude', 'timestamp', 'duration', 'user_id', 'location_id'],
        {
            'longitude': 'Float64',
            'latitude': 'Float64',
            'timestamp': 'Int64',
            'duration': 'Int64',
            'user_id': 'string',
            'location_id': 'string',
        },
    )


def test_get_empty_stop_df_xy_coordinates():
    """Test _get_empty_stop_df with x,y coordinates."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'x', 'y']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=False,
        x='x',
        y='y'
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['x', 'y', 'timestamp', 'duration'],
        {
            'x': 'Float64',
            'y': 'Float64',
            'timestamp': 'Int64',
            'duration': 'Int64',
        },
    )


def test_get_empty_stop_df_custom_traj_cols():
    """Test _get_empty_stop_df with custom traj_cols."""
    import nomad.stop_detection.utils as utils
    
    traj_cols = {
        'timestamp': 'unix_timestamp',
        'longitude': 'lon',
        'latitude': 'lat'
    }
    
    input_columns = ['unix_timestamp', 'lon', 'lat']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=traj_cols,
        keep_col_names=True,
        is_grid_based=False
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['lon', 'lat', 'unix_timestamp', 'duration'],
        {
            'lon': 'Float64',
            'lat': 'Float64',
            'unix_timestamp': 'Int64',
            'duration': 'Int64',
        },
    )


def test_get_empty_stop_df_grid_based():
    """Test _get_empty_stop_df for grid-based summarization."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'location_id']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=True
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['timestamp', 'duration', 'location_id'],
        {
            'timestamp': 'Int64',
            'duration': 'Int64',
            'location_id': 'string',
        },
    )


def test_get_empty_stop_df_grid_based_complete():
    """Test _get_empty_stop_df for grid-based with complete output."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'location_id']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=True,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=True
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['timestamp', 'duration', 'end_timestamp', 'n_pings', 'max_gap', 'location_id'],
        {
            'timestamp': 'Int64',
            'duration': 'Int64',
            'end_timestamp': 'Int64',
            'n_pings': 'Int64',
            'max_gap': 'Int64',
            'location_id': 'string',
        },
    )


def test_get_empty_stop_df_grid_based_with_geometry():
    """Test _get_empty_stop_df for grid-based with geometry."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'location_id', 'geometry']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=True,
        is_grid_based=True
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['timestamp', 'duration', 'location_id', 'geometry'],
        {
            'timestamp': 'Int64',
            'duration': 'Int64',
            'location_id': 'string',
            'geometry': 'object',
        },
    )


def test_get_empty_stop_df_keep_col_names_false():
    """Test _get_empty_stop_df with keep_col_names=False."""
    import nomad.stop_detection.utils as utils
    
    input_columns = ['timestamp', 'longitude', 'latitude']
    empty_df = utils._get_empty_stop_df(
        input_columns,
        complete_output=False,
        passthrough_cols=[],
        traj_cols=None,
        keep_col_names=False,
        is_grid_based=False
    )
    
    _assert_empty_stop_df(
        empty_df,
        ['longitude', 'latitude', 'start_timestamp', 'duration'],
        {
            'longitude': 'Float64',
            'latitude': 'Float64',
            'start_timestamp': 'Int64',
            'duration': 'Int64',
        },
    )


def test_get_empty_stop_df_matches_summarize_stop_schema_for_empty_and_clustered_input():
    """Test that empty and clustered summarize_stop paths share the exact output columns."""
    import nomad.stop_detection.utils as utils

    traj_cols = {
        "timestamp": "timestamp",
        "x": "x",
        "y": "y",
        "ha": "ha",
    }
    passthrough_cols = ["user_id"]
    base = pd.DataFrame(
        {
            "timestamp": [0, 60, 120, 180, 240, 300, 360],
            "x": [0.0, 0.1, 0.2, 0.25, 0.3, 5.0, 5.1],
            "y": [0.0, 0.1, 0.2, 0.25, 0.3, 5.0, 5.1],
            "ha": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "user_id": ["test_user"] * 7,
        }
    )

    empty_output = _summarize_stop_clusters(
        utils,
        base.assign(cluster=-1),
        complete_output=True,
        passthrough_cols=passthrough_cols,
        traj_cols=traj_cols,
    )
    clustered_output = _summarize_stop_clusters(
        utils,
        base.assign(cluster=[-1, -1, 0, 0, 0, 1, 1]),
        complete_output=True,
        passthrough_cols=passthrough_cols,
        traj_cols=traj_cols,
    )

    assert empty_output.empty
    assert not clustered_output.empty
    assert list(empty_output.columns) == list(clustered_output.columns)


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
