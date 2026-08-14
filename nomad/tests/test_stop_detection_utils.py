import pandas as pd
import pytest
import nomad.stop_detection.utils as utils

def _assert_empty_stop_df(empty_df, expected_columns, expected_dtypes):
    assert empty_df.empty
    # every stop table leads with the per-ping cluster label
    assert list(empty_df.columns) == ["cluster"] + expected_columns
    assert {col: str(dtype) for col, dtype in empty_df.dtypes.items()} == {
        "cluster": "Int64",
        **expected_dtypes,
    }


@pytest.mark.parametrize("keep_col_names", [True, False])
@pytest.mark.parametrize("complete_output", [False, True])
def test_summarize_stops_empty_output_matches_summarized_schema(complete_output, keep_col_names):
    """The no-cluster path must return the schema the summarized path produces."""
    traj_cols = {"timestamp": "timestamp", "x": "x", "y": "y", "ha": "ha"}
    passthrough_cols = ["user_id"]
    data = pd.DataFrame(
        {
            "timestamp": [0, 60, 120, 180, 240, 300, 360],
            "x": [0.0, 0.1, 0.2, 0.25, 0.3, 5.0, 5.1],
            "y": [0.0, 0.1, 0.2, 0.25, 0.3, 5.0, 5.1],
            "ha": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "user_id": ["test_user"] * 7,
        }
    )
    kwargs = dict(
        complete_output=complete_output,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
    )

    summarized = utils.summarize_stops(
        data, pd.Series([-1, -1, 0, 0, 0, 1, 1], name="cluster"), **kwargs
    )
    empty = utils.summarize_stops(
        data, pd.Series(-1, index=data.index, name="cluster"), **kwargs
    )

    assert not summarized.empty
    assert empty.empty
    assert list(empty.columns) == list(summarized.columns)
    assert empty.dtypes.equals(summarized.dtypes)


@pytest.mark.parametrize(
    "time_col,times",
    [
        pytest.param("timestamp", [0], id="timestamp"),
        pytest.param("datetime", pd.to_datetime([0], unit="s", utc=True), id="datetime"),
    ],
)
def test_summarize_stop_single_ping_reports_no_gap(time_col, times):
    """A one-ping cluster has no gap, and summarizing it must not raise."""
    group = pd.DataFrame({time_col: times, "x": [0.0], "y": [0.0]})

    summary = utils.summarize_stop(
        group, complete_output=True, traj_cols={time_col: time_col, "x": "x", "y": "y"}
    )

    assert summary["duration"] == 0
    assert summary["max_gap"] == 0


# Tests for _get_empty_stop_df function
def test_get_empty_stop_df_basic():
    """Test _get_empty_stop_df with basic parameters."""
    
    # Test basic case - should match summarize_stop output
    input_columns = ['timestamp', 'longitude', 'latitude']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
    
    input_columns = ['timestamp', 'longitude', 'latitude']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
            'diameter': 'Int64',
            'n_pings': 'Int64',
            'end_timestamp': 'Int64',
            'duration': 'Int64',
            'max_gap': 'Int64',
        },
    )


def test_get_empty_stop_df_with_passthrough():
    """Test _get_empty_stop_df with passthrough columns."""
    
    input_columns = ['timestamp', 'longitude', 'latitude', 'user_id', 'location_id']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
            'user_id': 'object',
            'location_id': 'object',
        },
    )


def test_get_empty_stop_df_xy_coordinates():
    """Test _get_empty_stop_df with x,y coordinates."""
    
    input_columns = ['timestamp', 'x', 'y']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
    
    traj_cols = {
        'timestamp': 'unix_timestamp',
        'longitude': 'lon',
        'latitude': 'lat'
    }
    
    input_columns = ['unix_timestamp', 'lon', 'lat']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
    
    input_columns = ['timestamp', 'location_id']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
            'location_id': 'object',
        },
    )


def test_get_empty_stop_df_grid_based_complete():
    """Test _get_empty_stop_df for grid-based with complete output."""
    
    input_columns = ['timestamp', 'location_id']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
            'location_id': 'object',
        },
    )


def test_get_empty_stop_df_grid_based_with_geometry():
    """Test _get_empty_stop_df for grid-based with geometry."""
    
    input_columns = ['timestamp', 'location_id', 'geometry']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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
            'location_id': 'object',
            'geometry': 'object',
        },
    )


def test_get_empty_stop_df_keep_col_names_false():
    """Test _get_empty_stop_df with keep_col_names=False."""
    
    input_columns = ['timestamp', 'longitude', 'latitude']
    empty_df = utils._get_empty_stop_df(
        pd.DataFrame(columns=input_columns),
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


def test_has_overlapping_stops_timestamp_detects_overlap():
    stops = pd.DataFrame(
        {
            "start_timestamp": [0, 100, 220],
            "duration": [2, 2, 1],
        }
    )

    assert utils.has_overlapping_stops(stops) is True


def test_has_overlapping_stops_timestamp_no_overlap_at_boundary():
    stops = pd.DataFrame(
        {
            "start_timestamp": [0, 120, 180],
            "duration": [2, 1, 1],
        }
    )

    assert utils.has_overlapping_stops(stops) is False


def test_has_overlapping_stops_datetime_with_end_columns():
    starts = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:03:00"])
    ends = pd.to_datetime(["2025-01-01 00:04:00", "2025-01-01 00:06:00"])
    stops = pd.DataFrame({"start_datetime": starts, "end_datetime": ends})

    assert utils.has_overlapping_stops(stops) is True


def test_has_overlapping_stops_raises_without_end_or_duration():
    stops = pd.DataFrame({"start_timestamp": [0, 60, 120]})

    with pytest.raises(ValueError, match=r"Missing required \(end or duration\)"):
        utils.has_overlapping_stops(stops)
