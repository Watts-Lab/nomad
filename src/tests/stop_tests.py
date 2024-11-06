import sys
import os
import pandas as pd
import numpy as np
import numpy.random as npr
import pytest
from pandas.testing import assert_frame_equal
from collections import defaultdict
sys.path.append(os.path.abspath("/Users/andresmondragon/nomad/daphme"))
import stop_detection as SD

TIME_THRESH = 120
DIST_THRESH = 100

@pytest.fixture
def pings_clusters_df():
    data_dict = {
        'cluster': [1, 1, 2, 2, 1, 1, 3, 3, 1],
        'x': [1000, 1020, 1150, 1180, 1300, 1255, 1500, 1550, 1000],
        'y': [2000, 2025, 2100, 2150, 2300, 2250, 2500, 2550, 2050]
    }
    data = pd.DataFrame(data_dict)
    
    data.index = pd.date_range(start='2024-01-01 00:00:00', periods=9, freq='H').astype(np.int64) // 10**9
    
    return data


@pytest.fixture
def only_pings_df():
    pings_dict = {
        'x': [1000, 1005, 1010, 1200, 1205, 1207, 1400, 1403, 1008, 1012, 1206, 1402],
        'y': [2000, 2002, 2005, 2200, 2202, 2203, 2400, 2401, 2008, 2003, 2204, 2402],
    }
    pings = pd.DataFrame(pings_dict)
    pings.index = pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:02:00',
                                  '2024-01-01 01:00:00', '2024-01-01 01:01:00', '2024-01-01 01:02:00',
                                  '2024-01-01 02:00:00', '2024-01-01 02:01:00', '2024-01-01 00:03:00',
                                  '2024-01-01 00:04:00', '2024-01-01 01:03:00', '2024-01-01 02:02:00'
                                 ]).astype(np.int64) // 10**9
    return pings


@pytest.fixture
def unassigned_output_df():
    unix_timestamp_index = pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:02:00',
                                           '2024-01-01 01:00:00', '2024-01-01 01:01:00', '2024-01-01 01:02:00',
                                           '2024-01-01 02:00:00', '2024-01-01 02:01:00', '2024-01-01 00:03:00',
                                           '2024-01-01 00:04:00', '2024-01-01 01:03:00', '2024-01-01 02:02:00'
                                          ]).astype(np.int64) // 10**9
    
    output = pd.DataFrame({'cluster': -1, 'core': -1}, index=unix_timestamp_index)
    return output

def test_extract_middle(pings_clusters_df):
    ans = (2, 4)

    result = SD.extract_middle(pings_clusters_df)

    assert ans == result


def test_find_neighbors(pings_clusters_df):
    ans = defaultdict(set)
    ans[1704067200].add(1704070800)
    ans[1704070800].add(1704067200)
    
    ans[1704074400].add(1704078000)
    ans[1704078000].add(1704074400)
    
    ans[1704081600].add(1704085200)
    ans[1704085200].add(1704081600)
    
    ans[1704088800].add(1704092400)
    ans[1704092400].add(1704088800)

    result = SD.find_neighbors(pings_clusters_df, TIME_THRESH, DIST_THRESH)
    
    assert ans == result

def test_process_clusters(only_pings_df, unassigned_output_df):
    correct_min_pts = 2
    incorrect_min_pts = 20
    
    true_result = SD.process_clusters(only_pings_df, TIME_THRESH, DIST_THRESH, correct_min_pts, unassigned_output_df)
    false_result = SD.process_clusters(only_pings_df, TIME_THRESH, DIST_THRESH, incorrect_min_pts, unassigned_output_df)
    
    assert true_result
    assert not false_result


def test_medoid():
    coords = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [6, 6]
    ])

    ans = np.array([3, 3])
    result = SD.medoid(coords)
    
    assert np.array_equal(ans, result)


def test_diameter():
    coords = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [6, 6]
    ])

    ans = 7.0710678118654755
    result = SD.diameter(coords)
    
    assert ans == result


def test_dbscan(only_pings_df):
    min_pts = 2

    data_dict = {
        'cluster': [0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 2],
        'core': [0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 2]
    }
    
    timestamps = [
        1704067200, 1704067260, 1704067320,
        1704070800, 1704070860, 1704070920,
        1704074400, 1704074460,
        1704073800, 1704074400,
        1704070980, 1704074520
    ]
    
    ans = pd.DataFrame(data_dict, index=timestamps)

    result = SD.dbscan(only_pings_df, TIME_THRESH, DIST_THRESH, min_pts)
    
    assert len(result) == len(ans)
    assert_frame_equal(result, ans)


def test_temporal_dbscan():
    pass


def test_lachesis():
    pass


def test_lachesis_patches():
    pass
    
# def main() -> None:
#     test_extract_middle()

# if __name__ == "__main__":
#     main()