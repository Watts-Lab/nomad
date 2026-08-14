# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HDBSCAN Profiling Analysis
#
# This notebook identifies and analyzes the bottleneck in the HDBSCAN implementation.

# %% [markdown]
# ## Setup

# %%
import sys
import pandas as pd
import numpy as np
import time
import cProfile
import pstats
from io import StringIO
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path().absolute().parent))

from nomad.stop_detection import hdbscan


# %% [markdown]
# ## Test Data Generator

# %%
def generate_test_data(n_points=1000, seed=42):
    """
    Generate test data with GUARANTEED unique timestamps.
    """
    np.random.seed(seed)
    
    # Random spatial coordinates
    x = np.random.uniform(-100, 100, n_points)
    y = np.random.uniform(-100, 100, n_points)
    
    # Generate UNIQUE timestamps using cumulative sum
    base_time = int(pd.Timestamp('2024-01-01').timestamp())
    intervals = np.random.randint(60, 300, n_points)  # 1-5 min apart
    timestamps = base_time + np.cumsum(intervals)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'x': x,
        'y': y,
        'user_id': 1
    })
    
    print(f"Generated {len(data)} points")
    
    return data

# Test it
test_data = generate_test_data(100)
test_data.head()

# %% [markdown]
# ## 1. cProfile Analysis
#
# Identify which functions consume the most time.

# %%
# Generate test data
data_500 = generate_test_data(500, seed=99)

# Profile with cProfile
profiler = cProfile.Profile()
profiler.enable()

labels = hdbscan.hdbscan_labels(
    data=data_500,
    time_thresh=30,
    min_pts=2,
    min_cluster_size=2,
    dur_min=5,
    traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y', 'user_id': 'user_id'}
)

profiler.disable()

# Print stats
s = StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.strip_dirs()
ps.sort_stats('cumulative')
ps.print_stats(40)

print("="*80)
print("cProfile Results - Top 40 Functions by Cumulative Time")
print("="*80)
print(s.getvalue())

# %% [markdown]
# ## 2. line_profiler Analysis
#
# Line-by-line profiling of suspected O(n²) functions.

# %%
# Install line_profiler if needed
# !pip install line_profiler -q

# %%
# %load_ext line_profiler

# %%
data_300 = generate_test_data(300, seed=88)

# %%
# Profile _compute_core_distance
print("="*80)
print("Line-by-line profile of _compute_core_distance()")
print("="*80)

# %lprun -f hdbscan._compute_core_distance hdbscan.hdbscan_labels(\
#     data=data_300,\
#     time_thresh=30,\
#     min_pts=2,\
#     min_cluster_size=2,\
#     dur_min=5,\
#     traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y', 'user_id': 'user_id'}\
# )

# %%
# Profile _build_hdbscan_graphs
print("="*80)
print("Line-by-line profile of _build_hdbscan_graphs()")
print("="*80)

# %lprun -f hdbscan._build_hdbscan_graphs hdbscan.hdbscan_labels(\
#     data=data_300,\
#     time_thresh=30,\
#     min_pts=2,\
#     min_cluster_size=2,\
#     dur_min=5,\
#     traj_cols={'timestamp': 'timestamp', 'x': 'x', 'y': 'y', 'user_id': 'user_id'}\
# )
