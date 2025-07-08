import numpy.random as npr

def gen_params_target_q(q_range=(0.3, 0.9), beta_dur_range=(25, 240), beta_ping_range=(2, 8), seed=None):
    rng = npr.default_rng(seed)
    c = rng.uniform(*q_range)
    beta_durations = rng.uniform(*beta_dur_range)
    beta_start = beta_durations / c

    lo, hi = beta_ping_range
    beta_ping_range_high = (lo+20, hi+20)
    beta_ping = min(
        (rng.uniform(*beta_ping_range) if rng.random() < 0.5 else rng.uniform(*beta_ping_range_high)),
        beta_durations
    )
    return {
        "beta_durations": beta_durations,
        "beta_start": beta_start,
        "beta_ping": beta_ping
    }

def gen_params_ranges(beta_start_range=(80, 450),beta_ping_range=(1.5, 20), beta_dur_range=(50, 300), seed=None):
    rng = npr.default_rng(seed)

    beta_durations = rng.uniform(*beta_dur_range)
    beta_start = rng.uniform(*beta_start_range)
    beta_ping = min(
        rng.uniform(*beta_ping_range),
        beta_durations
    )
    return {
        "beta_durations": beta_durations,
        "beta_start": beta_start,
        "beta_ping": beta_ping
    }

