import numpy.random as npr

def gen_params_target_q(q_range=(0.3, 0.9), beta_dur_range=(25, 240), seed=None):
    rng = npr.default_rng(seed)
    c = rng.uniform(*q_range)
    beta_durations = rng.uniform(*beta_dur_range)
    beta_start = beta_durations / c
    beta_ping = min(
        (rng.uniform(2, 8) if rng.random() < 0.5 else rng.uniform(22, 28)),
        beta_durations
    )
    return {
        "beta_durations": beta_durations,
        "beta_start": beta_start,
        "beta_ping": beta_ping
    }