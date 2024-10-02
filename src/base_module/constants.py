SCHEMA_NAMES = [
    "uid",
    "latitude",
    "longitude",
    "timestamp",
]
GRAVY_NAMES = [
    "identifier",
    "x",
    "y",
    "local_timestamp",
]
DEFAULT_LABELS = dict(zip(SCHEMA_NAMES, SCHEMA_NAMES))
GRAVY_LABELS = dict(zip(SCHEMA_NAMES, GRAVY_NAMES))
