SCHEMA_NAMES = [
    "id",
    "latitude",
    "longitude",
    "time",
]
GRAVY_NAMES = [
    "identifier",
    "x",
    "y",
    "local_timestamp",
]
DEFAULT_SCHEMA = dict(zip(SCHEMA_NAMES, SCHEMA_NAMES))
GRAVY_SCHEMA = dict(zip(SCHEMA_NAMES, GRAVY_NAMES))
