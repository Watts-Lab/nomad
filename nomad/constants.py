# TO DO: Add other schemas
# TO DO: Add tesselation cell default

DEFAULT_SCHEMA = {
    "user_id": "user_id",
    "latitude": "latitude",
    "longitude": "longitude",
    "datetime": "datetime",
    "start_datetime":"start_datetime",
    "end_datetime":"end_datetime",
    "start_timestamp":"start_timestamp",
    "end_timestamp":"end_timestamp",
    "timestamp": "timestamp",
    "x": "x",
    "y": "y",
    "geohash": "geohash",
    "tz_offset": "tz_offset",
    "duration" : "duration",
    "location" : "location"}

ALLOWED_BUILDINGS = {
    0: ['home'], 1: ['home'], 2: ['home'], 3: ['home'], 4: ['home'],
    5: ['home'], 6: ['home'], 7: ['home', 'park'],
    8: ['retail', 'work', 'park'], 9: ['work'], 10: ['work'], 11: ['work'],
    12: ['retail', 'park'], 13: ['retail', 'park'], 14: ['work'], 15: ['work'],
    16: ['work'], 17: ['work'], 18: ['retail', 'park', 'home'],
    19: ['retail', 'park', 'home'], 20: ['home'], 21: ['home'],
    22: ['home'], 23: ['home']
}

DEFAULT_SPEEDS = {'park': 2/1.96,
                  'home': 0.75/1.96,
                  'work': 0.75/1.96,
                  'retail': 1.75/1.96}

FAST_SPEEDS = {'park': 2.5/1.96,
               'home': 1/1.96,
               'work': 1/1.96,
               'retail': 2/1.96}

SLOW_SPEEDS = {'park': 1.5/1.96,
               'home': 0.5/1.96,
               'work': 0.5/1.96,
               'retail': 1.5/1.96}

DEFAULT_STILL_PROBS = {'park': 0.5,
                       'home': 0.9,
                       'work': 0.9,
                       'retail': 0.5}


FAST_STILL_PROBS = {'park': 0.1,
                    'home': 0.75,
                    'work': 0.75,
                    'retail': 0.2}

SLOW_STILL_PROBS = {'park': 0.75,
                    'home': 0.95,
                    'work': 0.95,
                    'retail': 0.75}

DEFAULT_STAY_PROBS = {'park': 1-((1/1)/4),
                      'retail': 1-((1/0.5)/4),
                      'work': 1-((1/7)/4),
                      'home': 1-((1/14)/4)}