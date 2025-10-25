import operator

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
    "date": "date",
    "utc_date": "date",
    "x": "x",
    "y": "y",
    "geohash": "geohash",
    "tz_offset": "tz_offset",
    "duration" : "duration",
    "ha":"ha",
    "h3_cell":"h3_cell",
    "location_id" : "location_id"}

SEC_PER_UNIT = {'s': 1, 'min': 60, 'h': 3_600, 'd': 86_400, 'w': 604_800}

ALLOWED_BUILDINGS = {
    0: ['home'], 1: ['home'], 2: ['home'], 3: ['home'], 4: ['home'],
    5: ['home'], 6: ['home'], 7: ['home', 'park'],
    8: ['retail', 'work', 'park'], 9: ['work'], 10: ['work'], 11: ['work'],
    12: ['retail', 'park'], 13: ['retail', 'park'], 14: ['work'], 15: ['work'],
    16: ['work'], 17: ['work'], 18: ['retail', 'park', 'home'],
    19: ['retail', 'park', 'home'], 20: ['home'], 21: ['home'],
    22: ['home'], 23: ['home']
}

FILTER_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
}

# For trajectory generation
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

# =============================================================================
# OPENSTREETMAP CONSTANTS
# =============================================================================

# Street network constants
STREET_HIGHWAY_TYPES = [
    'motorway',
    'trunk',
    'primary',
    'secondary',
    'tertiary',
    'unclassified',
    'residential',
    'living_street',
    'service'
]

STREET_EXCLUDED_SERVICE_TYPES = ['parking_aisle', 'driveway']
STREET_EXCLUDE_COVERED = True
STREET_EXCLUDE_TUNNELS = True
STREET_EXCLUDED_SURFACES = ['paving_stones']

# Park/green space tags for downloading
PARK_TAGS = {
    'leisure': ['park', 'recreation_ground', 'garden', 'playground', 'outdoor_seating', 
                'picnic_table', 'dog_park', 'pitch', 'swimming_pool'],
    'landuse': ['park', 'recreation_ground', 'grass', 'meadow', 'allotments', 'cemetery', 
                'village_green', 'greenfield'],
    'natural': ['wood', 'grassland', 'beach']
}

DEFAULT_CRS = "EPSG:4326"

# OSM tags to subtypes (canonical mapping)
OSM_BUILDING_TO_SUBTYPE = {
    'agricultural': 'agricultural',
    'barn': 'agricultural',
    'cowshed': 'agricultural',
    'farm': 'agricultural',
    'farm_auxiliary': 'agricultural',
    'farmhouse': 'agricultural',
    'glasshouse': 'agricultural',
    'greenhouse': 'agricultural',
    'silo': 'agricultural',
    'stable': 'agricultural',
    'sty': 'agricultural',
    'civic': 'civic',
    'fire_station': 'civic',
    'government': 'civic',
    'government_office': 'civic',
    'public': 'civic',
    'commercial': 'commercial',
    'hotel': 'hotel',
    'kiosk': 'commercial',
    'marketplace': 'commercial',
    'office': 'office',
    'restaurant': 'commercial',
    'retail': 'commercial',
    'shop': 'commercial',
    'supermarket': 'commercial',
    'warehouse': 'warehouse',
    'college': 'education',
    'kindergarten': 'education',
    'school': 'education',
    'university': 'education',
    'grandstand': 'entertainment',
    'pavilion': 'entertainment',
    'sports_centre': 'entertainment',
    'sports_hall': 'entertainment',
    'stadium': 'entertainment',
    'factory': 'industrial',
    'industrial': 'industrial',
    'manufacture': 'industrial',
    'clinic': 'medical',
    'hospital': 'medical',
    'bunker': 'military',
    'military': 'military',
    'allotment_house': 'outbuilding',
    'carport': 'outbuilding',
    'roof': 'outbuilding',
    'outbuilding': 'outbuilding',
    'shed': 'outbuilding',
    'cathedral': 'religious',
    'chapel': 'religious',
    'church': 'religious',
    'monastery': 'religious',
    'mosque': 'religious',
    'presbytery': 'religious',
    'religious': 'religious',
    'shrine': 'religious',
    'synagogue': 'religious',
    'temple': 'religious',
    'wayside_shrine': 'religious',
    'apartments': 'residential',
    'bungalow': 'residential',
    'cabin': 'residential',
    'detached': 'residential',
    'dormitory': 'residential',
    'duplex': 'residential',
    'dwelling_house': 'residential',
    'garage': 'garage',
    'garages': 'garage',
    'ger': 'residential',
    'house': 'residential',
    'houseboat': 'residential',
    'hut': 'residential',
    'residential': 'residential',
    'semi': 'residential',
    'semidetached_house': 'residential',
    'static_caravan': 'residential',
    'stilt_house': 'residential',
    'terrace': 'residential',
    'townhouse': 'residential',
    'trullo': 'residential',
    'beach_hut': 'service',
    'boathouse': 'service',
    'digester': 'service',
    'guardhouse': 'service',
    'service': 'service',
    'slurry_tank': 'service',
    'storage_tank': 'service',
    'toilets': 'service',
    'transformer_tower': 'service',
    'hangar': 'transportation',
    'parking': 'parking',
    'park': 'park',
    'train_station': 'transportation',
    'transportation': 'transportation',
}

OSM_AMENITY_TO_SUBTYPE = {
    'nursing_home': 'residential',
    'bus_station': 'transportation',
    'parking': 'parking',
    'fountain': 'park',  # Fountains are park features
    'place_of_worship': 'religious',
    'clinic': 'medical',
    'dentist': 'medical',
    'doctors': 'medical',
    'hospital': 'medical',
    'pharmacy': 'medical',
    'casino': 'entertainment',
    'conference_centre': 'entertainment',
    'events_venue': 'entertainment',
    'cinema': 'entertainment',
    'theatre': 'entertainment',
    'arts_centre': 'entertainment',
    'nightclub': 'entertainment',
    'bar': 'commercial',
    'cafe': 'commercial',
    'fast_food': 'commercial',
    'food_court': 'commercial',
    'fuel': 'commercial',
    'ice_cream': 'commercial',
    'pub': 'commercial',
    'restaurant': 'commercial',
    'animal_shelter': 'civic',
    'community_centre': 'civic',
    'courthouse': 'civic',
    'fire_station': 'civic',
    'library': 'civic',
    'police': 'civic',
    'post_office': 'civic',
    'public_bath': 'civic',
    'public_building': 'civic',
    'ranger_station': 'civic',
    'shelter': 'civic',
    'social_centre': 'civic',
    'townhall': 'civic',
    'veterinary': 'civic',
    'college': 'education',
    'driving_school': 'education',
    'kindergarten': 'education',
    'music_school': 'education',
    'school': 'education',
    'university': 'education',
}

OSM_TOURISM_TO_SUBTYPE = {
    'aquarium': 'entertainment',
    'attraction': 'entertainment',
    'gallery': 'entertainment',
    'museum': 'entertainment',
}

# Subtypes to category schemas
SUBTYPE_TO_GARDEN_CITY = {
    'agricultural': 'workplace',
    'civic': 'workplace',
    'commercial': 'retail',
    'education': 'workplace',
    'entertainment': 'retail',
    'garage': 'other',
    'hotel': 'residential',
    'industrial': 'workplace',
    'medical': 'workplace',
    'military': 'workplace',
    'office': 'workplace',
    'outbuilding': 'other',
    'parking': 'park',
    'religious': 'retail',
    'residential': 'residential',
    'service': 'other',
    'transportation': 'workplace',
    'warehouse': 'workplace',
    'park': 'park',
}

SUBTYPE_TO_GEOLIFE_PLUS = {
    'agricultural': 'unknown',
    'civic': 'unknown',
    'commercial': 'commercial',
    'education': 'school',
    'entertainment': 'commercial',
    'garage': 'unknown',
    'hotel': 'residential',
    'industrial': 'unknown',
    'medical': 'unknown',
    'military': 'unknown',
    'office': 'commercial',
    'outbuilding': 'unknown',
    'parking': 'unknown',
    'religious': 'unknown',
    'residential': 'residential',
    'service': 'unknown',
    'transportation': 'unknown',
    'warehouse': 'unknown',
    'park': 'unknown',
}

CATEGORY_SCHEMAS = {
    'garden_city': SUBTYPE_TO_GARDEN_CITY,
    'geolife_plus': SUBTYPE_TO_GEOLIFE_PLUS,
}

DEFAULT_CATEGORY_SCHEMA = 'garden_city'
DEFAULT_CRS = "EPSG:4326"

GARDEN_CITY_CATEGORIES = ['residential', 'retail', 'workplace', 'park', 'other']
GEOLIFE_PLUS_CATEGORIES = ['unknown', 'residential', 'commercial', 'school']
