"""
OpenStreetMap feature mapping constants for the NOMAD library.

This module contains comprehensive mappings from OSM tags to location types,
building categories, and business classifications. Based on the official OSM wiki:
https://wiki.openstreetmap.org/wiki/Map_features

The categorization system is designed around the user's requirements:
- location_type: Main categories (residential, retail, workplace, park, walkway, other)
- building_category: Structural building types
- business_category: Business/service classifications with NAICS equivalents
"""

# =============================================================================
# LOCATION TYPE MAPPINGS
# Maps OSM tags to the 5 main location types the user wants
# =============================================================================

LOCATION_TYPE_MAPPING = {
    "residential": {
        "building": [
            "residential", "house", "apartments", "hotel", "motel", "dormitory",
            "detached", "semidetached_house", "terrace", "bungalow", "cabin",
            "farm", "farmhouse", "static_caravan", "chalet", "vacation_home"
        ]
    },
    "retail": {
        "building": [
            "retail", "commercial", "shop", "supermarket", "mall", "marketplace",
            "department_store", "kiosk", "vending_machine"
        ],
        "amenity": [
            "restaurant", "cafe", "bar", "pub", "fast_food", "food_court",
            "marketplace", "shop", "department_store", "mall", "bank", "atm",
            "pharmacy", "hairdresser", "beauty", "place_of_worship",
            "community_centre", "library", "bookstore", "laundry", "dry_cleaning"
        ],
        "shop": True,  # All shop types are retail
        "office": ["bank", "insurance", "financial", "advertising_agency", "travel_agent"]
    },
    "workplace": {
        "building": [
            "office", "industrial", "commercial", "public", "hospital", "school",
            "university", "college", "government", "civic", "warehouse", "factory",
            "workshop", "laboratory", "studio", "courthouse", "fire_station",
            "police", "post_office", "bank", "research", "training", "conference_centre"
        ],
        "amenity": [
            "school", "university", "college", "hospital", "clinic", "doctors",
            "dentist", "pharmacy", "courthouse", "fire_station", "police",
            "post_office", "government", "public_building", "library",
            "research_institute", "training", "conference_centre", "community_centre",
            "social_facility", "arts_centre", "theatre", "cinema", "museum"
        ],
        "office": True,  # All office types are workplace
        "healthcare": True,  # All healthcare facilities are workplace
        "craft": True  # All craft workshops are workplace
    },
    "park": {
        "leisure": [
            "park", "recreation_ground", "garden", "playground", "nature_reserve",
            "wildlife_hide", "bird_hide", "outdoor_seating", "picnic_table",
            "dog_park", "pitch", "stadium", "sports_centre", "fitness_centre",
            "swimming_pool", "golf_course", "miniature_golf", "water_park",
            "theme_park", "zoo", "aquarium", "beach_resort"
        ],
        "landuse": [
            "park", "recreation_ground", "forest", "grass", "meadow", "allotments",
            "cemetery", "orchard", "vineyard", "farmland", "greenhouse_horticulture",
            "plant_nursery", "conservation", "village_green", "greenfield"
        ],
        "natural": [
            "wood", "forest", "grassland", "heath", "scrub", "fell", "bare_rock",
            "scree", "beach", "wetland", "marsh", "swamp", "water", "bay", "strait",
            "cape", "peninsula", "archipelago", "island", "islet"
        ],
        "amenity": [
            "grave_yard", "cemetery", "fountain", "bench", "waste_basket",
            "drinking_water", "shelter", "toilets"
        ],
        "boundary": ["national_park", "protected_area"]
    },
    "walkway": {
        "highway": [
            "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
            "residential", "service", "track", "path", "footway", "cycleway",
            "bridleway", "steps", "corridor", "elevator", "escalator",
            "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "tertiary_link", "living_street", "pedestrian", "road"
        ],
        "route": ["road", "bicycle", "foot", "hiking", "walking", "running"],
        "railway": ["rail", "light_rail", "subway", "monorail", "funicular"],
        "aeroway": ["runway", "taxiway"],
        "waterway": ["river", "canal", "stream", "ditch", "drain"]
    }
}

# =============================================================================
# BUILDING CATEGORIES
# Detailed structural classification of buildings
# =============================================================================

BUILDING_CATEGORIES = {
    # Residential buildings
    "residential": {"category": "residential", "subcategory": "multi_family"},
    "house": {"category": "residential", "subcategory": "single_family"},
    "apartments": {"category": "residential", "subcategory": "multi_family"},
    "hotel": {"category": "residential", "subcategory": "hospitality"},
    "motel": {"category": "residential", "subcategory": "hospitality"},
    "dormitory": {"category": "residential", "subcategory": "institutional"},
    "cabin": {"category": "residential", "subcategory": "recreational"},
    "farm": {"category": "residential", "subcategory": "agricultural"},
    "farmhouse": {"category": "residential", "subcategory": "agricultural"},
    "static_caravan": {"category": "residential", "subcategory": "mobile"},
    "detached": {"category": "residential", "subcategory": "single_family"},
    "semidetached_house": {"category": "residential", "subcategory": "multi_family"},
    "terrace": {"category": "residential", "subcategory": "multi_family"},
    "bungalow": {"category": "residential", "subcategory": "single_family"},
    "chalet": {"category": "residential", "subcategory": "recreational"},
    "vacation_home": {"category": "residential", "subcategory": "recreational"},

    # Retail and commercial buildings
    "retail": {"category": "retail", "subcategory": "general"},
    "commercial": {"category": "retail", "subcategory": "commercial"},
    "shop": {"category": "retail", "subcategory": "retail"},
    "supermarket": {"category": "retail", "subcategory": "grocery"},
    "mall": {"category": "retail", "subcategory": "shopping_center"},
    "department_store": {"category": "retail", "subcategory": "department_store"},
    "kiosk": {"category": "retail", "subcategory": "convenience"},
    "marketplace": {"category": "retail", "subcategory": "market"},
    "vending_machine": {"category": "retail", "subcategory": "vending"},

    # Workplace buildings
    "office": {"category": "workplace", "subcategory": "office"},
    "industrial": {"category": "workplace", "subcategory": "industrial"},
    "school": {"category": "workplace", "subcategory": "education"},
    "university": {"category": "workplace", "subcategory": "education"},
    "college": {"category": "workplace", "subcategory": "education"},
    "hospital": {"category": "workplace", "subcategory": "healthcare"},
    "government": {"category": "workplace", "subcategory": "government"},
    "civic": {"category": "workplace", "subcategory": "civic"},
    "warehouse": {"category": "workplace", "subcategory": "storage"},
    "factory": {"category": "workplace", "subcategory": "manufacturing"},
    "workshop": {"category": "workplace", "subcategory": "craft"},
    "laboratory": {"category": "workplace", "subcategory": "research"},
    "studio": {"category": "workplace", "subcategory": "creative"},
    "courthouse": {"category": "workplace", "subcategory": "legal"},
    "fire_station": {"category": "workplace", "subcategory": "emergency"},
    "police": {"category": "workplace", "subcategory": "emergency"},
    "post_office": {"category": "workplace", "subcategory": "postal"},
    "bank": {"category": "workplace", "subcategory": "financial"},
    "research": {"category": "workplace", "subcategory": "research"},
    "training": {"category": "workplace", "subcategory": "education"},
    "conference_centre": {"category": "workplace", "subcategory": "business"},

    # Public and institutional buildings
    "public": {"category": "workplace", "subcategory": "public"},
    "library": {"category": "workplace", "subcategory": "cultural"},
    "community_centre": {"category": "workplace", "subcategory": "community"},
    "place_of_worship": {"category": "workplace", "subcategory": "religious"},
    "cathedral": {"category": "workplace", "subcategory": "religious"},
    "chapel": {"category": "workplace", "subcategory": "religious"},
    "church": {"category": "workplace", "subcategory": "religious"},
    "mosque": {"category": "workplace", "subcategory": "religious"},
    "synagogue": {"category": "workplace", "subcategory": "religious"},
    "temple": {"category": "workplace", "subcategory": "religious"},
    "monastery": {"category": "workplace", "subcategory": "religious"},
    "convent": {"category": "workplace", "subcategory": "religious"},
    "museum": {"category": "workplace", "subcategory": "cultural"},
    "theatre": {"category": "workplace", "subcategory": "cultural"},
    "cinema": {"category": "workplace", "subcategory": "entertainment"},
    "arts_centre": {"category": "workplace", "subcategory": "cultural"},

    # Transportation and infrastructure
    "transportation": {"category": "workplace", "subcategory": "transportation"},
    "train_station": {"category": "workplace", "subcategory": "transportation"},
    "bus_station": {"category": "workplace", "subcategory": "transportation"},
    "airport": {"category": "workplace", "subcategory": "transportation"},
    "hangar": {"category": "workplace", "subcategory": "transportation"},

    # Sports and recreation
    "sports_hall": {"category": "workplace", "subcategory": "sports"},
    "stadium": {"category": "workplace", "subcategory": "sports"},
    "sports_centre": {"category": "workplace", "subcategory": "sports"},
    "gymnasium": {"category": "workplace", "subcategory": "sports"},
    "riding_hall": {"category": "workplace", "subcategory": "sports"},

    # Storage and utility
    "storage": {"category": "workplace", "subcategory": "storage"},
    "warehouse": {"category": "workplace", "subcategory": "storage"},
    "garage": {"category": "workplace", "subcategory": "storage"},
    "parking": {"category": "workplace", "subcategory": "transportation"},
    "carport": {"category": "workplace", "subcategory": "transportation"},

    # Military and security
    "bunker": {"category": "workplace", "subcategory": "military"},
    "military": {"category": "workplace", "subcategory": "military"},
    "guardhouse": {"category": "workplace", "subcategory": "security"},

    # Construction and temporary
    "construction": {"category": "workplace", "subcategory": "construction"},
    "temporary": {"category": "workplace", "subcategory": "temporary"},
    "container": {"category": "workplace", "subcategory": "temporary"},
}

# =============================================================================
# BUSINESS CLASSIFICATIONS
# Detailed business categorization with NAICS equivalents
# OSM doesn't use NAICS codes but we can map to equivalent classifications
# =============================================================================

BUSINESS_CLASSIFICATIONS = {
    # RETAIL CLASSIFICATIONS
    "supermarket": {"category": "retail", "subcategory": "grocery", "naics_equivalent": "445110"},
    "convenience": {"category": "retail", "subcategory": "convenience", "naics_equivalent": "445120"},
    "bakery": {"category": "retail", "subcategory": "food_specialty", "naics_equivalent": "445291"},
    "butcher": {"category": "retail", "subcategory": "food_specialty", "naics_equivalent": "445210"},
    "greengrocer": {"category": "retail", "subcategory": "food_specialty", "naics_equivalent": "445230"},
    "seafood": {"category": "retail", "subcategory": "food_specialty", "naics_equivalent": "445220"},
    "deli": {"category": "retail", "subcategory": "food_specialty", "naics_equivalent": "445110"},
    "department_store": {"category": "retail", "subcategory": "general_merchandise", "naics_equivalent": "452210"},
    "mall": {"category": "retail", "subcategory": "shopping_center", "naics_equivalent": "531120"},
    "clothes": {"category": "retail", "subcategory": "apparel", "naics_equivalent": "448140"},
    "shoes": {"category": "retail", "subcategory": "apparel", "naics_equivalent": "448210"},
    "jewelry": {"category": "retail", "subcategory": "specialty", "naics_equivalent": "448310"},
    "electronics": {"category": "retail", "subcategory": "electronics", "naics_equivalent": "443142"},
    "computer": {"category": "retail", "subcategory": "electronics", "naics_equivalent": "443142"},
    "mobile_phone": {"category": "retail", "subcategory": "electronics", "naics_equivalent": "443142"},
    "furniture": {"category": "retail", "subcategory": "home_furnishings", "naics_equivalent": "442110"},
    "hardware": {"category": "retail", "subcategory": "home_improvement", "naics_equivalent": "444130"},
    "garden_centre": {"category": "retail", "subcategory": "garden", "naics_equivalent": "444220"},
    "car": {"category": "retail", "subcategory": "automotive", "naics_equivalent": "441110"},
    "car_repair": {"category": "retail", "subcategory": "automotive_services", "naics_equivalent": "811111"},
    "bookstore": {"category": "retail", "subcategory": "books", "naics_equivalent": "451211"},
    "pharmacy": {"category": "retail", "subcategory": "health_beauty", "naics_equivalent": "446110"},
    "optician": {"category": "retail", "subcategory": "health_beauty", "naics_equivalent": "446130"},
    "cosmetics": {"category": "retail", "subcategory": "health_beauty", "naics_equivalent": "446120"},
    "hairdresser": {"category": "retail", "subcategory": "personal_services", "naics_equivalent": "812112"},
    "beauty": {"category": "retail", "subcategory": "personal_services", "naics_equivalent": "812112"},
    "restaurant": {"category": "retail", "subcategory": "food_service", "naics_equivalent": "722511"},
    "cafe": {"category": "retail", "subcategory": "food_service", "naics_equivalent": "722515"},
    "fast_food": {"category": "retail", "subcategory": "food_service", "naics_equivalent": "722513"},
    "bar": {"category": "retail", "subcategory": "food_service", "naics_equivalent": "722410"},
    "pub": {"category": "retail", "subcategory": "food_service", "naics_equivalent": "722410"},
    "bank": {"category": "retail", "subcategory": "financial_services", "naics_equivalent": "522110"},
    "atm": {"category": "retail", "subcategory": "financial_services", "naics_equivalent": "522320"},
    "laundry": {"category": "retail", "subcategory": "personal_services", "naics_equivalent": "812310"},
    "dry_cleaning": {"category": "retail", "subcategory": "personal_services", "naics_equivalent": "812320"},
    "bookstore": {"category": "retail", "subcategory": "books_media", "naics_equivalent": "451211"},
    "newsagent": {"category": "retail", "subcategory": "books_media", "naics_equivalent": "451212"},
    "stationery": {"category": "retail", "subcategory": "office_supplies", "naics_equivalent": "453210"},

    # OFFICE AND PROFESSIONAL SERVICES
    "company": {"category": "workplace", "subcategory": "corporate", "naics_equivalent": "551114"},
    "advertising_agency": {"category": "workplace", "subcategory": "professional_services", "naics_equivalent": "541810"},
    "architect": {"category": "workplace", "subcategory": "professional_services", "naics_equivalent": "541310"},
    "lawyer": {"category": "workplace", "subcategory": "professional_services", "naics_equivalent": "541110"},
    "accountant": {"category": "workplace", "subcategory": "professional_services", "naics_equivalent": "541211"},
    "insurance": {"category": "workplace", "subcategory": "financial_services", "naics_equivalent": "524210"},
    "financial": {"category": "workplace", "subcategory": "financial_services", "naics_equivalent": "523920"},
    "it": {"category": "workplace", "subcategory": "technology", "naics_equivalent": "541512"},
    "software": {"category": "workplace", "subcategory": "technology", "naics_equivalent": "541511"},
    "research": {"category": "workplace", "subcategory": "research", "naics_equivalent": "541720"},
    "newspaper": {"category": "workplace", "subcategory": "media", "naics_equivalent": "511110"},
    "radio": {"category": "workplace", "subcategory": "media", "naics_equivalent": "515112"},
    "telecommunication": {"category": "workplace", "subcategory": "telecom", "naics_equivalent": "517919"},
    "travel_agent": {"category": "workplace", "subcategory": "travel_services", "naics_equivalent": "561510"},

    # HEALTHCARE
    "hospital": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "622110"},
    "clinic": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "621493"},
    "doctors": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "621111"},
    "dentist": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "621210"},
    "pharmacy": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "446110"},
    "veterinary": {"category": "workplace", "subcategory": "veterinary", "naics_equivalent": "541940"},
    "physiotherapist": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "621340"},
    "psychotherapist": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "621330"},
    "optometrist": {"category": "workplace", "subcategory": "healthcare", "naics_equivalent": "621320"},

    # EDUCATION
    "school": {"category": "workplace", "subcategory": "education", "naics_equivalent": "611110"},
    "university": {"category": "workplace", "subcategory": "education", "naics_equivalent": "611310"},
    "college": {"category": "workplace", "subcategory": "education", "naics_equivalent": "611210"},
    "kindergarten": {"category": "workplace", "subcategory": "education", "naics_equivalent": "624410"},
    "childcare": {"category": "workplace", "subcategory": "education", "naics_equivalent": "624410"},
    "music_school": {"category": "workplace", "subcategory": "education", "naics_equivalent": "611610"},
    "language_school": {"category": "workplace", "subcategory": "education", "naics_equivalent": "611630"},

    # GOVERNMENT AND PUBLIC SERVICES
    "government": {"category": "workplace", "subcategory": "government", "naics_equivalent": "921110"},
    "police": {"category": "workplace", "subcategory": "public_safety", "naics_equivalent": "922120"},
    "fire_station": {"category": "workplace", "subcategory": "public_safety", "naics_equivalent": "922160"},
    "post_office": {"category": "workplace", "subcategory": "postal", "naics_equivalent": "491110"},
    "library": {"category": "workplace", "subcategory": "cultural", "naics_equivalent": "519120"},
    "courthouse": {"category": "workplace", "subcategory": "legal", "naics_equivalent": "922110"},
    "prison": {"category": "workplace", "subcategory": "correctional", "naics_equivalent": "922140"},
    "townhall": {"category": "workplace", "subcategory": "government", "naics_equivalent": "921110"},

    # RELIGIOUS AND COMMUNITY
    "place_of_worship": {"category": "workplace", "subcategory": "religious", "naics_equivalent": "813110"},
    "community_centre": {"category": "workplace", "subcategory": "community", "naics_equivalent": "624190"},
    "social_facility": {"category": "workplace", "subcategory": "social_services", "naics_equivalent": "624190"},
    "arts_centre": {"category": "workplace", "subcategory": "cultural", "naics_equivalent": "711310"},
    "theatre": {"category": "workplace", "subcategory": "cultural", "naics_equivalent": "711110"},
    "cinema": {"category": "workplace", "subcategory": "entertainment", "naics_equivalent": "512131"},
    "museum": {"category": "workplace", "subcategory": "cultural", "naics_equivalent": "712110"},
    "nightclub": {"category": "workplace", "subcategory": "entertainment", "naics_equivalent": "722410"},

    # TRANSPORTATION
    "train_station": {"category": "workplace", "subcategory": "transportation", "naics_equivalent": "485110"},
    "bus_station": {"category": "workplace", "subcategory": "transportation", "naics_equivalent": "485210"},
    "airport": {"category": "workplace", "subcategory": "transportation", "naics_equivalent": "481111"},
    "taxi": {"category": "workplace", "subcategory": "transportation", "naics_equivalent": "485310"},
    "car_rental": {"category": "workplace", "subcategory": "transportation", "naics_equivalent": "532111"},
    "fuel": {"category": "workplace", "subcategory": "transportation", "naics_equivalent": "447110"},
    "parking": {"category": "workplace", "subcategory": "transportation", "naics_equivalent": "812930"},
}

# =============================================================================
# AMENITY MAPPINGS
# Maps amenity tags to location types for more precise classification
# =============================================================================

AMENITY_TO_LOCATION_TYPE = {
    # RETAIL AMENITIES
    "restaurant": "retail",
    "cafe": "retail",
    "bar": "retail",
    "pub": "retail",
    "fast_food": "retail",
    "food_court": "retail",
    "marketplace": "retail",
    "shop": "retail",
    "department_store": "retail",
    "mall": "retail",
    "bank": "retail",
    "atm": "retail",
    "pharmacy": "retail",
    "hairdresser": "retail",
    "beauty": "retail",
    "laundry": "retail",
    "dry_cleaning": "retail",
    "bookstore": "retail",
    "newsagent": "retail",
    "stationery": "retail",

    # WORKPLACE AMENITIES
    "school": "workplace",
    "university": "workplace",
    "college": "workplace",
    "hospital": "workplace",
    "clinic": "workplace",
    "doctors": "workplace",
    "dentist": "workplace",
    "courthouse": "workplace",
    "fire_station": "workplace",
    "police": "workplace",
    "post_office": "workplace",
    "government": "workplace",
    "public_building": "workplace",
    "library": "workplace",
    "research_institute": "workplace",
    "training": "workplace",
    "conference_centre": "workplace",
    "community_centre": "workplace",
    "social_facility": "workplace",
    "arts_centre": "workplace",
    "theatre": "workplace",
    "cinema": "workplace",
    "museum": "workplace",
    "nightclub": "workplace",

    # PARK AMENITIES
    "grave_yard": "park",
    "cemetery": "park",
    "fountain": "park",
    "bench": "park",
    "waste_basket": "park",
    "drinking_water": "park",
    "shelter": "park",
    "toilets": "park",
}

# =============================================================================
# SHOP TYPE MAPPINGS
# Maps detailed shop types to retail subcategories
# =============================================================================

SHOP_TO_SUBCATEGORY = {
    # FOOD AND GROCERY
    "supermarket": "grocery",
    "convenience": "convenience",
    "bakery": "food_specialty",
    "butcher": "food_specialty",
    "greengrocer": "food_specialty",
    "seafood": "food_specialty",
    "deli": "food_specialty",
    "dairy": "food_specialty",
    "cheese": "food_specialty",
    "farm": "food_specialty",

    # GENERAL MERCHANDISE
    "department_store": "general_merchandise",
    "mall": "shopping_center",
    "general": "general_merchandise",
    "variety_store": "general_merchandise",

    # APPAREL AND ACCESSORIES
    "clothes": "apparel",
    "shoes": "apparel",
    "fashion": "apparel",
    "boutique": "apparel",
    "jewelry": "specialty",
    "watches": "specialty",
    "perfumery": "specialty",
    "bag": "apparel",
    "leather": "apparel",

    # ELECTRONICS AND TECHNOLOGY
    "electronics": "electronics",
    "computer": "electronics",
    "mobile_phone": "electronics",
    "telecommunication": "electronics",
    "hifi": "electronics",
    "camera": "electronics",
    "video_games": "electronics",

    # HOME AND GARDEN
    "furniture": "home_furnishings",
    "hardware": "home_improvement",
    "garden_centre": "garden",
    "doityourself": "home_improvement",
    "building_materials": "home_improvement",
    "paint": "home_improvement",
    "plumbing": "home_improvement",
    "electrical": "home_improvement",

    # AUTOMOTIVE
    "car": "automotive",
    "car_repair": "automotive_services",
    "car_parts": "automotive",
    "motorcycle": "automotive",
    "bicycle": "automotive",
    "tyres": "automotive",

    # BOOKS AND MEDIA
    "bookstore": "books_media",
    "newsagent": "books_media",
    "stationery": "office_supplies",
    "gift": "specialty",
    "toys": "specialty",
    "games": "specialty",

    # HEALTH AND BEAUTY
    "pharmacy": "health_beauty",
    "optician": "health_beauty",
    "cosmetics": "health_beauty",
    "hairdresser": "personal_services",
    "beauty": "personal_services",
    "tattoo": "personal_services",
    "massage": "personal_services",

    # FOOD SERVICE
    "restaurant": "food_service",
    "cafe": "food_service",
    "fast_food": "food_service",
    "bar": "food_service",
    "pub": "food_service",
    "ice_cream": "food_service",
    "pastry": "food_service",

    # FINANCIAL SERVICES
    "bank": "financial_services",
    "atm": "financial_services",

    # OTHER RETAIL
    "sports": "sports_outdoors",
    "outdoor": "sports_outdoors",
    "pet": "pet_supplies",
    "florist": "specialty",
    "alcohol": "food_specialty",
    "beverages": "food_specialty",
    "chocolate": "food_specialty",
    "coffee": "food_specialty",
    "tea": "food_specialty",
    "wine": "food_specialty",
    "kiosk": "convenience",
    "vending_machine": "convenience",
    "marketplace": "market",
    "travel_agency": "travel_services",
    "carpet": "home_furnishings",
    "curtain": "home_furnishings",
    "interior_decoration": "home_furnishings",
}

# =============================================================================
# OFFICE TYPE MAPPINGS
# Maps office types to workplace subcategories
# =============================================================================

OFFICE_TO_SUBCATEGORY = {
    "company": "corporate",
    "advertising_agency": "professional_services",
    "architect": "professional_services",
    "lawyer": "professional_services",
    "accountant": "professional_services",
    "insurance": "financial_services",
    "bank": "financial_services",
    "financial": "financial_services",
    "it": "technology",
    "software": "technology",
    "research": "research",
    "newspaper": "media",
    "radio": "media",
    "telecommunication": "telecom",
    "travel_agent": "travel_services",
    "association": "non_profit",
    "charity": "non_profit",
    "cooperative": "cooperative",
    "coworking": "coworking",
    "diplomatic": "government",
    "employment_agency": "professional_services",
    "estate_agent": "real_estate",
    "foundation": "non_profit",
    "government": "government",
    "moving_company": "logistics",
    "ngo": "non_profit",
    "notary": "professional_services",
    "political_party": "political",
    "quango": "government",
    "religion": "religious",
    "security": "security",
    "surveyor": "professional_services",
    "tax_advisor": "professional_services",
    "union": "labor",
}

# =============================================================================
# LANDUSE MAPPINGS
# Maps landuse types to park/open space categories
# =============================================================================

LANDUSE_TO_PARK_TYPE = {
    "park": "urban_park",
    "recreation_ground": "sports_field",
    "forest": "natural_forest",
    "grass": "grassland",
    "meadow": "meadow",
    "allotments": "community_garden",
    "cemetery": "cemetery",
    "orchard": "orchard",
    "vineyard": "vineyard",
    "farmland": "agricultural",
    "greenhouse_horticulture": "greenhouse",
    "plant_nursery": "nursery",
    "conservation": "conservation_area",
    "village_green": "village_green",
    "greenfield": "greenfield",
    "brownfield": "brownfield",
    "residential": "residential_area",
    "commercial": "commercial_area",
    "industrial": "industrial_area",
    "retail": "retail_area",
}

# =============================================================================
# HIGHWAY TYPE MAPPINGS
# Maps highway types to transportation categories
# =============================================================================

HIGHWAY_TO_TRANSPORT_TYPE = {
    "motorway": "highway",
    "trunk": "highway",
    "primary": "primary_road",
    "secondary": "secondary_road",
    "tertiary": "local_road",
    "unclassified": "local_road",
    "residential": "residential_street",
    "service": "service_road",
    "track": "track",
    "path": "footpath",
    "footway": "sidewalk",
    "cycleway": "bike_path",
    "bridleway": "bridle_path",
    "steps": "steps",
    "corridor": "indoor_corridor",
    "elevator": "elevator",
    "escalator": "escalator",
    "motorway_link": "highway_ramp",
    "trunk_link": "highway_ramp",
    "primary_link": "highway_ramp",
    "secondary_link": "highway_ramp",
    "tertiary_link": "highway_ramp",
    "living_street": "living_street",
    "pedestrian": "pedestrian_street",
    "road": "road",
}

# =============================================================================
# DEFAULT COORDINATE REFERENCE SYSTEM
# =============================================================================

DEFAULT_CRS = "EPSG:3857"  # Web Mercator projection

# =============================================================================
# FEATURE PRIORITY ORDER
# When multiple tags match, this determines which location type wins
# =============================================================================

LOCATION_TYPE_PRIORITY = ["residential", "retail", "workplace", "park", "walkway"]

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

VALID_LOCATION_TYPES = {"residential", "retail", "workplace", "park", "walkway", "other"}
VALID_GEOMETRY_TYPES = {"Point", "LineString", "Polygon", "MultiPolygon", "MultiLineString"}
VALID_AMENITY_TYPES = set(AMENITY_TO_LOCATION_TYPE.keys())
VALID_SHOP_TYPES = set(SHOP_TO_SUBCATEGORY.keys())
VALID_OFFICE_TYPES = set(OFFICE_TO_SUBCATEGORY.keys())
