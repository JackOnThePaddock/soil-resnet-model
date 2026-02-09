"""
Configuration for Australian Soil Data Extraction

Replace the placeholder values with your actual TERN/CSIRO API credentials.
DO NOT commit this file to version control with real credentials.
"""

# TERN Soil Data Federator API
TERN_API_BASE_URL = "https://esoil.io/TERNLandscapes/SoilDataFederatoR/SoilDataAPI"

# Replace with your registered credentials
# Demo credentials are limited to 5 records per query
TERN_API_USER = "Demo"  # Replace with your username
TERN_API_KEY = "Demo"   # Replace with your API key

# ANSIS Portal (fallback)
ANSIS_PORTAL_URL = "https://portal.ansis.net/"

# Database configuration
DATABASE_PATH = "soil_data.db"

# Query parameters
DEFAULT_UPPER_DEPTH = 0    # cm
DEFAULT_LOWER_DEPTH = 15   # cm
MIN_OBSERVATION_DATE = "2017-01-01"

# Property groups to query
PROPERTY_GROUPS = [
    "Soil pH",
    "Exchangeable Cations",
    "Organic Carbon",
    "CEC"
]

# Rayment & Lyons method codes (for reference)
METHOD_CODES = {
    "ph_cacl2": ["4A1"],           # pH in CaCl2
    "ph_water": ["4A2", "4B1"],    # pH in water
    "cec": ["15J1", "15E1"],       # Cation Exchange Capacity
    "organic_carbon": ["6A1", "6B1", "6B2"],  # Soil Organic Carbon
    "exchangeable_cations": ["15A1", "15C1", "15D1"]  # Ca, Mg, Na, K
}

# Unit conversion factors
# For converting mg/kg to cmol(+)/kg: divide by (atomic_weight * 10)
ATOMIC_WEIGHTS = {
    "Ca": 40.08,
    "Mg": 24.31,
    "Na": 22.99,
    "K": 39.10
}
