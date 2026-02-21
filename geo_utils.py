"""
geo_utils.py - Geolocation Medical Routing
===========================================
Given a user's GPS coordinates and the detected condition category,
finds nearby relevant specialists (dermatologists, orthopedics, etc.)
and returns structured results with distance, contact, and maps link.

Integration options:
OPTION A - Google Places API (recommended):
    Use the /place/nearbysearch/json endpoint with your API key.
    Set keyword to the specialist type and radius to 5000 metres.

OPTION B - OpenStreetMap Overpass API (free, no key required):
    Query nodes with amenity=clinic around the user coordinates.

OPTION C - Practo API / NHA ABDM registry (India-specific, official).

This file provides a realistic mock implementation with India-specific
hospital and specialist data for demo purposes.
"""

import math
import logging
import asyncio
from typing import List, Dict

log = logging.getLogger(__name__)

# Specialty mapping from condition category
CATEGORY_TO_SPECIALTY = {
    "dermatology":     {"specialist": "Dermatologist",      "search_keyword": "dermatologist skin clinic"},
    "orthopedics":     {"specialist": "Orthopedic Surgeon", "search_keyword": "orthopedic hospital"},
    "general_surgery": {"specialist": "General Surgeon",    "search_keyword": "surgical clinic"},
    "general":         {"specialist": "General Physician",  "search_keyword": "general physician clinic"},
}

# Mock specialist database (India-specific)
# Replace with live Google Places / Practo API data in production.
_MOCK_SPECIALISTS = [
    {
        "name": "Apollo Skin & Hair Clinic",
        "type": ["dermatology"],
        "address": "Jubilee Hills, Hyderabad, Telangana 500033",
        "phone": "+91-40-2360-7777",
        "rating": 4.7,
        "lat": 17.4315,
        "lon": 78.4072,
        "timings": "Mon-Sat: 9:00 AM - 8:00 PM",
        "insurance": ["CGHS", "Mediclaim", "Star Health"],
    },
    {
        "name": "KIMS Dermatology Department",
        "type": ["dermatology"],
        "address": "1-8-31/1, Minister Road, Secunderabad, Hyderabad 500003",
        "phone": "+91-40-4488-8000",
        "rating": 4.8,
        "lat": 17.4400,
        "lon": 78.4980,
        "timings": "24/7 OPD",
        "insurance": ["All major insurance accepted"],
    },
    {
        "name": "Dr. Reddy's Orthopedic Centre",
        "type": ["orthopedics"],
        "address": "Banjara Hills, Road No. 12, Hyderabad 500034",
        "phone": "+91-40-2335-1234",
        "rating": 4.6,
        "lat": 17.4239,
        "lon": 78.4378,
        "timings": "Mon-Sat: 10:00 AM - 7:00 PM",
        "insurance": ["CGHS", "ESI", "Mediclaim"],
    },
    {
        "name": "Yashoda Super Speciality Hospital",
        "type": ["dermatology", "orthopedics", "general_surgery", "general"],
        "address": "Raj Bhavan Road, Somajiguda, Hyderabad 500082",
        "phone": "+91-40-4567-4567",
        "rating": 4.9,
        "lat": 17.4254,
        "lon": 78.4564,
        "timings": "24/7",
        "insurance": ["All major insurance accepted"],
    },
    {
        "name": "Rainbow Children's Hospital",
        "type": ["general", "general_surgery"],
        "address": "Banjara Hills, Hyderabad 500034",
        "phone": "+91-40-4477-0000",
        "rating": 4.7,
        "lat": 17.4126,
        "lon": 78.4462,
        "timings": "24/7",
        "insurance": ["CGHS", "Arogya Sri", "Mediclaim"],
    },
    {
        "name": "Care Skin Institute",
        "type": ["dermatology"],
        "address": "Ameerpet, Hyderabad, Telangana 500016",
        "phone": "+91-40-2374-8888",
        "rating": 4.5,
        "lat": 17.4375,
        "lon": 78.4483,
        "timings": "Mon-Sat: 10:00 AM - 8:00 PM, Sun: 10:00 AM - 2:00 PM",
        "insurance": ["Mediclaim", "Star Health"],
    },
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in kilometres between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_maps_link(lat: float, lon: float, name: str) -> str:
    """Generate a Google Maps directions link."""
    encoded_name = name.replace(" ", "+")
    return "https://www.google.com/maps/dir/?api=1&destination=" + str(lat) + "," + str(lon)


async def find_nearby_specialists(
    lat: float,
    lon: float,
    condition_category: str,
    radius_km: float = 10.0,
    max_results: int = 5,
) -> List[Dict]:
    """
    Find nearby medical specialists relevant to the detected condition.

    Args:
        lat                : User latitude
        lon                : User longitude
        condition_category : Category from CNN inference e.g. dermatology
        radius_km          : Search radius in kilometres (default 10km)
        max_results        : Maximum number of results to return

    Returns:
        List of specialist dicts sorted by distance ascending.
    """
    await asyncio.sleep(0.1)

    specialty_info = CATEGORY_TO_SPECIALTY.get(
        condition_category,
        CATEGORY_TO_SPECIALTY["general"]
    )

    results = []
    for clinic in _MOCK_SPECIALISTS:
        if condition_category not in clinic["type"] and "general" not in clinic["type"]:
            if condition_category not in clinic["type"]:
                continue

        distance = _haversine_km(lat, lon, clinic["lat"], clinic["lon"])

        if distance > radius_km:
            continue

        results.append({
            "name":            clinic["name"],
            "specialist_type": specialty_info["specialist"],
            "address":         clinic["address"],
            "phone":           clinic["phone"],
            "rating":          clinic["rating"],
            "distance_km":     round(distance, 2),
            "timings":         clinic["timings"],
            "insurance":       clinic["insurance"],
            "maps_link":       _build_maps_link(clinic["lat"], clinic["lon"], clinic["name"]),
            "latitude":        clinic["lat"],
            "longitude":       clinic["lon"],
        })

    results.sort(key=lambda x: x["distance_km"])

    log.info("Found %d specialists within %skm for category=%s", len(results), radius_km, condition_category)

    final = []
    for r in results[:max_results]:
        r["disclaimer"] = (
            "Availability and specialisations should be confirmed directly "
            "with the facility before visiting. This data is for guidance only."
        )
        final.append(r)

    return final


async def get_emergency_contacts(state: str = "Telangana") -> Dict:
    """Return India emergency contacts relevant to the user state."""
    return {
        "national_emergency":     "112",
        "ambulance":              "108",
        "health_helpline":        "104",
        "poison_control":         "1800-116-117",
        "mental_health_helpline": "iCall: 9152987821",
        "state":                  state,
        "nhm_helpline":           "1800-180-1104",
    }
