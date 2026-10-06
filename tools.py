"""Ground Truth's tools, and the JSON that describes them to the model.

The idea comes from district-level informal-economy measurement at the IMF:
when a place has no recent business survey, open map data can stand in as a
proxy. These tools turn OpenStreetMap and World Bank data into a quick,
honest read of a local economy:

    locate_place            -> where is it? (OpenStreetMap Nominatim)
    scan_economic_footprint -> what economic activity is mapped there? (Overpass API)
    estimate_formality      -> how formal or informal does it look, and why? (original)
    check_map_coverage      -> can we trust the map in this area? (original)
    get_country_context     -> what do national statistics say? (World Bank API)

All data sources are free and need no API key.
"""

import json
import math
import time
from datetime import date

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

USER_AGENT = "GroundTruth/1.0 (+https://github.com/anamikam-772/gemini-web-tool-calling)"
HEADERS = {"User-Agent": USER_AGENT}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
# Public Overpass servers are shared and sometimes busy, so we try mirrors too.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
OVERPASS_TIMEOUT_S = 30  # how long to wait for each server (all are asked at the same time)
OVERPASS_BUDGET_S = 35  # total time to wait for the first good answer
FAILURE_MEMORY_S = 120  # remember a failed scan briefly, so the next tools fail fast instead of waiting again
# The ohsome API (HeiGIT, Heidelberg) is a research service built for counting OpenStreetMap features in an
# area. It runs on separate infrastructure from Overpass, so it is raced alongside the Overpass servers.
OHSOME_COUNT_URL = "https://api.ohsome.org/v1/elements/count"
WORLD_BANK_URL = "https://api.worldbank.org/v2/country/{code}/indicator/{indicator}"

MIN_RADIUS_M, MAX_RADIUS_M, DEFAULT_RADIUS_M = 200, 5000, 1500

# --- What we count on the map ---
# (key, Overpass filter, plain-English label, group)
# "formal" and "informal" signals feed the formality score; "coverage" signals
# tell us how thoroughly the area has been mapped at all.
SIGNALS = [
    ("banks", 'nwr["amenity"="bank"]', "banks", "formal"),
    ("atms", 'nwr["amenity"="atm"]', "ATMs", "formal"),
    ("offices", 'nwr["office"]', "offices", "formal"),
    ("big_retail", 'nwr["shop"~"^(supermarket|mall|department_store)$"]', "supermarkets and malls", "formal"),
    ("branded_shops", 'nwr["shop"]["brand"]', "chain-brand shops", "formal"),
    ("marketplaces", 'nwr["amenity"="marketplace"]', "open marketplaces", "informal"),
    ("kiosks", 'nwr["shop"="kiosk"]', "kiosks", "informal"),
    ("general_stores", 'nwr["shop"~"^(general|variety_store)$"]', "small general stores", "informal"),
    ("artisans", 'nwr["craft"]', "artisan workshops (tailors, carpenters, etc.)", "informal"),
    ("money_transfer", 'nwr["amenity"="money_transfer"]', "money-transfer / mobile-money agents", "informal"),
    ("clinics", 'nwr["amenity"~"^(clinic|hospital|doctors|pharmacy)$"]', "clinics, hospitals and pharmacies", "services"),
    ("schools", 'nwr["amenity"~"^(school|college|university)$"]', "schools and colleges", "services"),
    ("shops", 'nwr["shop"]', "shops of any kind", "coverage"),
    ("named_shops", 'nwr["shop"]["name"]', "shops with a name", "coverage"),
    ("amenities", 'nwr["amenity"]', "amenities of any kind", "coverage"),
    ("buildings", 'way["building"]', "building outlines", "coverage"),
    ("roads", 'way["highway"]', "road and path segments", "coverage"),
]

# The same signals written in the ohsome filter language.
OHSOME_FILTERS = {
    "banks": "amenity=bank",
    "atms": "amenity=atm",
    "offices": "office=*",
    "big_retail": "shop in (supermarket, mall, department_store)",
    "branded_shops": "shop=* and brand=*",
    "marketplaces": "amenity=marketplace",
    "kiosks": "shop=kiosk",
    "general_stores": "shop in (general, variety_store)",
    "artisans": "craft=*",
    "money_transfer": "amenity=money_transfer",
    "clinics": "amenity in (clinic, hospital, doctors, pharmacy)",
    "schools": "amenity in (school, college, university)",
    "shops": "shop=*",
    "named_shops": "shop=* and name=*",
    "amenities": "amenity=*",
    "buildings": "building=* and type:way",
    "roads": "highway=* and type:way",
}

# Weights reflect how much economic activity one mapped feature usually stands for.
# One open marketplace can hold hundreds of vendors; one bank branch anchors a formal
# financial district. They are deliberately simple so every number can be explained.
FORMALITY_WEIGHTS = {
    "banks": 3, "atms": 1, "offices": 1, "big_retail": 3, "branded_shops": 1,
    "marketplaces": 15, "kiosks": 1, "general_stores": 1, "artisans": 1, "money_transfer": 2,
}
MIN_EVIDENCE = 8  # below this much weighted evidence, a score would be noise
UNDERCOUNT_GAP = 20  # map score this many points below the national rate triggers an undercount check

_scan_cache: dict[tuple, tuple[float, dict]] = {}
_scan_failures: dict[tuple, tuple[float, str]] = {}
_last_server: dict[tuple, str] = {}  # which server answered each scan, reported in the results
_country_cache: dict[str, tuple[float, dict]] = {}
CACHE_SECONDS = 3600


def _error(message: str, **extra) -> str:
    """Errors are returned as data so the model can read them and decide what to do."""
    return json.dumps({"error": message, **extra})


def _check_coords(lat, lon) -> str | None:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return "lat and lon must be numbers. Call locate_place first to get them."
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return f"lat={lat}, lon={lon} is not a valid coordinate. Call locate_place to get correct ones."
    return None


def _clamp_radius(radius_m) -> int:
    try:
        radius = int(float(radius_m))
    except (TypeError, ValueError):
        radius = DEFAULT_RADIUS_M
    return max(MIN_RADIUS_M, min(MAX_RADIUS_M, radius))


def _map_data_note(from_cache: bool, lat: float = 0, lon: float = 0, radius: int = 0) -> str:
    server = _last_server.get((round(lat, 4), round(lon, 4), radius), "OpenStreetMap")
    if from_cache:
        return f"reused the map counts already fetched for this area from {server} (cached), no new request"
    return f"fetched live from {server}"


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --- Tool 1: locate_place ---


def locate_place(place: str) -> str:
    """Turn a place name into coordinates, a country code and a sensible scan radius."""
    place = (place or "").strip()
    if not place:
        return _error("place is empty. Pass a place name such as 'Kumasi Central Market, Ghana'.")

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 1},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        hits = resp.json()
    except (requests.RequestException, ValueError):
        hits = None  # fall back to the Open-Meteo geocoder below

    if hits:
        hit = hits[0]
        lat, lon = float(hit["lat"]), float(hit["lon"])
        south, north, west, east = (float(x) for x in hit["boundingbox"])
        half_diagonal = _haversine_m(south, west, north, east) / 2
        radius = int(round(max(500, min(3000, half_diagonal)) / 100) * 100)
        address = hit.get("address", {})
        return json.dumps({
            "name": hit.get("name") or place,
            "full_name": hit.get("display_name"),
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "country": address.get("country"),
            "country_code": (address.get("country_code") or "").upper() or None,
            "kind": f'{hit.get("category", "")}/{hit.get("type", "")}'.strip("/"),
            "suggested_radius_m": radius,
            "source": "OpenStreetMap Nominatim",
        })

    # Fallback: Open-Meteo knows cities and towns, not individual markets or streets.
    try:
        resp = requests.get(OPEN_METEO_GEOCODE_URL, params={"name": place, "count": 1}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except (requests.RequestException, ValueError) as e:
        return _error(f"Both geocoding services failed ({e}). Ask the user to try again shortly.")

    if not results:
        return _error(
            f"No place called '{place}' was found. Try adding the city and country "
            "(e.g. 'Adum, Kumasi, Ghana') or a nearby landmark."
        )
    hit = results[0]
    return json.dumps({
        "name": hit.get("name"),
        "full_name": ", ".join(x for x in [hit.get("name"), hit.get("admin1"), hit.get("country")] if x),
        "lat": round(hit["latitude"], 5),
        "lon": round(hit["longitude"], 5),
        "country": hit.get("country"),
        "country_code": (hit.get("country_code") or "").upper() or None,
        "kind": "populated place",
        "suggested_radius_m": DEFAULT_RADIUS_M,
        "source": "Open-Meteo geocoder (fallback)",
    })


# --- Tool 2: scan_economic_footprint ---


def _square(lat: float, lon: float, radius: int) -> tuple[float, float, float, float]:
    """South, west, north, east edges of a square centred on the point, `radius` metres from centre to each edge."""
    dlat = radius / 111_320
    dlon = radius / (111_320 * max(math.cos(math.radians(lat)), 0.01))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _square_area_km2(radius: int) -> float:
    return (2 * radius / 1000) ** 2


def _ask_overpass(url: str, query: str) -> dict:
    """Send the query to one Overpass server. Returns the counts or raises RuntimeError with a short reason."""
    try:
        resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=OVERPASS_TIMEOUT_S)
    except requests.RequestException as e:
        raise RuntimeError(type(e).__name__)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    try:
        elements = resp.json().get("elements", [])
    except ValueError:
        raise RuntimeError("unreadable response")
    if len(elements) != len(SIGNALS):
        raise RuntimeError("incomplete response")
    return {name: int(el.get("tags", {}).get("total", 0)) for (name, _, _, _), el in zip(SIGNALS, elements)}


def _ohsome_one(name: str, bbox: str) -> tuple[str, int]:
    resp = requests.get(
        OHSOME_COUNT_URL, params={"bboxes": bbox, "filter": OHSOME_FILTERS[name]}, headers=HEADERS, timeout=OVERPASS_TIMEOUT_S
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ohsome HTTP {resp.status_code}")
    try:
        return name, int(round(resp.json()["result"][-1]["value"]))
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("ohsome unreadable response")


def _ask_ohsome(south: float, west: float, north: float, east: float) -> dict:
    """Count every signal with the ohsome API, one small request per signal, run in parallel."""
    bbox = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"  # ohsome wants lon,lat order
    with ThreadPoolExecutor(max_workers=6) as pool:
        try:
            return dict(pool.map(lambda n: _ohsome_one(n, bbox), OHSOME_FILTERS))
        except requests.RequestException as e:
            raise RuntimeError(f"ohsome {type(e).__name__}")


def _overpass_counts(lat: float, lon: float, radius: int) -> tuple[dict, bool]:
    """Count every signal inside a square around a point, in one Overpass request.

    A bounding-box search uses the map's spatial index directly, which is much faster than a circular
    "around" search, so busy public servers answer in time more often. The same query goes to every
    server at once and the first good answer wins.

    Returns (counts, from_cache). Raises RuntimeError with advice the model can act on.
    """
    key = (round(lat, 4), round(lon, 4), radius)
    cached = _scan_cache.get(key)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1], True
    failed = _scan_failures.get(key)
    if failed and time.time() - failed[0] < FAILURE_MEMORY_S:
        raise RuntimeError(failed[1])

    south, west, north, east = _square(lat, lon, radius)
    query = (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_S}][bbox:{south:.6f},{west:.6f},{north:.6f},{east:.6f}];"
        + "".join(f"{flt};out count;" for _, flt, _, _ in SIGNALS)
    )

    problems = []
    pool = ThreadPoolExecutor(max_workers=len(OVERPASS_URLS) + 1)
    futures = {pool.submit(_ask_overpass, url, query): url.split("/")[2] for url in OVERPASS_URLS}
    futures[pool.submit(_ask_ohsome, south, west, north, east)] = "api.ohsome.org"
    try:
        for future in as_completed(futures, timeout=OVERPASS_BUDGET_S):
            try:
                counts = future.result()
            except RuntimeError as e:
                problems.append(str(e))
                continue
            except Exception as e:  # one source breaking in an unexpected way must not stop the others
                problems.append(f"{futures[future]} {type(e).__name__}")
                continue
            _scan_cache[key] = (time.time(), counts)
            _last_server[key] = futures[future]
            return counts, False
    except FuturesTimeout:
        problems.append("too slow")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    message = (
        f"None of the OpenStreetMap data services answered in time ({', '.join(problems) or 'no response'}). "
        f"They are shared and sometimes busy. Do not retry right away: tell the user to try again in a minute "
        f"or two, or to ask about a smaller area (radius_m {max(MIN_RADIUS_M, radius // 2)})."
    )
    _scan_failures[key] = (time.time(), message)
    raise RuntimeError(message)


def scan_economic_footprint(lat: float, lon: float, radius_m: int = DEFAULT_RADIUS_M) -> str:
    """Count mapped economic signals (markets, kiosks, banks, offices, clinics...) around a point."""
    if problem := _check_coords(lat, lon):
        return _error(problem)
    lat, lon, radius = float(lat), float(lon), _clamp_radius(radius_m)
    try:
        counts, from_cache = _overpass_counts(lat, lon, radius)
    except RuntimeError as e:
        return _error(str(e))

    area_km2 = _square_area_km2(radius)
    groups: dict[str, dict] = {"formal": {}, "informal": {}, "services": {}}
    for name, _, label, group in SIGNALS:
        if group in groups:
            groups[group][name] = {"label": label, "count": counts[name], "per_km2": round(counts[name] / area_km2, 1)}

    return json.dumps({
        "lat": lat,
        "lon": lon,
        "radius_m": radius,
        "area_km2": round(area_km2, 2),
        "formal_signals": groups["formal"],
        "informal_signals": groups["informal"],
        "public_services": groups["services"],
        "all_shops": counts["shops"],
        "all_amenities": counts["amenities"],
        "note": "Counts are what volunteers have mapped in OpenStreetMap, a proxy for activity, not a census.",
        "source": "OpenStreetMap data via Overpass or ohsome",
        "map_data": _map_data_note(from_cache, lat, lon, radius),
    })


# --- Tool 3 (original): estimate_formality ---


def _national_vulnerable_employment(country_code: str) -> dict | None:
    data = _country_indicators(country_code)
    if not data or "error" in data:
        return None
    item = data["indicators"].get("vulnerable_employment_pct")
    if not item or item.get("value") is None:
        return None
    return item


def estimate_formality(lat: float, lon: float, radius_m: int = DEFAULT_RADIUS_M, country_code: str | None = None) -> str:
    """Score how informal the local economy looks (0 = fully formal, 100 = fully informal) and explain why."""
    if problem := _check_coords(lat, lon):
        return _error(problem)
    lat, lon, radius = float(lat), float(lon), _clamp_radius(radius_m)
    try:
        counts, from_cache = _overpass_counts(lat, lon, radius)
    except RuntimeError as e:
        return _error(str(e))

    labels = {name: label for name, _, label, _ in SIGNALS}
    groups = {name: group for name, _, _, group in SIGNALS}
    evidence = {name: counts[name] * w for name, w in FORMALITY_WEIGHTS.items()}
    formal = sum(v for k, v in evidence.items() if groups[k] == "formal")
    informal = sum(v for k, v in evidence.items() if groups[k] == "informal")
    total = formal + informal

    if total < MIN_EVIDENCE:
        return _error(
            f"Only {total} weighted signals are mapped within {radius} m, too few for a meaningful score. "
            "Call check_map_coverage to see whether the area is simply unmapped, or retry with a larger "
            f"radius_m (up to {MAX_RADIUS_M}).",
            weighted_evidence=total,
        )

    local_score = round(100 * informal / total, 1)

    # Explain the score: each signal's share of all evidence and which way it pushes.
    drivers = sorted(
        (
            {
                "signal": labels[k],
                "count": counts[k],
                "push": "toward informal" if groups[k] == "informal" else "toward formal",
                "share_of_evidence_pct": round(100 * v / total, 1),
            }
            for k, v in evidence.items() if v > 0
        ),
        key=lambda d: d["share_of_evidence_pct"],
        reverse=True,
    )[:5]

    result = {
        "lat": lat,
        "lon": lon,
        "radius_m": radius,
        "local_informality_score": local_score,
        "local_band": _band(local_score),
        "source": "OpenStreetMap data via Overpass or ohsome (map counts) + World Bank API (national rate)",
        "map_data": _map_data_note(from_cache, lat, lon, radius),
        "weighted_evidence": total,
        "top_drivers": drivers,
        "method": (
            "Weighted count of informal-economy signals (open markets x15, mobile-money agents x2, kiosks, "
            "general stores, artisan workshops) versus formal signals (banks x3, supermarkets/malls x3, ATMs, "
            "offices, chain brands), as a share of all evidence."
        ),
    }

    # Small-area-estimation style shrinkage: blend the local map signal with the national
    # vulnerable-employment rate. The more map evidence, the more weight the local signal gets.
    if country_code:
        national = _national_vulnerable_employment(country_code)
        if national:
            w_local = round(total / (total + 30), 2)
            blended = round(w_local * local_score + (1 - w_local) * national["value"], 1)
            result.update({
                "national_benchmark": {
                    "vulnerable_employment_pct": national["value"],
                    "year": national["year"],
                    "source": "World Bank / ILO",
                },
                "blended_score": blended,
                "blended_band": _band(blended),
                "local_weight": w_local,
            })
            # Catch the classic mapping gap: markets and banks get mapped, but the kiosks, vendors and
            # mobile-money agents around them rarely do. If the map looks far more formal than the
            # country, and small informal businesses are almost absent, the score is probably too low.
            small_informal = sum(counts[k] for k in ("kiosks", "general_stores", "artisans", "money_transfer"))
            gap = national["value"] - local_score
            if gap >= UNDERCOUNT_GAP and small_informal <= max(3, 0.05 * counts["shops"]):
                result["undercount_warning"] = (
                    f"The map score ({local_score}) is {round(gap)} points below the national vulnerable-employment "
                    f"rate ({national['value']}%), and only {small_informal} small informal businesses (kiosks, "
                    "general stores, workshops, mobile-money agents) are mapped here. Informal activity is "
                    "probably under-mapped, so the real informality is likely higher than this score."
                )
        else:
            result["national_benchmark"] = f"No World Bank vulnerable-employment figure found for '{country_code}'."

    return json.dumps(result)


def _band(score: float) -> str:
    if score < 25:
        return "mostly formal"
    if score < 50:
        return "mixed, leaning formal"
    if score < 75:
        return "mixed, leaning informal"
    return "mostly informal"


# --- Tool 4 (original): check_map_coverage ---


def check_map_coverage(lat: float, lon: float, radius_m: int = DEFAULT_RADIUS_M) -> str:
    """Judge how completely the area is mapped, so we know how far to trust the other tools."""
    if problem := _check_coords(lat, lon):
        return _error(problem)
    lat, lon, radius = float(lat), float(lon), _clamp_radius(radius_m)
    try:
        c, from_cache = _overpass_counts(lat, lon, radius)
    except RuntimeError as e:
        return _error(str(e))

    area = _square_area_km2(radius)
    building_density = c["buildings"] / area
    road_density = c["roads"] / area
    poi_density = (c["shops"] + c["amenities"]) / area
    pois_per_100_buildings = 100 * (c["shops"] + c["amenities"]) / c["buildings"] if c["buildings"] else 0.0
    named_share = c["named_shops"] / c["shops"] if c["shops"] else None

    points, findings = 0, []

    if building_density >= 800:
        points += 2
    elif building_density >= 150:
        points += 1
        findings.append("Building outlines are only partly traced.")
    else:
        findings.append("Very few building outlines are mapped (or the area is genuinely empty).")

    if road_density >= 60:
        points += 2
    elif road_density >= 15:
        points += 1
    else:
        findings.append("The road network is sparsely mapped.")

    if poi_density >= 60:
        points += 2
    elif poi_density >= 10:
        points += 1
        findings.append("Shops and services are only partly mapped.")
    else:
        findings.append("Almost no shops or services are mapped.")

    if building_density >= 150 and pois_per_100_buildings < 1:
        findings.append(
            "Buildings are mapped but businesses are not: typical of satellite tracing with no ground survey. "
            "Economic counts here will undercount real activity, especially informal activity."
        )
    elif pois_per_100_buildings >= 3:
        points += 1

    if named_share is not None and c["shops"] >= 10:
        if named_share >= 0.6:
            points += 1
        elif named_share < 0.3:
            findings.append("Most mapped shops have no name, a sign of quick, low-detail mapping.")

    # points range 0..8
    if points >= 6:
        grade, confidence = "well mapped", "high"
    elif points >= 3:
        grade, confidence = "partially mapped", "medium"
    else:
        grade, confidence = "data desert", "low"
    # Economic estimates need mapped businesses, not just buildings and roads.
    if poi_density < 10 and confidence != "low":
        grade, confidence = "buildings mapped, businesses missing", "low"

    advice = {
        "high": "Map-based estimates here are reasonably reliable.",
        "medium": "Treat the formality score as indicative; real activity is likely higher than mapped.",
        "low": "Do not rely on map-based estimates here; lean on national statistics (get_country_context) instead.",
    }[confidence]

    return json.dumps({
        "lat": lat,
        "lon": lon,
        "radius_m": radius,
        "coverage_grade": grade,
        "confidence": confidence,
        "coverage_points": f"{points}/8",
        "metrics": {
            "buildings_per_km2": round(building_density, 1),
            "road_segments_per_km2": round(road_density, 1),
            "shops_and_amenities_per_km2": round(poi_density, 1),
            "shops_and_amenities_per_100_buildings": round(pois_per_100_buildings, 2),
            "share_of_shops_named": round(named_share, 2) if named_share is not None else None,
        },
        "findings": findings or ["No major gaps detected."],
        "advice": advice,
        "source": "OpenStreetMap data via Overpass or ohsome",
        "map_data": _map_data_note(from_cache, lat, lon, radius),
    })


# --- Tool 5: get_country_context ---

WB_INDICATORS = {
    "gdp_per_capita_usd": ("NY.GDP.PCAP.CD", "GDP per capita (current US$)"),
    "vulnerable_employment_pct": ("SL.EMP.VULN.ZS", "Vulnerable employment (% of total employment)"),
    "self_employed_pct": ("SL.EMP.SELF.ZS", "Self-employed (% of total employment)"),
    "urban_population_pct": ("SP.URB.TOTL.IN.ZS", "Urban population (% of total)"),
}


def _country_indicators(country_code: str) -> dict:
    code = (country_code or "").strip().upper()
    if not (code.isalpha() and len(code) in (2, 3)):
        return {"error": f"'{country_code}' is not a country code. Use the 2-letter code from locate_place, e.g. 'GH'."}

    cached = _country_cache.get(code)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]

    out, country_name = {}, None
    for key, (indicator, label) in WB_INDICATORS.items():
        try:
            resp = requests.get(
                WORLD_BANK_URL.format(code=code, indicator=indicator),
                params={"format": "json", "mrnev": 1},
                headers=HEADERS,
                timeout=10,
            )
            payload = resp.json()
        except (requests.RequestException, ValueError) as e:
            return {"error": f"World Bank API failed ({type(e).__name__}). Retry once, then answer without national context."}

        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            if isinstance(payload, list) and payload and "message" in payload[0]:
                return {"error": f"World Bank does not recognise country code '{code}'. Use the 2-letter ISO code from locate_place."}
            out[key] = {"label": label, "value": None, "year": None}
            continue
        row = payload[1][0]
        country_name = country_name or (row.get("country") or {}).get("value")
        value = row.get("value")
        out[key] = {
            "label": label,
            "value": round(value, 1) if isinstance(value, (int, float)) else None,
            "year": int(row["date"]) if str(row.get("date", "")).isdigit() else None,
        }

    years = [v["year"] for v in out.values() if v["year"]]
    newest = max(years) if years else None
    data = {
        "country": country_name or code,
        "country_code": code,
        "indicators": out,
        "latest_year": newest,
        "years_since_latest_data": (date.today().year - newest) if newest else None,
        "source": "World Bank World Development Indicators",
    }
    _country_cache[code] = (time.time(), data)
    return data


def get_country_context(country_code: str) -> str:
    """National benchmarks (GDP per capita, vulnerable and self-employment, urbanization) from the World Bank."""
    data = dict(_country_indicators(country_code))
    if "error" not in data and data["years_since_latest_data"] is not None and data["years_since_latest_data"] > 3:
        data["freshness_warning"] = (
            f"The newest figure is {data['years_since_latest_data']} years old; national data may not reflect today."
        )
    return json.dumps(data)


# --- What the model sees ---

_COORD_PARAMS = {
    "lat": {"type": "number", "description": "Latitude in decimal degrees, from locate_place."},
    "lon": {"type": "number", "description": "Longitude in decimal degrees, from locate_place."},
    "radius_m": {
        "type": "integer",
        "description": (
            f"Size of the area to scan, in meters from the centre to each edge of a square ({MIN_RADIUS_M}-{MAX_RADIUS_M}). "
            "Use suggested_radius_m from locate_place, and the same value across tools for one place."
        ),
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "locate_place",
            "description": (
                "Find a place anywhere in the world (city, district, market, street, landmark) and return its "
                "coordinates, country, 2-letter country_code and a suggested_radius_m for scanning. "
                "Always call this first for a place the conversation has not located yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {
                        "type": "string",
                        "description": "Place name, ideally with city and country, e.g. 'Kejetia Market, Kumasi, Ghana'.",
                    },
                },
                "required": ["place"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_economic_footprint",
            "description": (
                "Count the economic activity mapped in OpenStreetMap around a point: formal signals (banks, ATMs, "
                "offices, supermarkets, chain brands), informal signals (open markets, kiosks, general stores, "
                "artisan workshops, mobile-money agents) and public services (clinics, schools), with densities per km2."
            ),
            "parameters": {"type": "object", "properties": _COORD_PARAMS, "required": ["lat", "lon"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_formality",
            "description": (
                "Score how informal a local economy looks, from 0 (fully formal) to 100 (fully informal), using "
                "weighted map signals, and list the top drivers behind the score. Pass country_code to also get "
                "a blended score that shrinks the local estimate toward the national vulnerable-employment rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_COORD_PARAMS,
                    "country_code": {
                        "type": "string",
                        "description": "Optional 2-letter ISO country code from locate_place, e.g. 'GH'.",
                    },
                },
                "required": ["lat", "lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_map_coverage",
            "description": (
                "Judge how completely an area is mapped (buildings, roads, shops, named businesses) and return a "
                "coverage grade (well mapped / partially mapped / data desert) with a confidence level. Call this "
                "before stating conclusions about a place, because poorly mapped areas undercount activity."
            ),
            "parameters": {"type": "object", "properties": _COORD_PARAMS, "required": ["lat", "lon"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_country_context",
            "description": (
                "Get national benchmarks from the World Bank for a country: GDP per capita, vulnerable employment %, "
                "self-employment %, urban population %, each with its year, plus a warning if the data is stale. "
                "Call it in every investigation, once per country, so local results can be read against the national picture."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country_code": {
                        "type": "string",
                        "description": "2-letter (or 3-letter) ISO country code, e.g. 'GH' or 'NGA'.",
                    },
                },
                "required": ["country_code"],
            },
        },
    },
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {
    "locate_place": locate_place,
    "scan_economic_footprint": scan_economic_footprint,
    "estimate_formality": estimate_formality,
    "check_map_coverage": check_map_coverage,
    "get_country_context": get_country_context,
}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return _error(f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}")
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return _error(f"Bad arguments for {name}: {e}")
    except Exception as e:  # last line of defence: the model gets a message, the user gets an answer
        return _error(f"{name} failed unexpectedly ({type(e).__name__}: {e}). Try again or try another place.")
