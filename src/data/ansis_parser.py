"""ANSIS JSON-LD parser for nested soil data format."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_wkt_point(wkt_string: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse WKT POINT geometry to (longitude, latitude)."""
    if not wkt_string:
        return None, None
    match = re.search(r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", wkt_string)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def parse_depth_value(depth_obj: Dict) -> Optional[float]:
    """Parse depth value and convert to centimeters."""
    if not depth_obj or "result" not in depth_obj:
        return None
    result = depth_obj["result"]
    value = result.get("value")
    unit = result.get("unit", "")
    if value is None:
        return None
    if "M" in unit.upper() and "CM" not in unit.upper():
        return float(value) * 100
    return float(value)


def extract_property_value(
    prop_data: Any, preferred_methods: Optional[List[str]] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Extract property value, preferring specific analytical methods."""
    if prop_data is None:
        return None, None
    if isinstance(prop_data, list):
        if not prop_data:
            return None, None
        if preferred_methods:
            for item in prop_data:
                method = item.get("usedProcedure", "").replace("scm:", "").upper()
                if method in [m.upper() for m in preferred_methods]:
                    return item.get("result", {}).get("value"), method
        result = prop_data[0].get("result", {})
        method = prop_data[0].get("usedProcedure", "").replace("scm:", "")
        return result.get("value"), method
    elif isinstance(prop_data, dict):
        result = prop_data.get("result", {})
        method = prop_data.get("usedProcedure", "").replace("scm:", "")
        return result.get("value"), method
    return None, None


def parse_ansis_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a single ANSIS JSON file and extract soil observations."""
    observations = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "data" not in data:
        return observations

    for site in data["data"]:
        if site.get("type") != "ansis:SoilSite":
            continue
        site_id = site.get("id", "")
        lon, lat = parse_wkt_point(site.get("geometry", {}).get("result", ""))
        if lon is None or lat is None:
            continue

        scoped_ids = site.get("scopedIdentifier", [])
        external_id = scoped_ids[0].get("value") if scoped_ids else None
        authority = scoped_ids[0].get("authority") if scoped_ids else None

        for visit in site.get("siteVisit", []):
            visit_date = visit.get("startedAtTime") or visit.get("endedAtTime")
            if visit_date:
                try:
                    visit_date = datetime.fromisoformat(visit_date.replace("Z", "+00:00"))
                except Exception:
                    visit_date = None

            for profile in visit.get("soilProfile", []):
                for layer in profile.get("soilLayer", []):
                    upper_depth = parse_depth_value(layer.get("depthUpper"))
                    lower_depth = parse_depth_value(layer.get("depthLower"))
                    if upper_depth is None or lower_depth is None:
                        continue

                    obs = {
                        "site_id": site_id, "external_id": external_id, "authority": authority,
                        "latitude": lat, "longitude": lon,
                        "observation_date": visit_date.date() if visit_date else None,
                        "upper_depth_cm": upper_depth, "lower_depth_cm": lower_depth,
                        "dataset_source": "ANSIS",
                    }

                    ph_val, ph_method = extract_property_value(layer.get("ph"), ["4A1", "4B1"])
                    if ph_val is not None:
                        obs["ph_cacl2"] = ph_val
                        obs["method_ph"] = ph_method

                    for prop, key, col in [
                        ("exchangeableCalcium", "ca_cmol_kg", ["15A1", "15C1"]),
                        ("exchangeableMagnesium", "mg_cmol_kg", ["15A1", "15C1"]),
                        ("exchangeableSodium", "na_cmol_kg", ["15A1", "15C1"]),
                        ("exchangeablePotassium", "k_cmol_kg", ["15A1", "15C1"]),
                    ]:
                        val, _ = extract_property_value(layer.get(prop), col)
                        if val is not None:
                            obs[key] = val

                    toc_val, toc_method = extract_property_value(layer.get("totalOrganicCarbon"), ["6B2", "6A1"])
                    if toc_val is not None:
                        obs["soc_percent"] = toc_val

                    cec_val, _ = extract_property_value(
                        layer.get("cationExchangeCapacity") or layer.get("effectiveCEC"), ["15J1", "15E1"]
                    )
                    if cec_val is not None:
                        obs["cec_cmol_kg"] = cec_val

                    if any(obs.get(k) for k in ["ph_cacl2", "ca_cmol_kg", "soc_percent", "cec_cmol_kg"]):
                        observations.append(obs)

    return observations


def parse_ansis_directory(
    directory: str, upper_depth_max: float = 15, min_date: str = "2017-01-01",
) -> pd.DataFrame:
    """Parse all ANSIS JSON files in a directory."""
    directory = Path(directory)
    all_obs = []
    files = list(directory.iterdir())
    print(f"Found {len(files)} files to process")

    for i, fp in enumerate(files):
        if fp.is_file():
            try:
                all_obs.extend(parse_ansis_file(str(fp)))
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i+1}/{len(files)}, {len(all_obs)} observations")
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Warning: {fp.name}: {e}")

    if not all_obs:
        return pd.DataFrame()

    df = pd.DataFrame(all_obs)
    if "upper_depth_cm" in df.columns:
        df = df[df["upper_depth_cm"] <= upper_depth_max]
    if "observation_date" in df.columns and min_date:
        df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
        df = df[df["observation_date"] >= pd.to_datetime(min_date).date()]

    # Calculate ESP
    if "na_cmol_kg" in df.columns and "cec_cmol_kg" in df.columns:
        df["esp_percent"] = (df["na_cmol_kg"] / df["cec_cmol_kg"]) * 100
        df.loc[df["cec_cmol_kg"] == 0, "esp_percent"] = np.nan

    return df
