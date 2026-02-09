"""TERN Soil Data Federator API Client for Australian soil observations."""

import time
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests


TERN_API_BASE_URL = "https://esoil.io/TERNLandscapes/SoilDataFederatoR/SoilDataAPI"
ANSIS_PORTAL_URL = "https://portal.ansis.net/"


class TERNAPIClient:
    """Client for the TERN Soil Data Federator API."""

    def __init__(self, username: str = "Demo", api_key: str = "Demo",
                 base_url: str = TERN_API_BASE_URL):
        self.base_url = base_url
        self.username = username
        self.api_key = api_key
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                      timeout: int = 60) -> requests.Response:
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["usr"] = self.username
        params["key"] = self.api_key
        response = self.session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response

    def test_connection(self) -> bool:
        """Test if the API is available and returning real data."""
        try:
            response = self._make_request("DataSets", {"format": "json"})
            if response.status_code != 200:
                return False
            data = response.json()
            if isinstance(data, list) and len(data) == 1:
                first_item = str(data[0])
                if "superceded" in first_item.lower() or "ansis" in first_item.lower():
                    print("WARNING: TERN API has been deprecated.")
                    return False
            return True
        except requests.RequestException as e:
            print(f"API connection failed: {e}")
            return False

    def get_datasets(self) -> List[Dict]:
        return self._make_request("DataSets", {"format": "json"}).json()

    def get_property_groups(self) -> List[str]:
        return self._make_request("PropertyGroups", {"format": "json"}).json()

    def query_soil_data(
        self, property_group: Optional[str] = None,
        observed_property: Optional[str] = None,
        bbox: Optional[str] = None,
        upper_depth: int = 0, lower_depth: int = 15,
        num_to_return: int = 100000, output_format: str = "json",
    ) -> pd.DataFrame:
        """Query soil observation data."""
        params = {"format": output_format, "numToReturn": num_to_return}
        if property_group:
            params["observedPropertyGroup"] = property_group
        if observed_property:
            params["observedProperty"] = observed_property
        if bbox:
            params["bbox"] = bbox

        try:
            response = self._make_request("SoilData", params, timeout=120)
            if output_format == "json":
                data = response.json()
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict) and "data" in data:
                    return pd.DataFrame(data["data"])
                return pd.DataFrame([data]) if data else pd.DataFrame()
            elif output_format == "csv":
                return pd.read_csv(StringIO(response.text))
            else:
                raise ValueError(f"Unsupported format: {output_format}")
        except requests.RequestException as e:
            print(f"Error querying soil data: {e}")
            return pd.DataFrame()

    def query_all_property_groups(
        self, property_groups: List[str], **kwargs,
    ) -> List[pd.DataFrame]:
        """Query data for multiple property groups with rate limiting."""
        results = []
        for group in property_groups:
            print(f"Querying: {group}")
            df = self.query_soil_data(property_group=group, **kwargs)
            print(f"  Retrieved {len(df)} records")
            results.append(df)
            time.sleep(1)
        return results


def test_tern_api() -> bool:
    return TERNAPIClient().test_connection()


def fetch_soil_data_for_training(
    property_groups: Optional[List[str]] = None,
    upper_depth: int = 0, lower_depth: int = 15,
) -> List[pd.DataFrame]:
    """Fetch soil data suitable for ML training."""
    if property_groups is None:
        property_groups = ["Soil pH", "Exchangeable Cations", "Organic Carbon", "CEC"]
    client = TERNAPIClient()
    if not client.test_connection():
        print("TERN API unavailable. Use ANSIS portal: https://portal.ansis.net/")
        return []
    return client.query_all_property_groups(property_groups, upper_depth=upper_depth, lower_depth=lower_depth)
