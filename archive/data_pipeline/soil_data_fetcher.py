"""
TERN Soil Data Federator API Client

Queries the TERN API to fetch Australian soil observation data.
"""

import requests
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO
import time

from config import (
    TERN_API_BASE_URL,
    TERN_API_USER,
    TERN_API_KEY,
    PROPERTY_GROUPS,
    DEFAULT_UPPER_DEPTH,
    DEFAULT_LOWER_DEPTH
)


class TERNAPIClient:
    """Client for the TERN Soil Data Federator API."""

    def __init__(self, username: str = TERN_API_USER, api_key: str = TERN_API_KEY):
        self.base_url = TERN_API_BASE_URL
        self.username = username
        self.api_key = api_key
        self.session = requests.Session()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: int = 60
    ) -> requests.Response:
        """Make a request to the API."""
        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}

        # Add authentication
        params['usr'] = self.username
        params['key'] = self.api_key

        response = self.session.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        return response

    def test_connection(self) -> bool:
        """Test if the API is available and returning real data."""
        try:
            response = self._make_request("DataSets", {"format": "json"})
            if response.status_code != 200:
                return False

            # Check if API returns deprecation notice instead of data
            data = response.json()
            if isinstance(data, list) and len(data) == 1:
                first_item = str(data[0])
                if "superceded" in first_item.lower() or "ansis" in first_item.lower():
                    print("WARNING: TERN API has been deprecated.")
                    print("The API now returns: " + first_item[:200])
                    return False

            return True
        except requests.RequestException as e:
            print(f"API connection failed: {e}")
            return False

    def get_datasets(self) -> List[Dict]:
        """Get list of available datasets."""
        response = self._make_request("DataSets", {"format": "json"})
        return response.json()

    def get_property_groups(self) -> List[str]:
        """Get list of available property groups."""
        response = self._make_request("PropertyGroups", {"format": "json"})
        return response.json()

    def get_properties(
        self,
        property_group: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Get list of available soil properties.

        Args:
            property_group: Filter by property group (e.g., "Soil pH")
            verbose: Return detailed property info

        Returns:
            List of property definitions
        """
        params = {
            "format": "json",
            "verbose": "T" if verbose else "F"
        }

        if property_group:
            params["PropertyGroup"] = property_group

        response = self._make_request("Properties", params)
        return response.json()

    def query_soil_data(
        self,
        property_group: Optional[str] = None,
        observed_property: Optional[str] = None,
        dataset: Optional[str] = None,
        bbox: Optional[str] = None,
        upper_depth: int = DEFAULT_UPPER_DEPTH,
        lower_depth: int = DEFAULT_LOWER_DEPTH,
        num_to_return: int = 100000,
        output_format: str = "json"
    ) -> pd.DataFrame:
        """
        Query soil observation data.

        Args:
            property_group: Property group to query (e.g., "Soil pH")
            observed_property: Specific property code (e.g., "4A1")
            dataset: Filter by dataset provider
            bbox: Bounding box "minx;maxx;miny;maxy"
            upper_depth: Filter by upper depth (cm)
            lower_depth: Filter by lower depth (cm)
            num_to_return: Maximum number of records
            output_format: Response format (json, csv, xml)

        Returns:
            DataFrame with soil observations
        """
        params = {
            "format": output_format,
            "numToReturn": num_to_return
        }

        if property_group:
            params["observedPropertyGroup"] = property_group

        if observed_property:
            params["observedProperty"] = observed_property

        if dataset:
            params["DataSet"] = dataset

        if bbox:
            params["bbox"] = bbox

        # Note: Depth filtering may need to be done client-side
        # depending on API support

        try:
            response = self._make_request("SoilData", params, timeout=120)

            if output_format == "json":
                data = response.json()
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict) and 'data' in data:
                    return pd.DataFrame(data['data'])
                else:
                    return pd.DataFrame([data]) if data else pd.DataFrame()

            elif output_format == "csv":
                return pd.read_csv(StringIO(response.text))

            else:
                raise ValueError(f"Unsupported format: {output_format}")

        except requests.RequestException as e:
            print(f"Error querying soil data: {e}")
            return pd.DataFrame()

    def query_all_property_groups(
        self,
        property_groups: List[str] = PROPERTY_GROUPS,
        **kwargs
    ) -> List[pd.DataFrame]:
        """
        Query data for multiple property groups.

        Args:
            property_groups: List of property groups to query
            **kwargs: Additional parameters for query_soil_data

        Returns:
            List of DataFrames, one per property group
        """
        results = []

        for group in property_groups:
            print(f"Querying property group: {group}")
            df = self.query_soil_data(property_group=group, **kwargs)
            print(f"  Retrieved {len(df)} records")
            results.append(df)

            # Rate limiting - be nice to the API
            time.sleep(1)

        return results


def test_tern_api() -> bool:
    """Test TERN API connectivity."""
    client = TERNAPIClient()
    return client.test_connection()


def get_property_codes() -> Dict[str, List[Dict]]:
    """Get all available property codes organized by group."""
    client = TERNAPIClient()

    if not client.test_connection():
        print("Warning: API may be unavailable (deprecated)")
        return {}

    property_codes = {}

    try:
        groups = client.get_property_groups()
        print(f"Found {len(groups)} property groups")

        for group in groups:
            properties = client.get_properties(property_group=group)
            property_codes[group] = properties
            print(f"  {group}: {len(properties)} properties")

    except Exception as e:
        print(f"Error getting property codes: {e}")

    return property_codes


def fetch_soil_data_for_training(
    property_groups: List[str] = PROPERTY_GROUPS,
    upper_depth: int = DEFAULT_UPPER_DEPTH,
    lower_depth: int = DEFAULT_LOWER_DEPTH
) -> List[pd.DataFrame]:
    """
    Fetch soil data suitable for ML training.

    Args:
        property_groups: List of property groups to fetch
        upper_depth: Minimum upper depth (cm)
        lower_depth: Maximum lower depth (cm)

    Returns:
        List of DataFrames with soil data
    """
    client = TERNAPIClient()

    if not client.test_connection():
        print("Error: TERN API is unavailable.")
        print("Consider using ANSIS portal as fallback: https://portal.ansis.net/")
        return []

    return client.query_all_property_groups(
        property_groups=property_groups,
        upper_depth=upper_depth,
        lower_depth=lower_depth
    )


if __name__ == "__main__":
    # Test API connectivity
    print("Testing TERN Soil Data Federator API...")
    print("-" * 50)

    client = TERNAPIClient()

    if client.test_connection():
        print("API is available!")

        # Get datasets
        print("\nAvailable datasets:")
        datasets = client.get_datasets()
        for ds in datasets[:5]:  # Show first 5
            print(f"  - {ds}")

        # Get property groups
        print("\nAvailable property groups:")
        groups = client.get_property_groups()
        for group in groups:
            print(f"  - {group}")

        # Get properties for pH
        print("\nProperties for 'Soil pH':")
        ph_props = client.get_properties(property_group="Soil pH")
        for prop in ph_props[:5]:
            print(f"  - {prop}")

    else:
        print("API is unavailable (may be deprecated)")
        print("Please use ANSIS portal: https://portal.ansis.net/")
