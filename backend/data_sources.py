from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import httpx

# NY Open Data (Socrata) datasets for draw history:
# Powerball: d6yy-54nr
# Mega Millions: 5xaw-6ayf
# These are discoverable via data.gov and data.ny.gov.
# See: https://data.ny.gov/.../d6yy-54nr and .../5xaw-6ayf  (do not hard-link in UI text)
# NOTE: We'll use Socrata API endpoint: https://data.ny.gov/resource/{id}.json

SOCRATA_DOMAIN = "https://data.ny.gov/resource"

@dataclass(frozen=True)
class LotteryFeed:
    key: str
    name: str
    resource_id: str
    date_field: str
    number_fields: List[str]
    bonus_field: str

POWERBALL = LotteryFeed(
    key="powerball",
    name="Powerball",
    resource_id="d6yy-54nr",
    date_field="draw_date",
    number_fields=["winning_numbers"],
    bonus_field="mega_ball",  # This is the Powerball field
)

MEGAMILLIONS = LotteryFeed(
    key="megamillions",
    name="Mega Millions",
    resource_id="5xaw-6ayf",
    date_field="draw_date",
    number_fields=["winning_numbers"],
    bonus_field="mega_ball",  # This is the Mega Ball field
)

FEEDS = [POWERBALL, MEGAMILLIONS]


def _headers() -> Dict[str, str]:
    token = os.getenv("SOCRATA_APP_TOKEN", "").strip()
    headers: Dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["X-App-Token"] = token
    return headers


async def fetch_socrata_draws(feed: LotteryFeed, limit: int = 50000) -> List[Dict[str, Any]]:
    """
    Fetch full draw history as JSON rows from Socrata.
    We'll request all fields to handle varying schemas.
    """
    url = f"{SOCRATA_DOMAIN}/{feed.resource_id}.json"
    params = {
        "$order": f"{feed.date_field} ASC",
        "$limit": str(limit),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=_headers())
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Socrata response")
        return data