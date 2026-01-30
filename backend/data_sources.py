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

# Cached fallback data - stores manual/scraped results when Socrata fails
_FALLBACK_CACHE = {}

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
    Falls back to cached/manual data if Socrata fails.
    We'll request all fields to handle varying schemas.
    """
    url = f"{SOCRATA_DOMAIN}/{feed.resource_id}.json"
    params = {
        "$order": f"{feed.date_field} ASC",
        "$limit": str(limit),
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params=params, headers=_headers())
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise RuntimeError("Unexpected Socrata response")

            # Cache the successful data as fallback
            _FALLBACK_CACHE[feed.key] = data
            return data

    except Exception as e:
        print(f"[DATA_SOURCES] Socrata fetch failed for {feed.key}: {e}")

        # Try fallback: web scraping for Powerball
        if feed.key == "powerball":
            try:
                from backend.lottery_scraper import scrape_powerball_results
                print(f"[DATA_SOURCES] Attempting Powerball scrape as fallback...")
                scraped = await scrape_powerball_results(limit=limit)
                if scraped:
                    _FALLBACK_CACHE[feed.key] = scraped
                    print(f"[DATA_SOURCES] Powerball scrape succeeded: {len(scraped)} draws")
                    return scraped
            except Exception as scrape_err:
                print(f"[DATA_SOURCES] Powerball scrape failed: {scrape_err}")

        # Fall back to cached data if available
        if feed.key in _FALLBACK_CACHE:
            print(f"[DATA_SOURCES] Using cached data for {feed.key}")
            return _FALLBACK_CACHE[feed.key]

        # Last resort: return empty (ingest will skip)
        print(f"[DATA_SOURCES] No fallback data available for {feed.key}")
        return []


def add_manual_draws(feed_key: str, draws: List[Dict[str, Any]]) -> int:
    """
    Add manually uploaded draw data to fallback cache.
    Returns number of new draws added.
    """
    if feed_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[feed_key] = []

    existing_dates = {d.get('draw_date') for d in _FALLBACK_CACHE[feed_key]}
    added = 0

    for draw in draws:
        if draw.get('draw_date') not in existing_dates:
            _FALLBACK_CACHE[feed_key].append(draw)
            added += 1

    return added