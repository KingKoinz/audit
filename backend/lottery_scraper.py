"""
Web scraper for lottery results from official sources.
Fallback when Socrata API fails.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
from bs4 import BeautifulSoup
import re


async def scrape_powerball_results(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Scrape recent Powerball results from powerball.com
    Returns list of draw records with keys: draw_date, winning_numbers, powerball
    """
    results = []

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Fetch the previous results page
            url = "https://www.powerball.com/previous-results"
            r = await client.get(url)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, 'html.parser')

            # Find all result cards/containers
            # The page uses divs with specific classes for each draw
            result_items = soup.find_all('div', {'class': re.compile(r'.*result.*', re.I)})

            for item in result_items[:limit]:
                try:
                    # Try to extract draw date from the card
                    date_elem = item.find('span', {'class': re.compile(r'.*date.*', re.I)})
                    if not date_elem:
                        date_elem = item.find(string=re.compile(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)'))

                    if not date_elem:
                        continue

                    date_str = date_elem.text.strip() if hasattr(date_elem, 'text') else str(date_elem)
                    # Parse date like "Wed, Jan 28, 2026"
                    draw_date = _parse_powerball_date(date_str)
                    if not draw_date:
                        continue

                    # Extract winning numbers
                    numbers = []
                    number_elems = item.find_all('span', {'class': re.compile(r'.*number.*', re.I)})
                    for num_elem in number_elems:
                        try:
                            num = int(num_elem.text.strip())
                            if 1 <= num <= 69:
                                numbers.append(num)
                            elif 1 <= num <= 26:  # Could be powerball (1-26 range)
                                pass  # Will extract separately
                        except ValueError:
                            continue

                    if len(numbers) >= 5:
                        # Extract Powerball (usually larger number or marked separately)
                        pb_elem = item.find('span', {'class': re.compile(r'.*powerball.*', re.I)})
                        powerball = None
                        if pb_elem:
                            try:
                                powerball = int(pb_elem.text.strip())
                            except ValueError:
                                pass

                        if not powerball and len(numbers) > 5:
                            powerball = numbers[5]
                            numbers = numbers[:5]

                        if powerball:
                            results.append({
                                'draw_date': draw_date,
                                'winning_numbers': ' '.join(map(str, numbers[:5])),
                                'powerball': powerball
                            })

                except Exception as e:
                    # Skip malformed entries
                    continue

        return results

    except Exception as e:
        print(f"[SCRAPER] Powerball scrape failed: {e}")
        return []


def _parse_powerball_date(date_str: str) -> Optional[str]:
    """
    Parse date strings like "Wed, Jan 28, 2026" to ISO format "2026-01-28"
    """
    try:
        # Remove day of week if present
        date_str = re.sub(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s*', '', date_str)

        # Parse common formats
        for fmt in [
            '%b %d, %Y',      # Jan 28, 2026
            '%B %d, %Y',      # January 28, 2026
            '%m/%d/%Y',       # 01/28/2026
            '%Y-%m-%d',       # 2026-01-28
        ]:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.date().isoformat()
            except ValueError:
                continue

        return None
    except Exception:
        return None


def parse_lottery_csv(csv_content: str, game: str) -> List[Dict[str, Any]]:
    """
    Parse manually uploaded CSV with lottery results.
    Expected format:
    draw_date,n1,n2,n3,n4,n5,bonus_ball
    2026-01-28,5,12,23,35,42,17

    game: 'powerball' or 'megamillions'
    """
    results = []
    lines = csv_content.strip().split('\n')

    if not lines:
        return results

    # Skip header if present
    start = 1 if lines[0].lower().startswith('draw') else 0

    for line in lines[start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 7:
            continue

        try:
            draw_date = parts[0]
            # Validate date format
            datetime.fromisoformat(draw_date)

            numbers = [int(parts[i]) for i in range(1, 6)]
            bonus = int(parts[6])

            results.append({
                'draw_date': draw_date,
                'winning_numbers': ' '.join(map(str, numbers)),
                game.split('_')[0] if '_' in game else game: bonus  # 'powerball' or 'megamillions'
            })
        except (ValueError, IndexError):
            continue

    return results
