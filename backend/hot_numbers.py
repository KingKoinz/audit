"""
Hot Numbers Generator
Generates "statistically elevated" numbers based on recent frequency analysis.
NOT predictions - purely historical pattern analysis.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from backend.db import get_all_draws
from backend.audit import freq_counts, RANGES
import random

def generate_hot_numbers(feed_key: str, window_days: int = 30) -> Dict[str, Any]:
    """
    Generate 'hot numbers' based on recent frequency analysis.
    These are NOT predictions - they're historical frequency observations.

    Uses the most recent N draws (where N = window_days * 2 for twice-weekly drawings).
    """
    all_draws = get_all_draws(feed_key)
    if not all_draws:
        return {"error": "No data available"}

    # Use last N draws instead of time-based window
    # Assume 2 draws per week on average, so window_days translates to (window_days / 7 * 2) draws
    num_draws = max(10, int(window_days / 7 * 2))
    recent_draws = all_draws[:num_draws]  # get_all_draws returns newest first

    if len(recent_draws) < 10:
        return {"error": "Insufficient recent data"}

    # Get frequency counts for main numbers
    r = RANGES[feed_key]
    all_nums = []
    for draw in recent_draws:
        all_nums.extend(draw['numbers'])

    counts = freq_counts(all_nums, r['main_min'], r['main_max'])

    # Create frequency map
    freq_map = {}
    for i, count in enumerate(counts):
        number = i + r['main_min']
        freq_map[number] = count

    # Sort by frequency (descending)
    sorted_nums = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)

    # Get top 10 "hot" numbers
    hot_numbers = [num for num, freq in sorted_nums[:10]]

    # Select 5 from top 7 (adds slight randomness to keep it interesting)
    selected = random.sample(hot_numbers[:7], 5)
    selected.sort()

    # Get bonus number frequencies
    bonus_nums = [d['bonus'] for d in recent_draws]
    bonus_counts = freq_counts(bonus_nums, r['bonus_min'], r['bonus_max'])
    bonus_freq_map = {i + r['bonus_min']: count for i, count in enumerate(bonus_counts)}
    sorted_bonus = sorted(bonus_freq_map.items(), key=lambda x: x[1], reverse=True)
    hot_bonus = sorted_bonus[0][0]

    # Calculate "heat score" (how much above average)
    avg_freq = sum(counts) / len(counts)
    heat_scores = [freq_map[n] / avg_freq for n in selected]
    avg_heat = sum(heat_scores) / len(heat_scores)

    # Get date range of analyzed draws
    draw_dates = [datetime.fromisoformat(d['draw_date']) for d in recent_draws]
    oldest_date = min(draw_dates).strftime("%Y-%m-%d")
    newest_date = max(draw_dates).strftime("%Y-%m-%d")

    return {
        "feed": feed_key,
        "generated_at": datetime.now().isoformat(),
        "draws_analyzed": len(recent_draws),
        "date_range": {
            "oldest": oldest_date,
            "newest": newest_date
        },
        "main_numbers": selected,
        "bonus_number": hot_bonus,
        "heat_score": round(avg_heat, 2),
        "frequencies": {str(n): freq_map[n] for n in selected},
        "disclaimer": "These numbers showed elevated frequency in recent draws. This is historical analysis, NOT a prediction. Every number has equal odds in each draw."
    }
