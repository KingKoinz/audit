from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict

def get_next_draw_time(feed_key: str) -> Dict[str, any]:
    """
    Calculate next expected lottery draw based on schedule.
    Powerball: Wed/Sat 10:59 PM ET
    Mega Millions: Tue/Fri 11:00 PM ET
    """
    now = datetime.now()
    current_hour = now.hour

    if feed_key == "powerball":
        # Wednesday = 2, Saturday = 5
        draw_days = [2, 5]
        draw_time = "10:59 PM ET"
        draw_hour = 23  # 11 PM cutoff (draws at 10:59 PM)
    else:  # megamillions
        # Tuesday = 1, Friday = 4
        draw_days = [1, 4]
        draw_time = "11:00 PM ET"
        draw_hour = 23  # 11 PM cutoff

    # Find next draw day
    current_weekday = now.weekday()
    days_ahead = None

    for draw_day in draw_days:
        if draw_day == current_weekday:
            # Today is a draw day - check if draw hasn't happened yet
            if current_hour < draw_hour:
                days_ahead = 0  # Draw is TODAY
                break
            # Draw already happened today, continue to next draw day
        elif draw_day > current_weekday:
            days_ahead = draw_day - current_weekday
            break

    if days_ahead is None:
        # Next draw is next week (wrap around)
        days_ahead = (7 - current_weekday) + draw_days[0]

    next_draw = now + timedelta(days=days_ahead)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    next_draw_day = day_names[next_draw.weekday()]

    # Calculate more accurate hours away
    if days_ahead == 0:
        hours_away = max(0, draw_hour - current_hour)
        display_day = "TODAY"
    else:
        hours_away = (days_ahead * 24) - current_hour + draw_hour
        display_day = next_draw_day

    return {
        "next_date": next_draw.strftime("%Y-%m-%d"),
        "next_day": display_day,
        "draw_time": draw_time,
        "days_away": days_ahead,
        "hours_away": hours_away
    }
