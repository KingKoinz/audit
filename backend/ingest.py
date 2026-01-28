from __future__ import annotations

from typing import Any, Dict, List, Tuple
from datetime import datetime
from data_sources import FEEDS, fetch_socrata_draws
from parse import parse_winning_numbers_field
from db import init_db, upsert_draws

def normalize_date(s: str) -> str:
    # Socrata often returns ISO strings. We'll store ISO date-only where possible.
    # Example: '2010-02-03T00:00:00.000'
    try:
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        return dt.date().isoformat()
    except Exception:
        return s[:10]

async def ingest_all() -> Dict[str, Any]:
    init_db()
    summary = {"feeds": []}
    for feed in FEEDS:
        rows = await fetch_socrata_draws(feed)
        to_upsert: List[Tuple[str, str, int, int, int, int, int, int, str]] = []
        for r in rows:
            draw_date = normalize_date(r.get(feed.date_field, ""))
            wn = r.get("winning_numbers", "")
            if not wn:
                continue
            main, bonus_from_wn = parse_winning_numbers_field(wn)
            
            # If bonus not in winning_numbers, try multiple possible field names
            bonus = bonus_from_wn
            if bonus is None:
                # Try common field names for bonus ball
                for field_name in ['mega_ball', 'powerball', 'bonus_ball', feed.bonus_field]:
                    bonus_field_value = r.get(field_name)
                    if bonus_field_value:
                        try:
                            bonus = int(bonus_field_value)
                            break
                        except (ValueError, TypeError):
                            continue
            
            if bonus is None:
                continue
                
            mult = ""
            for mult_field in ['multiplier', 'power_play', 'megaplier']:
                mult_val = r.get(mult_field)
                if mult_val:
                    mult = str(mult_val)
                    break
                    
            to_upsert.append((feed.key, draw_date, main[0], main[1], main[2], main[3], main[4], bonus, mult))

        inserted = upsert_draws(to_upsert)
        summary["feeds"].append({"feed": feed.key, "rows_seen": len(rows), "rows_upserted": inserted})

        # Auto-validate any pending predictions for newly ingested draws
        if inserted > 0:
            try:
                from backend.research_journal import validate_prediction, init_research_db
                init_research_db()
                for row in to_upsert[-inserted:]:  # Only check newly inserted
                    feed_key, draw_date, n1, n2, n3, n4, n5, bonus, mult = row
                    actual_numbers = [n1, n2, n3, n4, n5]
                    result = validate_prediction(feed_key, draw_date, actual_numbers)
                    if result.get("status") == "validated":
                        print(f"[AUTO-VALIDATE] Validated prediction for {feed_key} {draw_date}: hot={result['hot_hits']}, cold={result['cold_hits']}, overdue={result['overdue_hits']}")
            except Exception as e:
                print(f"[AUTO-VALIDATE] Error validating predictions: {e}")

    return summary