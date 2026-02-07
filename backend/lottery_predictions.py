"""
Synthesize CANDIDATE patterns into actual lottery number predictions.
"""

from __future__ import annotations

import sqlite3
import json
import random
import re
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from collections import Counter

DB_PATH = Path("./data/research_journal.sqlite")


def get_top_candidate_patterns(feed_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get top CANDIDATE patterns from research log."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT hypothesis, test_method, p_value, effect_size, viable, ai_reasoning
        FROM research_log
        WHERE feed_key = ?
        AND viable = 1
        ORDER BY p_value ASC, effect_size DESC
        LIMIT ?
    """, (feed_key, limit))

    patterns = []
    for row in c.fetchall():
        patterns.append({
            "hypothesis": row[0],
            "test_method": row[1],
            "p_value": row[2],
            "effect_size": row[3],
            "viable": row[4],
            "reasoning": row[5]
        })

    conn.close()
    return patterns


def synthesize_lottery_prediction(
    feed_key: str,
    patterns: List[Dict],
    max_num: int,
    bonus_max: int
) -> Dict[str, Any]:
    """
    Synthesize CANDIDATE patterns into actual lottery number predictions.
    Combines insights from top viable patterns to select 5 main numbers + 1 bonus.
    """
    if not patterns:
        # Fallback: random prediction
        numbers = sorted(random.sample(range(1, max_num + 1), 5))
        bonus = random.randint(1, bonus_max)
        return {
            "numbers": numbers,
            "bonus": bonus,
            "reasoning": "Random selection (no patterns available)"
        }

    try:
        # Extract number hints from pattern hypotheses
        number_scores = Counter()

        for pattern in patterns:
            hypothesis = pattern.get("hypothesis", "").lower()

            # Extract mentioned numbers from hypothesis
            numbers_mentioned = re.findall(r'\b([1-9]\d?)\b', hypothesis)

            for num_str in numbers_mentioned:
                num = int(num_str)
                if 1 <= num <= max_num:
                    # Weight by effect size and p-value
                    weight = pattern.get("effect_size", 0.1) * max(0.1, 1.0 - pattern.get("p_value", 1.0))
                    number_scores[num] += weight

        # Get top 15 candidates (for diversity), then randomly sample 5
        if number_scores:
            top_candidates = sorted(
                [num for num, _ in number_scores.most_common(15)],
                key=lambda x: -number_scores[x]
            )
            # Randomly sample 5 from top 15 (maintains quality while adding diversity)
            if len(top_candidates) >= 5:
                top_numbers = sorted(random.sample(top_candidates, 5))
            else:
                top_numbers = sorted(top_candidates)
            # Ensure we have exactly 5
            while len(top_numbers) < 5:
                candidate = random.randint(1, max_num)
                if candidate not in top_numbers:
                    top_numbers.append(candidate)
            numbers = sorted(top_numbers)
        else:
            numbers = sorted(random.sample(range(1, max_num + 1), 5))

        # Select bonus using pattern insights
        bonus_candidates = []
        for pattern in patterns:
            hypothesis = pattern.get("hypothesis", "").lower()
            # Look for bonus/powerball mentions
            if "bonus" in hypothesis or "powerball" in hypothesis or "mega ball" in hypothesis:
                bonus_candidates.append(pattern.get("effect_size", 0.5))

        if bonus_candidates:
            # Randomize bonus around middle of range (40-60% of max) for diversity
            bonus_mid = int(bonus_max * 0.5)
            bonus_variance = max(1, bonus_mid // 3)
            bonus = max(1, min(bonus_max, bonus_mid + random.randint(-bonus_variance, bonus_variance)))
        else:
            bonus = random.randint(1, bonus_max)

        # Build reasoning
        pattern_names = [p.get("hypothesis", "")[:40] for p in patterns[:3]]
        reasoning = f"Based on {len(patterns)} viable patterns: {', '.join(pattern_names)}..."

        return {
            "numbers": numbers,
            "bonus": bonus,
            "reasoning": reasoning,
            "pattern_count": len(patterns)
        }

    except Exception as e:
        # Fallback
        numbers = sorted(random.sample(range(1, max_num + 1), 5))
        bonus = random.randint(1, bonus_max)
        return {
            "numbers": numbers,
            "bonus": bonus,
            "reasoning": f"Fallback prediction (error: {str(e)[:30]})",
            "pattern_count": 0
        }


def save_lottery_prediction(
    feed_key: str,
    draw_date: str,
    numbers: List[int],
    bonus: int,
    reasoning: str = ""
) -> int:
    """Save synthesized lottery prediction."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    now = datetime.now().isoformat()

    try:
        c.execute("""
            INSERT OR REPLACE INTO ai_predictions
            (feed_key, draw_date, hot_numbers, cold_numbers, overdue_numbers,
             hot_reasoning, cold_reasoning, overdue_reasoning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feed_key,
            draw_date,
            json.dumps(numbers),
            json.dumps([bonus]),
            json.dumps([]),
            reasoning,
            f"Bonus: {bonus}",
            "Synthesized from CANDIDATE patterns",
            now
        ))
        conn.commit()
        pred_id = c.lastrowid
    finally:
        conn.close()

    return pred_id


def get_latest_lottery_prediction(feed_key: str) -> Dict[str, Any]:
    """Get latest synthesized lottery prediction."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT hot_numbers, cold_numbers, overdue_numbers,
               hot_reasoning, cold_reasoning, draw_date
        FROM ai_predictions
        WHERE feed_key = ?
        ORDER BY draw_date DESC
        LIMIT 1
    """, (feed_key,))

    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "numbers": json.loads(row[0]),
        "bonus": json.loads(row[1])[0] if json.loads(row[1]) else None,
        "reasoning": row[3],
        "draw_date": row[5]
    }
