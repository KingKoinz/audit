from __future__ import annotations

import sqlite3
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

DB_PATH = Path("./data/research_journal.sqlite")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def init_research_db():
    """Initialize research journal database."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS research_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            feed_key TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            hypothesis TEXT NOT NULL,
            test_method TEXT NOT NULL,
            findings TEXT NOT NULL,
            p_value REAL,
            effect_size REAL,
            viable BOOLEAN,
            contradicts TEXT,
            ai_reasoning TEXT,
            data_window TEXT
        )
    """)
    
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_feed_timestamp
        ON research_log(feed_key, timestamp DESC)
    """)

    # Prediction tracking tables
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_key TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            draw_date TEXT NOT NULL,
            hot_numbers TEXT NOT NULL,
            cold_numbers TEXT NOT NULL,
            overdue_numbers TEXT NOT NULL,
            lookback_window INTEGER DEFAULT 30,
            created_at TEXT NOT NULL,
            validated BOOLEAN DEFAULT 0,
            UNIQUE(feed_key, draw_date)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            feed_key TEXT NOT NULL,
            draw_date TEXT NOT NULL,
            actual_numbers TEXT NOT NULL,
            hot_hits INTEGER DEFAULT 0,
            cold_hits INTEGER DEFAULT 0,
            overdue_hits INTEGER DEFAULT 0,
            total_numbers INTEGER DEFAULT 5,
            hot_hit_rate REAL,
            cold_hit_rate REAL,
            overdue_hit_rate REAL,
            validated_at TEXT NOT NULL,
            FOREIGN KEY (snapshot_id) REFERENCES prediction_snapshots(id)
        )
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_feed_date
        ON prediction_snapshots(feed_key, draw_date DESC)
    """)

    # Pursuit state tracking table
    c.execute("""
        CREATE TABLE IF NOT EXISTS pursuit_state (
            feed_key TEXT PRIMARY KEY,
            is_active BOOLEAN NOT NULL DEFAULT 0,
            target_hypothesis TEXT,
            target_test_method TEXT,
            target_parameters TEXT,
            discovery_level TEXT,
            pursuit_start_iteration INTEGER,
            pursuit_attempts INTEGER DEFAULT 0,
            last_p_value REAL,
            last_effect_size REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def log_research_iteration(
    feed_key: str,
    iteration: int,
    hypothesis: str,
    test_method: str,
    findings: Dict[str, Any],
    p_value: float,
    effect_size: float,
    viable: bool,
    contradicts: str,
    ai_reasoning: str,
    data_window: str
):
    """Store a research iteration."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO research_log
        (timestamp, feed_key, iteration, hypothesis, test_method, findings,
         p_value, effect_size, viable, contradicts, ai_reasoning, data_window)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        feed_key,
        iteration,
        hypothesis,
        test_method,
        json.dumps(convert_numpy_types(findings)),
        float(p_value) if p_value is not None else None,
        float(effect_size) if effect_size is not None else None,
        bool(viable),
        contradicts,
        ai_reasoning,
        data_window
    ))
    
    conn.commit()
    conn.close()

def get_recent_research(feed_key: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent research iterations for AI context."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    c.execute("""
        SELECT iteration, hypothesis, test_method, findings, p_value, effect_size, 
               viable, contradicts, ai_reasoning, data_window, timestamp
        FROM research_log
        WHERE feed_key = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (feed_key, limit))

    results = []
    for row in c.fetchall():
        findings = {}
        discovery = {}
        pursuit_mode = {}
        try:
            findings = json.loads(row[3]) if row[3] else {}
            discovery = findings.get('discovery', {})
            pursuit_mode = findings.get('pursuit_mode', {})
        except Exception:
            pass
        results.append({
            "iteration": row[0],
            "hypothesis": row[1],
            "test_method": row[2],
            "findings": findings,
            "discovery": discovery,
            "pursuit_mode": pursuit_mode,
            "p_value": row[4],
            "effect_size": row[5],
            "viable": bool(row[6]),
            "contradicts": row[7],
            "ai_reasoning": row[8],
            "data_window": row[9],
            "timestamp": row[10]
        })

    conn.close()
    return results

def get_iteration_count(feed_key: str) -> int:
    """Get current iteration count."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT MAX(iteration) FROM research_log WHERE feed_key = ?
    """, (feed_key,))

    result = c.fetchone()[0]
    conn.close()
    return result if result else 0

# ===== PURSUIT STATE MANAGEMENT =====

def get_pursuit_state(feed_key: str) -> Dict[str, Any]:
    """Get current pursuit state for a feed."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT is_active, target_hypothesis, target_test_method, target_parameters,
               discovery_level, pursuit_start_iteration, pursuit_attempts,
               last_p_value, last_effect_size
        FROM pursuit_state
        WHERE feed_key = ?
    """, (feed_key,))

    row = c.fetchone()
    conn.close()

    if not row:
        return {
            "is_active": False,
            "target_hypothesis": None,
            "target_test_method": None,
            "target_parameters": None,
            "discovery_level": None,
            "pursuit_start_iteration": None,
            "pursuit_attempts": 0,
            "last_p_value": None,
            "last_effect_size": None
        }

    return {
        "is_active": bool(row[0]),
        "target_hypothesis": row[1],
        "target_test_method": row[2],
        "target_parameters": json.loads(row[3]) if row[3] else None,
        "discovery_level": row[4],
        "pursuit_start_iteration": row[5],
        "pursuit_attempts": row[6],
        "last_p_value": row[7],
        "last_effect_size": row[8]
    }

def start_pursuit(
    feed_key: str,
    hypothesis: str,
    test_method: str,
    parameters: Dict[str, Any],
    discovery_level: str,
    current_iteration: int,
    p_value: float,
    effect_size: float
):
    """Enter pursuit mode to verify a CANDIDATE pattern."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    now = datetime.now().isoformat()

    c.execute("""
        INSERT OR REPLACE INTO pursuit_state
        (feed_key, is_active, target_hypothesis, target_test_method, target_parameters,
         discovery_level, pursuit_start_iteration, pursuit_attempts,
         last_p_value, last_effect_size, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        feed_key,
        True,
        hypothesis,
        test_method,
        json.dumps(parameters),
        discovery_level,
        current_iteration,
        1,  # First attempt
        p_value,
        effect_size,
        now,
        now
    ))

    conn.commit()
    conn.close()

def update_pursuit(
    feed_key: str,
    p_value: float,
    effect_size: float
):
    """Update pursuit state after a verification attempt."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        UPDATE pursuit_state
        SET pursuit_attempts = pursuit_attempts + 1,
            last_p_value = ?,
            last_effect_size = ?,
            updated_at = ?
        WHERE feed_key = ?
    """, (p_value, effect_size, datetime.now().isoformat(), feed_key))

    conn.commit()
    conn.close()

def end_pursuit(feed_key: str, reason: str = "completed"):
    """Exit pursuit mode."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        UPDATE pursuit_state
        SET is_active = 0,
            updated_at = ?
        WHERE feed_key = ?
    """, (datetime.now().isoformat(), feed_key))

    conn.commit()
    conn.close()


# ===== PREDICTION TRACKING =====

def classify_numbers_hot_cold_overdue(draws: List[Dict], max_num: int, lookback: int = 30) -> Dict[str, List[int]]:
    """
    Classify all numbers as hot, cold, or overdue based on recent history.

    - Hot: Appeared more than expected in last `lookback` draws
    - Cold: Appeared less than expected in last `lookback` draws
    - Overdue: Longest gap since last appearance
    """
    if len(draws) < lookback:
        lookback = len(draws)

    recent_draws = draws[:lookback]

    # Count appearances in recent draws
    from collections import Counter
    recent_numbers = [n for d in recent_draws for n in d["numbers"]]
    counts = Counter(recent_numbers)

    # Expected appearances: (lookback draws * 5 numbers per draw) / total numbers
    expected = (lookback * 5) / max_num

    hot = []
    cold = []
    neutral = []

    for num in range(1, max_num + 1):
        count = counts.get(num, 0)
        if count > expected * 1.5:  # 50% above expected
            hot.append(num)
        elif count < expected * 0.5:  # 50% below expected
            cold.append(num)
        else:
            neutral.append(num)

    # Find overdue numbers (longest gap since last appearance)
    last_seen = {n: float('inf') for n in range(1, max_num + 1)}
    for i, draw in enumerate(draws):
        for n in draw["numbers"]:
            if last_seen[n] == float('inf'):
                last_seen[n] = i

    # Overdue = top 15 numbers with longest gaps (not in hot)
    sorted_by_gap = sorted(last_seen.items(), key=lambda x: -x[1])
    overdue = [n for n, gap in sorted_by_gap[:15] if n not in hot]

    return {
        "hot": hot,
        "cold": cold,
        "overdue": overdue,
        "neutral": neutral
    }


def save_prediction_snapshot(
    feed_key: str,
    draw_date: str,
    hot_numbers: List[int],
    cold_numbers: List[int],
    overdue_numbers: List[int],
    lookback_window: int = 30
) -> int:
    """Save a prediction snapshot before a draw."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    now = datetime.now().isoformat()

    try:
        c.execute("""
            INSERT OR REPLACE INTO prediction_snapshots
            (feed_key, snapshot_date, draw_date, hot_numbers, cold_numbers,
             overdue_numbers, lookback_window, created_at, validated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            feed_key,
            now,
            draw_date,
            json.dumps(hot_numbers),
            json.dumps(cold_numbers),
            json.dumps(overdue_numbers),
            lookback_window,
            now
        ))
        conn.commit()
        snapshot_id = c.lastrowid
    finally:
        conn.close()

    return snapshot_id


def validate_prediction(feed_key: str, draw_date: str, actual_numbers: List[int]) -> Dict[str, Any]:
    """Validate a prediction against actual draw results."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Get the snapshot for this draw
    c.execute("""
        SELECT id, hot_numbers, cold_numbers, overdue_numbers
        FROM prediction_snapshots
        WHERE feed_key = ? AND draw_date = ? AND validated = 0
    """, (feed_key, draw_date))

    row = c.fetchone()
    if not row:
        conn.close()
        return {"status": "no_snapshot", "message": "No unvalidated snapshot for this draw"}

    snapshot_id = row[0]
    hot = json.loads(row[1])
    cold = json.loads(row[2])
    overdue = json.loads(row[3])

    # Calculate hits
    actual_set = set(actual_numbers)
    hot_hits = len(actual_set.intersection(hot))
    cold_hits = len(actual_set.intersection(cold))
    overdue_hits = len(actual_set.intersection(overdue))
    total = len(actual_numbers)

    # Calculate hit rates (what % of each category appeared)
    hot_hit_rate = hot_hits / len(hot) if hot else 0
    cold_hit_rate = cold_hits / len(cold) if cold else 0
    overdue_hit_rate = overdue_hits / len(overdue) if overdue else 0

    now = datetime.now().isoformat()

    # Save results
    c.execute("""
        INSERT INTO prediction_results
        (snapshot_id, feed_key, draw_date, actual_numbers, hot_hits, cold_hits,
         overdue_hits, total_numbers, hot_hit_rate, cold_hit_rate, overdue_hit_rate, validated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        snapshot_id, feed_key, draw_date, json.dumps(actual_numbers),
        hot_hits, cold_hits, overdue_hits, total,
        hot_hit_rate, cold_hit_rate, overdue_hit_rate, now
    ))

    # Mark snapshot as validated
    c.execute("""
        UPDATE prediction_snapshots SET validated = 1 WHERE id = ?
    """, (snapshot_id,))

    conn.commit()
    conn.close()

    return {
        "status": "validated",
        "hot_hits": hot_hits,
        "cold_hits": cold_hits,
        "overdue_hits": overdue_hits,
        "total": total,
        "hot_in_pool": len(hot),
        "cold_in_pool": len(cold),
        "overdue_in_pool": len(overdue)
    }


def get_prediction_stats(feed_key: str, limit: int = 50) -> Dict[str, Any]:
    """Get aggregate prediction statistics."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT
            COUNT(*) as total_predictions,
            SUM(hot_hits) as total_hot_hits,
            SUM(cold_hits) as total_cold_hits,
            SUM(overdue_hits) as total_overdue_hits,
            SUM(total_numbers) as total_numbers,
            AVG(hot_hit_rate) as avg_hot_rate,
            AVG(cold_hit_rate) as avg_cold_rate,
            AVG(overdue_hit_rate) as avg_overdue_rate
        FROM prediction_results
        WHERE feed_key = ?
        ORDER BY validated_at DESC
        LIMIT ?
    """, (feed_key, limit))

    row = c.fetchone()
    conn.close()

    if not row or row[0] == 0:
        return {
            "total_predictions": 0,
            "hot_accuracy": 0,
            "cold_accuracy": 0,
            "overdue_accuracy": 0,
            "baseline": 0.33,
            "message": "No validated predictions yet"
        }

    total_preds = row[0]
    # Calculate what % of winning numbers came from each category
    total_nums = row[4] or 1

    return {
        "total_predictions": total_preds,
        "hot_hits_total": row[1] or 0,
        "cold_hits_total": row[2] or 0,
        "overdue_hits_total": row[3] or 0,
        "hot_accuracy": (row[1] or 0) / total_nums * 100,
        "cold_accuracy": (row[2] or 0) / total_nums * 100,
        "overdue_accuracy": (row[3] or 0) / total_nums * 100,
        "avg_hot_hit_rate": row[5] or 0,
        "avg_cold_hit_rate": row[6] or 0,
        "avg_overdue_hit_rate": row[7] or 0,
        "baseline": 33.3  # Expected ~33% if random
    }


def get_recent_predictions(feed_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent prediction results for display."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT draw_date, actual_numbers, hot_hits, cold_hits, overdue_hits,
               total_numbers, validated_at
        FROM prediction_results
        WHERE feed_key = ?
        ORDER BY validated_at DESC
        LIMIT ?
    """, (feed_key, limit))

    results = []
    for row in c.fetchall():
        results.append({
            "draw_date": row[0],
            "actual_numbers": json.loads(row[1]),
            "hot_hits": row[2],
            "cold_hits": row[3],
            "overdue_hits": row[4],
            "total": row[5],
            "validated_at": row[6]
        })

    conn.close()
    return results
