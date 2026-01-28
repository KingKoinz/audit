"""
GAME ROTATION SYSTEM

Ensures AI research alternates between Powerball and Mega Millions
to maintain separate, non-mixed testing.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path("./data/research_journal.sqlite")

GAMES = ["powerball", "megamillions"]  # Rotate between both lotteries

def init_rotation_state():
    """Initialize game rotation tracking table."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS game_rotation (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_tested_game TEXT NOT NULL,
            last_rotation_time TEXT NOT NULL
        )
    """)

    # Initialize with megamillions if empty (so powerball goes first)
    c.execute("SELECT COUNT(*) FROM game_rotation")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO game_rotation (id, last_tested_game, last_rotation_time)
            VALUES (1, 'megamillions', ?)
        """, (datetime.now().isoformat(),))

    conn.commit()
    conn.close()

def get_next_game() -> str:
    """
    Get the next game to test (rotates between powerball and mega-millions).
    Returns the game that should be tested next.
    """
    init_rotation_state()

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("SELECT last_tested_game FROM game_rotation WHERE id = 1")
    row = c.fetchone()
    conn.close()

    if not row:
        return "powerball"

    last_game = row[0]

    # Rotate to next game
    current_idx = GAMES.index(last_game) if last_game in GAMES else 0
    next_idx = (current_idx + 1) % len(GAMES)
    next_game = GAMES[next_idx]

    return next_game

def record_game_tested(game: str):
    """Record that a game was just tested."""
    init_rotation_state()

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        UPDATE game_rotation
        SET last_tested_game = ?,
            last_rotation_time = ?
        WHERE id = 1
    """, (game, datetime.now().isoformat()))

    conn.commit()
    conn.close()

def get_current_game_state() -> dict:
    """Get current rotation state for display."""
    init_rotation_state()

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("SELECT last_tested_game, last_rotation_time FROM game_rotation WHERE id = 1")
    row = c.fetchone()
    conn.close()

    if not row:
        return {
            "last_tested": None,
            "next_game": "powerball",
            "last_rotation": None
        }

    last_game = row[0]
    next_game = get_next_game()

    return {
        "last_tested": last_game,
        "next_game": next_game,
        "last_rotation": row[1]
    }
