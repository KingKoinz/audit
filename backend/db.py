from __future__ import annotations

import os
import sqlite3
from typing import Iterable, List, Tuple, Optional, Dict, Any


def get_db_path() -> str:
    return os.getenv("DB_PATH", "./data/entropy.sqlite")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    conn = connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS draws (
                feed_key TEXT NOT NULL,
                draw_date TEXT NOT NULL,   -- ISO date or datetime
                n1 INTEGER NOT NULL,
                n2 INTEGER NOT NULL,
                n3 INTEGER NOT NULL,
                n4 INTEGER NOT NULL,
                n5 INTEGER NOT NULL,
                bonus INTEGER NOT NULL,    -- Powerball or Megaball
                multiplier TEXT,
                PRIMARY KEY (feed_key, draw_date)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_draws_feed_date
            ON draws(feed_key, draw_date);
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_draws(rows: Iterable[Tuple[str, str, int, int, int, int, int, int, str]]) -> int:
    conn = connect()
    inserted = 0
    try:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO draws
            (feed_key, draw_date, n1, n2, n3, n4, n5, bonus, multiplier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(rows),
        )
        inserted = cur.rowcount
        conn.commit()
        return inserted
    finally:
        conn.close()


def get_recent_draws(feed_key: str, limit: int) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT draw_date, n1, n2, n3, n4, n5, bonus, multiplier
            FROM draws
            WHERE feed_key = ?
            ORDER BY draw_date DESC
            LIMIT ?
            """,
            (feed_key, limit),
        )
        out = []
        for row in cur.fetchall():
            out.append(
                {
                    "draw_date": row[0],
                    "numbers": [row[1], row[2], row[3], row[4], row[5]],
                    "bonus": row[6],
                    "multiplier": row[7],
                }
            )
        return out
    finally:
        conn.close()


def get_all_draws(feed_key: str) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT draw_date, n1, n2, n3, n4, n5, bonus, multiplier
            FROM draws
            WHERE feed_key = ?
            ORDER BY draw_date ASC
            """,
            (feed_key,),
        )
        out = []
        for row in cur.fetchall():
            out.append(
                {
                    "draw_date": row[0],
                    "numbers": [row[1], row[2], row[3], row[4], row[5]],
                    "bonus": row[6],
                    "multiplier": row[7],
                }
            )
        return out
    finally:
        conn.close()


def get_latest_draw_date(feed_key: str) -> str:
    """Get the most recent draw date for staleness detection."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT draw_date FROM draws WHERE feed_key = ? ORDER BY draw_date DESC LIMIT 1",
            (feed_key,)
        )
        row = cur.fetchone()
        return row[0] if row else "Unknown"
    finally:
        conn.close()


def has_any_draws() -> bool:
    """Check if the database has any draw data."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM draws LIMIT 1")
        row = cur.fetchone()
        return row[0] > 0 if row else False
    finally:
        conn.close()