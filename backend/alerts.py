"""
ENTROPY Alert System
Handles notifications for significant discoveries (CANDIDATE+)
"""

import os
import json
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configuration - set these via environment variables
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")  # Where to send alerts

# Database for findings log
ALERTS_DB = Path("./data/alerts.sqlite")

def init_alerts_db():
    """Initialize alerts database."""
    ALERTS_DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(ALERTS_DB))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            feed_key TEXT NOT NULL,
            discovery_level TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            test_method TEXT NOT NULL,
            p_value REAL NOT NULL,
            effect_size REAL NOT NULL,
            persistence_count INTEGER DEFAULT 1,
            details TEXT,
            notified BOOLEAN DEFAULT 0,
            acknowledged BOOLEAN DEFAULT 0,
            notes TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT UNIQUE NOT NULL,
            p256dh TEXT NOT NULL,
            auth TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_findings_level
        ON findings(discovery_level, timestamp DESC)
    """)

    conn.commit()
    conn.close()

# Initialize on import
init_alerts_db()


def log_finding(
    feed_key: str,
    discovery_level: str,
    hypothesis: str,
    test_method: str,
    p_value: float,
    effect_size: float,
    persistence_count: int = 1,
    details: Optional[Dict] = None
) -> int:
    """
    Log a significant finding to the database.
    Returns the finding ID.
    """
    conn = sqlite3.connect(str(ALERTS_DB))
    c = conn.cursor()

    c.execute("""
        INSERT INTO findings
        (timestamp, feed_key, discovery_level, hypothesis, test_method,
         p_value, effect_size, persistence_count, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        feed_key,
        discovery_level,
        hypothesis,
        test_method,
        p_value,
        effect_size,
        persistence_count,
        json.dumps(details) if details else None
    ))

    finding_id = c.lastrowid
    conn.commit()
    conn.close()

    return finding_id


def get_findings(
    level_filter: Optional[str] = None,
    feed_key: Optional[str] = None,
    limit: int = 100,
    unacknowledged_only: bool = False
) -> List[Dict[str, Any]]:
    """Get findings from the log."""
    conn = sqlite3.connect(str(ALERTS_DB))
    c = conn.cursor()

    query = "SELECT * FROM findings WHERE 1=1"
    params = []

    if level_filter:
        query += " AND discovery_level = ?"
        params.append(level_filter)

    if feed_key:
        query += " AND feed_key = ?"
        params.append(feed_key)

    if unacknowledged_only:
        query += " AND acknowledged = 0"

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    c.execute(query, params)

    columns = ['id', 'timestamp', 'feed_key', 'discovery_level', 'hypothesis',
               'test_method', 'p_value', 'effect_size', 'persistence_count',
               'details', 'notified', 'acknowledged', 'notes']

    results = []
    for row in c.fetchall():
        finding = dict(zip(columns, row))
        if finding['details']:
            finding['details'] = json.loads(finding['details'])
        results.append(finding)

    conn.close()
    return results


def acknowledge_finding(finding_id: int, notes: str = "") -> bool:
    """Mark a finding as acknowledged."""
    conn = sqlite3.connect(str(ALERTS_DB))
    c = conn.cursor()

    c.execute("""
        UPDATE findings
        SET acknowledged = 1, notes = ?
        WHERE id = ?
    """, (notes, finding_id))

    success = c.rowcount > 0
    conn.commit()
    conn.close()
    return success


def send_email_alert(finding: Dict[str, Any]) -> bool:
    """Send email alert for a significant finding."""
    if not SMTP_USER or not ALERT_EMAIL:
        print("[ALERT] Email not configured - skipping notification")
        return False

    try:
        level = finding['discovery_level']
        emoji = {"CANDIDATE": "🔶", "VERIFIED": "🚨", "LEGENDARY": "👑"}.get(level, "📊")

        subject = f"{emoji} ENTROPY Alert: {level} - {finding['feed_key'].upper()}"

        body = f"""
ENTROPY LOTTERY ANOMALY DETECTION
=================================

{emoji} {level} DISCOVERY DETECTED {emoji}

Feed: {finding['feed_key'].upper()}
Time: {finding['timestamp']}

HYPOTHESIS:
{finding['hypothesis']}

STATISTICAL EVIDENCE:
- P-Value: {finding['p_value']:.6f}
- Effect Size: {finding['effect_size']:.4f}
- Persistence: {finding['persistence_count']} consecutive tests
- Test Method: {finding['test_method']}

{"=" * 40}

This finding requires your attention.
Review at: https://places-rides-journalists-api.trycloudflare.com/findings

---
ENTROPY - AI-powered Lottery Chaos Detector
        """

        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        # Mark as notified
        conn = sqlite3.connect(str(ALERTS_DB))
        c = conn.cursor()
        c.execute("UPDATE findings SET notified = 1 WHERE id = ?", (finding['id'],))
        conn.commit()
        conn.close()

        print(f"[ALERT] Email sent for {level} finding #{finding['id']}")
        return True

    except Exception as e:
        print(f"[ALERT] Email failed: {e}")
        return False


def save_push_subscription(endpoint: str, p256dh: str, auth: str) -> bool:
    """Save a push notification subscription."""
    conn = sqlite3.connect(str(ALERTS_DB))
    c = conn.cursor()

    try:
        c.execute("""
            INSERT OR REPLACE INTO push_subscriptions
            (endpoint, p256dh, auth, created_at)
            VALUES (?, ?, ?, ?)
        """, (endpoint, p256dh, auth, datetime.now().isoformat()))
        conn.commit()
        return True
    except Exception as e:
        print(f"[ALERT] Failed to save push subscription: {e}")
        return False
    finally:
        conn.close()


def get_push_subscriptions() -> List[Dict[str, str]]:
    """Get all push subscriptions."""
    conn = sqlite3.connect(str(ALERTS_DB))
    c = conn.cursor()

    c.execute("SELECT endpoint, p256dh, auth FROM push_subscriptions")

    results = [
        {"endpoint": row[0], "p256dh": row[1], "auth": row[2]}
        for row in c.fetchall()
    ]

    conn.close()
    return results


def send_push_notification(finding: Dict[str, Any]) -> int:
    """
    Send push notifications to all subscribers.
    Returns number of successful sends.
    Note: Requires pywebpush library and VAPID keys.
    """
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        print("[ALERT] pywebpush not installed - skipping push notifications")
        return 0

    VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
    VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
    VAPID_EMAIL = os.getenv("VAPID_EMAIL", "mailto:admin@entropy.app")

    if not VAPID_PRIVATE_KEY:
        print("[ALERT] VAPID keys not configured - skipping push notifications")
        return 0

    subscriptions = get_push_subscriptions()
    if not subscriptions:
        return 0

    level = finding['discovery_level']
    emoji = {"CANDIDATE": "🔶", "VERIFIED": "🚨", "LEGENDARY": "👑"}.get(level, "📊")

    payload = json.dumps({
        "title": f"{emoji} ENTROPY: {level} Found!",
        "body": f"{finding['feed_key'].upper()}: {finding['hypothesis'][:100]}...",
        "icon": "/Entropy.png",
        "badge": "/Entropy.png",
        "data": {
            "finding_id": finding['id'],
            "url": "/findings"
        }
    })

    success_count = 0
    for sub in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": {
                        "p256dh": sub["p256dh"],
                        "auth": sub["auth"]
                    }
                },
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={
                    "sub": VAPID_EMAIL
                }
            )
            success_count += 1
        except WebPushException as e:
            print(f"[ALERT] Push failed: {e}")
            # Remove invalid subscription
            if e.response and e.response.status_code in [404, 410]:
                conn = sqlite3.connect(str(ALERTS_DB))
                c = conn.cursor()
                c.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (sub["endpoint"],))
                conn.commit()
                conn.close()

    return success_count


def alert_discovery(
    feed_key: str,
    discovery_level: str,
    hypothesis: str,
    test_method: str,
    p_value: float,
    effect_size: float,
    persistence_count: int = 1,
    details: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Main alert function - logs finding and sends notifications.
    Only sends alerts when pattern has 3+ consecutive verification tests.
    """
    # Only log for significant findings
    if discovery_level not in ["CANDIDATE", "VERIFIED", "LEGENDARY"]:
        return {"logged": False, "reason": "Below alert threshold"}

    # Log the finding
    finding_id = log_finding(
        feed_key=feed_key,
        discovery_level=discovery_level,
        hypothesis=hypothesis,
        test_method=test_method,
        p_value=p_value,
        effect_size=effect_size,
        persistence_count=persistence_count,
        details=details
    )

    finding = {
        "id": finding_id,
        "timestamp": datetime.now().isoformat(),
        "feed_key": feed_key,
        "discovery_level": discovery_level,
        "hypothesis": hypothesis,
        "test_method": test_method,
        "p_value": p_value,
        "effect_size": effect_size,
        "persistence_count": persistence_count
    }

    # Only send notifications if pattern has 3+ consecutive tests
    email_sent = False
    push_count = 0

    if persistence_count >= 3:
        email_sent = send_email_alert(finding)
        push_count = send_push_notification(finding)

    return {
        "logged": True,
        "finding_id": finding_id,
        "email_sent": email_sent,
        "push_notifications": push_count,
        "persistence_count": persistence_count
    }
