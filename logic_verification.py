import sqlite3

DB_PATH = "./data/research_journal.sqlite"
FEED_KEY = "powerball"  # Change to your feed key if needed
PATTERN_SUBSTRING = ""  # Fill in part of the hypothesis to match


def verify_pattern_attempts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT hypothesis, p_value, effect_size, viable, iteration, timestamp
        FROM research_log
        WHERE feed_key = ?
          AND hypothesis LIKE ?
        ORDER BY iteration DESC
        LIMIT 5
    """, (FEED_KEY, f"%{PATTERN_SUBSTRING}%"))
    rows = c.fetchall()
    conn.close()

    print("Last 5 attempts for pattern:")
    for row in reversed(rows):
        print(f"Iter {row[4]} | p={row[1]:.4f} | effect={row[2]:.3f} | viable={bool(row[3])} | {row[5]}")

    # Check for 3+ consecutive viable
    consecutive = 0
    for row in reversed(rows):
        if row[3]:
            consecutive += 1
            if consecutive >= 3:
                print("Pattern should be VERIFIED (3+ consecutive viable).")
                return
        else:
            consecutive = 0
    print("Pattern correctly marked as INCONCLUSIVE after 5 attempts.")

if __name__ == "__main__":
    verify_pattern_attempts()
