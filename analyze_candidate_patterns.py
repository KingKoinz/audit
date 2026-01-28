import sqlite3
from collections import Counter

DB_PATH = './data/research_journal.sqlite'

# Connect to the database
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Query last 100 research iterations for both feeds
data = c.execute('''
    SELECT feed_key, iteration, hypothesis, test_method, p_value, effect_size, viable
    FROM research_log
    WHERE feed_key IN ('powerball','megamillions')
    ORDER BY iteration DESC
    LIMIT 100
''').fetchall()
conn.close()

# Analyze candidate patterns (p < 0.01, effect_size > 0.2, viable)
candidates = [row for row in data if row[4] is not None and row[5] is not None and row[4] < 0.01 and abs(row[5]) > 0.2 and row[6]]

feed_counts = Counter(row[0] for row in candidates)

total_counts = Counter(row[0] for row in data)

print('--- Candidate Pattern Frequency (last 100 tests) ---')
for feed in ['powerball', 'megamillions']:
    print(f'{feed}: {feed_counts[feed]} candidate patterns out of {total_counts[feed]} total tests')

print('\nDetails:')
for row in candidates:
    print(f'Feed: {row[0]}, Iter: {row[1]}, p={row[4]:.4f}, effect={row[5]:.3f}, Test: {row[3]}, Hypothesis: {row[2][:60]}...')
