"""
Analyze persistence of elevated number frequencies in lottery data.
Checks if number 28's "hotness" is temporary clustering or persistent bias.
"""

import sqlite3
from collections import Counter
from pathlib import Path
import numpy as np
from math import sqrt, erfc

def _chisquare(observed, expected):
    """Simple chi-square implementation (no scipy dependency)."""
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    p = erfc(sqrt(chi2 / 2))
    return chi2, p

DB_PATH = Path("./data/entropy.sqlite")

def analyze_number_persistence(feed_key="powerball", target_number=28, window_size=50):
    """
    Analyze if a number shows persistent elevated frequency across rolling windows.

    Returns:
        - Overall frequency vs expected
        - Consistency across different time windows
        - Chi-square tests for different periods
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Get all draws ordered by date
    cursor.execute("""
        SELECT n1, n2, n3, n4, n5 FROM draws
        WHERE feed_key = ?
        ORDER BY draw_date ASC
    """, (feed_key,))

    draws = cursor.fetchall()
    conn.close()

    if not draws:
        return {"error": "No data found"}

    # Parse all numbers
    all_draws_numbers = []
    for draw in draws:
        numbers = list(draw)  # n1, n2, n3, n4, n5
        all_draws_numbers.append(numbers)

    total_draws = len(all_draws_numbers)

    # Overall statistics
    all_numbers = [n for draw in all_draws_numbers for n in draw]
    overall_count = all_numbers.count(target_number)
    expected_per_number = len(all_numbers) / 69

    # Rolling window analysis
    window_counts = []
    window_chi_squares = []
    window_p_values = []

    for i in range(0, len(all_draws_numbers) - window_size + 1, window_size):
        window_draws = all_draws_numbers[i:i+window_size]
        window_numbers = [n for draw in window_draws for n in draw]
        window_count = window_numbers.count(target_number)

        expected_in_window = len(window_numbers) / 69

        # Chi-square for this window
        observed = [window_count, len(window_numbers) - window_count]
        expected = [expected_in_window, len(window_numbers) - expected_in_window]
        chi2, p = _chisquare(observed, expected)

        window_counts.append(window_count)
        window_chi_squares.append(chi2)
        window_p_values.append(p)

    # Most recent window (last 50 draws)
    recent_draws = all_draws_numbers[-window_size:]
    recent_numbers = [n for draw in recent_draws for n in draw]
    recent_count = recent_numbers.count(target_number)
    recent_expected = len(recent_numbers) / 69

    # Overall chi-square
    observed_overall = [overall_count, len(all_numbers) - overall_count]
    expected_overall = [expected_per_number, len(all_numbers) - expected_per_number]
    overall_chi2, overall_p = _chisquare(observed_overall, expected_overall)

    # Persistence check: How many windows show p < 0.05?
    significant_windows = sum(1 for p in window_p_values if p < 0.05)
    very_significant_windows = sum(1 for p in window_p_values if p < 0.01)

    return {
        "number": target_number,
        "total_draws": total_draws,
        "overall_statistics": {
            "observed_count": overall_count,
            "expected_count": expected_per_number,
            "frequency_percent": (overall_count / len(all_numbers)) * 100,
            "expected_percent": (1 / 69) * 100,
            "deviation": overall_count - expected_per_number,
            "chi_square": overall_chi2,
            "p_value": overall_p
        },
        "recent_window": {
            "draws": window_size,
            "observed_count": recent_count,
            "expected_count": recent_expected,
            "deviation": recent_count - recent_expected,
            "z_score": (recent_count - recent_expected) / np.sqrt(recent_expected) if recent_expected > 0 else 0
        },
        "persistence_analysis": {
            "total_windows_tested": len(window_counts),
            "windows_with_p_lt_0.05": significant_windows,
            "windows_with_p_lt_0.01": very_significant_windows,
            "persistence_rate": (significant_windows / len(window_counts)) * 100 if window_counts else 0,
            "average_count_per_window": np.mean(window_counts) if window_counts else 0,
            "std_dev_per_window": np.std(window_counts) if window_counts else 0,
            "min_count": min(window_counts) if window_counts else 0,
            "max_count": max(window_counts) if window_counts else 0
        },
        "window_details": [
            {
                "window_num": i+1,
                "count": window_counts[i],
                "chi2": window_chi_squares[i],
                "p_value": window_p_values[i],
                "significant": window_p_values[i] < 0.05
            }
            for i in range(len(window_counts))
        ],
        "verdict": determine_verdict(
            overall_p,
            significant_windows,
            len(window_counts),
            recent_count,
            recent_expected
        )
    }

def determine_verdict(overall_p, sig_windows, total_windows, recent_count, recent_expected):
    """Determine if this is random clustering or persistent bias."""

    persistence_rate = (sig_windows / total_windows) * 100 if total_windows > 0 else 0
    recent_deviation = recent_count - recent_expected

    if overall_p > 0.05:
        return "RANDOM: No overall statistical significance. This is normal randomness."
    elif overall_p < 0.05 and persistence_rate < 20:
        return "CLUSTERING: Statistically significant overall, but not persistent across time windows. Likely temporary cluster."
    elif overall_p < 0.01 and persistence_rate >= 20 and persistence_rate < 50:
        return "INTERESTING: Some persistence detected. Warrants continued monitoring."
    elif overall_p < 0.01 and persistence_rate >= 50:
        return "ANOMALY: High persistence across multiple time windows. Potential systematic bias."
    elif recent_deviation > 4 and overall_p > 0.05:
        return "RECENT SPIKE: Strong recent deviation but not historically persistent. Temporary hot streak."
    else:
        return "UNCLEAR: Mixed signals. Continue monitoring."

if __name__ == "__main__":
    result = analyze_number_persistence("powerball", 28, 50)

    print("=" * 70)
    print(f"PERSISTENCE ANALYSIS: Number {result['number']}")
    print("=" * 70)

    print(f"\nOVERALL STATISTICS ({result['total_draws']} total draws):")
    print(f"  Observed:  {result['overall_statistics']['observed_count']}")
    print(f"  Expected:  {result['overall_statistics']['expected_count']:.2f}")
    print(f"  Deviation: {result['overall_statistics']['deviation']:+.2f}")
    print(f"  Chi-square = {result['overall_statistics']['chi_square']:.4f}")
    print(f"  p-value = {result['overall_statistics']['p_value']:.6e}")

    print(f"\nRECENT 50 DRAWS:")
    print(f"  Observed:  {result['recent_window']['observed_count']}")
    print(f"  Expected:  {result['recent_window']['expected_count']:.2f}")
    print(f"  Deviation: {result['recent_window']['deviation']:+.2f}")
    print(f"  Z-score:   {result['recent_window']['z_score']:.2f} sigma")

    print(f"\nPERSISTENCE ANALYSIS:")
    print(f"  Windows tested:           {result['persistence_analysis']['total_windows_tested']}")
    print(f"  Significant (p<0.05):     {result['persistence_analysis']['windows_with_p_lt_0.05']}")
    print(f"  Very significant (p<0.01): {result['persistence_analysis']['windows_with_p_lt_0.01']}")
    print(f"  Persistence rate:         {result['persistence_analysis']['persistence_rate']:.1f}%")
    print(f"  Avg count per window:     {result['persistence_analysis']['average_count_per_window']:.2f}")
    print(f"  Std deviation:            {result['persistence_analysis']['std_dev_per_window']:.2f}")
    print(f"  Range:                    {result['persistence_analysis']['min_count']}-{result['persistence_analysis']['max_count']}")

    print(f"\nVERDICT:")
    print(f"  {result['verdict']}")

    print(f"\nWINDOW-BY-WINDOW BREAKDOWN:")
    for window in result['window_details']:
        sig_marker = "WARNING" if window['significant'] else "OK"
        print(f"  [{sig_marker:7s}] Window {window['window_num']:2d}: count={window['count']:2d}, chi2={window['chi2']:6.2f}, p={window['p_value']:.4f}")

    print("\n" + "=" * 70)
