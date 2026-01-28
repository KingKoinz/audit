from __future__ import annotations
# --- Advanced Statistical Pattern Tests ---
import math
from scipy.stats import entropy as scipy_entropy

def test_number_clustering(draws: List[Dict]) -> Dict[str, Any]:
    """
    Test hypothesis: Numbers cluster together more than expected (low spread).
    """
    spreads = []
    for draw in draws:
        nums = sorted(draw["numbers"])
        spreads.append(nums[-1] - nums[0])
    mean_spread = np.mean(spreads)
    std_spread = np.std(spreads)
    # Compare to random expectation (simulate or use historical mean)
    expected_spread = 20  # Placeholder, should be estimated from random draws
    effect_size = (mean_spread - expected_spread) / (std_spread if std_spread else 1)
    return {
        "mean_spread": mean_spread,
        "std_spread": std_spread,
        "expected_spread": expected_spread,
        "effect_size": effect_size,
        "viable": abs(effect_size) > 0.5
    }

def test_streaks(draws: List[Dict]) -> Dict[str, Any]:
    """
    Test hypothesis: Some numbers appear in streaks (consecutive draws) more than random.
    """
    from collections import defaultdict
    streaks = defaultdict(int)
    last_seen = {}
    for i, draw in enumerate(draws):
        for n in draw["numbers"]:
            if n in last_seen and last_seen[n] == i-1:
                streaks[n] += 1
            last_seen[n] = i
    max_streak = max(streaks.values()) if streaks else 0
    mean_streak = np.mean(list(streaks.values())) if streaks else 0
    return {
        "max_streak": max_streak,
        "mean_streak": mean_streak,
        "viable": max_streak > 2 or mean_streak > 1
    }

def test_entropy(draws: List[Dict], feed_key: str) -> Dict[str, Any]:
    """
    Test hypothesis: The entropy of number distribution is lower/higher than expected.
    """
    from backend.audit import RANGES
    r = RANGES[feed_key]
    max_num = r["main_max"]
    all_numbers = [n for d in draws for n in d["numbers"] if 1 <= n <= max_num]
    counts = np.bincount(all_numbers, minlength=max_num + 1)[1:max_num + 1]
    probs = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts)/len(counts)
    ent = scipy_entropy(probs, base=2)
    expected_entropy = math.log2(r["main_max"])
    effect_size = (ent - expected_entropy) / expected_entropy
    return {
        "entropy": ent,
        "expected_entropy": expected_entropy,
        "effect_size": effect_size,
        "viable": abs(effect_size) > 0.05
    }

def test_markov_chain(draws: List[Dict]) -> Dict[str, Any]:
    """
    Test hypothesis: The appearance of a number depends on the previous draw (first-order Markov property).
    """
    from collections import defaultdict
    transitions = defaultdict(lambda: defaultdict(int))
    prev_nums = set()
    for draw in draws:
        curr_nums = set(draw["numbers"])
        for n in prev_nums:
            for m in curr_nums:
                transitions[n][m] += 1
        prev_nums = curr_nums
    # Calculate transition probabilities and look for strong dependencies
    strong_links = 0
    for n, targets in transitions.items():
        total = sum(targets.values())
        if total == 0: continue
        probs = [v/total for v in targets.values()]
        if max(probs) > 0.2:  # Arbitrary threshold for strong dependency
            strong_links += 1
    return {
        "strong_links": strong_links,
        "viable": strong_links > 0
    }

def test_pairs_triplets(draws: List[Dict]) -> Dict[str, Any]:
    """
    Test hypothesis: Certain pairs or triplets of numbers appear together more than expected.
    """
    from collections import Counter
    pair_counts = Counter()
    triplet_counts = Counter()
    for draw in draws:
        nums = sorted(draw["numbers"])
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pair = (nums[i], nums[j])
                pair_counts[pair] += 1
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                for k in range(j+1, len(nums)):
                    triplet = (nums[i], nums[j], nums[k])
                    triplet_counts[triplet] += 1
    most_common_pair = pair_counts.most_common(1)[0] if pair_counts else ((),0)
    most_common_triplet = triplet_counts.most_common(1)[0] if triplet_counts else ((),0)
    return {
        "most_common_pair": most_common_pair,
        "most_common_triplet": most_common_triplet,
        "viable": most_common_pair[1] > 2 or most_common_triplet[1] > 1
    }

def test_positional_bias(draws: List[Dict], position: int, feed_key: str) -> Dict[str, Any]:
    """
    Test hypothesis: Certain numbers are more likely to appear in a specific position (e.g., first ball).
    """
    from backend.audit import RANGES
    r = RANGES[feed_key]
    pos_counts = np.zeros(r["main_max"]+1)
    for draw in draws:
        nums = draw["numbers"]
        if len(nums) > position:
            pos_counts[nums[position]] += 1
    most_common = int(np.argmax(pos_counts))
    return {
        "position": position,
        "most_common": most_common,
        "count": int(pos_counts[most_common]),
        "viable": pos_counts[most_common] > (len(draws)/r["main_max"])*2  # Arbitrary: 2x expected
    }


from typing import Dict, List, Any, Tuple
import numpy as np
from collections import Counter
from datetime import datetime

def test_digit_ending_hypothesis(draws: List[Dict], target_digit: int, feed_key: str) -> Dict[str, Any]:
    """
    Test hypothesis: Numbers ending in X are drawn more/less frequently.
    Example: "Numbers ending in 7 appear more often than expected"
    """
    from backend.audit import RANGES, _chisquare
    
    max_num = RANGES[feed_key]["main_max"]
    
    # Count numbers ending in target_digit
    all_numbers = []
    for draw in draws:
        all_numbers.extend(draw["numbers"])
    
    ending_in_target = sum(1 for n in all_numbers if n % 10 == target_digit and n <= max_num)
    total_numbers = len(all_numbers)
    
    # Expected: numbers ending in target_digit / total possible numbers
    possible_with_digit = len([n for n in range(1, max_num + 1) if n % 10 == target_digit])
    expected_freq = possible_with_digit / max_num
    expected_count = total_numbers * expected_freq
    
    # Chi-square test
    observed = [ending_in_target, total_numbers - ending_in_target]
    expected = [expected_count, total_numbers - expected_count]
    chi2, p_value = _chisquare(observed, expected)
    
    effect_size = (ending_in_target - expected_count) / expected_count if expected_count > 0 else 0
    
    return {
        "observed": ending_in_target,
        "expected": expected_count,
        "total": total_numbers,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_consecutive_numbers_hypothesis(draws: List[Dict]) -> Dict[str, Any]:
    """
    Test hypothesis: Consecutive numbers (e.g., 12, 13) appear together more than random.
    """
    from backend.audit import _chisquare
    
    consecutive_count = 0
    total_draws = len(draws)
    
    for draw in draws:
        sorted_nums = sorted(draw["numbers"])
        has_consecutive = any(sorted_nums[i+1] - sorted_nums[i] == 1 for i in range(len(sorted_nums)-1))
        if has_consecutive:
            consecutive_count += 1
    
    # Expected by randomness (rough estimate: ~40% chance)
    expected_rate = 0.4
    expected_count = total_draws * expected_rate
    
    observed = [consecutive_count, total_draws - consecutive_count]
    expected = [expected_count, total_draws - expected_count]
    chi2, p_value = _chisquare(observed, expected)
    
    effect_size = (consecutive_count - expected_count) / expected_count if expected_count > 0 else 0
    
    return {
        "draws_with_consecutive": consecutive_count,
        "total_draws": total_draws,
        "rate": consecutive_count / total_draws,
        "expected_rate": expected_rate,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_sum_range_hypothesis(draws: List[Dict], low: int, high: int) -> Dict[str, Any]:
    """
    Test hypothesis: Draw sums fall within specific range more than random.
    Example: "Sums between 100-150 are more common"
    """
    from backend.audit import _chisquare

    # Calculate actual sums from all draws
    all_sums = [sum(draw["numbers"]) for draw in draws]
    total_draws = len(draws)

    sums_in_range = sum(1 for s in all_sums if low <= s <= high)

    # Calculate expected rate based on actual sum distribution (empirical)
    # For 5 numbers from 1-70: min=15, max=340, typical range ~100-250
    # Use actual data to estimate expected rate for this range
    min_sum = min(all_sums) if all_sums else 15
    max_sum = max(all_sums) if all_sums else 340
    total_sum_range = max_sum - min_sum + 1

    # Expected rate = proportion of possible range covered
    range_width = high - low + 1
    expected_rate = range_width / total_sum_range if total_sum_range > 0 else 0.33

    # Clamp expected rate to reasonable bounds
    expected_rate = max(0.01, min(0.99, expected_rate))
    expected_count = total_draws * expected_rate

    observed = [sums_in_range, total_draws - sums_in_range]
    expected = [expected_count, total_draws - expected_count]
    chi2, p_value = _chisquare(observed, expected)

    effect_size = (sums_in_range - expected_count) / expected_count if expected_count > 0 else 0

    return {
        "draws_in_range": sums_in_range,
        "total_draws": total_draws,
        "rate": sums_in_range / total_draws,
        "expected_rate": expected_rate,
        "range": f"{low}-{high}",
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_even_odd_imbalance_hypothesis(draws: List[Dict]) -> Dict[str, Any]:
    """
    Test hypothesis: Even/odd numbers are imbalanced in draws.
    """
    from backend.audit import _chisquare
    
    even_counts = []
    for draw in draws:
        even_count = sum(1 for n in draw["numbers"] if n % 2 == 0)
        even_counts.append(even_count)
    
    # Expected: 2.5 even numbers per 5-number draw
    expected_even_per_draw = 2.5
    total_draws = len(draws)
    avg_even = np.mean(even_counts)
    
    # Simple test: is average even count significantly different from 2.5?
    observed = [sum(even_counts), total_draws * 5 - sum(even_counts)]
    expected = [expected_even_per_draw * total_draws, (5 - expected_even_per_draw) * total_draws]
    chi2, p_value = _chisquare(observed, expected)
    
    effect_size = (avg_even - expected_even_per_draw) / expected_even_per_draw
    
    return {
        "avg_even_per_draw": avg_even,
        "expected_even": expected_even_per_draw,
        "total_draws": total_draws,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_bonus_ball_correlation_hypothesis(draws: List[Dict], feed_key: str) -> Dict[str, Any]:
    """
    Test hypothesis: Bonus ball correlates with main number patterns.
    """
    from backend.audit import _chisquare
    
    # Count how often bonus ball is near (within 5) of a main number
    correlation_count = 0
    total_draws = len(draws)
    
    for draw in draws:
        bonus = draw["bonus"]
        main_nums = draw["numbers"]
        if any(abs(bonus - m) <= 5 for m in main_nums):
            correlation_count += 1
    
    # Expected (random assumption)
    expected_rate = 0.5
    expected_count = total_draws * expected_rate
    
    observed = [correlation_count, total_draws - correlation_count]
    expected = [expected_count, total_draws - expected_count]
    chi2, p_value = _chisquare(observed, expected)
    
    effect_size = (correlation_count - expected_count) / expected_count if expected_count > 0 else 0
    
    return {
        "correlated_draws": correlation_count,
        "total_draws": total_draws,
        "rate": correlation_count / total_draws,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_day_of_week_number_bias(draws: List[Dict], target_number: int, target_day: int) -> Dict[str, Any]:
    """
    Test hypothesis: Specific number appears more frequently on specific day of week.
    Example: "Number 28 appears more often on Wednesdays"

    Args:
        target_number: The number to track (e.g., 28)
        target_day: Day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)
    """
    from backend.audit import _chisquare

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Count occurrences of target number on target day vs other days
    target_day_count = 0
    target_day_total_nums = 0
    other_days_count = 0
    other_days_total_nums = 0

    for draw in draws:
        # Parse date
        try:
            draw_date = datetime.fromisoformat(draw["draw_date"])
            day_of_week = draw_date.weekday()
        except:
            continue

        nums = draw["numbers"]
        num_count = nums.count(target_number)

        if day_of_week == target_day:
            target_day_count += num_count
            target_day_total_nums += len(nums)
        else:
            other_days_count += num_count
            other_days_total_nums += len(nums)

    # Calculate frequencies
    target_day_freq = target_day_count / target_day_total_nums if target_day_total_nums > 0 else 0
    other_days_freq = other_days_count / other_days_total_nums if other_days_total_nums > 0 else 0

    # Expected frequency (same across all days)
    total_count = target_day_count + other_days_count
    total_nums = target_day_total_nums + other_days_total_nums
    expected_freq = total_count / total_nums if total_nums > 0 else 0

    expected_target_day = expected_freq * target_day_total_nums
    expected_other_days = expected_freq * other_days_total_nums

    # Chi-square test
    if expected_target_day > 0 and expected_other_days > 0:
        observed = [target_day_count, other_days_count]
        expected = [expected_target_day, expected_other_days]
        chi2, p_value = _chisquare(observed, expected)
        effect_size = (target_day_count - expected_target_day) / expected_target_day
    else:
        chi2, p_value, effect_size = 0, 1.0, 0

    return {
        "target_number": target_number,
        "target_day": day_names[target_day],
        "target_day_occurrences": target_day_count,
        "target_day_expected": expected_target_day,
        "other_days_occurrences": other_days_count,
        "other_days_expected": expected_other_days,
        "target_day_frequency": target_day_freq,
        "other_days_frequency": other_days_freq,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_month_number_bias(draws: List[Dict], target_number: int, target_month: int) -> Dict[str, Any]:
    """
    Test hypothesis: Specific number appears more frequently in specific month.
    Example: "Number 7 appears more often in July"

    Args:
        target_number: The number to track
        target_month: Month (1=January, 2=February, ..., 12=December)
    """
    from backend.audit import _chisquare

    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    target_month_count = 0
    target_month_total_nums = 0
    other_months_count = 0
    other_months_total_nums = 0

    for draw in draws:
        try:
            draw_date = datetime.fromisoformat(draw["draw_date"])
            month = draw_date.month
        except:
            continue

        nums = draw["numbers"]
        num_count = nums.count(target_number)

        if month == target_month:
            target_month_count += num_count
            target_month_total_nums += len(nums)
        else:
            other_months_count += num_count
            other_months_total_nums += len(nums)

    # Calculate frequencies
    target_month_freq = target_month_count / target_month_total_nums if target_month_total_nums > 0 else 0
    other_months_freq = other_months_count / other_months_total_nums if other_months_total_nums > 0 else 0

    # Expected frequency
    total_count = target_month_count + other_months_count
    total_nums = target_month_total_nums + other_months_total_nums
    expected_freq = total_count / total_nums if total_nums > 0 else 0

    expected_target_month = expected_freq * target_month_total_nums
    expected_other_months = expected_freq * other_months_total_nums

    # Chi-square test
    if expected_target_month > 0 and expected_other_months > 0:
        observed = [target_month_count, other_months_count]
        expected = [expected_target_month, expected_other_months]
        chi2, p_value = _chisquare(observed, expected)
        effect_size = (target_month_count - expected_target_month) / expected_target_month
    else:
        chi2, p_value, effect_size = 0, 1.0, 0

    return {
        "target_number": target_number,
        "target_month": month_names[target_month],
        "target_month_occurrences": target_month_count,
        "target_month_expected": expected_target_month,
        "other_months_occurrences": other_months_count,
        "other_months_expected": expected_other_months,
        "target_month_frequency": target_month_freq,
        "other_months_frequency": other_months_freq,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_seasonal_number_bias(draws: List[Dict], target_number: int, target_season: str) -> Dict[str, Any]:
    """
    Test hypothesis: Specific number appears more frequently in specific season.
    Example: "Number 13 appears more often in winter"

    Args:
        target_number: The number to track
        target_season: "winter", "spring", "summer", or "fall"
    """
    from backend.audit import _chisquare

    # Define seasons by month (Northern Hemisphere)
    seasons = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11]
    }

    target_season = target_season.lower()
    if target_season not in seasons:
        return {"error": f"Invalid season: {target_season}"}

    target_months = seasons[target_season]

    target_season_count = 0
    target_season_total_nums = 0
    other_seasons_count = 0
    other_seasons_total_nums = 0

    for draw in draws:
        try:
            draw_date = datetime.fromisoformat(draw["draw_date"])
            month = draw_date.month
        except:
            continue

        nums = draw["numbers"]
        num_count = nums.count(target_number)

        if month in target_months:
            target_season_count += num_count
            target_season_total_nums += len(nums)
        else:
            other_seasons_count += num_count
            other_seasons_total_nums += len(nums)

    # Calculate frequencies
    target_season_freq = target_season_count / target_season_total_nums if target_season_total_nums > 0 else 0
    other_seasons_freq = other_seasons_count / other_seasons_total_nums if other_seasons_total_nums > 0 else 0

    # Expected frequency
    total_count = target_season_count + other_seasons_count
    total_nums = target_season_total_nums + other_seasons_total_nums
    expected_freq = total_count / total_nums if total_nums > 0 else 0

    expected_target_season = expected_freq * target_season_total_nums
    expected_other_seasons = expected_freq * other_seasons_total_nums

    # Chi-square test
    if expected_target_season > 0 and expected_other_seasons > 0:
        observed = [target_season_count, other_seasons_count]
        expected = [expected_target_season, expected_other_seasons]
        chi2, p_value = _chisquare(observed, expected)
        effect_size = (target_season_count - expected_target_season) / expected_target_season
    else:
        chi2, p_value, effect_size = 0, 1.0, 0

    return {
        "target_number": target_number,
        "target_season": target_season,
        "target_season_occurrences": target_season_count,
        "target_season_expected": expected_target_season,
        "other_seasons_occurrences": other_seasons_count,
        "other_seasons_expected": expected_other_seasons,
        "target_season_frequency": target_season_freq,
        "other_seasons_frequency": other_seasons_freq,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_weekend_vs_weekday_bias(draws: List[Dict], target_number: int) -> Dict[str, Any]:
    """
    Test hypothesis: Specific number appears more frequently on weekends vs weekdays.
    Example: "Number 21 appears more often on weekend draws"
    """
    from backend.audit import _chisquare

    weekend_count = 0
    weekend_total_nums = 0
    weekday_count = 0
    weekday_total_nums = 0

    for draw in draws:
        try:
            draw_date = datetime.fromisoformat(draw["draw_date"])
            day_of_week = draw_date.weekday()
            is_weekend = day_of_week in [5, 6]  # Saturday, Sunday
        except:
            continue

        nums = draw["numbers"]
        num_count = nums.count(target_number)

        if is_weekend:
            weekend_count += num_count
            weekend_total_nums += len(nums)
        else:
            weekday_count += num_count
            weekday_total_nums += len(nums)

    # Calculate frequencies
    weekend_freq = weekend_count / weekend_total_nums if weekend_total_nums > 0 else 0
    weekday_freq = weekday_count / weekday_total_nums if weekday_total_nums > 0 else 0

    # Expected frequency
    total_count = weekend_count + weekday_count
    total_nums = weekend_total_nums + weekday_total_nums
    expected_freq = total_count / total_nums if total_nums > 0 else 0

    expected_weekend = expected_freq * weekend_total_nums
    expected_weekday = expected_freq * weekday_total_nums

    # Chi-square test
    if expected_weekend > 0 and expected_weekday > 0:
        observed = [weekend_count, weekday_count]
        expected = [expected_weekend, expected_weekday]
        chi2, p_value = _chisquare(observed, expected)
        effect_size = (weekend_count - expected_weekend) / expected_weekend
    else:
        chi2, p_value, effect_size = 0, 1.0, 0

    return {
        "target_number": target_number,
        "weekend_occurrences": weekend_count,
        "weekend_expected": expected_weekend,
        "weekday_occurrences": weekday_count,
        "weekday_expected": expected_weekday,
        "weekend_frequency": weekend_freq,
        "weekday_frequency": weekday_freq,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1
    }

def test_temporal_persistence_hypothesis(draws: List[Dict], target_number: int, window_size: int = 30) -> Dict[str, Any]:
    """
    Test hypothesis: When a number shows elevated frequency, does it persist over time?
    Example: "Number 28's elevated frequency persists for 30+ consecutive draws"

    This tests if 'hot' numbers stay hot (temporal autocorrelation).
    """
    from backend.audit import _chisquare

    if len(draws) < window_size * 2:
        return {"error": "Not enough data for temporal persistence test"}

    # Split into rolling windows and check if elevated periods persist
    windows_with_elevation = 0
    consecutive_elevated_windows = 0
    max_consecutive_streak = 0
    current_streak = 0

    expected_per_number = window_size * 5 / 69  # 5 numbers per draw, 69 total numbers

    for i in range(len(draws) - window_size + 1):
        window = draws[i:i+window_size]
        window_nums = [n for draw in window for n in draw["numbers"]]
        count = window_nums.count(target_number)

        # Is this window elevated? (more than 1.5x expected)
        if count > expected_per_number * 1.5:
            windows_with_elevation += 1
            current_streak += 1
            max_consecutive_streak = max(max_consecutive_streak, current_streak)
        else:
            current_streak = 0

    total_windows = len(draws) - window_size + 1
    persistence_rate = windows_with_elevation / total_windows if total_windows > 0 else 0

    # Chi-square: does the number of elevated windows exceed random expectation?
    # We'd expect ~5-10% of windows to be elevated by chance
    expected_elevated = total_windows * 0.075  # 7.5% baseline

    observed = [windows_with_elevation, total_windows - windows_with_elevation]
    expected = [expected_elevated, total_windows - expected_elevated]
    chi2, p_value = _chisquare(observed, expected)

    effect_size = (windows_with_elevation - expected_elevated) / expected_elevated if expected_elevated > 0 else 0

    return {
        "target_number": target_number,
        "window_size": window_size,
        "total_windows_tested": total_windows,
        "elevated_windows": windows_with_elevation,
        "expected_elevated_windows": expected_elevated,
        "persistence_rate": persistence_rate,
        "max_consecutive_elevated_streak": max_consecutive_streak,
        "chi2": chi2,
        "p_value": p_value,
        "effect_size": effect_size,
        "viable": p_value < 0.01 and abs(effect_size) > 0.1 and max_consecutive_streak >= 3
    }

# Registry of available pattern tests
PATTERN_TESTS = {
    "digit_ending": test_digit_ending_hypothesis,
    "consecutive_numbers": test_consecutive_numbers_hypothesis,
    "sum_range": test_sum_range_hypothesis,
    "even_odd_imbalance": test_even_odd_imbalance_hypothesis,
    "bonus_correlation": test_bonus_ball_correlation_hypothesis,
    "day_of_week_bias": test_day_of_week_number_bias,
    "month_bias": test_month_number_bias,
    "seasonal_bias": test_seasonal_number_bias,
    "weekend_weekday_bias": test_weekend_vs_weekday_bias,
    "temporal_persistence": test_temporal_persistence_hypothesis,
    # Advanced tests:
    "number_clustering": test_number_clustering,
    "streaks": test_streaks,
    "entropy": test_entropy,
    "markov_chain": test_markov_chain,
    "pairs_triplets": test_pairs_triplets,
    "positional_bias": test_positional_bias
}
