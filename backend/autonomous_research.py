# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import re
import numpy as np
import math
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
from backend.research_journal import (
    init_research_db, log_research_iteration,
    get_recent_research, get_iteration_count,
    get_pursuit_state, start_pursuit, update_pursuit, end_pursuit,
    generate_verification_windows, migrate_db
)
from backend.pattern_tests import PATTERN_TESTS
from backend.db import get_all_draws
from backend.discovery_framework import classify_discovery, track_persistence
from backend.alerts import alert_discovery


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization, aggressively handle NaN/infinity."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Replace NaN and infinity with safe values for JSON
        try:
            if math.isnan(val):
                return 1.0  # Default for failed tests
            elif math.isinf(val):
                return 1.0 if val > 0 else 0.0
        except (TypeError, ValueError):
            return 1.0
        return val
    elif isinstance(obj, float):
        # Handle native Python floats too - AGGRESSIVE checking
        try:
            if math.isnan(obj):
                return 1.0
            elif math.isinf(obj):
                return 1.0 if obj > 0 else 0.0
        except (TypeError, ValueError):
            return 1.0
        return obj
    elif isinstance(obj, np.ndarray):
        # Convert array and clean each element
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        # Recursively clean dict values
        cleaned = {}
        for key, value in obj.items():
            try:
                cleaned[key] = convert_numpy_types(value)
            except Exception:
                # If conversion fails, use safe default
                cleaned[key] = None
        return cleaned
    elif isinstance(obj, list):
        # Recursively clean list items
        try:
            return [convert_numpy_types(item) for item in obj]
        except Exception:
            return []
    else:
        return obj


def load_user_avenues():
    """Load user-suggested research avenues from user_patterns.json"""
    import os
    avenues_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_patterns.json")
    if os.path.exists(avenues_file):
        try:
            with open(avenues_file, "r") as f:
                data = json.load(f)
                return [a for a in data.get("avenues", []) if a.get("status") == "pending"]
        except Exception:
            return []
    return []


def mark_avenue_investigating(avenue_id: str):
    """Mark a user avenue as being investigated"""
    import os
    avenues_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_patterns.json")
    if os.path.exists(avenues_file):
        try:
            with open(avenues_file, "r") as f:
                data = json.load(f)
            for avenue in data.get("avenues", []):
                if avenue.get("id") == avenue_id:
                    avenue["status"] = "investigating"
                    break
            with open(avenues_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# Deep dive investigation queue (in-memory, per feed)
_deep_dive_queue = {"powerball": [], "megamillions": []}


def generate_parameter_variations(test_method: str, base_params: dict) -> list:
    """Generate parameter variations for deep-dive investigation of a pattern"""
    variations = []

    if test_method == "sum_range":
        low = base_params.get("low", 100)
        high = base_params.get("high", 150)
        width = high - low
        # Test adjacent ranges
        variations.append({"low": low - width, "high": low, "variation": "lower_adjacent"})
        variations.append({"low": high, "high": high + width, "variation": "upper_adjacent"})
        # Test narrower range (center)
        center = (low + high) // 2
        variations.append({"low": center - width//4, "high": center + width//4, "variation": "narrow_center"})

    elif test_method in ["day_of_week_bias", "month_bias", "seasonal_bias"]:
        target_num = base_params.get("target_number", 7)
        # Test nearby numbers
        for offset in [-1, 1, -5, 5]:
            new_num = target_num + offset
            if 1 <= new_num <= 70:
                variations.append({**base_params, "target_number": new_num, "variation": f"nearby_{offset:+d}"})

    elif test_method == "digit_ending":
        digit = base_params.get("digit", 7)
        # Test other digits
        for d in [(digit - 1) % 10, (digit + 1) % 10, (digit + 5) % 10]:
            variations.append({"digit": d, "variation": f"digit_{d}"})

    elif test_method == "temporal_persistence":
        target_num = base_params.get("target_number", 7)
        window = base_params.get("window_size", 30)
        # Test different window sizes
        for w in [15, 50, 100]:
            if w != window:
                variations.append({"target_number": target_num, "window_size": w, "variation": f"window_{w}"})

    elif test_method == "positional_bias":
        position = base_params.get("position", 0)
        # Test other positions
        for p in range(5):
            if p != position:
                variations.append({"position": p, "variation": f"position_{p}"})

    return variations


def queue_deep_dive(feed_key: str, test_method: str, base_params: dict, hypothesis: str):
    """Queue parameter variations for deep-dive investigation"""
    variations = generate_parameter_variations(test_method, base_params)
    for var in variations:
        _deep_dive_queue[feed_key].append({
            "test_method": test_method,
            "parameters": var,
            "base_hypothesis": hypothesis,
            "variation": var.pop("variation", "unknown")
        })


def get_next_deep_dive(feed_key: str):
    """Get next queued deep-dive investigation"""
    if _deep_dive_queue.get(feed_key):
        return _deep_dive_queue[feed_key].pop(0)
    return None


def has_deep_dive_queue(feed_key: str) -> bool:
    """Check if there are queued deep-dive investigations"""
    return len(_deep_dive_queue.get(feed_key, [])) > 0


def execute_custom_test(draws, custom_logic: str, parameters: dict, feed_key: str) -> Dict[str, Any]:
    """
    Execute a custom test method described by AI.
    This is a statistical framework that interprets AI's test description.
    """
    from backend.audit import RANGES
    
    # Extract all main numbers from draws
    all_numbers = []
    for draw in draws:
        nums = draw["numbers"]
        if isinstance(nums, str):
            # Defensive: handle legacy or malformed data
            nums = [int(x.strip()) for x in nums.split(",") if x.strip().isdigit()]
        all_numbers.extend(nums)
    
    # Generic statistical tests based on common keywords in custom_logic
    logic_lower = custom_logic.lower()
    
    # Modular arithmetic test
    if "modulo" in logic_lower or "mod" in logic_lower:
        from scipy import stats

        mod_value = parameters.get("modulus", 7)
        observed = [n % mod_value for n in all_numbers]
        expected_prob = 1.0 / mod_value
        observed_counts = np.bincount(observed, minlength=mod_value)
        expected_counts = np.full(mod_value, len(all_numbers) * expected_prob)

        # Use scipy's proper chi-square test
        chi_square, p_value = stats.chisquare(observed_counts, expected_counts)
        effect_size = np.max(np.abs(observed_counts - expected_counts)) / np.mean(expected_counts) if np.mean(expected_counts) > 0 else 0

        return {
            "viable": p_value < 0.05 and abs(effect_size) > 0.1,
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "details": f"Modulo {mod_value} distribution: {observed_counts.tolist()}"
        }
    
    # Gap analysis test
    elif "gap" in logic_lower or "spacing" in logic_lower:
        from scipy import stats

        target_number = parameters.get("target_number", 1)
        gaps = []
        last_pos = None
        for i, draw in enumerate(draws):
            if target_number in draw["numbers"]:
                if last_pos is not None:
                    gaps.append(i - last_pos)
                last_pos = i

        if len(gaps) < 2:
            return {"viable": False, "p_value": 1.0, "effect_size": 0, "details": "Insufficient gaps"}

        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps, ddof=1) if len(gaps) > 1 else 1
        # Expected gap for random occurrence: total_draws / occurrences
        occurrences = len(gaps) + 1
        expected_gap = len(draws) / occurrences if occurrences > 0 else mean_gap

        # Use proper t-test for comparing mean gap to expected
        t_stat, p_value = stats.ttest_1samp(gaps, expected_gap) if std_gap > 0 else (0, 1.0)
        effect_size = abs(mean_gap - expected_gap) / expected_gap if expected_gap > 0 else 0

        return {
            "viable": p_value < 0.05 and abs(effect_size) > 0.1,
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "details": f"Mean gap: {mean_gap:.2f}, Expected: {expected_gap:.2f}"
        }
    
    # Ratio test
    elif "ratio" in logic_lower:
        from scipy import stats

        # Ratio of high:low numbers
        max_num = RANGES[feed_key]["main_max"]
        mid_point = max_num // 2
        high_count = sum(1 for n in all_numbers if n > mid_point)
        low_count = sum(1 for n in all_numbers if n <= mid_point)
        total = high_count + low_count

        # Use proper binomial test
        p_value = stats.binomtest(high_count, total, 0.5, alternative='two-sided').pvalue if total > 0 else 1.0
        expected = total / 2
        effect_size = abs(high_count - expected) / expected if expected > 0 else 0

        return {
            "viable": p_value < 0.05 and abs(effect_size) > 0.1,
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "details": f"High: {high_count}, Low: {low_count}, Ratio: {high_count/low_count if low_count > 0 else 'inf'}"
        }
    
    # Generic fallback: basic distribution test
    else:
        from scipy import stats

        max_num = RANGES[feed_key]["main_max"]
        # Filter to valid range and count
        valid_numbers = [n for n in all_numbers if 1 <= n <= max_num]
        if not valid_numbers:
            return {
                "viable": False,
                "p_value": 1.0,
                "effect_size": 0.0,
                "details": "No valid numbers in range"
            }
        observed_counts = np.bincount(valid_numbers, minlength=max_num + 1)[1:max_num + 1]  # Exactly max_num elements
        expected_count = len(valid_numbers) / max_num
        expected_counts = np.full(max_num, expected_count)

        # Use scipy's proper chi-square test
        chi_square, p_value = stats.chisquare(observed_counts, expected_counts)
        effect_size = np.max(np.abs(observed_counts - expected_counts)) / expected_count if expected_count > 0 else 0

        return {
            "viable": p_value < 0.05 and abs(effect_size) > 0.1,
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "details": f"Custom distribution test across {max_num} numbers"
        }

def run_autonomous_research(feed_key: str) -> Dict[str, Any]:
    """
    Autonomous AI research agent that:
    1. Reviews past 50 analyses
    2. Proposes new pattern hypothesis via Claude
    3. Tests the hypothesis
    4. Reports findings
    5. Stores in research journal
    6. Checks for contradictions
    """
    init_research_db()
    migrate_db()

    try:
        import anthropic
    except ImportError:
        return {
            "status": "error",
            "message": "Anthropic API not available"
        }
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "disabled",
            "message": "AI research disabled. Add ANTHROPIC_API_KEY"
        }
    
    # Get iteration count
    iteration = get_iteration_count(feed_key) + 1

    # Check pursuit state
    pursuit = get_pursuit_state(feed_key)
    in_pursuit_mode = pursuit["is_active"]

    # Get research history for context (increased for better persistence)
    history = get_recent_research(feed_key, limit=100)

    # Get all draws for testing
    draws = get_all_draws(feed_key)
    total_draws = len(draws)

    # VERIFICATION MODE: Test on locked, pre-selected windows for independent verification
    draws_window = draws
    if in_pursuit_mode and pursuit.get("verification_windows"):
        # Use the pre-locked verification window for this attempt
        verification_windows = pursuit["verification_windows"]
        attempt_num = pursuit["pursuit_attempts"]  # 0-indexed

        if 0 <= attempt_num < len(verification_windows):
            window = verification_windows[attempt_num]
            start_date = window["start_date"]
            end_date = window["end_date"]
            filtered_draws = [
                d for d in draws
                if start_date <= d.get("draw_date", "") <= end_date
            ]
            if filtered_draws:
                draws_window = filtered_draws
                print(f"[VERIFICATION] Attempt {attempt_num + 1}/{len(verification_windows)}: Testing on locked window {start_date} to {end_date} ({len(draws_window)} draws)")
            else:
                print(f"[VERIFICATION] No draws in window {start_date} to {end_date}, using all draws")
        else:
            print(f"[VERIFICATION] Attempt {attempt_num} exceeds window count {len(verification_windows)}")
    elif in_pursuit_mode:
        # Fallback: Use most recent 50 draws if no windows available
        draws_window = draws[-50:] if len(draws) > 50 else draws
        print(f"[VERIFICATION] No verification windows available, testing on last {len(draws_window)} draws")

    # === PARTIAL WIN PATTERN ANALYSIS (3+, 4+, 5+ matches) ===
    def count_partial_wins(candidate_numbers, all_draws, min_match=3):
        """
        For a given candidate number set, count how many times it would have matched 3, 4, or 5 numbers in historical draws.
        Returns a dict: {3: count, 4: count, 5: count}
        """
        from collections import Counter
        match_counts = Counter()
        for draw in all_draws:
            draw_nums = set(draw["numbers"])
            matches = len(set(candidate_numbers) & draw_nums)
            if matches >= min_match:
                match_counts[matches] += 1
        return dict(match_counts)

    # Try to extract candidate numbers from parameters if possible
    candidate_numbers = None
    if 'parameters' in locals() and parameters:
        # Try common parameter keys
        for key in ['numbers', 'candidate_numbers', 'pattern_numbers', 'number_set']:
            if key in parameters and isinstance(parameters[key], (list, tuple)):
                candidate_numbers = parameters[key]
                break
    # If not found, try to parse from hypothesis string (e.g., "Pattern: [3, 12, 27, 44, 56]")
    if not candidate_numbers and 'hypothesis_data' in locals() and 'hypothesis' in hypothesis_data:
        import re
        m = re.search(r'\[(\d+(?:,\s*\d+)*)\]', hypothesis_data['hypothesis'])
        if m:
            candidate_numbers = [int(x) for x in m.group(1).split(',')]

    partial_win_stats = {}
    elevated_probability = {}
    if candidate_numbers and len(candidate_numbers) >= 3:
        partial_win_stats = count_partial_wins(candidate_numbers, draws, min_match=3)
        # Calculate expected probability for 3, 4, 5 matches (approximate, for documentation)
        r = None
        from backend.audit import RANGES
        if feed_key in RANGES:
            r = RANGES[feed_key]
        n = len(candidate_numbers)
        total_draws = len(draws)
        # Hypergeometric probability for k matches (approximate, not accounting for bonus balls)
        def hypergeom_prob(N, K, n, k):
            from math import comb
            return comb(K, k) * comb(N-K, n-k) / comb(N, n)
        expected = {}
        if r:
            N = r["main_max"]
            K = n
            n_draw = 5  # Powerball/MM main numbers per draw
            for k in [3,4,5]:
                expected[k] = total_draws * hypergeom_prob(N, K, n_draw, k)
        # Compare observed to expected, flag if observed > 2x expected (arbitrary threshold)
        for k in [3,4,5]:
            obs = partial_win_stats.get(k, 0)
            exp = expected.get(k, 0)
            if exp > 0 and obs > 2*exp:
                elevated_probability[k] = {
                    "observed": obs,
                    "expected": exp,
                    "note": f"Observed {obs} is more than double expected ({exp:.2f}) for {k} matches."
                }


    # Build context summary for Claude
    history_summary = "\n".join([
        f"- Iteration {h['iteration']}: {h['hypothesis']} → p={h['p_value']:.4f}, viable={h['viable']}"
        for h in history[-20:]  # Last 20 (increased for better pattern tracking)
    ])

    if not history_summary:
        history_summary = "No previous research yet. You're starting fresh!"

    # Prompt Claude to propose a new hypothesis
    client = anthropic.Anthropic(api_key=api_key)

    # Load user-suggested research avenues
    user_avenues = load_user_avenues()
    user_avenues_section = ""
    if user_avenues and not in_pursuit_mode:
        avenues_text = "\n".join([
            f"  - **{a['id']}**: {a['question']}\n    Details: {a.get('details', 'N/A')}"
            for a in user_avenues[:3]  # Show top 3 pending
        ])
        user_avenues_section = f"""
🎯 **USER-SUGGESTED RESEARCH AVENUES (HIGH PRIORITY):**
These are statistical questions the user wants investigated. Consider exploring these:

{avenues_text}

If you investigate one of these, design an appropriate test to answer the question.

"""

    # Build pursuit mode instructions if active
    pursuit_instructions = ""
    if in_pursuit_mode:
        pursuit_instructions = f"""🔬 **VERIFICATION MODE ACTIVE** 🔬\n\nYou are currently verifying a CANDIDATE pattern that needs persistence testing.\n\n**PATTERN UNDER INVESTIGATION:**\n- Hypothesis: {pursuit['target_hypothesis']}\n- Test Method: {pursuit['target_test_method']}\n- Parameters: {json.dumps(pursuit['target_parameters'])}\n- Discovery Level: {pursuit['discovery_level']}\n- Attempts so far: {pursuit['pursuit_attempts']}/5\n- Last p-value: {pursuit['last_p_value']:.6f}\n- Last effect size: {pursuit['last_effect_size']:.4f}\n\n**YOUR MISSION FOR THIS ITERATION:**\nRE-TEST THE EXACT SAME PATTERN to check if it persists. Use the same test_method and parameters.\nThis is attempt #{pursuit['pursuit_attempts'] + 1} of verification.\n\n**WHAT TO EXPECT:**\n- If p < 0.05: Pattern still shows significance (continue verification)\n- If p > 0.05: Pattern dissolved (FALSE POSITIVE - verification failed)\n- If persists 3+ times: Pattern becomes VERIFIED ANOMALY\n\nDO NOT deviate from this test. Verification requires consistency."""
    else:
        pursuit_instructions = "**EXPLORATION MODE:** You are free to test any creative pattern hypothesis."

    # In pursuit mode, we don't ask Claude to generate - we use the original pattern and test on different data
    if in_pursuit_mode:
        # Force exact re-test of original pattern
        hypothesis_data = {
            "hypothesis": pursuit['target_hypothesis'],
            "test_method": pursuit['target_test_method'],
            "parameters": json.loads(pursuit['target_parameters']) if isinstance(pursuit['target_parameters'], str) else pursuit['target_parameters'],
            "reasoning": f"Verification attempt {pursuit['pursuit_attempts'] + 1}/5: Re-testing original pattern on independent data window",
            "creativity_score": 5  # Not creative, just rigorous
        }
        # Skip Claude generation entirely in pursuit mode - go straight to testing
        ai_response = json.dumps(hypothesis_data)
    else:
        prompt = f"""You are an autonomous statistical research agent analyzing {feed_key.upper()} lottery data.

{pursuit_instructions}
{user_avenues_section}
**YOUR MISSION:** Propose a NEW, CREATIVE pattern hypothesis to test against the entire history of {total_draws} draws.


**AVAILABLE PRE-BUILT TEST METHODS:**

**Basic Pattern Tests:**
1. digit_ending: Test if numbers ending in X (0-9) appear more/less often - USE THIS for digit frequency patterns! Examples: "numbers ending in 5", "multiples of 5 (ending in 0 or 5)", "low numbers (ending in 1-3)"
2. consecutive_numbers: Test if consecutive numbers cluster
3. sum_range: Test if draw sums favor specific ranges (REQUIRES parameters: {{"low": <int>, "high": <int>}})
    - Example: "parameters": {{"low": 100, "high": 150}}
4. even_odd_imbalance: Test if even/odd distribution is biased
5. bonus_correlation: Test if bonus ball correlates with main numbers

**Advanced Pattern Tests (NEW!):**
6. number_clustering: Test if numbers cluster together more than expected (low spread)
7. streaks: Test if some numbers appear in streaks (consecutive draws) more than random
8. entropy: Test if the entropy of number distribution is lower/higher than expected
9. markov_chain: Test if the appearance of a number depends on the previous draw (first-order Markov property)
10. pairs_triplets: Test if certain pairs or triplets of numbers appear together more than expected
11. positional_bias: Test if certain numbers are more likely to appear in a specific position (e.g., first ball)

**Temporal/Date-Correlation Tests:**
12. day_of_week_bias: Test if SPECIFIC NUMBER appears more on SPECIFIC DAY
13. month_bias: Test if SPECIFIC NUMBER appears more in SPECIFIC MONTH
14. seasonal_bias: Test if SPECIFIC NUMBER appears more in SPECIFIC SEASON
15. weekend_weekday_bias: Test if SPECIFIC NUMBER appears more on weekends vs weekdays
16. temporal_persistence: Test if elevated number frequencies PERSIST over time

**PREFER PRE-BUILT METHODS FIRST** - Only use "custom" if none of the pre-built methods fit your hypothesis!
- If you want to test a specific number frequency: use digit_ending or another pre-built method
- If your pattern fits into these categories, use the pre-built tests - they're optimized and reliable

**OR INVENT YOUR OWN CUSTOM TEST - EXPLORE DIVERSE PATTERN CATEGORIES:**
(Use these ONLY when pre-built methods won't work, and always provide detailed custom_test_logic)

**Category A: Temporal & Sequential Patterns (USE THE NEW DATE-CORRELATION TESTS ABOVE!)**
- Number-specific day-of-week correlations (use day_of_week_bias test!)
- Number-specific seasonal patterns (use seasonal_bias test!)
- Month-specific number biases (use month_bias test!)
- Weekend vs weekday correlations (use weekend_weekday_bias test!)
- Temporal persistence of "hot" numbers (use temporal_persistence test!)
- Draw sequence momentum (do recent trends continue?)

**Category B: Positional & Structural Patterns**
- First ball vs last ball distributions
- Position-specific biases (slot 1 favors low numbers?)
- Spread/variance between min and max in a draw
- Number of "gaps" between consecutive numbers
- Clustering in specific number ranges (1-20 vs 50-69)

**Category C: Mathematical Properties**
- Fibonacci sequence membership
- Perfect squares, cubes, or other powers
- Numbers divisible by specific values (3, 7, 11)
- Digit sum patterns (sum of digits = 7?)
- Palindromic numbers (11, 22, 33)
- Triangular numbers, factorials, other sequences

**Category D: Statistical Anomalies**
- Autocorrelation between consecutive draws
- Runs and streaks (same number appearing multiple draws in row)
- Recency bias (recently drawn numbers reappear faster?)
- Lottery ball "wear" hypothesis (older balls drawn more?)
- Machine/operator effects (different outcomes per draw location)

**Category E: Chaos & Entropy**
- Shannon entropy of number distributions
- Kolmogorov complexity approximations
- Benford's law violations
- Spectral analysis (frequency domain patterns)
- Fractal dimension of draw sequences

---

🔴 **EXPLOIT HUNTER MODE - THINK LIKE A SECURITY AUDITOR** 🔴

You are not just testing random patterns - you are PROBING FOR VULNERABILITIES in the lottery system.
Ask yourself: "How could this system be broken? Where are the weak points?"

**EXPLOIT CATEGORY 1: Draw Mechanics (HIGHEST YIELD)**
- Ball wear degradation: Heavily-used balls vs. rarely-used balls - detection possible!
- Machine mechanism wear: Mechanical arms degrade → detectible frequency shifts
- Ball weight/condition: Older balls processed differently by mechanism
- Temperature effects: Ambient conditions affect drawing consistency
- Pressure/humidity: Environmental factors create measurable anomalies
- Machine aging: Specific draw machines show degradation patterns over months/years

**EXPLOIT CATEGORY 2: Bonus Ball Mechanics**
- Separate drawing mechanism: Powerball/Mega Ball use DIFFERENT equipment
- Independent bias: Bonus drawn separately = independent wear patterns
- Pool differences: Bonus drawn from different numbered ball set (1-26 vs 1-15)
- Mechanism variance: Bonus drawing equipment may age differently
- Frequency anomalies: Bonus numbers show different frequency distribution
- Correlation weakness: Bonus-main correlation reveals mechanism differences

**EXPLOIT CATEGORY 3: Temporal Vulnerabilities**
- Equipment warm-up: First draw of the day different from later draws?
- Seasonal equipment behavior: Temperature/humidity affecting mechanics?
- Maintenance windows: Patterns after equipment servicing?
- Operator fatigue: Late-night draws different from morning draws?
- Draw frequency: Rushed drawings vs. properly-paced ones?

**EXPLOIT CATEGORY 4: Human Factor Exploits**
- Operator patterns: Same operator = same subtle biases?
- Ball loading sequence: Does load order affect selection?
- Verification gaps: Patterns in unverified vs. verified draws?
- Location bias: Different draw locations = different outcomes?
- Procedure drift: Has the process changed over time?

**EXPLOIT CATEGORY 5: Historical Transitions (DOCUMENTABLE EXPLOITS)**
- Equipment changes: Powerball format changed 2015, 2019 - check anomalies!
- Machine replacements: When did specific draw machines get swapped?
- Frequency changes: Rule changes → verification windows with bias possible
- Transition windows: New equipment typically needs calibration period (2-4 weeks)
- Before/after analysis: Compare draw patterns across documented change dates
- Jackpot level changes: Different draw frequency → different wear patterns
- Public records: Illinois Lottery publishes official equipment/rule changes

**YOUR ADVERSARIAL MISSION:**
Think like a penetration tester. Don't just test if patterns exist - test if the SYSTEM HAS FLAWS.
The most valuable finding is not "number 7 is lucky" but "the RNG shows modulo bias on Wednesdays."

---

**YOUR RESEARCH HISTORY (last 20 iterations):**
{history_summary}


**CRITICAL DIVERSITY REQUIREMENTS:**
1. You MUST NOT repeat the same hypothesis or reasoning as any of the last 5 iterations. If you do, your response will be rejected and you must try again.
2. AVOID patterns tested in last 10 iterations - BE RADICALLY DIFFERENT!
3. If you see multiple number theory tests (primes, squares, etc.) in recent history, SWITCH CATEGORIES ENTIRELY
4. Rotate between categories: temporal → positional → mathematical → statistical → chaos
5. If diversity_score would be low, ABANDON your first idea and pick from a different category
6. Make it entertaining - you're performing for a livestream!
7. Every 5 iterations, test something COMPLETELY different from your usual approach"""

    # --- ENFORCE UNIQUE HYPOTHESIS/REASONING (no repeats from last 5) ---
    def is_unique_hypothesis_reasoning(new_hypothesis, new_reasoning, new_method, history, overused):
        # Check for exact hypothesis repeat (expanded to 50 iterations)
        for h in history[-50:]:
            if not h: continue
            if h.get('hypothesis', '').strip() == new_hypothesis.strip():
                return False
            if h.get('ai_reasoning', '').strip() == new_reasoning.strip():
                return False

        # ENFORCE METHOD DIVERSITY - reject if method was used in last 10 iterations
        recent_methods = [h.get('test_method', '') for h in history[-10:] if h]
        if new_method in recent_methods:
            return False

        # BLOCK "custom" method if used too often (catch-all abuse prevention)
        custom_count = sum(1 for h in history[-15:] if h and h.get('test_method', '') == 'custom')
        if new_method == 'custom' and custom_count >= 2:
            return False

        # ENFORCE PATTERN DIVERSITY - reject similar hypothesis patterns
        # These patterns catch variations like "divisible by 7" vs "divisible by 11"
        repetitive_patterns = [
            'divisible by', 'multiple of', 'factor of', 'modulo', 'mod ',
            'ending in', 'ends in', 'digit ',
            'fibonacci', 'prime', 'perfect square',
        ]
        new_hyp_lower = new_hypothesis.lower()

        # Check if we're repeating ANY pattern from the last 50 iterations
        # This catches "multiple of 7", "multiple of 5", "divisible by X", etc.
        for h in history[-50:]:
            if not h: continue
            hist_lower = h.get('hypothesis', '').lower()
            # Check if any repetitive pattern matches between new and historical hypothesis
            for pattern in repetitive_patterns:
                if pattern in new_hyp_lower and pattern in hist_lower:
                    # Both contain the same pattern - likely a repeat of the same family
                    # Additionally check: are they testing the same number in that pattern?
                    # Extract number after pattern if present
                    new_num = new_hyp_lower[new_hyp_lower.find(pattern) + len(pattern):].split()[0] if pattern in new_hyp_lower else ""
                    hist_num = hist_lower[hist_lower.find(pattern) + len(pattern):].split()[0] if pattern in hist_lower else ""
                    if new_num and hist_num and new_num == hist_num:
                        # Same pattern with same number - this is a repeat
                        return False
                    elif not new_num or not hist_num:
                        # Pattern without number (like "fibonacci", "prime") - block if appeared recently
                        if pattern in ['fibonacci', 'prime', 'perfect square']:
                            return False

        # ENFORCE CATEGORY DIVERSITY - reject if same category 2+ times in last 3
        category_keywords = {
            'number_theory': ['prime', 'fibonacci', 'square', 'palindrome', 'divisible', 'digit', 'modulo', 'multiple', 'factor'],
            'temporal': ['day', 'week', 'month', 'year', 'season', 'time', 'weekend', 'weekday', 'temporal', 'date'],
            'positional': ['first', 'last', 'position', 'slot', 'order', 'positional', 'index'],
            'statistical': ['correlation', 'autocorrelation', 'recency', 'streak', 'runs', 'markov', 'entropy', 'frequency', 'distribution'],
            'structural': ['sum', 'range', 'even', 'odd', 'consecutive', 'cluster', 'pair', 'triplet', 'gap', 'spacing'],
            'draw_mechanics': ['ball', 'wear', 'weight', 'degradation', 'machine', 'mechanism', 'physical', 'temperature', 'pressure', 'aging', 'replacement'],
            'bonus_mechanics': ['bonus', 'powerball', 'mega ball', 'mega_ball', 'ball bias', 'separate pool', 'different mechanism'],
            'historical_transitions': ['transition', 'change', 'upgrade', 'replacement', 'rule change', 'equipment change', 'jackpot level', 'period change', 'window']
        }
        new_category = None
        for cat, keywords in category_keywords.items():
            if any(kw in new_hyp_lower for kw in keywords):
                new_category = cat
                break

        if new_category:
            same_cat_count = 0
            for h in history[-15:]:  # Extended from 3 to 15 iterations for better category diversity
                if not h: continue
                hist_lower = h.get('hypothesis', '').lower()
                for cat, keywords in category_keywords.items():
                    if any(kw in hist_lower for kw in keywords):
                        if cat == new_category:
                            same_cat_count += 1
                        break
            if same_cat_count >= 3:  # Block if same category appears 3+ times in last 15
                return False

        return True

    # --- Track recent test methods to enforce diversity ---
    recent_methods = [h.get('test_method', 'unknown') for h in history[-10:]]
    method_counts = {}
    for m in recent_methods:
        method_counts[m] = method_counts.get(m, 0) + 1

    # Find overused methods (used more than 3 times in last 10)
    overused_methods = [m for m, count in method_counts.items() if count >= 3]

    # Find methods NOT used recently (encourage diversity)
    all_methods = ['digit_ending', 'consecutive_numbers', 'sum_range', 'even_odd_imbalance',
                   'bonus_correlation', 'number_clustering', 'streaks', 'entropy',
                   'markov_chain', 'pairs_triplets', 'positional_bias',
                   'day_of_week_bias', 'month_bias', 'seasonal_bias',
                   'weekend_weekday_bias', 'temporal_persistence']
    underused_methods = [m for m in all_methods if m not in recent_methods]

    # --- Detect recent categories to force rotation ---
    category_keywords = {
        'number_theory': ['prime', 'fibonacci', 'square', 'palindrome', 'divisible', 'digit', 'modulo', 'multiple', 'factor'],
        'temporal': ['day', 'week', 'month', 'year', 'season', 'time', 'weekend', 'weekday', 'temporal', 'date'],
        'positional': ['first', 'last', 'position', 'slot', 'order', 'positional', 'index'],
        'statistical': ['correlation', 'autocorrelation', 'recency', 'streak', 'runs', 'markov', 'entropy', 'frequency', 'distribution'],
        'structural': ['sum', 'range', 'even', 'odd', 'consecutive', 'cluster', 'pair', 'triplet', 'gap', 'spacing'],
        'draw_mechanics': ['ball', 'wear', 'weight', 'degradation', 'machine', 'mechanism', 'physical', 'temperature', 'pressure', 'aging', 'replacement'],
        'bonus_mechanics': ['bonus', 'powerball', 'mega ball', 'mega_ball', 'ball bias', 'separate pool', 'different mechanism'],
        'historical_transitions': ['transition', 'change', 'upgrade', 'replacement', 'rule change', 'equipment change', 'jackpot level', 'period change', 'window']
    }
    recent_categories = []
    for h in history[-5:]:
        if not h: continue
        hist_lower = h.get('hypothesis', '').lower()
        for cat, keywords in category_keywords.items():
            if any(kw in hist_lower for kw in keywords):
                recent_categories.append(cat)
                break
    # Find the dominant category
    from collections import Counter
    cat_counts = Counter(recent_categories)
    stuck_category = cat_counts.most_common(1)[0][0] if cat_counts else None
    stuck_count = cat_counts.most_common(1)[0][1] if cat_counts else 0

    # --- AI GENERATION LOOP: retry up to 5 times for uniqueness ---
    max_retries = 5
    for attempt in range(max_retries):
        # In pursuit mode, skip Claude - use original hypothesis set above
        if in_pursuit_mode:
            ai_response = json.dumps(hypothesis_data)
            break  # Only one attempt needed - exact re-test

        # Build context summary for Claude (must be inside loop for freshness)
        history_summary = "\n".join([
            f"- Iteration {h['iteration']}: [{h.get('test_method', '?')}] {h['hypothesis'][:50]}... → p={h['p_value']:.4f}, viable={h['viable']}"
            for h in history[-20:]
        ])
        if not history_summary:
            history_summary = "No previous research yet. You're starting fresh!"

        # Build diversity instructions
        diversity_instructions = ""
        if stuck_count >= 3:
            all_categories = list(category_keywords.keys())
            other_categories = [c for c in all_categories if c != stuck_category]
            diversity_instructions += f"\n🚨 **CRITICAL: SWITCH CATEGORIES NOW!** You've been stuck in '{stuck_category}' for {stuck_count} iterations.\n"
            diversity_instructions += f"**MANDATORY:** Use one of these categories instead: {', '.join(other_categories)}\n"
        if overused_methods:
            diversity_instructions += f"\n⚠️ **AVOID THESE OVERUSED METHODS:** {', '.join(overused_methods)} (used too often recently)\n"
        if underused_methods:
            diversity_instructions += f"\n✅ **TRY THESE UNDERUSED METHODS:** {', '.join(underused_methods[:5])} (not tested recently)\n"
        # Add random seed to encourage variety
        import random
        random_seed = random.randint(1000, 9999)
        diversity_instructions += f"\n🎲 Randomness seed: {random_seed} - Use this to inspire a novel approach!\n"

        # Build pursuit mode instructions if active
        if in_pursuit_mode:
            pursuit_instructions = """**VERIFICATION MODE ACTIVE**\n\nYou are currently verifying a CANDIDATE pattern that needs persistence testing.\n\n**PATTERN UNDER INVESTIGATION:**\n- Hypothesis: {target_hypothesis}\n- Test Method: {target_test_method}\n- Parameters: {target_parameters}\n- Discovery Level: {discovery_level}\n- Attempts so far: {pursuit_attempts}/5\n- Last p-value: {last_p_value:.6f}\n- Last effect size: {last_effect_size:.4f}\n\n**YOUR MISSION FOR THIS ITERATION:**\nRE-TEST THE EXACT SAME PATTERN to check if it persists. Use the same test_method and parameters.\nThis is attempt #{next_attempt} of verification.\n\n**WHAT TO EXPECT:**\n- If p < 0.05: Pattern still shows significance (continue verification)\n- If p > 0.05: Pattern dissolved (FALSE POSITIVE - verification failed)\n- If persists 3+ times: Pattern becomes VERIFIED ANOMALY\n\nDO NOT deviate from this test. Verification requires consistency.""".format(
                target_hypothesis=pursuit['target_hypothesis'],
                target_test_method=pursuit['target_test_method'],
                target_parameters=json.dumps(pursuit['target_parameters']),
                discovery_level=pursuit['discovery_level'],
                pursuit_attempts=pursuit['pursuit_attempts'],
                last_p_value=pursuit['last_p_value'],
                last_effect_size=pursuit['last_effect_size'],
                next_attempt=pursuit['pursuit_attempts'] + 1
            )
        else:
            pursuit_instructions = "**EXPLORATION MODE:** You are free to test any creative pattern hypothesis."

        prompt = f"""You are an autonomous statistical research agent analyzing {feed_key.upper()} lottery data.

{pursuit_instructions}
{diversity_instructions}

**YOUR MISSION:** Propose a NEW, CREATIVE pattern hypothesis to test against the entire history of {total_draws} draws.


**AVAILABLE PRE-BUILT TEST METHODS:**

**Basic Pattern Tests:**
1. digit_ending: Test if numbers ending in X (0-9) appear more/less often - USE THIS for digit frequency patterns! Examples: "numbers ending in 5", "multiples of 5 (ending in 0 or 5)", "low numbers (ending in 1-3)"
2. consecutive_numbers: Test if consecutive numbers cluster
3. sum_range: Test if draw sums favor specific ranges (REQUIRES parameters: {{"low": <int>, "high": <int>}})
    - Example: "parameters": {{"low": 100, "high": 150}}
4. even_odd_imbalance: Test if even/odd distribution is biased
5. bonus_correlation: Test if bonus ball correlates with main numbers

**Advanced Pattern Tests (NEW!):**
6. number_clustering: Test if numbers cluster together more than expected (low spread)
7. streaks: Test if some numbers appear in streaks (consecutive draws) more than random
8. entropy: Test if the entropy of number distribution is lower/higher than expected
9. markov_chain: Test if the appearance of a number depends on the previous draw (first-order Markov property)
10. pairs_triplets: Test if certain pairs or triplets of numbers appear together more than expected
11. positional_bias: Test if certain numbers are more likely to appear in a specific position (e.g., first ball)

**Temporal/Date-Correlation Tests:**
12. day_of_week_bias: Test if SPECIFIC NUMBER appears more on SPECIFIC DAY
13. month_bias: Test if SPECIFIC NUMBER appears more in SPECIFIC MONTH
14. seasonal_bias: Test if SPECIFIC NUMBER appears more in SPECIFIC SEASON
15. weekend_weekday_bias: Test if SPECIFIC NUMBER appears more on weekends vs weekdays
16. temporal_persistence: Test if elevated number frequencies PERSIST over time

**PREFER PRE-BUILT METHODS FIRST** - Only use "custom" if none of the pre-built methods fit your hypothesis!
- If you want to test a specific number frequency: use digit_ending or another pre-built method
- If your pattern fits into these categories, use the pre-built tests - they're optimized and reliable

**OR INVENT YOUR OWN CUSTOM TEST - EXPLORE DIVERSE PATTERN CATEGORIES:**
(Use these ONLY when pre-built methods won't work, and always provide detailed custom_test_logic)

**Category A: Temporal & Sequential Patterns (USE THE NEW DATE-CORRELATION TESTS ABOVE!)**
- Number-specific day-of-week correlations (use day_of_week_bias test!)
- Number-specific seasonal patterns (use seasonal_bias test!)
- Month-specific number biases (use month_bias test!)
- Weekend vs weekday correlations (use weekend_weekday_bias test!)
- Temporal persistence of "hot" numbers (use temporal_persistence test!)
- Draw sequence momentum (do recent trends continue?)

**Category B: Positional & Structural Patterns**
- First ball vs last ball distributions
- Position-specific biases (slot 1 favors low numbers?)
- Spread/variance between min and max in a draw
- Number of "gaps" between consecutive numbers
- Clustering in specific number ranges (1-20 vs 50-69)

**Category C: Mathematical Properties**
- Fibonacci sequence membership
- Perfect squares, cubes, or other powers
- Numbers divisible by specific values (3, 7, 11)
- Digit sum patterns (sum of digits = 7?)
- Palindromic numbers (11, 22, 33)
- Triangular numbers, factorials, other sequences

**Category D: Statistical Anomalies**
- Autocorrelation between consecutive draws
- Runs and streaks (same number appearing multiple draws in row)
- Recency bias (recently drawn numbers reappear faster?)
- Lottery ball "wear" hypothesis (older balls drawn more?)
- Machine/operator effects (different outcomes per draw location)

**Category E: Chaos & Entropy**
- Shannon entropy of number distributions
- Kolmogorov complexity approximations
- Benford's law violations
- Spectral analysis (frequency domain patterns)
- Fractal dimension of draw sequences

**YOUR RESEARCH HISTORY (last 20 iterations):**
{history_summary}

**CRITICAL DIVERSITY REQUIREMENTS:**
1. You MUST NOT repeat the same hypothesis or reasoning as any of the last 5 iterations. If you do, your response will be rejected and you must try again.
2. AVOID patterns tested in last 10 iterations - BE RADICALLY DIFFERENT!
3. If you see multiple number theory tests (primes, squares, etc.) in recent history, SWITCH CATEGORIES ENTIRELY
4. Rotate between categories: temporal → positional → mathematical → statistical → chaos
5. If diversity_score would be low, ABANDON your first idea and pick from a different category
6. Make it entertaining - you're performing for a livestream!
7. Every 5 iterations, test something COMPLETELY different from your usual approach

**OUTPUT FORMAT (STRICT JSON - NO NEWLINES IN STRINGS):**
{{
  "hypothesis": "Clear single-line statement of what you're testing",
  "test_method": "digit_ending|consecutive_numbers|sum_range|even_odd_imbalance|bonus_correlation|day_of_week_bias|month_bias|seasonal_bias|weekend_weekday_bias|temporal_persistence|custom",
    "parameters": {{"digit": 7}} OR {{"low": 100, "high": 150}} OR {{"target_number": 28, "target_day": 2}} OR {{"target_number": 7, "target_month": 7}} OR {{"target_number": 13, "target_season": "winter"}} OR {{}},
  "custom_test_logic": "IF test_method=custom, describe your statistical test methodology in detail",
  "reasoning": "Multi-sentence explanation on a single line. What made you think of this? Why is it interesting? Be entertaining!",
  "iteration": {iteration},
  "next_interval_seconds": 120,
  "interval_reasoning": "Why this timing? Explain your decision.",
  "creativity_score": 7
}}

**⚠️ CRITICAL: CUSTOM TEST RULE:**
- If you choose test_method="custom", you MUST provide the custom_test_logic field
- custom_test_logic should describe the EXACT statistical methodology in a single line
- Example: "Count how many draws have sum > 200 vs sum <= 200, calculate chi-square statistic, report p-value"
- If you cannot describe a clear custom test logic, CHOOSE A PRE-BUILT METHOD INSTEAD!
- Responses with test_method="custom" but NO custom_test_logic will be REJECTED and you'll have to try again

**IMPORTANT NOTES ON TEMPORAL TESTS:**
- When testing if a SPECIFIC NUMBER correlates with dates, use the temporal tests (day_of_week_bias, month_bias, etc.)
- ALWAYS specify target_number in parameters for temporal tests
- For day_of_week_bias: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
- For month_bias: January=1, February=2, ..., December=12
- For seasonal_bias: Use "winter", "spring", "summer", or "fall" (lowercase)
- These tests check if ELEVATED STATISTICAL OCCURRENCES of numbers correlate with draw dates!

**CRITICAL JSON RULES:**
- All string values MUST be on a single line (no line breaks inside strings)
- Escape any quotes inside strings with backslash
- Keep it simple and valid JSON
- Set creativity_score (1-10) based on how novel your hypothesis is
- If test_method="custom", MUST include custom_test_logic field

**CRITICAL: DECIDE YOUR NEXT RESEARCH INTERVAL (30 sec to 30 min)**
- If you found VIABLE pattern: Speed up (60-120s) to verify persistence
- If patterns are STABLE/RANDOM: Slow down (600-1800s) to conserve resources
- If you detected CONTRADICTION: Investigate immediately (30-60s)
- If near draw time window: Active research (120-300s)
- If data is stale (>3 days): Slow pace (900-1800s)

**TIME DILATION PRINCIPLE:** Pattern emergence depends on temporal density. Your cadence = your strategic decision!

Propose your next hypothesis NOW with your chosen interval. Be autonomous and CREATIVE!"""

        # Skip Claude generation entirely if in pursuit mode
        if not in_pursuit_mode:
            # Increase temperature on retries to get more variety
            temp = 0.7 + (attempt * 0.2)  # 0.7, 0.9, 1.1 on retries
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=min(temp, 1.0),  # Cap at 1.0
                messages=[{"role": "user", "content": prompt}]
            )
            ai_response = message.content[0].text.strip()
            try:
                hypothesis_data = json.loads(ai_response)
            except Exception:
                continue  # Retry if JSON parse fails

            # If we got here, we have hypothesis_data from Claude
            new_hypothesis = hypothesis_data.get('hypothesis', '')
            new_reasoning = hypothesis_data.get('reasoning', '')
            new_method = hypothesis_data.get('test_method', 'custom')

            # Check uniqueness
            if not is_unique_hypothesis_reasoning(new_hypothesis, new_reasoning, new_method, history, overused_methods):
                continue  # Retry if not unique
        else:
            # In pursuit mode - hypothesis_data already set
            new_hypothesis = hypothesis_data.get('hypothesis', '')
            new_reasoning = hypothesis_data.get('reasoning', '')
            new_method = hypothesis_data.get('test_method', 'custom')

            # Check diversity - same category repetition is NOT allowed (in exploration mode only)
            new_hyp_lower = new_hypothesis.lower()
            category_keywords = {
                'number_theory': ['prime', 'fibonacci', 'square', 'palindrome', 'divisible', 'digit', 'modulo', 'multiple', 'factor'],
                'temporal': ['day', 'week', 'month', 'year', 'season', 'time', 'weekend', 'weekday', 'temporal', 'date'],
                'positional': ['first', 'last', 'position', 'slot', 'order', 'positional', 'index'],
                'statistical': ['correlation', 'autocorrelation', 'recency', 'streak', 'runs', 'markov', 'entropy', 'frequency', 'distribution'],
                'structural': ['sum', 'range', 'even', 'odd', 'consecutive', 'cluster', 'pair', 'triplet', 'gap', 'spacing']
            }
            new_category = None
            for cat, keywords in category_keywords.items():
                if any(kw in new_hyp_lower for kw in keywords):
                    new_category = cat
                    break

            if new_category:
                same_cat_count = 0
                for h in history[-3:]:  # Check last 3
                    if not h: continue
                    hist_lower = h.get('hypothesis', '').lower()
                    for cat, keywords in category_keywords.items():
                        if any(kw in hist_lower for kw in keywords):
                            if cat == new_category:
                                same_cat_count += 1
                            break

                # STRICT: Reject if same category appears 2+ times in last 3
                if same_cat_count >= 2:
                    if attempt < max_retries - 1:
                        continue  # Retry with different category

        # Quick validation inside loop to enable retry
        # Check if custom test is missing required field
        if hypothesis_data.get("test_method") == "custom":
            if not hypothesis_data.get("custom_test_logic"):
                if attempt < max_retries - 1:
                    continue  # Retry - Claude forgot to provide custom_test_logic

        # DIVERSITY CHECK INSIDE LOOP - can trigger retry
        if not in_pursuit_mode:
            # Detect current hypothesis category
            hypothesis_lower = hypothesis_data.get("hypothesis", "").lower()
            current_category = None

            # Category keywords (must include all exploit types)
            category_keywords = {
                'number_theory': ['prime', 'fibonacci', 'square', 'palindrome', 'divisible', 'digit', 'modulo', 'factorial', 'triangular'],
                'temporal': ['day', 'week', 'month', 'year', 'season', 'time', 'date', 'weekend', 'weekday'],
                'positional': ['first', 'last', 'position', 'slot', 'order', 'sequence', 'spread', 'range'],
                'statistical': ['correlation', 'autocorrelation', 'recency', 'streak', 'runs', 'bias', 'wear'],
                'structural': ['sum', 'range', 'even', 'odd', 'consecutive', 'cluster', 'pair', 'triplet', 'gap', 'spacing'],
                'draw_mechanics': ['ball', 'wear', 'weight', 'degradation', 'machine', 'mechanism', 'physical', 'temperature', 'pressure', 'aging', 'replacement'],
                'bonus_mechanics': ['bonus', 'powerball', 'mega ball', 'mega_ball', 'ball bias', 'separate pool', 'different mechanism'],
                'historical_transitions': ['transition', 'change', 'upgrade', 'replacement', 'rule change', 'equipment change', 'jackpot level', 'period change', 'window']
            }

            for category, keywords in category_keywords.items():
                if any(keyword in hypothesis_lower for keyword in keywords):
                    current_category = category
                    break

            # Check similarity to extended history (last 50 iterations, not just 10)
            # This catches repeats across longer time spans
            same_category_count = 0
            too_similar = False

            for h in history[-50:]:
                hist_lower = h.get("hypothesis", "").lower()

                # Word overlap similarity - lowered threshold from 50% to 30% to catch subtle repeats
                hyp_words = set(hypothesis_lower.split())
                hist_words = set(hist_lower.split())
                if hyp_words and hist_words:
                    overlap = len(hyp_words & hist_words) / len(hyp_words | hist_words)
                    if overlap > 0.3:  # Lowered from 0.5 to catch patterns like "digit 7" variants
                        # Too similar to recent test
                        too_similar = True
                        break

                # Category similarity
                if current_category:
                    hist_category = None
                    for category, keywords in category_keywords.items():
                        if any(keyword in hist_lower for keyword in keywords):
                            hist_category = category
                            break
                    if hist_category == current_category:
                        same_category_count += 1

            # Retry if too similar or same category too often
            if too_similar or same_category_count >= 3:
                if attempt < max_retries - 1:
                    continue  # Retry with different hypothesis
                else:
                    # Max retries hit - ACCEPT anyway with warning (fallback like Test B custom test)
                    # Better to run an imperfect test than fail completely
                    print(f"[DIVERSITY FALLBACK] Max retries hit, accepting hypothesis despite diversity concern")
                    break

        # Accept this response
        break

    # Parse the response (ai_response works for both pursuit and non-pursuit modes)
    try:
        response_text = ai_response

        # Extract JSON from response (Claude might wrap it in markdown)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        # Aggressively clean the JSON before parsing
        # Replace literal newlines and control characters within string values
        def clean_json_string(text):
            """Remove newlines and control chars from JSON string values."""
            result = []
            in_string = False
            escape_next = False
            for i, char in enumerate(text):
                if escape_next:
                    result.append(char)
                    escape_next = False
                elif char == '\\':
                    result.append(char)
                    escape_next = True
                elif char == '"':
                    result.append(char)
                    in_string = not in_string
                elif in_string and char in '\n\r\t':
                    # Remove newlines, carriage returns, and tabs inside strings
                    continue
                else:
                    result.append(char)
            # If we end inside a string, forcibly close it
            if in_string:
                result.append('"')
            return ''.join(result)

        cleaned_text = clean_json_string(response_text)

        # Try parsing the cleaned JSON
        try:
            hypothesis_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Last resort: try with strict=False
            decoder = json.JSONDecoder(strict=False)
            hypothesis_data = decoder.decode(cleaned_text)
    
    except json.JSONDecodeError as e:
        # Return the response so we can see what's wrong
        return {
            "status": "error",
            "message": f"Failed to parse AI hypothesis: {str(e)}",
            "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "hint": "Claude generated invalid JSON. Check for unescaped quotes or newlines in strings"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse AI hypothesis: {str(e)}",
            "raw_response": response_text[:500] if 'response_text' in locals() else "No response"
        }
    
    # PRE-EXECUTION VALIDATION
    validation_errors = []
    
    # Validate required fields
    if not hypothesis_data.get("hypothesis"):
        validation_errors.append("Missing hypothesis field")
    if not hypothesis_data.get("test_method"):
        validation_errors.append("Missing test_method field")
    
    # Calculate diversity metrics for logging (post-acceptance)
    # Diversity check already happened in loop - this is just for display/logging
    diversity_score = 5  # Default
    same_category_count = 0
    current_category = "unknown"
    diversity_warning = ""

    # Recalculate for logging purposes
    hypothesis_lower = hypothesis_data.get("hypothesis", "").lower()
    category_keywords = {
        'number_theory': ['prime', 'fibonacci', 'square', 'palindrome', 'divisible', 'digit', 'modulo', 'factorial', 'triangular'],
        'temporal': ['day', 'week', 'month', 'year', 'season', 'time', 'date', 'weekend', 'weekday'],
        'positional': ['first', 'last', 'position', 'slot', 'order', 'sequence', 'spread', 'range'],
        'statistical': ['correlation', 'autocorrelation', 'recency', 'streak', 'runs', 'bias', 'wear'],
        'structural': ['sum', 'range', 'even', 'odd', 'consecutive', 'cluster', 'pair', 'triplet', 'gap', 'spacing'],
        'draw_mechanics': ['ball', 'wear', 'weight', 'degradation', 'machine', 'mechanism', 'physical', 'temperature', 'pressure', 'aging', 'replacement'],
        'bonus_mechanics': ['bonus', 'powerball', 'mega ball', 'mega_ball', 'ball bias', 'separate pool', 'different mechanism'],
        'historical_transitions': ['transition', 'change', 'upgrade', 'replacement', 'rule change', 'equipment change', 'jackpot level', 'period change', 'window']
    }

    for category, keywords in category_keywords.items():
        if any(keyword in hypothesis_lower for keyword in keywords):
            current_category = category
            break

    for h in history[-50:]:  # Expanded from -10 to catch category repetition over longer spans
        hist_lower = h.get("hypothesis", "").lower()
        if current_category:
            for category, keywords in category_keywords.items():
                if any(keyword in hist_lower for keyword in keywords):
                    if category == current_category:
                        same_category_count += 1
                    break

    if in_pursuit_mode and same_category_count >= 3:
        diversity_warning = f"[VERIFICATION MODE] Testing same {current_category} category {same_category_count}x (allowed for verification)"
    
    # Execute the test
    test_method = hypothesis_data["test_method"]
    parameters = hypothesis_data.get("parameters", {})
    
    # Handle custom test methods
    if test_method == "custom":
        custom_logic = hypothesis_data.get("custom_test_logic", "")
        if not custom_logic:
            # FALLBACK: Convert to a sensible pre-built method based on hypothesis
            hypothesis_lower = hypothesis_data.get("hypothesis", "").lower()
            fallback_method = "digit_ending"  # Default fallback
            fallback_params = {"digit": 5}

            # Detect what the hypothesis is about and choose appropriate fallback
            if any(x in hypothesis_lower for x in ["sum", "total", "range"]):
                fallback_method = "sum_range"
                fallback_params = {"low": 100, "high": 150}
            elif any(x in hypothesis_lower for x in ["even", "odd"]):
                fallback_method = "even_odd_imbalance"
                fallback_params = {}
            elif any(x in hypothesis_lower for x in ["consecutive", "cluster", "spread"]):
                fallback_method = "number_clustering"
                fallback_params = {}
            elif any(x in hypothesis_lower for x in ["streak", "repeat"]):
                fallback_method = "streaks"
                fallback_params = {}
            elif any(x in hypothesis_lower for x in ["entropy", "randomness", "distribution"]):
                fallback_method = "entropy"
                fallback_params = {}

            # Use fallback method
            hypothesis_data["test_method"] = fallback_method
            hypothesis_data["parameters"] = fallback_params
            test_method = fallback_method
            parameters = fallback_params
            print(f"[FALLBACK] Custom test without logic converted to {fallback_method} for: {hypothesis_data.get('hypothesis', '')[:60]}...")
        
        # Execute custom test via AI interpretation (safe fallback)
        try:
            results = execute_custom_test(draws, custom_logic, parameters, feed_key)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Custom test execution failed: {str(e)}",
                "custom_logic": custom_logic
            }
    
    elif test_method not in PATTERN_TESTS:
        return {
            "status": "error",
            "message": f"Unknown test method: {test_method}. Use one of {list(PATTERN_TESTS.keys())} or 'custom'"
        }
    
    else:
        # Run pre-built test with parameters
        test_func = PATTERN_TESTS[test_method]
        
        try:
            # Handle different test methods and their parameters
            if test_method == "digit_ending":
                results = test_func(draws_window, parameters.get("digit", 7), feed_key)
            elif test_method == "sum_range":
                if "low" not in parameters or "high" not in parameters:
                    return {
                        "status": "error",
                        "message": "Test execution failed: Missing required parameter 'low' or 'high' for sum_range test"
                    }
                results = test_func(draws_window, parameters["low"], parameters["high"])
            elif test_method == "bonus_correlation":
                results = test_func(draws_window, feed_key)
            elif test_method == "day_of_week_bias":
                results = test_func(draws_window, parameters.get("target_number", 7), parameters.get("target_day", 0))
            elif test_method == "month_bias":
                results = test_func(draws_window, parameters.get("target_number", 7), parameters.get("target_month", 1))
            elif test_method == "seasonal_bias":
                results = test_func(draws_window, parameters.get("target_number", 7), parameters.get("target_season", "summer"))
            elif test_method == "weekend_weekday_bias":
                results = test_func(draws_window, parameters.get("target_number", 7))
            elif test_method == "temporal_persistence":
                results = test_func(draws_window, parameters.get("target_number", 7), parameters.get("window_size", 30))
            elif test_method == "positional_bias":
                results = test_func(draws_window, parameters.get("position", 0), feed_key)
            elif test_method == "entropy":
                results = test_func(draws_window, feed_key)
            else:
                # Default: tests that only need draws
                results = test_func(draws_window)
        except KeyError as e:
            return {
                "status": "error",
                "message": f"Test execution failed: Missing required parameter {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Test execution failed: {str(e)}"
            }
    
    # Check for contradictions with past research (check last 20 iterations)
    contradicts = ""
    for past in history[-20:]:
        if past["hypothesis"].lower() in hypothesis_data["hypothesis"].lower():
            if past["viable"] != results["viable"]:
                contradicts = f"Contradicts iteration {past['iteration']}"
    
    # Ensure results has required fields with defaults and clean NaN/infinity
    if "p_value" not in results:
        results["p_value"] = 1.0
    else:
        # Clean NaN/infinity from p_value
        p_val = results["p_value"]
        if isinstance(p_val, (float, np.floating)):
            if math.isnan(p_val) or math.isinf(p_val):
                results["p_value"] = 1.0

    if "effect_size" not in results:
        results["effect_size"] = 0.0
    else:
        # Clean NaN/infinity from effect_size
        e_val = results["effect_size"]
        if isinstance(e_val, (float, np.floating)):
            if math.isnan(e_val):
                results["effect_size"] = 0.0
            elif math.isinf(e_val):
                results["effect_size"] = 0.0

    if "viable" not in results:
        results["viable"] = False

    # ENHANCE VERIFICATION MODE REASONING WITH ACTUAL RESULTS
    if in_pursuit_mode:
        viable_status = "✓ SIGNIFICANT" if results["viable"] else "✗ NOT SIGNIFICANT"
        original_hypothesis = pursuit['target_hypothesis']

        # Show what's being tested + verification attempt results
        hypothesis_data["reasoning"] = f"[VERIFICATION] Re-testing: {original_hypothesis[:90]}... | Attempt {pursuit['pursuit_attempts'] + 1}/5: {viable_status} | p={results['p_value']:.6f}, effect={results['effect_size']:.4f}"

    # Track persistence for verification
    persistence_count = track_persistence(feed_key, hypothesis_data["hypothesis"], results["p_value"])

    # Classify the discovery
    discovery = classify_discovery(
        p_value=results["p_value"],
        effect_size=results["effect_size"],
        persistence_count=persistence_count
    )

    # Add discovery to findings for storage and ensure it's clean
    # Clean any NaN/infinity from discovery object
    if isinstance(discovery, dict):
        for key, val in discovery.items():
            if isinstance(val, (float, np.floating)):
                if math.isnan(val) or math.isinf(val):
                    discovery[key] = 0.0 if math.isnan(val) else (1.0 if val > 0 else 0.0)
    results["discovery"] = discovery

    # Log the research iteration
    log_research_iteration(
        feed_key=feed_key,
        iteration=iteration,
        hypothesis=hypothesis_data["hypothesis"],
        test_method=test_method,
        findings=results,
        p_value=results["p_value"],
        effect_size=results["effect_size"],
        viable=results["viable"],
        contradicts=contradicts,
        ai_reasoning=hypothesis_data["reasoning"],
        data_window=f"{total_draws} draws"
    )

    # ===== PURSUIT STATE MANAGEMENT =====
    pursuit_mode_message = ""

    if in_pursuit_mode:
        # We're in verification mode - check if pattern persisted
        if results["p_value"] < 0.05:
            # Pattern still significant - update pursuit state
            update_pursuit(feed_key, results["p_value"], results["effect_size"])
            updated_pursuit = get_pursuit_state(feed_key)

            if updated_pursuit["pursuit_attempts"] >= 5:
                # Max attempts reached
                if discovery["level"] in ["VERIFIED", "LEGENDARY"]:
                    pursuit_mode_message = f"✅ VERIFICATION COMPLETE: Pattern VERIFIED after {updated_pursuit['pursuit_attempts']} tests!"
                    end_pursuit(feed_key, "verified")
                    # ALERT: Pattern verified through pursuit!
                    alert_discovery(
                        feed_key=feed_key,
                        discovery_level=discovery["level"],
                        hypothesis=hypothesis_data["hypothesis"],
                        test_method=test_method,
                        p_value=results["p_value"],
                        effect_size=results["effect_size"],
                        persistence_count=updated_pursuit["pursuit_attempts"],
                        details={"parameters": parameters, "iteration": iteration, "verification": "complete"}
                    )
                else:
                    pursuit_mode_message = f"⏹️ VERIFICATION ENDED: Max attempts (5) reached. Pattern inconclusive."
                    end_pursuit(feed_key, "max_attempts")
            elif discovery["level"] in ["VERIFIED", "LEGENDARY"]:
                # Pattern verified with 3+ persistence
                pursuit_mode_message = f"🚨 PATTERN VERIFIED! Persisted across {persistence_count} tests. Exiting verification mode."
                end_pursuit(feed_key, "verified")
                # ALERT: Pattern verified through persistence!
                alert_discovery(
                    feed_key=feed_key,
                    discovery_level=discovery["level"],
                    hypothesis=hypothesis_data["hypothesis"],
                    test_method=test_method,
                    p_value=results["p_value"],
                    effect_size=results["effect_size"],
                    persistence_count=persistence_count,
                    details={"parameters": parameters, "iteration": iteration, "verification": "persistence"}
                )
            else:
                # Continue verification
                pursuit_mode_message = f"🔬 VERIFICATION MODE: Pattern persists (attempt {updated_pursuit['pursuit_attempts']}/5). Continue testing..."
        else:
            # Pattern dissolved - false positive
            pursuit_mode_message = f"❌ FALSE POSITIVE DETECTED: Pattern dissolved (p={results['p_value']:.4f}). Exiting verification mode."
            end_pursuit(feed_key, "disproven")

    else:
        # Not in pursuit mode - check if we should enter it
        if discovery["level"] == "CANDIDATE":
            # Generate locked verification windows for independent testing
            verification_windows = generate_verification_windows(draws)

            # Start pursuit mode for this candidate pattern
            start_pursuit(
                feed_key=feed_key,
                hypothesis=hypothesis_data["hypothesis"],
                test_method=test_method,
                parameters=parameters,
                discovery_level=discovery["level"],
                current_iteration=iteration,
                p_value=results["p_value"],
                effect_size=results["effect_size"],
                verification_windows=verification_windows
            )
            pursuit_mode_message = "🔶 CANDIDATE DETECTED: Entering VERIFICATION MODE. Next iteration will re-test this pattern."

            # ALERT: Log and notify about the candidate
            alert_discovery(
                feed_key=feed_key,
                discovery_level="CANDIDATE",
                hypothesis=hypothesis_data["hypothesis"],
                test_method=test_method,
                p_value=results["p_value"],
                effect_size=results["effect_size"],
                persistence_count=persistence_count,
                details={"parameters": parameters, "iteration": iteration}
            )

        elif discovery["level"] in ["VERIFIED", "LEGENDARY"]:
            # Immediately verified (rare - would need existing persistence)
            pursuit_mode_message = f"🚨 {discovery['label'].upper()}: Pattern immediately verified!"

            # ALERT: Major discovery - log and notify immediately
            alert_discovery(
                feed_key=feed_key,
                discovery_level=discovery["level"],
                hypothesis=hypothesis_data["hypothesis"],
                test_method=test_method,
                p_value=results["p_value"],
                effect_size=results["effect_size"],
                persistence_count=persistence_count,
                details={"parameters": parameters, "iteration": iteration, "immediate": True}
            )

    # Get final pursuit state for response
    final_pursuit_state = get_pursuit_state(feed_key)

    # AGGRESSIVELY CLEAN all float values before response - walk entire results dict
    def clean_dict_recursive(d):
        """Recursively clean NaN/infinity from dict"""
        if not isinstance(d, dict):
            return d
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, (float, np.floating)):
                try:
                    if math.isnan(v) or math.isinf(v):
                        v = 1.0 if 'p_value' in str(k) else 0.0
                except:
                    v = 0.0
            elif isinstance(v, dict):
                v = clean_dict_recursive(v)
            elif isinstance(v, list):
                cleaned_list = []
                for item in v:
                    if isinstance(item, dict):
                        cleaned_list.append(clean_dict_recursive(item))
                    elif isinstance(item, (float, np.floating)):
                        try:
                            if math.isnan(item) or math.isinf(item):
                                cleaned_list.append(0.0)
                            else:
                                cleaned_list.append(item)
                        except:
                            cleaned_list.append(0.0)
                    else:
                        cleaned_list.append(item)
                v = cleaned_list
            cleaned[k] = v
        return cleaned

    if isinstance(results, dict):
        results = clean_dict_recursive(results)

    # Convert all numpy types to native Python types for JSON serialization
    # Ensure all required fields are present with defaults if missing

    # In verification mode, make sure the original hypothesis is shown
    display_hypothesis = hypothesis_data.get("hypothesis", "No hypothesis generated.")
    if in_pursuit_mode:
        display_hypothesis = pursuit.get('target_hypothesis', display_hypothesis)

    return convert_numpy_types({
        "status": "success",
        "iteration": iteration,
        "hypothesis": display_hypothesis,
        "reasoning": hypothesis_data.get("reasoning", "No reasoning provided."),
        "test_method": test_method or "unknown",
        "results": results or {},
        "viable": results.get("viable", False) if isinstance(results, dict) else False,
        "contradicts": contradicts or "",
        "p_value": results.get("p_value", 1.0) if isinstance(results, dict) else 1.0,
        "effect_size": results.get("effect_size", 0) if isinstance(results, dict) else 0,
        "next_interval_seconds": hypothesis_data.get("next_interval_seconds", 120),
        "interval_reasoning": hypothesis_data.get("interval_reasoning", "Default 2min interval"),
        "data_window": f"{total_draws} draws analyzed",
        "parameters": parameters or {},
        "discovery": discovery or {},
        "creativity_score": hypothesis_data.get("creativity_score", 5),
        "diversity_score": diversity_score if diversity_score is not None else 5,
        "diversity_warning": diversity_warning or "",
        "pattern_category": current_category or "unknown",
        "category_repetition_count": same_category_count if same_category_count is not None else 0,
        "custom_test_logic": hypothesis_data.get("custom_test_logic", None),
        "persistence_count": persistence_count if persistence_count is not None else 0,
        "partial_win_stats": partial_win_stats,
        "elevated_partial_win_probability": elevated_probability,
        "pursuit_mode": {
            "active": final_pursuit_state.get("is_active", False),
            "message": pursuit_mode_message or "",
            "attempts": final_pursuit_state.get("pursuit_attempts", 0) if final_pursuit_state.get("is_active", False) else 0,
            "max_attempts": 5,
            # Include the original hypothesis being verified
            "original_hypothesis": final_pursuit_state.get("target_hypothesis", "") if final_pursuit_state.get("is_active", False) else "",
            "original_test_method": final_pursuit_state.get("target_test_method", "") if final_pursuit_state.get("is_active", False) else "",
            "verification_status": f"Testing pattern from discovery attempt {final_pursuit_state.get('discovery_iteration', '?')}" if final_pursuit_state.get("is_active", False) else ""
        }
    })
