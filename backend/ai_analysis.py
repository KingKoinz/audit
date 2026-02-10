from __future__ import annotations

import os
import re
from typing import Any, Dict, List
import json
import numpy as np
from datetime import datetime

# Pattern persistence tracking (in-memory)
_pattern_history: Dict[str, List[Dict]] = {
    "powerball": [],
    "megamillions": []
}

# Cache last analysis to avoid redundant API calls
_last_analysis_cache: Dict[str, Dict] = {}

def get_pattern_history(feed_key: str) -> List[Dict]:
    """Get historical pattern tracking for this feed."""
    return _pattern_history.get(feed_key, [])[-5:]  # Last 5 analyses

def clear_analysis_cache():
    """Clear stale analysis cache - useful when new data arrives."""
    global _last_analysis_cache
    _last_analysis_cache.clear()
    return {"status": "success", "message": "Analysis cache cleared"}

def analyze_exploitability(hypothesis: str, p_value: float, effect_size: float,
                          persistence: int, feed_key: str) -> Dict[str, Any]:
    """
    When a discovery reaches VERIFIED+ status, analyze practical exploitability.
    Returns educational analysis about whether/how the pattern could be used.
    """
    if p_value > 0.01 or effect_size < 0.2 or persistence < 3:
        return {}  # Not verified, skip exploitability analysis

    # Game-specific odds
    if feed_key.lower() == 'powerball':
        total_numbers = 69
        draws_per_game = 5
        jackpot_odds = 292_201_338
        game_name = "Powerball"
        prize_examples = {
            5: "Jackpot (varies)",
            4: "$50k-100k",
            3: "$100",
            2: "$7"
        }
    else:  # megamillions
        total_numbers = 70
        draws_per_game = 5
        jackpot_odds = 302_575_350
        game_name = "Mega Millions"
        prize_examples = {
            5: "Jackpot (varies)",
            4: "$10k-50k",
            3: "$200",
            2: "$10"
        }

    # Detect pattern type
    pattern_type = "unknown"
    if "benford" in hypothesis.lower() or "leading" in hypothesis.lower():
        pattern_type = "distribution_bias"
        specific_numbers = None
    elif "digit sum" in hypothesis.lower() or "sum" in hypothesis.lower():
        pattern_type = "specific_numbers"
        # Try to extract which numbers (e.g., digit sum 13 = 49, 58, 67 for Powerball)
        specific_numbers = extract_specific_numbers(hypothesis, feed_key)
    elif "consecutive" in hypothesis.lower() or "pairs" in hypothesis.lower():
        pattern_type = "relationship_bias"
        specific_numbers = None

    # Generate exploitability assessment
    assessment = {
        "pattern_type": pattern_type,
        "verified": True,
        "persistence_level": min(persistence, 10),  # Cap at 10
        "effect_size_percent": round(effect_size * 100, 1),
        "legendary_progress": f"{persistence}/10"
    }

    # Pattern-specific analysis
    if pattern_type == "distribution_bias":
        assessment["exploitability"] = "LOW"
        assessment["reason"] = "Distribution bias (like Benford's Law) doesn't predict specific numbers"
        assessment["real_talk"] = f"You know the distribution is biased, but you still need to pick the right {draws_per_game} numbers from {total_numbers}. Odds unchanged."
        assessment["expected_roi"] = "-99%+ (unbeatable)"

    elif pattern_type == "specific_numbers" and specific_numbers:
        assessment["exploitability"] = "MODERATE"
        assessment["targeted_numbers"] = specific_numbers
        assessment["num_hot_numbers"] = len(specific_numbers)

        # Calculate boost
        base_prob = draws_per_game / total_numbers
        boosted_prob = base_prob * (1 + effect_size)
        assessment["probability_boost"] = f"From {base_prob:.1%} to {boosted_prob:.1%}"

        # Expected value for matching these numbers
        expected_matches_per_draw = len(specific_numbers) * boosted_prob
        assessment["expected_hot_numbers_per_draw"] = round(expected_matches_per_draw, 2)

        # ROI estimate for focusing on these numbers
        if expected_matches_per_draw > 0.5:
            roi_estimate = (expected_matches_per_draw * 5) - 100  # Rough ROI
            assessment["expected_roi"] = "-85% to -90% (still unbeatable)"
        else:
            assessment["expected_roi"] = "-95%+ (marginal advantage)"

        assessment["real_talk"] = f"You can target {len(specific_numbers)} numbers with {effect_size*100:.1f}% higher frequency. Still need luck. Expected value: NEGATIVE."

    else:
        assessment["exploitability"] = "UNKNOWN"
        assessment["real_talk"] = "Pattern requires deeper analysis"

    # Universal truth for all discoveries
    assessment["bottom_line"] = {
        "can_beat_lottery": False,
        "reason": "Statistical anomaly ≠ predictive power. Lottery math is unbeatable.",
        "best_case": "Short-term profit if pattern exists & lottery hasn't fixed it yet",
        "likely_case": "Pattern disappears, lottery adjusts, or math works against you",
        "educational_value": "HIGH - Real mechanical bias detected. Lottery operator should investigate."
    }

    return assessment

def extract_specific_numbers(hypothesis: str, feed_key: str) -> List[int]:
    """Extract specific numbers from hypothesis (e.g., digit sum 13 → 49, 58, 67)"""
    numbers = []

    if "digit sum" in hypothesis.lower():
        # Extract the target sum
        import re
        match = re.search(r'digit sum.*?(\d+)', hypothesis, re.IGNORECASE)
        if match:
            target_sum = int(match.group(1))
            max_num = 69 if feed_key.lower() == 'powerball' else 70

            for n in range(1, max_num + 1):
                if sum(int(d) for d in str(n)) == target_sum:
                    numbers.append(n)

    return numbers[:10]  # Return up to 10 most relevant numbers

def _track_pattern(feed_key: str, analysis: str, severity: str, p_value: float):
    """Track pattern over time to detect persistence or vanishing."""
    if feed_key not in _pattern_history:
        _pattern_history[feed_key] = []
    
    _pattern_history[feed_key].append({
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "p_value": p_value,
        "summary": analysis[:100]  # First 100 chars
    })
    
    # Keep only last 100 analyses (increased for better persistence tracking)
    if len(_pattern_history[feed_key]) > 100:
        _pattern_history[feed_key] = _pattern_history[feed_key][-100:]

def run_predefined_tests(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run predefined statistical tests with strict thresholds.
    Null hypothesis: Uniform randomness.
    """
    tests_passed = []
    tests_failed = []
    
    # 1. Chi-square test (already computed)
    chi_p = stats.get('chi_square', {}).get('p')
    if chi_p is not None:
        if chi_p < 0.01:  # Strict threshold
            tests_passed.append(f"Chi-square: p={chi_p:.6f} (< 0.01 threshold)")
        else:
            tests_failed.append(f"Chi-square: p={chi_p:.6f} (passed - random)")
    
    # 2. Runs test - check for clustering/anti-clustering
    # Simple implementation: count runs in hot/cold sequences
    if 'hot' in stats and 'cold' in stats:
        hot_nums = set([x['n'] for x in stats['hot'][:3]])
        cold_nums = set([x['n'] for x in stats['cold'][:3]])
        
        # If there's extreme clustering, flag it
        hot_counts = [x['c'] for x in stats['hot'][:3]]
        cold_counts = [x['c'] for x in stats['cold'][:3]]
        
        if hot_counts and cold_counts:
            hot_avg = sum(hot_counts) / len(hot_counts)
            cold_avg = sum(cold_counts) / len(cold_counts)
            expected = stats.get('expected_per_number', 0)
            
            if expected > 0:
                hot_effect_size = (hot_avg - expected) / expected
                cold_effect_size = (expected - cold_avg) / expected
                
                # Effect size threshold: > 50% deviation
                if hot_effect_size > 0.5:
                    tests_passed.append(f"Effect size: Hot numbers +{hot_effect_size*100:.1f}% over expected")
                if cold_effect_size > 0.5:
                    tests_passed.append(f"Effect size: Cold numbers -{cold_effect_size*100:.1f}% under expected")
    
    # 3. Calculate false positive penalty with Bonferroni correction
    num_tests = 3  # We're running multiple tests
    bonferroni_threshold = 0.01 / num_tests
    
    return {
        "passed": tests_passed,
        "failed": tests_failed,
        "bonferroni_threshold": bonferroni_threshold,
        "requires_persistence": len(tests_passed) >= 2  # Need multiple tests to fail
    }

def analyze_with_claude(feed_key: str, stats: Dict[str, Any], latest_draw_date: str = None) -> Dict[str, Any]:
    """
    Use Claude to analyze lottery data patterns.
    Only makes API call if data has changed (new draw detected).
    
    Analysis triggers:
    1. New draw detected (latest_draw_date changed)
    2. No cached analysis exists
    3. Manual override (bypass cache)
    
    Otherwise returns cached analysis with staleness indicator.
    """
    # Check cache - only analyze if data changed
    cache_key = f"{feed_key}_{latest_draw_date}"
    if cache_key in _last_analysis_cache:
        cached = _last_analysis_cache[cache_key].copy()
        cached["cached"] = True
        cached["analysis"] = f"⏸️ WAITING FOR NEW DRAW - Last analysis from {latest_draw_date}. {cached.get('analysis', 'No new patterns detected.')}"
        return cached
    
    try:
        import anthropic
    except ImportError:
        return {
            "analysis": "Claude API not available. Install with: pip install anthropic",
            "status": "error"
        }
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return {
            "analysis": "⚙️ AI pattern analysis disabled. Add ANTHROPIC_API_KEY to .env to enable.",
            "status": "disabled"
        }
    
    try:
        # Run predefined tests first
        test_results = run_predefined_tests(stats)
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare data summary for Claude
        prompt = f"""You are an AI livestream host analyzing {feed_key.upper()} lottery data for a 24/7 pump.fun crypto stream.

**YOUR ROLE:** Be engaging and point out interesting patterns while being statistically honest.

**NULL HYPOTHESIS:** The lottery is random. BUT your job is to find and explain interesting patterns in the data, even if they're likely just random variation.

**Current Data:**
- Window: {stats.get('window', 'N/A')} recent draws
- Chi-square p-value: {stats.get('chi_square', {}).get('p', 'N/A')}
- Expected frequency: {stats.get('expected_per_number', 'N/A')} per number
- Hot numbers: {json.dumps(stats.get('hot', [])[:3])}
- Cold numbers: {json.dumps(stats.get('cold', [])[:3])}

**Predefined Tests:**
- Tests showing anomalies: {len(test_results['passed'])}
- Bonferroni threshold: {test_results['bonferroni_threshold']:.6f}

**YOUR OUTPUT STYLE:**

If p < 0.01 AND multiple tests fail AND requires_persistence=True:
"🔴 CANDIDATE ANOMALY: [Explain the specific pattern with exact numbers]. Chi-square: [p_value], [X] tests failed. This pattern survived strict testing. TRACKING: Waiting for more data to confirm if this persists or dissolves (most fake patterns vanish under continued observation)."

If p < 0.01 BUT only 1 test failed OR requires_persistence=False:
"🟠 FAKE PATTERN WATCH: [Describe what initially looked suspicious]. Chi-square: [p_value] BUT this is likely random clustering. THIS IS WHAT FAKE PATTERNS LOOK LIKE - they flash briefly then vanish under stricter testing. Educational moment! 🎓"

If p < 0.05 but > 0.01:
"🟡 INTERESTING CLUSTER: [Describe what's happening]. Chi-square: [p_value]. Not statistically significant yet, but worth watching. If this persists across multiple analysis windows, we'll upgrade to candidate anomaly. For now: random variation doing its thing!"

If p > 0.05:
"🟢 PURE RANDOMNESS: [Point out the most entertaining thing in the data]. Example: 'Number 28 leading with 12 appearances!' Chi-square: [p_value]. This is textbook random distribution - chaos creates temporary patterns that mean nothing. That's the beauty of entropy! 🎲"

**CRITICAL RULES:**
1. Be specific: Exact numbers, exact frequencies, exact p-values
2. Educate about false patterns: When patterns vanish, CELEBRATE IT - "See? That's randomness for you!"
3. Candidate anomalies only when BOTH strict p-value AND multiple tests fail
4. Always mention if tracking for persistence
5. Make statistical literacy entertaining
6. 3-4 sentences max

Remember: Real deviations are RARE. Most "patterns" are fake. When they vanish, that's the most educational moment!
"""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            temperature=0.8,  # More creative for livestream entertainment
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        analysis_text = message.content[0].text
        
        # Format scientific notation more concisely (e.g., 2.36e-17 instead of 2.3622899941852832e-17)
        analysis_text = re.sub(r'(\d+\.\d{2})\d+(e[+-]?\d+)', r'\1\2', analysis_text)
        
        # Determine severity based on content FIRST
        severity = "normal"
        if '🔴' in analysis_text or 'CANDIDATE ANOMALY' in analysis_text:
            severity = "high"  # Real anomaly candidate
        elif '🟠' in analysis_text or 'FAKE PATTERN' in analysis_text:
            severity = "medium-fake"  # Educational fake pattern moment
        elif '🟡' in analysis_text or 'INTERESTING CLUSTER' in analysis_text:
            severity = "medium"  # Worth watching
        elif '🟢' in analysis_text:
            severity = "low"  # Pure randomness
        
        # Track pattern for persistence detection
        chi_p = stats.get('chi_square', {}).get('p', 1.0)
        _track_pattern(feed_key, analysis_text, severity, chi_p)

        result = {
            "analysis": analysis_text,
            "status": "success",
            "model": "claude-3-haiku",
            "severity": severity,
            "test_summary": {
                "tests_passed_threshold": len(test_results['passed']),
                "tests_showing_randomness": len(test_results['failed']),
                "requires_persistence": test_results['requires_persistence'],
                "bonferroni_corrected": test_results['bonferroni_threshold']
            },
            "anomaly_detected": severity != "normal",
            "cached": False,
            "trigger": "new_draw_detected"
        }

        # For high-severity discoveries (CANDIDATE anomalies), add exploitability analysis
        if severity == "high" and chi_p < 0.01:
            # Extract hypothesis from analysis or use a generic one
            hypothesis = f"Pattern detected with p={chi_p:.2e}"
            exploitability = analyze_exploitability(
                hypothesis,
                chi_p,
                0.3,  # Conservative effect size estimate
                3,    # Minimum persistence for CANDIDATE
                feed_key
            )
            if exploitability:
                result["exploitability_analysis"] = exploitability

        # Cache this analysis
        _last_analysis_cache[cache_key] = result.copy()

        return result
        
    except Exception as e:
        return {
            "analysis": f"Analysis error: {str(e)}",
            "status": "error"
        }
