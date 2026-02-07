"""
DISCOVERY VERIFICATION FRAMEWORK

Defines what qualifies as "finding something" and how to verify it.
"""

from typing import Dict, Any
from datetime import datetime

# Discovery severity levels
DISCOVERY_LEVELS = {
    "NOISE": {
        "criteria": "p > 0.05",
        "label": "Random Noise",
        "color": "#888",
        "action": "Discard",
        "emoji": "💨"
    },
    "INTERESTING": {
        "criteria": "0.01 < p < 0.05, effect_size > 0.1",
        "label": "Interesting Cluster",
        "color": "#fbbf24",
        "action": "Monitor",
        "emoji": "👀"
    },
    "CANDIDATE": {
        "criteria": "p < 0.01, effect_size > 0.2, single test",
        "label": "Candidate Anomaly",
        "color": "#ff8c00",
        "action": "Verify with persistence",
        "emoji": "🔶"
    },
    "VERIFIED": {
        "criteria": "p < 0.01, effect_size > 0.2, persists 3+ tests",
        "label": "VERIFIED ANOMALY",
        "color": "#ff4444",
        "action": "Alert & investigate",
        "emoji": "🚨"
    },
    "LEGENDARY": {
        "criteria": "p < 0.001, effect_size > 0.5, persists 10+ tests",
        "label": "LEGENDARY DISCOVERY",
        "color": "#ff00ff",
        "action": "MAJOR ALERT - Possible lottery malfunction",
        "emoji": "👑"
    }
}

def validate_window_consistency(effect_sizes: list, p_values: list, window_sizes: list) -> Dict[str, Any]:
    """
    Validate that a pattern is consistent across verification windows (not inflated in small samples).

    Args:
        effect_sizes: List of effect sizes from each verification window test
        p_values: List of p-values from each verification window test
        window_sizes: List of sample sizes for each window

    Returns:
        Dict with 'valid': bool, 'reason': str, 'metrics': dict
    """
    if len(effect_sizes) < 2:
        return {"valid": True, "reason": "Not enough windows to validate", "metrics": {}}

    # Check effect size stability (should not vary wildly)
    min_effect = min([abs(e) for e in effect_sizes])
    max_effect = max([abs(e) for e in effect_sizes])
    avg_effect = sum([abs(e) for e in effect_sizes]) / len(effect_sizes)

    # Calculate coefficient of variation (CV) for effect size
    if avg_effect > 0:
        effect_cv = ((max_effect - min_effect) / avg_effect) * 100
    else:
        effect_cv = 0

    # Effect size should be consistent (CV < 50% = moderate stability)
    # Higher CV suggests pattern is inflated in specific windows
    effect_stable = effect_cv < 50

    # Check p-value trend (should not improve suspiciously)
    # If p-value keeps getting better, might be p-hacking/overfitting
    p_improving = all(p_values[i] <= p_values[i-1] for i in range(1, len(p_values)))
    p_hacking_risk = p_improving and len(p_values) > 2

    # Check minimum window size (need adequate power)
    min_window_size = min(window_sizes) if window_sizes else 0
    adequate_power = min_window_size >= 5  # At least 5 draws per window

    metrics = {
        "effect_size_cv": round(effect_cv, 1),  # Coefficient of variation
        "effect_size_range": (round(min_effect, 3), round(max_effect, 3)),
        "effect_size_avg": round(avg_effect, 3),
        "p_value_trend": "improving" if p_improving else "variable",
        "min_window_size": min_window_size,
        "num_windows_tested": len(effect_sizes)
    }

    # Determine validity
    reasons = []
    if not effect_stable:
        reasons.append(f"Effect size varies too much (CV={effect_cv:.1f}%)")
    if p_hacking_risk:
        reasons.append("P-value keeps improving (potential p-hacking)")
    if not adequate_power:
        reasons.append(f"Windows too small ({min_window_size} draws minimum needed 5+)")

    if reasons:
        return {
            "valid": False,
            "reason": " | ".join(reasons),
            "metrics": metrics
        }

    return {
        "valid": True,
        "reason": "Pattern consistent across windows",
        "metrics": metrics
    }


def classify_discovery(p_value: float, effect_size: float, persistence_count: int = 1,
                       effect_sizes_history: list = None, p_values_history: list = None,
                       window_sizes_history: list = None) -> Dict[str, Any]:
    """
    Classify a finding based on statistical criteria.

    Args:
        p_value: Statistical significance
        effect_size: Magnitude of deviation from expected
        persistence_count: How many consecutive tests showed this pattern
        effect_sizes_history: Effect sizes from each window test (for validation)
        p_values_history: P-values from each window test (for validation)
        window_sizes_history: Sample sizes for each window

    Returns:
        Discovery classification with presentation metadata
    """
    
    # Legendary (extremely rare - would be headline news)
    if p_value < 0.001 and abs(effect_size) > 0.5 and persistence_count >= 10:
        return {
            "level": "LEGENDARY",
            "verified": True,
            "confidence": 99.9,
            **DISCOVERY_LEVELS["LEGENDARY"]
        }
    
    # Verified (strong evidence with persistence)
    # STRICT: ALL THREE conditions must be true - do not relax any
    if p_value < 0.01 and abs(effect_size) > 0.2 and persistence_count >= 3:
        # If we have window history, validate consistency before marking VERIFIED
        window_validation = None
        if effect_sizes_history and p_values_history and window_sizes_history:
            window_validation = validate_window_consistency(
                effect_sizes_history, p_values_history, window_sizes_history
            )
            # Only mark VERIFIED if window validation passes
            if not window_validation["valid"]:
                # Pattern meets statistical criteria but fails window consistency
                return {
                    "level": "CANDIDATE",
                    "verified": False,
                    "confidence": 95.0,
                    "needs_persistence": True,
                    "validation_issue": window_validation["reason"],
                    **DISCOVERY_LEVELS["CANDIDATE"]
                }

        result = {
            "level": "VERIFIED",
            "verified": True,
            "confidence": 99.0,
            **DISCOVERY_LEVELS["VERIFIED"]
        }
        if window_validation:
            result["window_validation"] = window_validation
        return result
    
    # Candidate (needs verification)
    # Requires: p < 0.01 AND effect_size > 0.2 (effect_size MUST be substantial)
    if p_value < 0.01 and abs(effect_size) > 0.2:
        return {
            "level": "CANDIDATE",
            "verified": False,
            "confidence": 95.0,
            "needs_persistence": True,
            **DISCOVERY_LEVELS["CANDIDATE"]
        }
    
    # Interesting (worth watching)
    if 0.01 < p_value < 0.05 and abs(effect_size) > 0.1:
        return {
            "level": "INTERESTING",
            "verified": False,
            "confidence": 90.0,
            **DISCOVERY_LEVELS["INTERESTING"]
        }

    # Edge case: p < 0.01 but weak effect (should not reach here, but catch it)
    if p_value < 0.01 and abs(effect_size) > 0.1:
        return {
            "level": "INTERESTING",
            "verified": False,
            "confidence": 85.0,
            "note": "Statistically significant but weak effect size - does not meet CANDIDATE threshold",
            **DISCOVERY_LEVELS["INTERESTING"]
        }

    # Noise (random variation)
    return {
        "level": "NOISE",
        "verified": False,
        "confidence": 0.0,
        **DISCOVERY_LEVELS["NOISE"]
    }

def track_persistence(feed_key: str, hypothesis: str, p_value: float) -> int:
    """
    Track how many consecutive tests showed the same pattern.
    Returns persistence count for verification.
    """
    # This would query the research journal for similar hypotheses
    # with significant results in recent history
    from backend.research_journal import get_recent_research
    
    recent = get_recent_research(feed_key, limit=20)
    persistence = 0
    
    # Count consecutive tests with similar hypothesis and p < 0.05
    for r in reversed(recent):
        if hypothesis.lower() in r['hypothesis'].lower() and r['p_value'] < 0.05:
            persistence += 1
        else:
            break  # Pattern broke
    
    return persistence + 1  # Include current test
