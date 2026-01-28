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

def classify_discovery(p_value: float, effect_size: float, persistence_count: int = 1) -> Dict[str, Any]:
    """
    Classify a finding based on statistical criteria.
    
    Args:
        p_value: Statistical significance
        effect_size: Magnitude of deviation from expected
        persistence_count: How many consecutive tests showed this pattern
        
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
    if p_value < 0.01 and abs(effect_size) > 0.2 and persistence_count >= 3:
        return {
            "level": "VERIFIED",
            "verified": True,
            "confidence": 99.0,
            **DISCOVERY_LEVELS["VERIFIED"]
        }
    
    # Candidate (needs verification)
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
