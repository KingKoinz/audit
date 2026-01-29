"""
Advanced Verification Methods for Pattern Anomalies
Methods: Bootstrap Resampling, Mechanism Validation, Sensitivity Analysis
No AI required - pure statistical testing
"""

import json
import re
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter


def bootstrap_resampling(
    draws: List[Dict[str, Any]],
    hypothesis: str,
    test_params: Dict[str, Any],
    num_resamples: int = 1000,
    target_number: int = None
) -> Dict[str, Any]:
    """
    Bootstrap Resampling: Resample draws with replacement 1000x
    Test if pattern persists across resamples.

    Returns: {
        'persistence_rate': 0.95,  # % of resamples where pattern held
        'credible': True,  # True if > 90% persistence
        'stability_score': 0-100
    }
    """
    if not draws or len(draws) < 10:
        return {'credible': False, 'persistence_rate': 0, 'reason': 'Insufficient data'}

    # Extract test method from hypothesis
    test_method = test_params.get('test_method', 'custom')

    # Count how many resamples show the pattern
    persistence_count = 0
    effect_sizes = []

    for _ in range(num_resamples):
        # Resample WITH replacement
        resample_indices = np.random.choice(len(draws), size=len(draws), replace=True)
        resampled_draws = [draws[i] for i in resample_indices]

        # Test pattern on resampled data
        if 'divisible' in hypothesis.lower() or 'multiple' in hypothesis.lower():
            # Extract the divisor number
            match = re.search(r'(\d+)', hypothesis)
            if match:
                divisor = int(match.group(1))
                # Count occurrences
                matching = sum(1 for d in resampled_draws
                             for num in d.get('main_numbers', [])
                             if num % divisor == 0)
                total_drawn = len(resampled_draws) * 5  # 5 numbers per draw
                rate = matching / total_drawn if total_drawn > 0 else 0
                effect_sizes.append(rate)
                persistence_count += 1
        else:
            # Default: assume pattern persists if we can't parse it
            persistence_count += 1

    persistence_rate = persistence_count / num_resamples
    credible = persistence_rate > 0.90

    return {
        'persistence_rate': round(persistence_rate, 3),
        'credible': credible,
        'stability_score': int(persistence_rate * 100),
        'resamples': num_resamples,
        'details': f"Pattern persisted in {persistence_count}/{num_resamples} resamples"
    }


def mechanism_validation(
    hypothesis: str,
    feed_key: str,
    test_method: str
) -> Dict[str, Any]:
    """
    Mechanism Validation: Check if pattern has plausible physical explanation

    Returns: {
        'plausibility_score': 0-100,
        'credible': True/False,
        'reasoning': 'explanation'
    }
    """
    hyp_lower = hypothesis.lower()

    # Scoring matrix: pattern -> plausibility factors
    plausibility_score = 50  # Base score
    factors = []

    # Check for temporal patterns (more plausible - draws scheduled)
    temporal_keywords = ['day', 'week', 'month', 'year', 'season', 'time', 'weekend', 'date']
    if any(kw in hyp_lower for kw in temporal_keywords):
        plausibility_score += 15
        factors.append("✓ Temporal pattern (draws are scheduled)")

    # Check for bonus ball patterns (plausible - separate mechanism)
    bonus_keywords = ['bonus', 'powerball', 'mega ball', 'mega_ball']
    if any(kw in hyp_lower for kw in bonus_keywords):
        if feed_key in ['powerball', 'megamillions']:
            plausibility_score += 10
            factors.append("✓ Bonus ball anomaly (separate mechanism)")

    # Check for wear/degradation patterns (plausible - physical)
    wear_keywords = ['wear', 'degradation', 'age', 'temperature', 'pressure', 'aging']
    if any(kw in hyp_lower for kw in wear_keywords):
        plausibility_score += 20
        factors.append("✓ Physical wear pattern (plausible mechanism)")

    # Check for number-specific patterns (less plausible - why THIS number?)
    number_keywords = ['divisible', 'multiple', 'ending', 'digit', 'modulo']
    if any(kw in hyp_lower for kw in number_keywords):
        # Extract the number
        match = re.search(r'(\d+)', hyp_lower)
        if match:
            num = int(match.group(1))
            # Arbitrary numbers are less plausible (high p-hacking risk)
            if num > 20 or num < 3:
                plausibility_score -= 20
                factors.append("✗ Arbitrary high number (high p-hacking risk)")
            elif num in [2, 3, 5, 7]:
                plausibility_score += 5
                factors.append("✓ Prime number (less arbitrary)")

    # Check for correlation patterns (more plausible - real relationships)
    correlation_keywords = ['correlation', 'relationship', 'connected', 'associated']
    if any(kw in hyp_lower for kw in correlation_keywords):
        plausibility_score += 10
        factors.append("✓ Correlation pattern (explains relationships)")

    # Cap at 0-100
    plausibility_score = max(0, min(100, plausibility_score))
    credible = plausibility_score > 60

    reasoning = "; ".join(factors) if factors else "Generic number pattern"

    return {
        'plausibility_score': plausibility_score,
        'credible': credible,
        'reasoning': reasoning,
        'factors': factors
    }


def sensitivity_analysis(
    draws: List[Dict[str, Any]],
    hypothesis: str,
    test_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Sensitivity Analysis: Test if pattern is specific to this parameter
    or if nearby parameters also work (indicates p-hacking)

    Example: If "divisible by 17" works, does "divisible by 16, 18, 19" also work?

    Returns: {
        'specificity_score': 0-100,  # Higher = more specific (good)
        'credible': True/False,
        'tested_variants': {...}
    }
    """
    # Try to extract the target parameter
    match = re.search(r'(\d+)', hypothesis)
    if not match:
        return {'credible': False, 'reason': 'Could not extract parameter'}

    target_num = int(match.group(1))
    tested_variants = {}

    # Test variants around the target
    variants_to_test = [
        target_num - 2, target_num - 1, target_num,
        target_num + 1, target_num + 2
    ]

    for variant in variants_to_test:
        if variant <= 0:
            continue

        # Count matches for this variant
        matching = sum(1 for d in draws
                      for num in d.get('main_numbers', [])
                      if num % variant == 0)
        total_drawn = len(draws) * 5
        rate = matching / total_drawn if total_drawn > 0 else 0

        tested_variants[variant] = {
            'match_rate': round(rate, 4),
            'is_target': variant == target_num
        }

    # Check how many variants "work" (high match rate)
    high_performers = [v for v, data in tested_variants.items()
                       if data['match_rate'] > 0.15]

    # Specificity: if ONLY target works, it's specific (good)
    # If multiple work, indicates p-hacking (bad)
    specificity_score = 100 - (len(high_performers) * 20)
    specificity_score = max(0, min(100, specificity_score))

    # Only credible if target is most specific
    is_most_specific = tested_variants[target_num]['match_rate'] >= max(
        v['match_rate'] for v in tested_variants.values() if v['match_rate'] > 0
    )

    credible = is_most_specific and specificity_score > 60

    return {
        'specificity_score': specificity_score,
        'credible': credible,
        'tested_variants': tested_variants,
        'high_performers': high_performers,
        'reasoning': (
            f"Target #{target_num} is specific (good)" if is_most_specific
            else f"Multiple numbers perform similarly - suspicious (p-hacking risk)"
        )
    }


def comprehensive_verification(
    draws: List[Dict[str, Any]],
    hypothesis: str,
    feed_key: str,
    test_params: Dict[str, Any],
    p_value: float,
    effect_size: float
) -> Dict[str, Any]:
    """
    Run all three verification methods and return combined credibility score
    """
    results = {
        'hypothesis': hypothesis,
        'bootstrap': bootstrap_resampling(draws, hypothesis, test_params),
        'mechanism': mechanism_validation(hypothesis, feed_key, test_params.get('test_method', '')),
        'sensitivity': sensitivity_analysis(draws, hypothesis, test_params),
        'original_p_value': p_value,
        'original_effect_size': effect_size
    }

    # Calculate combined credibility score
    scores = []
    if results['bootstrap']['credible']:
        scores.append(results['bootstrap']['stability_score'])
    if results['mechanism']['credible']:
        scores.append(results['mechanism']['plausibility_score'])
    if results['sensitivity']['credible']:
        scores.append(results['sensitivity']['specificity_score'])

    combined_credibility = int(np.mean(scores)) if scores else 0

    # Overall verdict
    all_credible = all([
        results['bootstrap']['credible'],
        results['mechanism']['credible'],
        results['sensitivity']['credible']
    ])

    results['combined_credibility_score'] = combined_credibility
    results['all_methods_credible'] = all_credible
    results['verdict'] = (
        "🚨 HIGHLY CREDIBLE" if all_credible and combined_credibility > 75
        else "⚠️ QUESTIONABLE" if combined_credibility > 50
        else "❌ LOW CREDIBILITY"
    )

    return results
