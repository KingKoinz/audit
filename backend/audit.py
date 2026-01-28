from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os
import numpy as np

# Simple chi-square implementation (no scipy dependency needed)
def _chisquare(observed, expected):
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    # Degrees of freedom
    df = len(observed) - 1
    # Approximate p-value using normal approximation (good enough for our purpose)
    from math import sqrt, erfc
    p = erfc(sqrt(chi2 / 2))
    return chi2, p

# Default ranges (current game rules; historical rules changed over decades,
# but for visualization/audit we keep it simple and focus on modern-era behavior).
RANGES = {
    "powerball": {"main_min": 1, "main_max": 69, "bonus_min": 1, "bonus_max": 26},
    "megamillions": {"main_min": 1, "main_max": 70, "bonus_min": 1, "bonus_max": 25},
}

def rolling_window(draws_desc: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    return draws_desc[:k]

def freq_counts(nums: List[int], min_n: int, max_n: int) -> List[int]:
    counts = [0] * (max_n - min_n + 1)
    for n in nums:
        if min_n <= n <= max_n:
            counts[n - min_n] += 1
    return counts

def chi_square_uniform(counts: List[int]) -> Dict[str, Any]:
    arr = np.array(counts, dtype=float)
    if arr.sum() == 0:
        return {"chi2": None, "p": None}
    expected = np.ones_like(arr) * (arr.sum() / len(arr))
    chi2, p = _chisquare(arr, expected)
    return {"chi2": float(chi2), "p": float(p)}

def hot_cold_overdue(draws_desc: List[Dict[str, Any]], feed_key: str, window: int = 50) -> Dict[str, Any]:
    r = RANGES[feed_key]
    w = draws_desc[:window]
    main_nums = [n for d in w for n in d["numbers"]]
    counts = freq_counts(main_nums, r["main_min"], r["main_max"])

    # hot/cold based on window counts
    idx_sorted = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    hot = [{"n": i + r["main_min"], "c": counts[i]} for i in idx_sorted[:8]]
    cold = [{"n": i + r["main_min"], "c": counts[i]} for i in idx_sorted[-8:]]

    # overdue: how many draws since last seen (windowed, purely descriptive)
    last_seen = {n: None for n in range(r["main_min"], r["main_max"] + 1)}
    for di, d in enumerate(draws_desc):  # desc order: di=0 is most recent
        for n in d["numbers"]:
            # Only track numbers within current game range (rules changed over time)
            if r["main_min"] <= n <= r["main_max"]:
                if last_seen[n] is None:
                    last_seen[n] = di
    overdue_sorted = sorted(last_seen.items(), key=lambda kv: (kv[1] if kv[1] is not None else 10**9), reverse=True)
    overdue = [{"n": n, "draws_since": (ds if ds is not None else None)} for n, ds in overdue_sorted[:8]]

    # debunk: compare observed window counts to expectation
    # Expectation: each draw picks 5 numbers without replacement from range; approximate expected count:
    # E[count] ≈ window * 5 / range_size
    range_size = (r["main_max"] - r["main_min"] + 1)
    expected = window * 5.0 / range_size
    # A quick z-score style heuristic (not a claim of signal)
    z = []
    for i, c in enumerate(counts):
        # Approx var for binomial approximation: n*p*(1-p) where n=window*5 trials, p=1/range_size
        ntrials = window * 5.0
        p = 1.0 / range_size
        var = ntrials * p * (1 - p)
        zs = 0.0 if var <= 0 else (c - expected) / np.sqrt(var)
        z.append(float(zs))
    # We'll also compute a chi-square p-value for the entire window distribution:
    chi = chi_square_uniform(counts)

    return {
        "window": window,
        "expected_per_number": expected,
        "chi_square": chi,
        "hot": hot,
        "cold": cold,
        "overdue": overdue,
        "z_scores": z,
        "counts": counts,
        "min": r["main_min"],
        "max": r["main_max"],
    }

def heatmap_matrix(draws_desc: List[Dict[str, Any]], feed_key: str, window: int) -> Dict[str, Any]:
    r = RANGES[feed_key]
    w = draws_desc[:window]
    # rows = draws (most recent at top), cols = number range, cell=1 if number appeared in that draw
    cols = list(range(r["main_min"], r["main_max"] + 1))
    mat = []
    dates = []
    for d in w:
        row = [0] * len(cols)
        for n in d["numbers"]:
            if r["main_min"] <= n <= r["main_max"]:
                row[n - r["main_min"]] = 1
        mat.append(row)
        dates.append(d["draw_date"])
    return {"dates": dates, "cols": cols, "matrix": mat}

def monte_carlo_band(draws_desc: List[Dict[str, Any]], feed_key: str, sims: int = 2000) -> Dict[str, Any]:
    r = RANGES[feed_key]
    # Build observed cumulative deviation from expectation across time:
    # For each draw i, compute total count per number so far, compare to expected i*5/range.
    draws = list(reversed(draws_desc))  # oldest->newest
    n_draws = len(draws)
    range_size = (r["main_max"] - r["main_min"] + 1)
    expected_per_draw = 5.0 / range_size

    obs_counts = np.zeros(range_size, dtype=float)
    obs_dev_series = []
    for i, d in enumerate(draws, start=1):
        for n in d["numbers"]:
            if r["main_min"] <= n <= r["main_max"]:
                obs_counts[n - r["main_min"]] += 1.0
        exp_counts = i * expected_per_draw
        # deviation metric: max abs deviation across numbers
        dev = float(np.max(np.abs(obs_counts - exp_counts)))
        obs_dev_series.append(dev)

    # Simulate bands under randomness (approx: draw 5 unique numbers each draw)
    sim_devs = np.zeros((sims, n_draws), dtype=float)
    rng = np.random.default_rng(42)

    for s in range(sims):
        counts = np.zeros(range_size, dtype=float)
        for i in range(n_draws):
            picks = rng.choice(range_size, size=5, replace=False)
            counts[picks] += 1.0
            exp = (i + 1) * expected_per_draw
            sim_devs[s, i] = np.max(np.abs(counts - exp))

    lo = np.quantile(sim_devs, 0.05, axis=0).tolist()
    hi = np.quantile(sim_devs, 0.95, axis=0).tolist()

    return {
        "x": list(range(1, n_draws + 1)),
        "obs": obs_dev_series,
        "band_lo": lo,
        "band_hi": hi,
    }