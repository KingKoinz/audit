from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from backend.db import init_db, get_recent_draws, get_all_draws, has_any_draws
from backend.audit import hot_cold_overdue, heatmap_matrix, monte_carlo_band
from backend.research_journal import init_research_db
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="ENTROPY Audit")

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.on_event("startup")
async def _startup():
    init_db()
    init_research_db()  # Initialize research journal

    # Auto-ingest data on first startup if database is empty
    if not has_any_draws():
        print("[STARTUP] Database is empty. Auto-ingesting historical lottery data...")
        from backend.ingest import ingest_all
        try:
            summary = await ingest_all()
            print(f"[STARTUP] Auto-ingestion complete: {summary}")
        except Exception as e:
            print(f"[STARTUP] Auto-ingestion failed: {e}")
            print("[STARTUP] You can manually run: ingest.bat")
    else:
        print("[STARTUP] Database has existing data. Skipping auto-ingestion.")

    # Daily auto-update: Check and update hot/cold/overdue for both games
    try:
        print("[STARTUP] Running daily predictions update...")
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        await daily_predictions_update()
        print("[STARTUP] Daily predictions update complete!")
        sys.stdout.flush()
    except Exception as e:
        import traceback
        print(f"[STARTUP] Daily predictions update error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()


async def daily_predictions_update():
    """
    Check both Powerball and Mega Millions for new draws.
    1. Validate any unvalidated past predictions against actual results
    2. Create new prediction snapshot for the NEXT upcoming draw
    Runs on startup - designed to catch up even if app was down for days.
    """
    from backend.research_journal import (
        classify_numbers_hot_cold_overdue,
        save_prediction_snapshot,
        validate_prediction,
        generate_claude_ai_predictions,
        save_ai_prediction,
    )
    from backend.schedule import get_next_draw_time
    from backend.audit import RANGES
    import sqlite3
    from pathlib import Path
    from datetime import datetime, timezone
    import sys

    feeds = ["powerball", "megamillions"]
    db_path = Path("./data/research_journal.sqlite")

    for feed_key in feeds:
        try:
            # Get all draws for this feed
            all_draws = get_all_draws(feed_key)
            if not all_draws:
                print(f"[DAILY-UPDATE] No draws found for {feed_key}")
                sys.stdout.flush()
                continue

            # Sort by date descending to get most recent first
            all_draws_sorted = sorted(all_draws, key=lambda x: x.get("draw_date", ""), reverse=True)

            # Build a dict of draw results by date for quick lookup
            draws_by_date = {d.get("draw_date", ""): d.get("numbers", []) for d in all_draws_sorted}

            print(f"[DAILY-UPDATE] {feed_key.upper()}: Latest draw in DB is {all_draws_sorted[0].get('draw_date', 'unknown')}")
            sys.stdout.flush()

            conn = sqlite3.connect(str(db_path))
            c = conn.cursor()

            # STEP 1: Validate any unvalidated past predictions
            c.execute("""
                SELECT id, draw_date FROM prediction_snapshots
                WHERE feed_key = ? AND validated = 0
                ORDER BY draw_date ASC
            """, (feed_key,))
            unvalidated = c.fetchall()

            for snapshot_id, snapshot_draw_date in unvalidated:
                # Check if we have actual draw results for this date
                # Match by date prefix (YYYY-MM-DD)
                date_prefix = snapshot_draw_date[:10] if snapshot_draw_date else ""
                actual_numbers = None

                for draw_date, numbers in draws_by_date.items():
                    if draw_date.startswith(date_prefix):
                        actual_numbers = numbers
                        break

                if actual_numbers:
                    print(f"[DAILY-UPDATE] Validating snapshot {snapshot_id} for {feed_key} draw {date_prefix}")
                    sys.stdout.flush()
                    result = validate_prediction(feed_key, date_prefix, actual_numbers)
                    if result.get("status") == "validated":
                        print(f"[DAILY-UPDATE] Validated: hot={result['hot_hits']}/{result['hot_in_pool']}, cold={result['cold_hits']}/{result['cold_in_pool']}, overdue={result['overdue_hits']}/{result['overdue_in_pool']}")
                    else:
                        print(f"[DAILY-UPDATE] Validation result: {result}")
                    sys.stdout.flush()

            # STEP 2: Create new snapshot for NEXT upcoming draw
            next_draw_info = get_next_draw_time(feed_key)
            next_draw_date = next_draw_info.get("next_date", "")  # Get YYYY-MM-DD

            if next_draw_date:
                # Check if we already have a snapshot for this upcoming draw
                c.execute("""
                    SELECT id FROM prediction_snapshots
                    WHERE feed_key = ? AND draw_date LIKE ?
                """, (feed_key, f"{next_draw_date}%"))
                existing = c.fetchone()

                if not existing:
                    # Create new snapshot with current hot/cold/overdue
                    max_num = RANGES[feed_key]["main_max"]
                    classifications = classify_numbers_hot_cold_overdue(all_draws_sorted, max_num, lookback=30)

                    snapshot_id = save_prediction_snapshot(
                        feed_key=feed_key,
                        draw_date=next_draw_date,
                        hot_numbers=classifications["hot"],
                        cold_numbers=classifications["cold"],
                        overdue_numbers=classifications["overdue"],
                        lookback_window=30
                    )
                    print(f"[DAILY-UPDATE] Created snapshot {snapshot_id} for {feed_key} next draw {next_draw_date}")
                    print(f"[DAILY-UPDATE] Hot: {classifications['hot'][:5]}... Cold: {classifications['cold'][:5]}... Overdue: {classifications['overdue'][:5]}...")
                    sys.stdout.flush()
                else:
                    print(f"[DAILY-UPDATE] Snapshot already exists for {feed_key} draw {next_draw_date}")
                    sys.stdout.flush()

                # STEP 3: Generate Claude AI predictions for this draw (hot/cold/overdue)
                try:
                    print(f"[DAILY-UPDATE] Generating Claude AI predictions for {feed_key} draw {next_draw_date}...")
                    sys.stdout.flush()

                    ai_predictions = generate_claude_ai_predictions(feed_key, all_draws_sorted, RANGES[feed_key]["main_max"], lookback=100)

                    ai_pred_id = save_ai_prediction(
                        feed_key=feed_key,
                        draw_date=next_draw_date,
                        hot_numbers=ai_predictions.get("hot", []),
                        cold_numbers=ai_predictions.get("cold", []),
                        overdue_numbers=ai_predictions.get("overdue", []),
                        hot_reasoning=ai_predictions.get("hot_reasoning", ""),
                        cold_reasoning=ai_predictions.get("cold_reasoning", ""),
                        overdue_reasoning=ai_predictions.get("overdue_reasoning", "")
                    )

                    print(f"[DAILY-UPDATE] Claude AI Prediction for {feed_key} {next_draw_date}:")
                    print(f"[DAILY-UPDATE]   Hot: {ai_predictions.get('hot', [])} - {ai_predictions.get('hot_reasoning', '')[:80]}")
                    print(f"[DAILY-UPDATE]   Cold: {ai_predictions.get('cold', [])} - {ai_predictions.get('cold_reasoning', '')[:80]}")
                    print(f"[DAILY-UPDATE]   Overdue: {ai_predictions.get('overdue', [])} - {ai_predictions.get('overdue_reasoning', '')[:80]}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[DAILY-UPDATE] Error generating AI predictions for {feed_key}: {e}")
                    sys.stdout.flush()

                # STEP 4: Generate lottery number predictions based on CANDIDATE patterns
                try:
                    from backend.lottery_predictions import get_top_candidate_patterns, synthesize_lottery_prediction, save_lottery_prediction

                    print(f"[DAILY-UPDATE] Generating lottery predictions from CANDIDATE patterns for {feed_key} {next_draw_date}...")
                    sys.stdout.flush()

                    patterns = get_top_candidate_patterns(feed_key, limit=20)
                    if patterns:
                        max_num = RANGES[feed_key]["main_max"]
                        bonus_max = RANGES[feed_key]["bonus_max"]
                        prediction = synthesize_lottery_prediction(feed_key, patterns, max_num, bonus_max)

                        save_lottery_prediction(
                            feed_key=feed_key,
                            draw_date=next_draw_date,
                            numbers=prediction["numbers"],
                            bonus=prediction["bonus"],
                            reasoning=prediction["reasoning"]
                        )

                        print(f"[DAILY-UPDATE] Lottery prediction for {feed_key} {next_draw_date}: {prediction['numbers']} + {prediction['bonus']}")
                        sys.stdout.flush()
                    else:
                        print(f"[DAILY-UPDATE] No CANDIDATE patterns found for {feed_key} yet")
                        sys.stdout.flush()
                except Exception as e:
                    print(f"[DAILY-UPDATE] Error generating lottery predictions for {feed_key}: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()

            conn.close()

        except Exception as e:
            import traceback
            print(f"[DAILY-UPDATE] Error for {feed_key}: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()

    print("[DAILY-UPDATE] Complete for all feeds")
    sys.stdout.flush()
    sys.stderr.flush()

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("frontend/index.html")

@app.get("/Entropy.png")
def entropy_logo():
    return FileResponse("Entropy.png")

@app.get("/entropy-video.mp4")
def entropy_video():
    return FileResponse("Entropy_move.mp4", media_type="video/mp4")

@app.get("/powerball.png")
def powerball_logo():
    return FileResponse("powerball.png")

@app.get("/megamillions.png")
def megamillions_logo():
    return FileResponse(".github/logo_MM_233x110.png")

@app.get("/illinois-lottery.png")
def illinois_logo():
    return FileResponse("illinois-lottery.png")

@app.get("/sw.js")
def service_worker():
    return FileResponse("frontend/sw.js", media_type="application/javascript")

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/recent/{feed_key}")
def recent(feed_key: str):
    rolling = int(os.getenv("ROLLING_DRAWS", "120"))
    draws = get_recent_draws(feed_key, rolling)
    return {"feed": feed_key, "rolling": rolling, "draws": draws}

@app.get("/api/bait/{feed_key}")
def bait(feed_key: str):
    draws = get_recent_draws(feed_key, 5000)
    window = 50
    result = hot_cold_overdue(draws, feed_key, window=window)

    # Add draw metadata
    from backend.db import get_latest_draw_date
    latest_draw = get_latest_draw_date(feed_key)

    # Add Claude analysis with smart caching (only calls API on new draws)
    from backend.ai_analysis import analyze_with_claude, get_pattern_history
    ai_result = analyze_with_claude(feed_key, result, latest_draw_date=latest_draw)
    result["ai_analysis"] = ai_result
    result["latest_draw_date"] = latest_draw
    result["pattern_history"] = get_pattern_history(feed_key)

    # Draw schedule
    import datetime
    from backend.schedule import get_next_draw_time
    next_draw_info = get_next_draw_time(feed_key)
    result["next_draw"] = next_draw_info

    # Auto-snapshot: Create prediction snapshot if next draw is within 2 hours
    try:
        from backend.research_journal import (
            classify_numbers_hot_cold_overdue,
            save_prediction_snapshot,
            init_research_db
        )
        from backend.audit import RANGES
        import sqlite3
        from pathlib import Path

        # Check if next draw is within 2 hours
        next_draw_utc = next_draw_info.get("next_draw_utc")
        if next_draw_utc:
            next_dt = datetime.datetime.fromisoformat(next_draw_utc.replace('Z', '+00:00'))
            now = datetime.datetime.now(datetime.timezone.utc)
            hours_until_draw = (next_dt - now).total_seconds() / 3600

            if 0 < hours_until_draw <= 2:
                # Check if snapshot already exists for this draw
                init_research_db()
                db_path = Path("./data/research_journal.sqlite")
                conn = sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute("""
                    SELECT id FROM prediction_snapshots
                    WHERE feed_key = ? AND draw_date = ?
                """, (feed_key, next_draw_utc))
                existing = c.fetchone()
                conn.close()

                if not existing:
                    # Create snapshot
                    max_num = RANGES[feed_key]["main_max"]
                    all_draws = get_all_draws(feed_key)
                    classifications = classify_numbers_hot_cold_overdue(all_draws, max_num, lookback=30)

                    snapshot_id = save_prediction_snapshot(
                        feed_key=feed_key,
                        draw_date=next_draw_utc,
                        hot_numbers=classifications["hot"],
                        cold_numbers=classifications["cold"],
                        overdue_numbers=classifications["overdue"],
                        lookback_window=30
                    )
                    print(f"[AUTO-SNAPSHOT] Created prediction snapshot {snapshot_id} for {feed_key} draw at {next_draw_utc}")
                    result["prediction_snapshot_created"] = True
    except Exception as e:
        print(f"[AUTO-SNAPSHOT] Error: {e}")

    return {"feed": feed_key, **result}

@app.get("/api/monte/{feed_key}")
def monte(feed_key: str):
    try:
        draws = get_recent_draws(feed_key, 500)
        result = monte_carlo_band(draws, feed_key, sims=500)

        # === Add candidate/verification summary stats ===
        from backend.research_journal import get_recent_research
        history = get_recent_research(feed_key, limit=100)

        # Count candidates - use discovery.level if available, else fallback to p-value check
        candidate_count = 0
        verification_count = 0
        for h in history:
            discovery_level = h.get('discovery', {}).get('level')
            p_val = h.get('p_value', 1.0)
            effect = abs(h.get('effect_size', 0))

            if discovery_level == 'CANDIDATE':
                candidate_count += 1
            elif discovery_level in ('VERIFIED', 'LEGENDARY'):
                verification_count += 1
            elif discovery_level is None:
                # Fallback for old data without discovery field
                if p_val < 0.01 and effect > 0.2:
                    candidate_count += 1

            # Also count pursuit mode as verification
            if h.get('pursuit_mode', {}).get('active'):
                verification_count += 1

        total = len(history) if history else 1
        verif_rate = verification_count / total if total else 0

        # For frontend compatibility, use Mega/Power naming
        extra = {}
        if feed_key == 'megamillions':
            extra = {
                'megaCandidateCount': candidate_count,
                'megaVerifRate': verif_rate
            }
        elif feed_key == 'powerball':
            extra = {
                'powerCandidateCount': candidate_count,
                'powerVerifRate': verif_rate
            }
        return {"feed": feed_key, **result, **extra}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "feed": feed_key}

@app.get("/api/heat/{feed_key}")
def heat(feed_key: str):
    try:
        rolling = int(os.getenv("ROLLING_DRAWS", "120"))
        draws = get_recent_draws(feed_key, rolling)
        if not draws:
            raise ValueError(f"No draws found for feed_key '{feed_key}' (rolling={rolling})")
        heatmap_data = heatmap_matrix(draws, feed_key, window=rolling)
        # Also calculate frequency for the same window
        from backend.audit import hot_cold_overdue
        freq_data = hot_cold_overdue(draws, feed_key, window=rolling)
        return {
            "feed": feed_key, 
            **heatmap_data,
            "counts": freq_data["counts"],
            "min": freq_data["min"],
            "max": freq_data["max"]
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[HEATMAP ERROR] {tb}")
        return {"error": str(e), "traceback": tb, "feed": feed_key}

@app.get("/api/research/{feed_key}")
def research(feed_key: str):
    """
    Autonomous AI research endpoint.
    Proposes and tests new pattern hypotheses for a specific game.
    """
    try:
        from backend.autonomous_research import run_autonomous_research
        from backend.research_journal import get_recent_research
        from backend.game_rotation import record_game_tested

        # Run one research iteration
        result = run_autonomous_research(feed_key)

        # Record that this game was tested
        record_game_tested(feed_key)

        # Get recent history
        history = get_recent_research(feed_key, limit=10)

        # Expose key test execution values for monitoring
        test_values = {
            "iteration": result.get("iteration"),
            "hypothesis": result.get("hypothesis"),
            "test_method": result.get("test_method"),
            "custom_test_logic": result.get("custom_test_logic"),
            "p_value": result.get("p_value"),
            "effect_size": result.get("effect_size"),
            "viable": result.get("viable"),
            "next_interval_seconds": result.get("next_interval_seconds"),
        }
        return {
            "feed": feed_key,
            "current_research": result,
            "recent_history": history,
            "test_values": test_values
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[RESEARCH ERROR] {tb}")
        return {"error": str(e), "traceback": tb, "feed": feed_key}

@app.get("/api/research-auto")
def research_auto():
    """
    Autonomous AI research with automatic game rotation.
    Alternates between Powerball and Mega Millions to keep results separate.
    """
    try:
        from backend.autonomous_research import run_autonomous_research
        from backend.research_journal import get_recent_research
        from backend.game_rotation import get_next_game, record_game_tested

        # Test A: Powerball, Test B: Mega Millions (for contrast)
        feed_A = 'powerball'
        feed_B = 'megamillions'

        print(f"[AUTO-RESEARCH] Running Test A on POWERBALL and Test B on MEGAMILLIONS")

        # Run persistent, independent research for each game
        resultA = run_autonomous_research(feed_A, test_variant='A') if 'test_variant' in run_autonomous_research.__code__.co_varnames else run_autonomous_research(feed_A)
        resultB = run_autonomous_research(feed_B, test_variant='B') if 'test_variant' in run_autonomous_research.__code__.co_varnames else run_autonomous_research(feed_B)

        # Record both games as tested (for rotation logic)
        record_game_tested(feed_A)
        record_game_tested(feed_B)

        # Get recent history for each game
        historyA = get_recent_research(feed_A, limit=5)
        historyB = get_recent_research(feed_B, limit=5)

        # Map feed keys to display names
        game_names = {
            "powerball": "Powerball",
            "megamillions": "Mega Millions"
        }

        # Always return two research results, with error placeholders if needed
        def safe_result(res, idx):
            if not res or res.get('status') == 'error' or 'error' in res:
                return {
                    'status': 'error',
                    'message': res.get('message', res.get('error', f'AI research unavailable (test {idx+1})')) if res else f'AI research unavailable (test {idx+1})',
                    'viable': False
                }
            return res

        rA = safe_result(resultA, 0)
        rB = safe_result(resultB, 1)

        # Expose key test execution values for both tests
        test_values_A = {
            "iteration": rA.get("iteration"),
            "hypothesis": rA.get("hypothesis"),
            "test_method": rA.get("test_method"),
            "custom_test_logic": rA.get("custom_test_logic"),
            "p_value": rA.get("p_value"),
            "effect_size": rA.get("effect_size"),
            "viable": rA.get("viable"),
            "next_interval_seconds": rA.get("next_interval_seconds"),
        }
        test_values_B = {
            "iteration": rB.get("iteration"),
            "hypothesis": rB.get("hypothesis"),
            "test_method": rB.get("test_method"),
            "custom_test_logic": rB.get("custom_test_logic"),
            "p_value": rB.get("p_value"),
            "effect_size": rB.get("effect_size"),
            "viable": rB.get("viable"),
            "next_interval_seconds": rB.get("next_interval_seconds"),
        }
        return {
            "feeds": [feed_A, feed_B],
            "games": [f"{game_names.get(feed_A, feed_A.title())} (Test A)", f"{game_names.get(feed_B, feed_B.title())} (Test B)"],
            "research": [rA, rB],
            "recent_histories": [historyA, historyB],
            "test_values": [test_values_A, test_values_B],
            "error": (rA.get('message') if rA.get('status') == 'error' else '') + (rB.get('message') if rB.get('status') == 'error' else '')
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[AUTO-RESEARCH ERROR] {tb}")
        return {"error": str(e), "traceback": tb}

@app.get("/api/hot-numbers/{feed_key}")
def get_hot_numbers(feed_key: str, window_days: int = 90):
    """
    Generate 'hot numbers' based on recent frequency analysis.

    DISCLAIMER: This is statistical analysis of historical data only.
    It is NOT a prediction system. All lottery numbers have equal
    probability in each draw regardless of past frequency.

    For entertainment and educational purposes only.

    If you or someone you know has a gambling problem, call:
    National Problem Gambling Helpline: 1-800-522-4700
    """
    from backend.hot_numbers import generate_hot_numbers

    try:
        result = generate_hot_numbers(feed_key, window_days=window_days)
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[HOT NUMBERS ERROR] {tb}")
        return {"error": str(e), "traceback": tb}


# ===== PREDICTION TRACKING ENDPOINTS =====

@app.get("/api/predictions/{feed_key}")
def get_predictions(feed_key: str):
    """
    Get prediction tracking statistics and recent results.
    Shows how well hot/cold/overdue classifications predict actual draws.
    """
    from backend.research_journal import (
        get_prediction_stats,
        get_recent_predictions,
        init_research_db
    )

    try:
        init_research_db()  # Ensure tables exist
        stats = get_prediction_stats(feed_key)
        recent = get_recent_predictions(feed_key, limit=10)

        return {
            "feed": feed_key,
            "stats": stats,
            "recent_predictions": recent,
            "interpretation": {
                "hot_edge": stats.get("hot_accuracy", 0) - 33.3,
                "cold_edge": stats.get("cold_accuracy", 0) - 33.3,
                "overdue_edge": stats.get("overdue_accuracy", 0) - 33.3,
                "note": "Positive edge = category outperforming random baseline"
            }
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[PREDICTIONS ERROR] {tb}")
        return {"error": str(e), "feed": feed_key}


@app.get("/api/predictions/snapshot/{feed_key}")
def create_prediction_snapshot(feed_key: str):
    """
    Create a prediction snapshot for the next draw.
    Classifies all numbers as hot/cold/overdue based on current data.
    """
    from backend.research_journal import (
        classify_numbers_hot_cold_overdue,
        save_prediction_snapshot,
        init_research_db
    )
    from backend.audit import RANGES
    from backend.schedule import get_next_draw_time
    import datetime

    try:
        init_research_db()

        # Get all draws
        draws = get_all_draws(feed_key)
        if not draws:
            return {"error": "No draws found", "feed": feed_key}

        max_num = RANGES[feed_key]["main_max"]

        # Classify numbers
        classifications = classify_numbers_hot_cold_overdue(draws, max_num, lookback=30)

        # Get next draw date
        next_draw = get_next_draw_time(feed_key)
        draw_date = next_draw.get("next_draw_utc", datetime.datetime.now().isoformat())

        # Save snapshot
        snapshot_id = save_prediction_snapshot(
            feed_key=feed_key,
            draw_date=draw_date,
            hot_numbers=classifications["hot"],
            cold_numbers=classifications["cold"],
            overdue_numbers=classifications["overdue"],
            lookback_window=30
        )

        return {
            "feed": feed_key,
            "snapshot_id": snapshot_id,
            "draw_date": draw_date,
            "classifications": {
                "hot": classifications["hot"],
                "hot_count": len(classifications["hot"]),
                "cold": classifications["cold"],
                "cold_count": len(classifications["cold"]),
                "overdue": classifications["overdue"],
                "overdue_count": len(classifications["overdue"])
            }
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[SNAPSHOT ERROR] {tb}")
        return {"error": str(e), "feed": feed_key}


@app.get("/api/lottery-prediction/{feed_key}")
def get_lottery_prediction(feed_key: str):
    """
    Get AI-generated lottery number prediction based on viable CANDIDATE patterns.
    Synthesizes top patterns into actual lottery predictions (5 main numbers + bonus).

    DISCLAIMER: For research analysis only. Not gambling advice.
    All lottery numbers have equal probability in each draw.
    """
    from backend.lottery_predictions import (
        get_top_candidate_patterns,
        synthesize_lottery_prediction,
        save_lottery_prediction,
        get_latest_lottery_prediction
    )
    from backend.audit import RANGES
    from backend.schedule import get_next_draw_time
    import datetime

    try:
        # Get feed configuration
        if feed_key not in RANGES:
            return {"error": f"Unknown feed: {feed_key}"}

        max_num = RANGES[feed_key]["main_max"]
        bonus_max = RANGES[feed_key]["bonus_max"]

        # Get next draw date
        next_draw_info = get_next_draw_time(feed_key)
        next_draw_date = next_draw_info.get("next_date", "")

        # Check if we already have a prediction for this draw
        latest_pred = get_latest_lottery_prediction(feed_key)
        if latest_pred and latest_pred.get("draw_date", "").startswith(next_draw_date):
            # Return existing prediction
            return {
                "feed": feed_key,
                "draw_date": next_draw_date,
                "prediction": {
                    "numbers": latest_pred.get("numbers", []),
                    "bonus": latest_pred.get("bonus"),
                    "reasoning": latest_pred.get("reasoning", "")
                },
                "status": "existing"
            }

        # Generate new prediction from CANDIDATE patterns
        patterns = get_top_candidate_patterns(feed_key, limit=20)

        if not patterns:
            return {
                "feed": feed_key,
                "draw_date": next_draw_date,
                "prediction": None,
                "message": "No viable CANDIDATE patterns yet. System still building research..."
            }

        prediction = synthesize_lottery_prediction(feed_key, patterns, max_num, bonus_max)

        # Save prediction
        save_lottery_prediction(
            feed_key=feed_key,
            draw_date=next_draw_date,
            numbers=prediction["numbers"],
            bonus=prediction["bonus"],
            reasoning=prediction["reasoning"]
        )

        return {
            "feed": feed_key,
            "draw_date": next_draw_date,
            "prediction": {
                "numbers": prediction["numbers"],
                "bonus": prediction["bonus"],
                "reasoning": prediction["reasoning"]
            },
            "patterns_used": prediction.get("pattern_count", len(patterns)),
            "status": "generated",
            "disclaimer": "Research analysis only - not gambling advice"
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[LOTTERY PREDICTION ERROR] {tb}")
        return {"error": str(e), "feed": feed_key}


@app.post("/api/predictions/validate/{feed_key}")
def validate_prediction_endpoint(feed_key: str, draw_date: str, numbers: str):
    """
    Validate a prediction against actual draw results.
    numbers should be comma-separated (e.g., "7,12,23,45,67")
    """
    from backend.research_journal import validate_prediction, init_research_db

    try:
        init_research_db()
        actual_numbers = [int(n.strip()) for n in numbers.split(",")]

        result = validate_prediction(feed_key, draw_date, actual_numbers)

        return {
            "feed": feed_key,
            "draw_date": draw_date,
            "actual_numbers": actual_numbers,
            "validation": result
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[VALIDATE ERROR] {tb}")
        return {"error": str(e), "feed": feed_key}


# ===== PREDICTION ARENA (BETTING PREVIEW) =====

@app.get("/bet", response_class=HTMLResponse)
def betting_page():
    """Serve the prediction arena / betting preview page."""
    return FileResponse("frontend/bet.html")


@app.get("/api/arena/schedule")
def arena_schedule():
    """Get next draw times for all games in the arena."""
    from backend.schedule import get_next_draw_time

    return {
        "powerball": get_next_draw_time("powerball"),
        "megamillions": get_next_draw_time("megamillions"),
        # Future: Add more games here
        # "il_pick3": get_next_draw_time("il_pick3"),
    }


# In-memory suggestions storage (for demo - would use DB in production)
_suggestions = [
    {"id": 1, "author": "cryptodev", "text": "Add IL Pick 3 and Pick 4 games for more frequent betting action!", "votes": 12, "timestamp": "2026-01-25"},
    {"id": 2, "author": "degenking", "text": "Parlay bets - combine multiple predictions for bigger multipliers", "votes": 8, "timestamp": "2026-01-26"},
    {"id": 3, "author": "numberswiz", "text": "Show AI's hot/cold classifications directly on betting page", "votes": 6, "timestamp": "2026-01-27"},
]
_suggestion_id = 4


@app.get("/api/arena/suggestions")
def get_suggestions():
    """Get community suggestions for the arena."""
    return {"suggestions": sorted(_suggestions, key=lambda x: x["votes"], reverse=True)}


@app.post("/api/arena/suggestions")
def add_suggestion(author: str = "Anonymous", text: str = ""):
    """Add a new community suggestion."""
    global _suggestion_id

    if not text or len(text.strip()) < 5:
        return {"error": "Suggestion text too short"}

    import datetime
    suggestion = {
        "id": _suggestion_id,
        "author": author.replace("@", "")[:20] or "Anonymous",
        "text": text[:500],
        "votes": 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    _suggestions.append(suggestion)
    _suggestion_id += 1

    return {"success": True, "suggestion": suggestion}


@app.post("/api/arena/suggestions/{suggestion_id}/vote")
def vote_suggestion(suggestion_id: int):
    """Upvote a suggestion."""
    for s in _suggestions:
        if s["id"] == suggestion_id:
            s["votes"] += 1
            return {"success": True, "votes": s["votes"]}
    return {"error": "Suggestion not found"}


# ============== FINDINGS LOG & ALERTS ==============

from backend.alerts import (
    get_findings, acknowledge_finding,
    save_push_subscription, get_push_subscriptions
)

@app.get("/findings", response_class=HTMLResponse)
def findings_page():
    """Serve the findings log page."""
    return FileResponse("frontend/findings.html")


@app.get("/api/findings")
def api_get_findings(
    level: str = None,
    feed_key: str = None,
    limit: int = 100,
    unacknowledged: bool = False
):
    """Get findings from the alert log."""
    findings = get_findings(
        level_filter=level,
        feed_key=feed_key,
        limit=limit,
        unacknowledged_only=unacknowledged
    )
    return {"findings": findings, "count": len(findings)}


@app.post("/api/findings/{finding_id}/acknowledge")
def api_acknowledge_finding(finding_id: int, notes: str = ""):
    """Acknowledge a finding."""
    success = acknowledge_finding(finding_id, notes)
    return {"success": success}


@app.get("/api/findings/stats")
def api_findings_stats():
    """Get findings statistics."""
    all_findings = get_findings(limit=1000)

    stats = {
        "total": len(all_findings),
        "by_level": {},
        "by_feed": {},
        "unacknowledged": 0
    }

    for f in all_findings:
        level = f['discovery_level']
        feed = f['feed_key']

        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
        stats["by_feed"][feed] = stats["by_feed"].get(feed, 0) + 1

        if not f['acknowledged']:
            stats["unacknowledged"] += 1

    return stats


# ============== PUSH NOTIFICATIONS ==============

import os
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")

@app.get("/api/push/vapid-key")
def get_vapid_key():
    """Get the VAPID public key for push subscriptions."""
    return {"publicKey": VAPID_PUBLIC_KEY}


@app.post("/api/push/subscribe")
def subscribe_push(subscription: dict):
    """Subscribe to push notifications."""
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys", {})
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")

    if not all([endpoint, p256dh, auth]):
        return {"error": "Invalid subscription data"}

    success = save_push_subscription(endpoint, p256dh, auth)
    return {"success": success}


@app.delete("/api/push/subscribe")
def unsubscribe_push(endpoint: str):
    """Unsubscribe from push notifications."""
    import sqlite3
    from pathlib import Path
    ALERTS_DB = Path("./data/alerts.sqlite")

    conn = sqlite3.connect(str(ALERTS_DB))
    c = conn.cursor()
    c.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()

    return {"success": deleted}


# ============== MANUAL LOTTERY IMPORT ==============

from pydantic import BaseModel

class CSVImportRequest(BaseModel):
    csv_content: str

@app.post("/api/ingest/manual/{feed_key}")
async def ingest_manual_lottery(feed_key: str, request: CSVImportRequest):
    """
    Manually import lottery results via CSV content.
    Directly inserts into database, bypassing fallback cache.

    CSV Format:
    draw_date,n1,n2,n3,n4,n5,bonus_ball
    2026-01-28,5,12,23,35,42,17
    2026-01-29,8,15,27,39,48,22

    Feed keys: 'powerball' or 'megamillions'
    """
    from backend.lottery_scraper import parse_lottery_csv
    from backend.data_sources import FEEDS
    from backend.db import upsert_draws, init_db
    from backend.research_journal import validate_prediction, init_research_db

    # Validate feed_key
    feed_names = {f.key for f in FEEDS}
    if feed_key not in feed_names:
        return {"error": f"Invalid feed_key. Must be one of: {', '.join(feed_names)}"}

    try:
        # Initialize databases
        init_db()
        init_research_db()

        # Parse CSV
        csv_content = request.csv_content
        draws = parse_lottery_csv(csv_content, feed_key)
        if not draws:
            return {"error": "No valid draws parsed from CSV"}

        # Convert parsed draws to database format
        to_upsert = []
        for draw in draws:
            draw_date = draw.get('draw_date')
            wn = draw.get('winning_numbers', '')
            main_nums = [int(n) for n in wn.split()]
            bonus = draw.get(feed_key)  # Either 'powerball' or 'megamillions' key

            if len(main_nums) >= 5 and bonus:
                to_upsert.append((
                    feed_key,
                    draw_date,
                    main_nums[0], main_nums[1], main_nums[2], main_nums[3], main_nums[4],
                    bonus,
                    ""  # multiplier field
                ))

        # Insert directly into database
        inserted = upsert_draws(to_upsert)
        print(f"[MANUAL-IMPORT] Inserted {inserted} draws for {feed_key}")

        # Auto-validate predictions for the newly inserted draws
        validated_count = 0
        for row in to_upsert:
            feed_key_col, draw_date, n1, n2, n3, n4, n5, bonus, mult = row
            actual_numbers = [n1, n2, n3, n4, n5]
            result = validate_prediction(feed_key_col, draw_date, actual_numbers)
            if result.get("status") == "validated":
                validated_count += 1
                print(f"[MANUAL-IMPORT] Validated prediction for {feed_key_col} {draw_date}")

        return {
            "status": "success",
            "feed_key": feed_key,
            "draws_parsed": len(draws),
            "draws_inserted": inserted,
            "predictions_validated": validated_count
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[MANUAL-IMPORT ERROR] {tb}")
        return {"error": str(e)}


@app.get("/api/ingest/status")
def get_ingest_status():
    """
    Get the status of data sources and cached data.
    Shows which games have fallback data cached.
    """
    from backend.data_sources import _FALLBACK_CACHE, FEEDS

    status = {
        "socrata_fallback_active": bool(_FALLBACK_CACHE),
        "cached_feeds": {},
        "available_feeds": [f.key for f in FEEDS]
    }

    for feed_key, data in _FALLBACK_CACHE.items():
        if data:
            dates = [d.get('draw_date') for d in data if d.get('draw_date')]
            status["cached_feeds"][feed_key] = {
                "count": len(data),
                "latest_date": max(dates) if dates else None,
                "earliest_date": min(dates) if dates else None
            }

    return status