# Temporal Pattern Analysis - Date/Number Correlation Tests

## What We Added

The AI autonomous research system can now test if **specific numbers** show **elevated statistical occurrences** that **correlate with draw dates** (days of the week, months, seasons, etc.).

## New Test Methods

### 1. Day of Week Correlation (`day_of_week_bias`)
**Tests if a specific number appears more frequently on a specific day of the week.**

**Example Hypothesis:**
- "Number 28 appears more frequently on Wednesdays"
- "Number 7 is drawn more often on Saturdays"

**Parameters:**
```json
{
  "test_method": "day_of_week_bias",
  "parameters": {
    "target_number": 28,
    "target_day": 2
  }
}
```

**Day Codes:**
- 0 = Monday
- 1 = Tuesday
- 2 = Wednesday
- 3 = Thursday
- 4 = Friday
- 5 = Saturday
- 6 = Sunday

**What It Tests:**
- Compares frequency of target number on target day vs all other days
- Chi-square test for statistical significance
- Reports p-value and effect size

---

### 2. Month Correlation (`month_bias`)
**Tests if a specific number appears more frequently in a specific month.**

**Example Hypothesis:**
- "Number 7 appears more often in July"
- "Number 13 is drawn more frequently in October"

**Parameters:**
```json
{
  "test_method": "month_bias",
  "parameters": {
    "target_number": 7,
    "target_month": 7
  }
}
```

**Month Codes:**
- 1 = January
- 2 = February
- ...
- 12 = December

---

### 3. Seasonal Correlation (`seasonal_bias`)
**Tests if a specific number appears more frequently in a specific season.**

**Example Hypothesis:**
- "Number 13 appears more often during winter months"
- "Number 28 is drawn more frequently in summer"

**Parameters:**
```json
{
  "test_method": "seasonal_bias",
  "parameters": {
    "target_number": 13,
    "target_season": "winter"
  }
}
```

**Season Definitions (Northern Hemisphere):**
- `"winter"` = December, January, February
- `"spring"` = March, April, May
- `"summer"` = June, July, August
- `"fall"` = September, October, November

---

### 4. Weekend vs Weekday (`weekend_weekday_bias`)
**Tests if a specific number appears more frequently on weekends vs weekdays.**

**Example Hypothesis:**
- "Number 21 appears more often on weekend draws"
- "Number 42 is drawn more frequently on weekdays"

**Parameters:**
```json
{
  "test_method": "weekend_weekday_bias",
  "parameters": {
    "target_number": 21
  }
}
```

**Weekend Definition:**
- Weekend = Saturday, Sunday
- Weekday = Monday-Friday

---

### 5. Temporal Persistence (`temporal_persistence`)
**Tests if elevated number frequencies persist over time (temporal autocorrelation).**

**Example Hypothesis:**
- "Number 28's elevated frequency persists for 30+ consecutive draws"
- "When number 7 becomes 'hot', it stays hot for extended periods"

**Parameters:**
```json
{
  "test_method": "temporal_persistence",
  "parameters": {
    "target_number": 28,
    "window_size": 30
  }
}
```

**What It Tests:**
- Splits historical data into rolling windows
- Checks if elevated periods cluster together (persistence)
- Reports maximum consecutive elevated streak
- Tests if "hot" numbers stay hot over time

---

## How It Works

### Statistical Methodology

All temporal tests use the same statistical framework:

1. **Split draws by time category** (e.g., Wednesdays vs other days)
2. **Count target number occurrences** in each category
3. **Calculate expected frequency** assuming randomness (uniform distribution)
4. **Chi-square test** to determine if observed differs from expected
5. **Effect size calculation** to measure magnitude of deviation
6. **Viability threshold**: p < 0.01 AND |effect_size| > 0.1

### Example Output

```json
{
  "target_number": 28,
  "target_day": "Wednesday",
  "target_day_occurrences": 45,
  "target_day_expected": 32.5,
  "other_days_occurrences": 124,
  "other_days_expected": 136.5,
  "target_day_frequency": 0.0287,
  "other_days_frequency": 0.0179,
  "chi2": 8.42,
  "p_value": 0.0037,
  "effect_size": 0.385,
  "viable": true
}
```

This would indicate:
- Number 28 appeared 45 times on Wednesdays (expected: 32.5)
- 38.5% elevation above expected
- p-value of 0.0037 (statistically significant)
- **Pattern is viable** - warrants further investigation

---

## How the AI Uses These Tests

The autonomous research AI will now:

1. **Automatically detect temporal patterns** by testing various number/date combinations
2. **Follow up on elevated numbers** (like number 28) with temporal correlation tests
3. **Test persistence** when it finds a "hot" number
4. **Rotate between pattern categories** to ensure diverse testing

### Example AI Research Flow:

1. **Iteration 1**: Detects number 28 has elevated frequency (12 occurrences in last 50 draws)
2. **Iteration 2**: Tests `day_of_week_bias` - "Does number 28 appear more on Wednesdays?"
3. **Iteration 3**: Tests `temporal_persistence` - "Does number 28's hotness persist over time?"
4. **Iteration 4**: Tests `month_bias` - "Does number 28 appear more in January?"
5. **Iteration 5**: Tests `seasonal_bias` - "Does number 28 appear more in winter?"

---

## Why This Matters for Auditing

### What We're Looking For:

**Random Lottery (Expected):**
- No temporal correlations
- Numbers equally likely on all days/months/seasons
- No persistence of elevated frequencies
- All p-values > 0.05

**Suspicious Lottery (Would Trigger Alerts):**
- Specific numbers consistently appear more on specific days
- Strong temporal persistence (hot numbers stay hot)
- Seasonal biases that persist year-over-year
- p-values < 0.01 with large effect sizes

### Real-World Implications:

If we found a pattern like:
- "Number 28 appears 50% more often on Wednesdays" (p < 0.001)
- This pattern persists across 5+ years of data
- Effect size > 0.5 (large magnitude)

This would suggest:
- Possible physical bias (ball wear, machine calibration)
- Drawing procedure irregularities
- Non-random number generation

---

## Technical Details

### Code Location:
- Test implementations: `backend/pattern_tests.py`
- AI prompt updates: `backend/autonomous_research.py`
- Registry: `PATTERN_TESTS` dictionary

### Dependencies:
- NumPy for statistical calculations
- datetime for date parsing
- Chi-square test from `backend.audit._chisquare`

### Date Parsing:
- Draws must have `date` field in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- Uses Python's `datetime.fromisoformat()` for parsing
- Extracts weekday (0-6) and month (1-12) for temporal analysis

---

## Testing It Manually

You can test these patterns manually using the persistence analysis script:

```bash
python analyze_persistence.py
```

This will show:
- Overall frequency of number 28 across all draws
- Recent window statistics
- Persistence across rolling windows
- Verdict on whether it's clustering or bias

---

## Next Steps

The AI will now automatically:
1. Detect elevated number frequencies
2. Test temporal correlations (day, month, season)
3. Check persistence over time
4. Report findings with statistical significance

Monitor the **AI Autonomous Research Lab** section on the dashboard to see these tests in action!
