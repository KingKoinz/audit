const REFRESH_SECONDS = 120;  // 2 minutes with cheap Haiku model
let DYNAMIC_INTERVAL = 300;  // AI-controlled cadence (starts at 5min, reduces from ~$35/day to ~$2.50/day)

// Track streaming loops to prevent duplicates
const streamingLoops = {};

// 100+ Lottery Myths Database
const LOTTERY_MYTHS = [
  {myth: "Hot numbers are more likely to appear again", reality: "Each draw is independent. Past frequency doesn't affect future odds."},
  {myth: "Overdue numbers are 'due' to appear", reality: "Numbers have no memory. Every draw has equal probability."},
  {myth: "Lucky numbers increase your chances", reality: "All number combinations have exactly the same odds."},
  {myth: "Quick picks have lower odds than manual selection", reality: "Selection method doesn't change probability—every combo is equally likely."},
  {myth: "Patterns like 1-2-3-4-5 never win", reality: "Sequential numbers have the same odds as any other combination."},
  {myth: "Playing the same numbers eventually pays off", reality: "Each draw is independent—past plays don't accumulate advantage."},
  {myth: "Buying more tickets at once is luckier", reality: "More tickets = more chances, but timing doesn't matter."},
  {myth: "Numbers that recently won won't appear again soon", reality: "Past winners have equal probability in every future draw."},
  {myth: "Birthdays are unlucky because everyone uses them", reality: "Odds stay the same, but prizes might split more if you win."},
  {myth: "Certain stores are luckier than others", reality: "Store location has zero effect on random number generation."},
  {myth: "Drawing days affect which numbers appear", reality: "Day of the week doesn't influence RNG equipment."},
  {myth: "Numbers drawn together tend to repeat", reality: "No correlation exists between numbers in different draws."},
  {myth: "Skipping draws increases future odds", reality: "Not playing has no effect on probability when you do play."},
  {myth: "Computer algorithms can predict winners", reality: "True randomness is mathematically unpredictable."},
  {myth: "Odd/even ratios reveal patterns", reality: "Each number's odds are independent of odd/even status."},
  {myth: "Sum ranges indicate winning zones", reality: "All sum values are equally probable across time."},
  {myth: "Low and high number mixing improves odds", reality: "Distribution strategy doesn't change fundamental probability."},
  {myth: "Dreams predict winning numbers", reality: "Coincidence bias makes random matches feel meaningful."},
  {myth: "Lucky charms influence outcomes", reality: "Physical objects can't affect random number generators."},
  {myth: "Buying tickets at specific times helps", reality: "Purchase time has no connection to draw results."},
  {myth: "Repeating digits are less likely to win", reality: "11-22-33-44-55 has identical odds to any other combo."},
  {myth: "Jackpot size affects number distribution", reality: "Prize amount doesn't influence RNG mechanics."},
  {myth: "Playing fewer numbers increases odds", reality: "You must match all required numbers—fewer picks = zero wins."},
  {myth: "Numbers ending in 7 are luckier", reality: "Digit endings have no special properties."},
  {myth: "Avoiding multiples of 5 improves chances", reality: "Arithmetic patterns don't affect probability."},
  {myth: "Cold numbers offer better value", reality: "All numbers have equal expected value over time."},
  {myth: "Tracking trends helps predict future draws", reality: "Past data cannot forecast truly random events."},
  {myth: "Prime numbers win more often", reality: "Mathematical properties don't influence random selection."},
  {myth: "Clusters of consecutive numbers never win", reality: "Consecutive sequences have equal probability."},
  {myth: "Spreading across the range improves odds", reality: "Number distribution strategy is irrelevant to probability."},
  {myth: "Playing near-miss numbers next time works", reality: "Being close doesn't create future advantage."},
  {myth: "Bonus ball patterns indicate main number trends", reality: "Bonus draws are independent random events."},
  {myth: "Symmetrical patterns have lower odds", reality: "Visual patterns don't affect mathematical probability."},
  {myth: "Numbers divisible by 3 cluster together", reality: "Divisibility has no correlation in random draws."},
  {myth: "Full coverage strategies guarantee eventual wins", reality: "You'd spend far more than any jackpot to cover all combos."},
  {myth: "Recent jackpot winners mean lower odds now", reality: "Past winners don't reduce probability for future players."},
  {myth: "Moon phases affect lottery outcomes", reality: "Celestial events have zero impact on RNG systems."},
  {myth: "Playing on your birthday increases luck", reality: "Personal dates have no connection to random draws."},
  {myth: "Numbers from previous week won't repeat", reality: "Each draw is independent—repeats are equally likely."},
  {myth: "Choosing unpopular numbers improves returns", reality: "Odds stay the same, but unpopular numbers might mean less prize splitting."},
  {myth: "Wheeling systems beat the odds", reality: "Wheeling covers more combos but doesn't change base probability."},
  {myth: "Statistical analysis reveals winning formulas", reality: "Randomness has no formula—analysis shows patterns that don't persist."},
  {myth: "Numbers that haven't appeared in years are due", reality: "Long gaps don't create pressure for numbers to appear."},
  {myth: "Playing all odd or all even numbers is bad", reality: "Any valid combination has identical probability."},
  {myth: "Machines have hot and cold cycles", reality: "RNG systems don't have predictable cycles."},
  {myth: "Buying tickets early in the roll is luckier", reality: "Position in ticket printing has no effect on draws."},
  {myth: "Anniversary dates bring good fortune", reality: "Meaningful dates don't influence random outcomes."},
  {myth: "Avoiding past winners makes you more likely to win", reality: "Exclusion strategies don't improve probability."},
  {myth: "Numbers below 31 win less often", reality: "All numbers have equal long-term frequency."},
  {myth: "Random number generators can be hacked", reality: "Modern lottery RNGs are cryptographically secure."},
  {myth: "Playing during off-peak hours improves odds", reality: "Player volume doesn't affect draw mechanics."},
  {myth: "Multiples of 10 rarely appear together", reality: "Multiples have the same odds as non-multiples."},
  {myth: "Certain number combinations are 'prettier' and luckier", reality: "Aesthetic appeal has no correlation with probability."},
  {myth: "Negative visualization prevents wins", reality: "Your thoughts cannot influence random physical processes."},
  {myth: "Lottery is a tax on people who are bad at math", reality: "It's entertainment with known odds—playing for fun is valid."},
  {myth: "Winning twice is statistically impossible", reality: "Unlikely but not impossible—some people have won multiple times."},
  {myth: "Numbers that make geometric patterns win more", reality: "Visual patterns on tickets don't affect random draws."},
  {myth: "Fibonacci sequences have special lottery power", reality: "Mathematical sequences don't influence random selection."},
  {myth: "Picking numbers in ascending order reduces chances", reality: "Order doesn't matter—only the combination itself."},
  {myth: "Numbers that form words on keypad are unlucky", reality: "Keypad layouts have no connection to lottery draws."},
  {myth: "Using significant dates limits your pool badly", reality: "True, but it doesn't change per-ticket probability."},
  {myth: "Professionals use secret systems to win", reality: "No legitimate system exists—winners are always random."},
  {myth: "Numbers with cultural significance perform better", reality: "Cultural meaning doesn't affect random number generation."},
  {myth: "Avoiding mirror numbers improves odds", reality: "Symmetrical properties are irrelevant to probability."},
  {myth: "Playing the same combo for years builds up karma", reality: "Probability has no memory or cosmic justice."},
  {myth: "Computer-generated picks are less random", reality: "Good RNGs produce better randomness than human selection."},
  {myth: "Numbers that add up to specific totals win more", reality: "Sum values follow normal distribution—no magic totals."},
  {myth: "Lottery machines can be influenced by temperature", reality: "Modern RNGs are electronic and environmentally stable."},
  {myth: "Playing opposite of last week's numbers helps", reality: "Inversion strategies don't change probability."},
  {myth: "Numbers drawn in different positions correlate", reality: "Draw position doesn't create relationships between numbers."},
  {myth: "Skipping the first draw after rollover is smart", reality: "Rollover size doesn't affect outcome probability."},
  {myth: "Using numerology increases winning chances", reality: "Mystical number systems have no basis in probability."},
  {myth: "Numbers that rhyme or sound similar cluster", reality: "Phonetic properties don't influence random draws."},
  {myth: "Lottery balls wear out and get drawn less", reality: "Equipment is regularly tested and replaced for fairness."},
  {myth: "Changing your numbers jinxes them", reality: "No such thing as jinxes in random systems."},
  {myth: "Playing during Mercury retrograde is unlucky", reality: "Astrology has no effect on random number generation."},
  {myth: "Numbers that appeared in your fortune cookie will win", reality: "Random suggestions have no special predictive power."},
  {myth: "Visualization techniques manifest winning numbers", reality: "Mental focus cannot alter random physical processes."},
  {myth: "Studying number theory reveals lottery secrets", reality: "Academic math confirms randomness—it doesn't break it."},
  {myth: "Numbers that sound lucky in other languages win", reality: "Linguistic associations have zero impact on draws."},
  {myth: "Lottery outcomes follow Benford's Law", reality: "True randomness actually doesn't follow Benford's distribution."},
  {myth: "Machines located in unlucky buildings perform differently", reality: "Building history has no effect on RNG equipment."},
  {myth: "Playing numbers from recent news events is strategic", reality: "Current events don't influence random draws."},
  {myth: "Numbers that are perfect squares are special", reality: "Mathematical properties don't create lottery advantages."},
  {myth: "Avoiding numbers that lost last time works", reality: "All non-winning numbers have equal odds next time."},
  {myth: "Lottery officials know which numbers will win", reality: "RNG systems ensure even operators cannot predict outcomes."},
  {myth: "Playing during commercials brings luck", reality: "Your viewing habits don't affect random number generation."},
  {myth: "Numbers that appear in pi are more random", reality: "Irrational numbers don't make lottery picks better."},
  {myth: "Syndicates have insider advantages", reality: "Groups just buy more tickets—no special probability boost."},
  {myth: "Numbers matching your license plate are lucky", reality: "Personal identifiers have no connection to lottery RNGs."},
  {myth: "Playing when you're happy increases odds", reality: "Emotional state cannot influence random outcomes."},
  {myth: "Certain ethnic groups are luckier with numbers", reality: "Demographics have absolutely no effect on probability."},
  {myth: "Numbers that form crosses on tickets win more", reality: "Geometric shapes on play slips don't matter."},
  {myth: "Avoiding 13 improves your chances", reality: "Superstition doesn't change mathematical probability."},
  {myth: "Playing on holidays increases luck", reality: "Calendar dates don't affect random number generation."},
  {myth: "Numbers seen repeatedly in daily life are signs", reality: "Confirmation bias makes you notice numbers more."},
  {myth: "Lottery is rigged for government revenue", reality: "Independent audits ensure fairness—transparency is mandatory."},
  {myth: "Using a random generator for picks is cheating luck", reality: "Randomness is randomness—the source doesn't matter."},
  {myth: "Numbers that lost 100 times are finally due", reality: "Gambler's fallacy—each draw is independent."},
  {myth: "Playing fewer lines with more draws helps", reality: "Total tickets matter—timing doesn't change probability."},
  {myth: "Lottery numbers follow quantum mechanics", reality: "Classical probability fully explains lottery outcomes."},
  {myth: "Winners were just more deserving", reality: "Morality and cosmic justice don't influence random draws."},
  {myth: "Playing with found money brings extra luck", reality: "Money source has zero connection to draw results."},
  {myth: "Certain ball machines favor specific numbers", reality: "Machines are rigorously tested for uniform distribution."},
  {myth: "Playing near bankruptcy makes you luckier", reality: "Financial desperation doesn't alter probability."},
  {myth: "Numbers from addresses of past winners are hot", reality: "Winner addresses are random and don't predict anything."},
  {myth: "Using significant life dates creates destiny", reality: "Personal meaning doesn't influence random systems."},
  {myth: "Lottery apps have better odds than retail", reality: "Purchase method doesn't change draw probability."},
  {myth: "Playing the inverse of hot numbers beats the system", reality: "Complementary strategies don't change odds."},
  {myth: "Mathematical prodigies can calculate winning numbers", reality: "Intelligence cannot predict true randomness."},
  {myth: "Lottery curse proves winning is bad luck", reality: "Confirmation bias and poor money management explain 'curses'."},
  {myth: "Playing when the jackpot is huge improves odds", reality: "Jackpot size doesn't change probability of winning."},
  {myth: "Retailers know which tickets will win", reality: "Scratch tickets maybe have patterns, but draw games are pure RNG."},
  {myth: "Numbers that weren't picked by others win more", reality: "Other people's choices don't affect the draw."},
  {myth: "Wearing lucky colors when buying tickets helps", reality: "Clothing choices cannot influence random number generation."},
  {myth: "Lottery outcomes can be predicted with AI", reality: "Machine learning on random data produces random predictions."},
  {myth: "Playing at the stroke of midnight is magical", reality: "Superstitious timing has no effect on draws hours later."},
  {myth: "Numbers that form your age are destined", reality: "Age-based numbers have no special lottery properties."}
];

const VIEWS = [
  { key: 'heat', title: 'Heatmap (Rolling Draw Presence)' },
  { key: 'hist', title: 'Histogram (Rolling Frequency)' },
];

// ===== ROTATING AI HEADLINES =====
const TITLES = [
  // Core
  "Can lottery randomness be cracked?",
  "Can randomness fail under scrutiny?",
  "Is randomness as random as we think?",
  "What would a crack in randomness look like?",
  "If randomness broke, would we see it?",
  "Assuming randomness. Attempting to disprove.",
  "Patterns detected. Tested. Rejected.",
  "Most patterns fail inspection.",
  "Noise creates convincing illusions.",
  "Randomness doesn't explain itself.",

  // Mega Millions
  "Can Mega Millions randomness be disproved?",
  "Does Mega Millions behave like true randomness?",
  "Are Mega Millions streaks statistically meaningful?",
  "What patterns should exist in Mega Millions?",
  "Is Mega Millions indistinguishable from noise?",
  "Can Mega Millions fail entropy tests?",
  "How often should Mega Millions appear anomalous?",
  "Does Mega Millions enter verification unusually often?",
  "Are Mega Millions clusters inevitable?",
  "What would a real Mega Millions anomaly look like?",

  // Powerball
  "Can Powerball randomness be falsified?",
  "Does Powerball deviate from expected entropy?",
  "Are Powerball streaks meaningful or inevitable?",
  "What would a Powerball anomaly look like?",
  "How often should Powerball trigger verification?",
  "Is Powerball statistically noisier than expected?",
  "Does Powerball exhibit excess clustering?",
  "Are Powerball deviations persistent or transient?",
  "What patterns should Powerball never show?",
  "Does Powerball behave like a fair system?",

  // System / audit tone
  "Testing randomness in public.",
  "Live audit of lottery entropy.",
  "All apparent signals undergo verification.",
  "Verification precedes interpretation.",
  "Extraordinary claims require extraordinary evidence.",
  "If this worked, it wouldn't be quiet.",
  "The null hypothesis is stubborn.",
  "Randomness is boring. That's the point.",
  "Silence is expected.",
  "Entropy resists intuition."
];

let titleIndex = Math.floor(Math.random() * TITLES.length);
let verificationActive = false;

function rotateTitle() {
  if (verificationActive) return;
  const titleEl = document.getElementById("ai-title");
  if (!titleEl) return;

  titleEl.classList.add("fade");
  setTimeout(() => {
    titleIndex = (titleIndex + 1) % TITLES.length;
    titleEl.textContent = TITLES[titleIndex];
    titleEl.classList.remove("fade");
  }, 600);
}

function enterVerificationMode(message, durationMs = 15000) {
  const titleEl = document.getElementById("ai-title");
  const bannerEl = document.getElementById("verification-banner");
  if (!titleEl || !bannerEl) return;

  verificationActive = true;

  bannerEl.textContent = message ||
    "🔎 VERIFICATION MODE — ANALYZING STATISTICAL DEVIATION";

  bannerEl.classList.remove("hidden");

  titleEl.classList.add("fade");
  setTimeout(() => {
    titleEl.textContent = "Verification in progress…";
    titleEl.classList.remove("fade");
  }, 600);

  setTimeout(() => {
    bannerEl.classList.add("hidden");
    verificationActive = false;
    rotateTitle();
  }, durationMs);
}

// Start title rotation on load (30 seconds for slower, deliberate feel)
document.addEventListener('DOMContentLoaded', () => {
  setInterval(rotateTitle, 30000);
});

let viewIndex = 0;

function $(id){ return document.getElementById(id); }

function fmtTime(){
  const d = new Date();
  return d.toLocaleString();
}

async function jget(url){
  const r = await fetch(url);
  if(!r.ok) throw new Error(await r.text());
  return await r.json();
}

// Rotate random myths
function rotateMyths(){
  const container = $('mythContainer');
  if (!container) return;

  // Fisher-Yates shuffle for true randomness
  const shuffled = [...LOTTERY_MYTHS];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const selected = shuffled.slice(0, 2);

  console.log('Rotating myths:', selected.map(m => m.myth.substring(0, 30)));

  container.innerHTML = selected.map(m => `
    <div class="myth-box">
      <div class="myth-icon">❌</div>
      <div class="myth-text">
        <strong>MYTH:</strong> "${m.myth}"<br>
        <strong>REALITY:</strong> ${m.reality}
      </div>
    </div>
  `).join('');
}

// Fetch and display prediction tracking stats
async function loadPredictionStats() {
  const container = $('predictionContainer');
  if (!container) return;

  try {
    // Fetch stats for both feeds
    const [pbRes, mmRes] = await Promise.all([
      fetch('/api/predictions/powerball'),
      fetch('/api/predictions/megamillions')
    ]);

    const pbData = await pbRes.json();
    const mmData = await mmRes.json();

    const formatEdge = (edge) => {
      if (edge === undefined || edge === null || isNaN(edge)) return '<span style="color:#888;">N/A</span>';
      const color = edge > 0 ? '#00ff88' : (edge < 0 ? '#ff6b35' : '#888');
      const sign = edge > 0 ? '+' : '';
      return `<span style="color:${color};font-weight:700;">${sign}${edge.toFixed(1)}%</span>`;
    };

    const renderFeedStats = (data, name) => {
      const stats = data.stats || {};
      const interp = data.interpretation || {};
      const total = stats.total_predictions || 0;

      if (total === 0) {
        return `
          <div class="pred-feed">
            <div class="pred-feed-title">${name}</div>
            <div class="pred-no-data">No predictions tracked yet</div>
          </div>
        `;
      }

      return `
        <div class="pred-feed">
          <div class="pred-feed-title">${name}</div>
          <div class="pred-row">
            <span class="pred-label hot">Hot Edge:</span>
            ${formatEdge(interp.hot_edge)}
          </div>
          <div class="pred-row">
            <span class="pred-label cold">Cold Edge:</span>
            ${formatEdge(interp.cold_edge)}
          </div>
          <div class="pred-row">
            <span class="pred-label overdue">Overdue Edge:</span>
            ${formatEdge(interp.overdue_edge)}
          </div>
          <div class="pred-total">${total} predictions tracked</div>
        </div>
      `;
    };

    container.innerHTML = `
      <div class="prediction-header">
        <div class="pred-icon">📊</div>
        <div class="pred-title">Category Prediction Accuracy</div>
      </div>
      <div class="pred-subtitle">Comparing hot/cold/overdue classifications vs actual draws</div>
      <div class="pred-feeds">
        ${renderFeedStats(pbData, '🔴 Powerball')}
        ${renderFeedStats(mmData, '🟡 Mega Millions')}
      </div>
      <div class="pred-baseline">Baseline: 33.3% (random chance)</div>
      <div class="pred-note">Positive edge = category outperforming random</div>
    `;
  } catch (err) {
    console.error('Failed to load prediction stats:', err);
    container.innerHTML = `
      <div class="prediction-header">
        <div class="pred-icon">📊</div>
        <div class="pred-title">Category Prediction Tracking</div>
      </div>
      <div class="pred-error">Loading prediction data...</div>
    `;
  }
}

// Panel rotation between myths and predictions
let currentPanel = 'myth';
function rotatePanels() {
  const mythContainer = $('mythContainer');
  const predContainer = $('predictionContainer');
  const dots = document.querySelectorAll('.panel-indicator .dot');

  if (!mythContainer || !predContainer) return;

  if (currentPanel === 'myth') {
    // Switch to predictions
    mythContainer.style.display = 'none';
    predContainer.style.display = 'block';
    currentPanel = 'prediction';
    loadPredictionStats(); // Refresh data when showing
  } else {
    // Switch to myths
    predContainer.style.display = 'none';
    mythContainer.style.display = 'block';
    currentPanel = 'myth';
    rotateMyths(); // Refresh myths when showing
  }

  // Update indicator dots
  dots.forEach(dot => {
    dot.classList.toggle('active', dot.dataset.panel === currentPanel);
  });
}

function renderBait(b){
  const chiP = b.chi_square?.p;
  const verdict = (chiP !== null && chiP !== undefined && chiP < 0.01)
    ? 'STATISTICALLY UNUSUAL'
    : 'NORMAL RANDOM VARIATION';

  const vColor = (verdict.startsWith('STATISTICALLY')) ? 'var(--warn)' : 'var(--good)';

  const hot = b.hot.slice(0, 5).map(x => `<span style="background:#ff6b35;color:#000;padding:2px 6px;border-radius:4px;font-weight:700;margin:2px;">${x.n}</span>`).join(' ');
  const cold = b.cold.slice(0, 5).map(x => `<span style="background:#4a5568;color:#fff;padding:2px 6px;border-radius:4px;font-weight:700;margin:2px;">${x.n}</span>`).join(' ');
  const overdue = b.overdue.slice(0, 5).map(x => `<span style="background:#6b46c1;color:#fff;padding:2px 6px;border-radius:4px;font-weight:700;margin:2px;">${x.n}</span>`).join(' ');

  // AI Analysis section
  let aiSection = '';
  if (b.ai_analysis) {
    const aiStatus = b.ai_analysis.status;
    const severity = b.ai_analysis.severity || 'normal';
    let aiColor = '#6496ff';
    let aiIcon = '🤖';
    let bgColor = 'rgba(100,150,255,0.08)';
    let borderColor = 'rgba(100,150,255,0.3)';
    
    if (aiStatus === 'success') {
      // Color code by severity
      if (severity === 'high') {
        aiIcon = '🔴';
        aiColor = '#ff4444';
        bgColor = 'rgba(255,68,68,0.15)';
        borderColor = 'rgba(255,68,68,0.5)';
      } else if (severity === 'medium-fake') {
        aiIcon = '🟠';
        aiColor = '#ff8c00';
        bgColor = 'rgba(255,140,0,0.12)';
        borderColor = 'rgba(255,140,0,0.4)';
      } else if (severity === 'medium') {
        aiIcon = '🟡';
        aiColor = '#fbbf24';
        bgColor = 'rgba(251,191,36,0.12)';
        borderColor = 'rgba(251,191,36,0.4)';
      } else if (severity === 'low') {
        aiIcon = '🟢';
        aiColor = '#00ff88';
        bgColor = 'rgba(0,255,136,0.08)';
        borderColor = 'rgba(0,255,136,0.3)';
      } else {
        aiIcon = '🔵';
        aiColor = '#6496ff';
      }
    } else if (aiStatus === 'error') {
      aiIcon = '⚠️';
      aiColor = '#ff6b35';
    } else if (aiStatus === 'disabled') {
      aiIcon = '⚙️';
      aiColor = '#888';
    }
    
    // Latest draw date & staleness indicator
    let dataStatus = '';
    if (b.latest_draw_date) {
      const daysDiff = Math.floor((new Date() - new Date(b.latest_draw_date)) / (1000 * 60 * 60 * 24));
      const stalenessColor = daysDiff > 3 ? '#ff6b35' : '#00ff88';
      const cacheIndicator = b.ai_analysis.cached ? '💾 CACHED - ' : '🔍 LIVE ANALYSIS - ';
      dataStatus = `<div style="font-size:10px; color:${stalenessColor}; margin-bottom:6px;">${cacheIndicator}📅 Latest Draw: ${b.latest_draw_date} (${daysDiff} days ago)</div>`;
    }
    
    // Next draw countdown
    let nextDrawInfo = '';
    if (b.next_draw) {
      const nd = b.next_draw;
      nextDrawInfo = `<div style="font-size:10px; color:#6496ff; margin-bottom:6px;">⏰ Next Draw: ${nd.next_day} ${nd.next_date} at ${nd.draw_time} (${nd.hours_away}h away)</div>`;
    }
    
    // Pattern history tracking
    let historySection = '';
    if (b.pattern_history && b.pattern_history.length > 1) {
      const prevSeverity = b.pattern_history[b.pattern_history.length - 2].severity;
      const currSeverity = severity;
      let persistenceNote = '';
      if (prevSeverity === currSeverity && currSeverity !== 'normal') {
        persistenceNote = '<div style="font-size:10px; color:#ff8c00; margin-top:6px;">⚡ Pattern persisting across multiple analyses!</div>';
      } else if (prevSeverity !== 'normal' && currSeverity === 'normal') {
        persistenceNote = '<div style="font-size:10px; color:#00ff88; margin-top:6px;">✨ Previous pattern VANISHED - textbook randomness!</div>';
      }
      historySection = persistenceNote;
    }
    
    let testSummary = '';
    if (b.ai_analysis.test_summary) {
      const ts = b.ai_analysis.test_summary;
      testSummary = `
        <div style="font-size:10px; color:var(--muted); margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1);">
          <strong>Statistical Tests:</strong>
          Anomalies: ${ts.tests_passed_threshold} | 
          Random: ${ts.tests_showing_randomness} | 
          Bonferroni: ${ts.bonferroni_corrected.toFixed(6)}
          ${ts.requires_persistence ? ' | ⏱️ Needs persistence check' : ''}
        </div>
      `;
    }
    
    aiSection = `
      <div style="grid-column: 1/-1; padding:14px; background:${bgColor}; border-radius:8px; border:2px solid ${borderColor}; margin-top:10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        ${dataStatus}
        ${nextDrawInfo}
        <div style="font-size:13px; color:${aiColor}; margin-bottom:8px; font-weight:700;">
          ${aiIcon} LIVE AI PATTERN ANALYSIS
        </div>
        <div style="font-size:13px; line-height:1.7; color:#fff; font-weight:500;">
          ${b.ai_analysis.analysis.replace(/\n/g, '<br>')}
        </div>
        ${historySection}
        ${testSummary}
        ${aiStatus === 'success' ? `<div style="font-size:9px; color:var(--muted); margin-top:8px; font-style:italic; opacity:0.7;">Model: ${b.ai_analysis.model} | Updates every ${REFRESH_SECONDS}s | Analyzing ${b.window} recent draws</div>` : ''}
      </div>
    `;
  }

  // Usual chart panel only
  $('baitPanel').innerHTML = `
    <div style="grid-column: 1/-1; padding:10px; background:rgba(255,255,255,0.03); border-radius:8px;">
      <div style="font-size:13px; margin-bottom:8px;"><strong>Statistical Test Result:</strong></div>
      <div style="font-size:18px; font-weight:700; color:${vColor}; margin-bottom:6px;">${verdict}</div>
      <div style="font-size:11px; color:var(--muted);">
        χ² p-value: ${(chiP ?? NaN).toFixed(6)} | 
        Window: ${b.window} draws | 
        Expected: ${b.expected_per_number.toFixed(1)} per number
      </div>
    </div>
    
    ${aiSection}
    
    <div class="box">
      <div style="font-size:13px; margin-bottom:8px;"><strong>🔥 "Hot" Numbers</strong></div>
      <div style="font-size:11px; color:var(--muted); margin-bottom:8px;">Most frequent in last ${b.window} draws</div>
      <div style="margin-top:8px;">${hot}</div>
      <div style="margin-top:10px; font-size:10px; color:#ff6b35; border-top:1px solid var(--line); padding-top:8px;">
        ⚠️ This does NOT mean they're more likely next time!
      </div>
    </div>
    
    <div class="box">
      <div style="font-size:13px; margin-bottom:8px;"><strong>❄️ "Cold" Numbers</strong></div>
      <div style="font-size:11px; color:var(--muted); margin-bottom:8px;">Least frequent in last ${b.window} draws</div>
      <div style="margin-top:8px;">${cold}</div>
      <div style="margin-top:10px; font-size:10px; color:#6496ff; border-top:1px solid var(--line); padding-top:8px;">
        ⚠️ They're not "due" to appear—each draw is independent!
      </div>
    </div>
    
    <div class="box" style="grid-column: 1/-1;">
      <div style="font-size:13px; margin-bottom:8px;"><strong>⏰ "Overdue" Numbers</strong></div>
      <div style="font-size:11px; color:var(--muted); margin-bottom:8px;">Haven't appeared for the longest time</div>
      <div style="margin-top:8px;">${overdue}</div>
      <div style="margin-top:10px; font-size:10px; color:#b19cd9; border-top:1px solid var(--line); padding-top:8px;">
        ⚠️ MYTH BUSTED: Numbers don't have memory. Past draws don't affect future probability. Every number has equal odds every single draw!
      </div>
    </div>
  `;
}

function renderResearch(r) {
  if (!r || !r.current_research) {
    $('researchOut').innerHTML = '<div style="color:#888;">🔬 AI Research initializing...</div>';
    return;
  }

  // Game label banner with lottery-specific logos
  const gameDisplayName = r.game_display || (r.feed === 'powerball' ? 'Powerball' : (r.feed === 'megamillions' ? 'Mega Millions' : 'Unknown Game'));
  const gameColor = r.feed === 'powerball' ? '#e03131' : (r.feed === 'megamillions' ? '#37b24d' : '#fbbf24');
  // Unify logo sizing for both games
  const gameLogo = r.feed === 'powerball' ?
    '<img src="/powerball.png" alt="Powerball" style="height:44px; aspect-ratio:2.5/1; vertical-align:middle; object-fit:contain;">' :
    (r.feed === 'megamillions' ?
      '<img src="/megamillions.png" alt="Mega Millions" style="height:44px; aspect-ratio:2.5/1; vertical-align:middle; object-fit:contain;">' :
      '🎱');
  const gameBanner = `
    <div style="background: linear-gradient(135deg, ${gameColor}22 0%, ${gameColor}44 100%); border: 2px solid ${gameColor}; padding: 10px; border-radius: 8px; margin-bottom: 12px;">
      <div style="font-size:16px; color:${gameColor}; font-weight:900; text-align:center; display:flex; align-items:center; justify-content:center; gap:12px;">
        ${gameLogo}
        <span>TESTING: ${gameDisplayName.toUpperCase()}</span>
        ${gameLogo}
      </div>
      ${r.rotation_info ? `<div style="font-size:10px; color:#aaa; text-align:center; margin-top:4px;">Auto-rotating between ${r.rotation_info.games_in_rotation.join(' ⟷ ')}</div>` : ''}
    </div>
  `;
  
  const curr = r.current_research;
  const status = curr.status;
  if (status !== 'success') {
    $('researchOut').innerHTML = `<div style="color:#ff6b35;">⚠️ Research: ${curr.message || 'Error'}</div>`;
    return;
  }
  // Defensive: fallback for missing fields
  const hypothesis = curr.hypothesis || 'No hypothesis generated.';
  const reasoning = curr.reasoning || 'No reasoning provided.';
  const test_method = curr.test_method || 'Unknown';
  const viable = typeof curr.viable === 'boolean' ? curr.viable : false;
  const p_value = curr.p_value !== undefined ? curr.p_value : 1.0;
  const effect_size = curr.effect_size !== undefined ? curr.effect_size : 0;
  // Viable or not?
  const viableColor = viable ? '#00ff88' : '#888';
  const viableIcon = viable ? '✅' : '❌';
  const viableText = viable ? 'VIABLE PATTERN DETECTED' : 'NOT VIABLE - RANDOM VARIATION';
  

  // === PARTIAL WIN PATTERN ANALYSIS (3+, 4+, 5+) ===
  let partialWinHtml = '';
  if (curr.partial_win_stats && (curr.partial_win_stats[3] || curr.partial_win_stats[4] || curr.partial_win_stats[5])) {
    partialWinHtml = `<div style="margin:10px 0 0 0; padding:8px; background:rgba(0,255,136,0.07); border-radius:6px; border-left:3px solid #00ff88;">
      <div style="font-size:11px; color:#00ff88; font-weight:700; margin-bottom:2px;">🎯 PARTIAL WIN PATTERN ANALYSIS</div>`;
    [3,4,5].forEach(k => {
      if (curr.partial_win_stats[k]) {
        let elev = '';
        if (curr.elevated_partial_win_probability && curr.elevated_partial_win_probability[k]) {
          elev = `<span style='color:#ffeb3b; font-weight:700;' title='Elevated probability: ${curr.elevated_partial_win_probability[k].note}'>⬆️</span>`;
        }
        partialWinHtml += `<div style='font-size:12px; color:#fff; margin-bottom:2px;'>${elev} <strong>${k} MATCHES:</strong> ${curr.partial_win_stats[k]} times`;
        if (curr.elevated_partial_win_probability && curr.elevated_partial_win_probability[k]) {
          const note = curr.elevated_partial_win_probability[k].note;
          partialWinHtml += `<div style='font-size:10px; color:#ffeb3b; margin-left:18px;'>${note}</div>`;
        }
        partialWinHtml += `</div>`;
      }
    });
    partialWinHtml += `</div>`;
  }

  // Pursuit Mode Banner with animated progress, tooltips, and live animation
  let pursuitBanner = '';
  if (curr.pursuit_mode && curr.pursuit_mode.active) {
    const attempts = curr.pursuit_mode.attempts;
    const maxAttempts = curr.pursuit_mode.max_attempts;
    let attemptDots = '';
    // Enhancement 2: Tooltips for each attempt dot
    if (curr.pursuit_mode.attempt_results && Array.isArray(curr.pursuit_mode.attempt_results)) {
      attemptDots = curr.pursuit_mode.attempt_results.map((res, idx) => {
        let color = '#aaa';
        let icon = '●';
        let anim = 'pursuit-dot';
        let tooltip = '';
        if (curr.pursuit_mode.attempt_details && curr.pursuit_mode.attempt_details[idx]) {
          const d = curr.pursuit_mode.attempt_details[idx];
          tooltip = `data-tooltip="Attempt #${idx+1}: ${d.result ? 'PASS' : 'FAIL'} | p=${d.p_value ?? 'N/A'} | ${d.timestamp ?? ''}"`;
        }
        if (res === true) { color = '#00ff88'; icon = '✔'; anim += ' pass'; }
        else if (res === false) { color = '#ff6b35'; icon = '✖'; anim += ' fail'; }
        else { color = '#6496ff'; icon = '●'; }
        // Enhancement 1: Animate the most recent dot
        if (idx === attempts-1 && (res === true || res === false)) anim += ' live-anim';
        return `<span class="${anim}" style="font-size:18px;margin:0 2px;color:${color};" ${tooltip}>${icon}</span>`;
      }).join('');
    } else {
      attemptDots = Array.from({length: maxAttempts}, (_, i) => {
        let color = i < attempts ? '#6496ff' : '#222';
        let anim = 'pursuit-dot';
        return `<span class="${anim}" style="font-size:18px;margin:0 2px;color:${color};">●</span>`;
      }).join('');
    }
    // Enhancement 4: Animated banner/confetti for candidate/verified
    let flashClass = '';
    let confettiHtml = '';
    if (curr.discovery && (curr.discovery.level === 'CANDIDATE' || curr.discovery.level === 'VERIFIED')) {
      flashClass = 'glow-flash';
      confettiHtml = `<div id="confetti" class="confetti"></div>`;
      // Trigger headline verification banner
      const gameName = gameDisplayName.toUpperCase();
      const level = curr.discovery.level;
      enterVerificationMode(
        `🔎 VERIFICATION MODE — ${gameName} ${level} PATTERN UNDER REVIEW`,
        18000
      );
    }
    // Enhancement 10: Live status dot
    let statusDot = '<span class="status-dot processing" style="position:absolute;top:10px;right:10px;"></span>';
    pursuitBanner = `
      <div class="${flashClass}" style="background: linear-gradient(135deg, #6496ff22 0%, #6496ff44 100%); border: 2px solid #6496ff; padding: 12px; border-radius: 8px; margin-bottom: 12px; position:relative; overflow:hidden;">
        ${statusDot}
        <div style="font-size:15px; color:#6496ff; font-weight:900; margin-bottom:6px;">
          🔬 VERIFICATION MODE ACTIVE
        </div>
        <div style="font-size:12px; color:#ddd; line-height:1.6; margin-bottom:8px;">
          Systematically re-testing pattern for persistence verification
        </div>
        <div style="font-size:11px; color:#aaa; margin-bottom:6px;">
          <strong>Attempt:</strong> ${attempts}/${maxAttempts}
        </div>
        <div style="font-size:22px; letter-spacing:2px; margin-bottom:4px;">${attemptDots}</div>
        ${confettiHtml}
      </div>
    `;
  }

  // Pursuit Mode Message (status updates)
  let pursuitMessage = '';
  if (curr.pursuit_mode && curr.pursuit_mode.message) {
    const msg = curr.pursuit_mode.message;
    let msgColor = '#6496ff';
    if (msg.includes('VERIFIED') || msg.includes('✅')) msgColor = '#00ff88';
    if (msg.includes('FALSE POSITIVE') || msg.includes('❌')) msgColor = '#ff6b35';
    if (msg.includes('CANDIDATE') || msg.includes('🔶')) msgColor = '#ff8c00';

    pursuitMessage = `
      <div style="background: ${msgColor}22; border-left: 4px solid ${msgColor}; padding: 10px; margin-bottom: 12px; border-radius: 4px;">
        <div style="font-size:13px; color:${msgColor}; font-weight:700;">
          ${msg}
        </div>
      </div>
    `;
  }
  
  // Creativity & Diversity Metrics
  let metricsSection = '';
  if (curr.creativity_score !== undefined || curr.diversity_score !== undefined) {
    const creativity = curr.creativity_score || 5;
    const diversity = curr.diversity_score || 5;
    const creativityBar = '█'.repeat(creativity) + '░'.repeat(10 - creativity);
    const diversityBar = '█'.repeat(diversity) + '░'.repeat(10 - diversity);
    
    // Color code diversity based on score
    const diversityColor = diversity >= 7 ? '#00ff88' : diversity >= 4 ? '#ff8c00' : '#ff6b35';
    const diversityEmoji = diversity >= 7 ? '🌈' : diversity >= 4 ? '⚠️' : '🔴';
    
    const warningNote = curr.diversity_warning ? `<div style="margin-top:6px; padding:6px; background:rgba(255,140,0,0.2); border-radius:4px; font-size:9px; color:#ff8c00;">⚠️ ${curr.diversity_warning}</div>` : '';
    
    metricsSection = `
      <div style="margin-top:10px; padding:10px; background:rgba(100,150,255,0.08); border-radius:6px; border-left:3px solid #6496ff;">
        <div style="font-size:10px; color:#6496ff; font-weight:700; margin-bottom:6px;">🎯 AI PATTERN QUALITY METRICS</div>
        <div style="font-size:10px; color:#ddd; line-height:1.6; font-family:monospace;">
          <div style="margin-bottom:4px;">💡 Creativity: ${creativity}/10 <span style="color:#00ff88;">${creativityBar}</span></div>
          <div style="margin-bottom:4px;">${diversityEmoji} Diversity: ${diversity}/10 <span style="color:${diversityColor};">${diversityBar}</span></div>
          ${curr.custom_test_logic ? '<div style="margin-top:4px; color:#ff00ff;">⚡ CUSTOM TEST METHOD</div>' : ''}
        </div>
        ${warningNote}
      </div>
    `;
  }
  
  // AI Reasoning Process - show the thinking with streaming effect
  let reasoningProcess = '';
  if (curr.reasoning) {
    // Unescape backslash sequences from JSON
    const cleanReasoning = curr.reasoning.replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\');
    reasoningProcess = `
      <div style="margin-top:12px; padding:12px; background:rgba(100,150,255,0.08); border-radius:6px; border-left:3px solid #6496ff;" class="fade-in">
        <div style="font-size:11px; color:#6496ff; font-weight:700; margin-bottom:6px;">
          <span class="status-dot processing"></span>🧠 AI REASONING PROCESS<span class="thinking-dots"></span>
        </div>
        <div id="ai-reasoning-text" style="font-size:11px; color:#ddd; line-height:1.6; font-family:monospace; white-space:pre-wrap;">${cleanReasoning}</div>
      </div>
    `;
  }
  
  // Test execution details
  let testDetails = '';
  if (curr.results) {
    const r = curr.results;
    const details = [];
    
    if (r.observed !== undefined && r.expected !== undefined) {
      details.push(`Observed: ${r.observed} | Expected: ${r.expected.toFixed(2)}`);
    }
    if (r.draws_with_consecutive !== undefined) {
      details.push(`Consecutive found in ${r.draws_with_consecutive}/${r.total_draws} draws`);
    }
    if (r.draws_in_range !== undefined) {
      details.push(`Draws in range: ${r.draws_in_range}/${r.total_draws} (${(r.rate * 100).toFixed(1)}%)`);
    }
    if (r.avg_even_per_draw !== undefined) {
      details.push(`Avg even/draw: ${r.avg_even_per_draw.toFixed(2)} (expected: ${r.expected_even})`);
    }
    if (r.correlated_draws !== undefined) {
      details.push(`Correlated: ${r.correlated_draws}/${r.total_draws} (${(r.rate * 100).toFixed(1)}%)`);
    }
    
    testDetails = `
      <div style="margin-top:10px; padding:10px; background:rgba(0,0,0,0.3); border-radius:6px; font-family:monospace;">
        <div style="font-size:10px; color:#00ff88; font-weight:700; margin-bottom:6px;">📊 TEST EXECUTION</div>
        ${details.length > 0 ?
          details.map(d => `<div style="font-size:14px; color:#fff; font-weight:700; background:rgba(0,255,136,0.10); border-radius:4px; padding:2px 6px; margin-bottom:2px;">${d.replace(/(\d+\.\d+|\d+)/g, '<span style=\"color:#00ff88; font-weight:900;\">$1</span>')}</div>`).join('')
          : '<div style="color:#888; font-size:12px; font-style:italic;">No test values available</div>'}
      </div>
    `;
  }
  
  // Contradiction warning
  let contradictWarning = '';
  if (curr.contradicts) {
    contradictWarning = `<div style="background:rgba(255,107,53,0.2); padding:8px; border-radius:4px; margin-top:8px; border-left:3px solid #ff6b35;">
      ⚠️ <strong>CONTRADICTION:</strong> ${curr.contradicts}
    </div>`;
  }
  
  // Research history
  let historyHtml = '';
  if (r.recent_history && r.recent_history.length > 0) {
    const historyItems = r.recent_history.slice(0, 5).map(h => {
      const hIcon = h.viable ? '✅' : '❌';
      // Unescape and show more characters to differentiate hypotheses
      const cleanHypothesis = h.hypothesis.replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\');
      const shortHypothesis = cleanHypothesis.length > 100 ? cleanHypothesis.substring(0, 100) + '...' : cleanHypothesis;
      return `<div style="font-size:10px; color:#aaa; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05); line-height:1.4;">
        ${hIcon} <strong>Iter ${h.iteration}:</strong> ${shortHypothesis} <span style="color:#666;">(p=${h.p_value?.toFixed(4) || 'N/A'})</span>
      </div>`;
    }).join('');
    
    // Count verified discoveries in recent history
    const verifiedCount = r.recent_history.filter(h => h.viable && h.p_value < 0.01).length;
    const hallOfFame = verifiedCount > 0 ? `<div style="font-size:10px; color:#00ff88; margin-top:6px; font-weight:700;">🏆 ${verifiedCount} candidate anomalies detected in last ${r.recent_history.length} tests</div>` : '';
    
    historyHtml = `
      <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.1);">
        <div style="font-size:11px; color:var(--accent); margin-bottom:6px;"><strong>📊 RECENT RESEARCH HISTORY</strong></div>
        ${historyItems}
        ${hallOfFame}
      </div>
    `;
  }
  
  // Unescape text for display
  const cleanHypothesis = curr.hypothesis.replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  const cleanIntervalReasoning = curr.interval_reasoning ? curr.interval_reasoning.replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\') : '';
  
  // Enhancement 5: Research Timeline (last N attempts)
  let timelineHtml = '';
  if (r.recent_history && r.recent_history.length > 0) {
    const N = Math.min(10, r.recent_history.length);
    const timelineItems = r.recent_history.slice(0, N).map((h, idx) => {
      const hIcon = h.viable ? '✔' : '✖';
      const color = h.viable ? '#00ff88' : '#ff6b35';
      const tooltip = `Attempt #${h.iteration}: ${h.viable ? 'PASS' : 'FAIL'} | p=${h.p_value?.toFixed(4) || 'N/A'} | ${h.timestamp || ''}`;
      return `<span class="timeline-dot" style="color:${color};" data-tooltip="${tooltip}" onclick="window.showAttemptDetail && window.showAttemptDetail(${h.iteration})">${hIcon}</span>`;
    }).join('');
    timelineHtml = `
      <div style="margin:10px 0 0 0; text-align:center;">
        <div style="font-size:10px; color:#6496ff; font-weight:700; margin-bottom:2px;">Research Timeline (last ${N})</div>
        <div class="timeline-bar">${timelineItems}</div>
      </div>
    `;
  }

  $('researchOut').innerHTML = `
    ${gameBanner}
    ${pursuitBanner}
    ${pursuitMessage}
    ${metricsSection}
    <div style="font-size:12px; color:${viableColor}; margin-bottom:8px; font-weight:700;">
      ${viableIcon} ITERATION #${curr.iteration}: ${viableText}
    </div>
    <div style="font-size:13px; line-height:1.6; margin-bottom:8px; color:#fff; font-weight:600;">
      ${cleanHypothesis}
    </div>
    ${partialWinHtml}
    ${reasoningProcess}
    ${testDetails}
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; font-size:10px; color:#aaa; margin-top:10px;">
      <div><strong>Test Method:</strong> ${curr.test_method}</div>
      <div><strong>p-value:</strong> ${curr.p_value?.toFixed(6) || 'N/A'}</div>
      <div><strong>Effect Size:</strong> ${curr.effect_size?.toFixed(4) || 'N/A'}</div>
      <div><strong>Status:</strong> ${curr.viable ? 'Statistically significant' : 'Random variation'}</div>
    </div>
    <div style="margin-top:10px; padding:10px; background:rgba(100,150,255,0.1); border-radius:6px; border-left:3px solid #6496ff;">
      <div style="font-size:11px; color:#6496ff; font-weight:700; margin-bottom:4px;">⏱️ AI CADENCE CONTROL</div>
      <div style="font-size:10px; color:#aaa; line-height:1.5;">
        <strong>Next Research:</strong> ${curr.next_interval_seconds}s (${Math.round(curr.next_interval_seconds/60)}min)<br>
        <strong>AI Decision:</strong> ${cleanIntervalReasoning}
      </div>
    </div>
    ${timelineHtml}
    ${contradictWarning}
    ${historyHtml}
  `;

  // Enhancement 4: Confetti animation for candidate/verified
  if (document.getElementById('confetti')) {
    launchConfetti();
  }

  // Enhancement 2: Tooltips for attempt dots and timeline
  enableTooltips();
  // Enhancement 1: Animate the most recent dot
  animateLiveAttemptDot();
}
// Enhancement 4: Confetti animation function
function launchConfetti() {
  const confettiEl = document.getElementById('confetti');
  if (!confettiEl) return;
  confettiEl.innerHTML = '';
  for (let i = 0; i < 24; i++) {
    const div = document.createElement('div');
    div.className = 'confetti-piece';
    div.style.left = Math.random()*100 + '%';
    div.style.background = `hsl(${Math.random()*360},90%,60%)`;
    div.style.animationDelay = (Math.random()*0.7)+'s';
    confettiEl.appendChild(div);
  }
  setTimeout(() => { if(confettiEl) confettiEl.innerHTML = ''; }, 1800);
}

// Enhancement 2: Tooltips for attempt dots and timeline
function enableTooltips() {
  document.querySelectorAll('[data-tooltip]').forEach(el => {
    el.onmouseenter = function(e) {
      let tip = document.createElement('div');
      tip.className = 'custom-tooltip';
      tip.textContent = el.getAttribute('data-tooltip');
      document.body.appendChild(tip);
      const rect = el.getBoundingClientRect();
      tip.style.left = (rect.left + window.scrollX + rect.width/2 - tip.offsetWidth/2) + 'px';
      tip.style.top = (rect.top + window.scrollY - 28) + 'px';
      el._tip = tip;
    };
    el.onmouseleave = function() {
      if (el._tip) { el._tip.remove(); el._tip = null; }
    };
  });
}

// Enhancement 1: Animate the most recent attempt dot
function animateLiveAttemptDot() {
  const live = document.querySelector('.pursuit-dot.live-anim');
  if (live) {
    live.classList.remove('live-anim');
    void live.offsetWidth; // force reflow
    live.classList.add('live-anim');
  }
}

function plotHeat(h){
  // Create frequency heatmap - count occurrences per number
  const numCols = h.cols.length;
  const numRows = h.matrix.length;
  
  // Calculate frequency for each number across all draws in the window
  const freqData = new Array(numCols).fill(0);
  for (let row = 0; row < numRows; row++) {
    for (let col = 0; col < numCols; col++) {
      freqData[col] += h.matrix[row][col];
    }
  }
  
  // Clear any Plotly chart before adding HTML
  const chartEl = $('chartMain');
  if (chartEl && chartEl.data) {
    Plotly.purge('chartMain');
  }
  
  // Create heat squares grid
  const maxCount = Math.max(...freqData);
  const minCount = Math.min(...freqData);
  
  function getHeatColor(count) {
    const ratio = (count - minCount) / (maxCount - minCount || 1);
    if (ratio < 0.2) return '#1a0505';
    if (ratio < 0.4) return '#5a0000';
    if (ratio < 0.6) return '#b91c1c';
    if (ratio < 0.8) return '#f97316';
    return '#fbbf24';
  }
  
  let html = '<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(45px, 1fr)); gap:4px; padding:10px;">';
  
  for (let i = 0; i < h.cols.length; i++) {
    const num = h.cols[i];
    const count = freqData[i];
    const color = getHeatColor(count);

    // Determine if this is a hot or cold number based on count
    const isHot = count >= (maxCount * 0.7);
    const isCold = count <= (minCount + (maxCount - minCount) * 0.3);
    const animClass = isHot ? 'hot-number number-item' : (isCold ? 'cold-number number-item' : 'number-item');

    html += `
      <div class="${animClass}" style="
        background:${color};
        aspect-ratio:1;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        border-radius:6px;
        border:1px solid rgba(0,0,0,0.3);
        font-weight:700;
        font-size:13px;
        color:#fff;
        text-shadow:0 1px 2px rgba(0,0,0,0.8);
        cursor:pointer;
        transition:transform 0.2s;
      "
      title="Number ${num}: ${count} appearances"
      onmouseover="this.style.transform='scale(1.1)'"
      onmouseout="this.style.transform='scale(1)'">
        <div style="font-size:15px;">${num}</div>
        <div style="font-size:9px; color:rgba(255,255,255,0.7);">${count}</div>
      </div>
    `;
  }
  
  html += '</div>';
  
  // Add legend
  html += `
    <div style="display:flex; align-items:center; gap:8px; padding:10px; font-size:11px; color:#aaa;">
      <span>Cold</span>
      <div style="display:flex; gap:2px;">
        <div style="width:20px;height:20px;background:#1a0505;border-radius:3px;"></div>
        <div style="width:20px;height:20px;background:#5a0000;border-radius:3px;"></div>
        <div style="width:20px;height:20px;background:#b91c1c;border-radius:3px;"></div>
        <div style="width:20px;height:20px;background:#f97316;border-radius:3px;"></div>
        <div style="width:20px;height:20px;background:#fbbf24;border-radius:3px;"></div>
      </div>
      <span>Hot</span>
    </div>
  `;
  
  // Rotate between: full-panel entropy video, heatmap grid, and blue horizontal bar chart
  function renderHeatmapPanel() {
    // Clean up any existing Plotly chart first
    const freqEl = document.getElementById('chartFreq');
    if (freqEl) {
      try { Plotly.purge('chartFreq'); } catch(e) {}
      freqEl.remove();
    }
    // Clear the container completely
    chartEl.innerHTML = '';

    const choice = Math.floor(Math.random() * 3);
    if (choice === 0) {
      // Video panel
      chartEl.innerHTML = `
        <div style="display:flex;justify-content:center;align-items:center;height:320px;width:100%;background:rgba(0,0,0,0.18);border-radius:12px;overflow:hidden;">
          <video autoplay loop muted playsinline style="height:90%;width:auto;max-width:95%;max-height:95%;filter:drop-shadow(0 0 24px #00d9ff);border-radius:8px;">
            <source src="/entropy-video.mp4" type="video/mp4">
          </video>
        </div>
      `;
    } else if (choice === 1) {
      // Heatmap grid
      chartEl.innerHTML = html;
    } else {
      // Blue horizontal bar chart
      chartEl.innerHTML = `<div id="chartFreq" style="height:320px;width:100%;"></div>`;
      const x = h.cols;
      const y = h.cols.map((_, i) => h.matrix.reduce((sum, row) => sum + row[i], 0));
      const data = [{
        type: 'bar',
        x,
        y,
        marker: {
          color: '#00d9ff',
          line: {color: '#00a8cc', width: 1}
        },
        hovertemplate: 'Number %{x}<br>Count %{y}<extra></extra>'
      }];
      const layout = {
        margin: {l: 60, r: 30, t: 10, b: 50},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#0a0a0a',
        xaxis: {
          title: {text: 'Number', font: {size: 13, color: '#e8ecf3'}},
          gridcolor: '#1a1a1a',
          tickfont: {size: 10, color: '#aaa'},
          dtick: 5
        },
        yaxis: {
          title: {text: 'Frequency', font: {size: 13, color: '#e8ecf3'}},
          gridcolor: '#1a1a1a',
          tickfont: {size: 10, color: '#aaa'}
        },
        font: {color: '#e8ecf3', family: 'ui-sans-serif, system-ui'}
      };
      // Use newPlot instead of react for clean render
      Plotly.newPlot('chartFreq', data, layout, {displayModeBar:false, responsive:true});
    }
  }

  // Always clear and restart the timer to avoid multiple timers
  if (chartEl._entropyRotateTimer) {
    clearInterval(chartEl._entropyRotateTimer);
    chartEl._entropyRotateTimer = null;
  }
  renderHeatmapPanel();
  chartEl._entropyRotateTimer = setInterval(renderHeatmapPanel, 30000);
}

// Frequency bar removed - heatmap now shows all info in squares
function plotFrequencyBar(h) {
  // Deprecated - using heat squares instead
  return;
}

function plotHistogram(b){
  const x = [];
  const y = [];
  for(let i=b.min; i<=b.max; i++){
    x.push(i);
    y.push(b.counts[i - b.min]);
  }
  

  // Intermittently show entropy logo between heatmap and histogram
  const chartEl = $('chartMain');
  if (chartEl) {
    if (Math.random() < 0.33) { // ~33% chance to show logo
      chartEl.innerHTML = `<div style="display:flex;justify-content:center;align-items:center;height:60px;"><img src="/entropy.png" alt="Entropy" style="height:48px;filter:drop-shadow(0 0 8px #00d9ff);margin-bottom:8px;"></div>`;
      setTimeout(() => {
        Plotly.react('chartMain', [{type:'bar',x,y,marker:{color:'#00d9ff',line:{color:'#00a8cc',width:1}},hovertemplate:'Number %{x}<br>Count %{y}<extra></extra>'}],
          {margin:{l:50,r:20,t:10,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',xaxis:{title:{text:'Number',font:{size:14}},gridcolor:'#333'},yaxis:{title:{text:`Count (last ${b.window} draws)`,font:{size:14}},gridcolor:'#333'},font:{color:'#e8ecf3',family:'ui-sans-serif,system-ui'}},
          {displayModeBar:false,responsive:true});
      }, 1200);
      return;
    } else {
      chartEl.innerHTML = '';
    }
  }

  const data = [{
    type: 'bar',
    x, y,
    marker: {
      color: '#00d9ff',
      line: {color: '#00a8cc', width: 1}
    },
    hovertemplate: 'Number %{x}<br>Count %{y}<extra></extra>'
  }];

  const layout = {
    margin: {l: 50, r: 20, t: 10, b: 40},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      title: {text: 'Number', font: {size: 14}},
      gridcolor: '#333'
    },
    yaxis: {
      title: {text: `Count (last ${b.window} draws)`, font: {size: 14}},
      gridcolor: '#333'
    },
    font: {color: '#e8ecf3', family: 'ui-sans-serif, system-ui'}
  };

  Plotly.react('chartMain', data, layout, {displayModeBar:false, responsive:true});
}

function plotMonte(m){
  const band = {
    x: m.x.concat([...m.x].reverse()),
    y: m.band_hi.concat([...m.band_lo].reverse()),
    fill: 'toself',
    fillcolor: 'rgba(100, 150, 255, 0.2)',
    type: 'scatter',
    hoverinfo: 'skip',
    line: {width: 0},
    name: '5–95% band',
    showlegend: true
  };
  const obs = {
    x: m.x,
    y: m.obs,
    type: 'scatter',
    mode: 'lines',
    line: {color: '#ff6b35', width: 2},
    name: 'Observed',
    showlegend: true
  };

  const layout = {
    margin: {l: 50, r: 20, t: 10, b: 40},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      title: {text: 'Draw count (older → newer)', font: {size: 12}},
      gridcolor: '#333'
    },
    yaxis: {
      title: {text: 'Max abs deviation', font: {size: 12}},
      gridcolor: '#333'
    },
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(17, 19, 26, 0.8)',
      bordercolor: '#444',
      borderwidth: 1
    },
    font: {color: '#e8ecf3', family: 'ui-sans-serif, system-ui'}
  };

  // Alternate between Monte Carlo chart and statistical anomaly summary every 30 seconds
  function renderMontePanel(choice) {
    const chartEl = document.getElementById('chartMonte');
    const verdictBox = document.getElementById('monteVerdict').parentElement;
    if (choice === 0) {
      // Show only the Monte Carlo chart
      // Replace chartMonte with a fresh div to ensure Plotly attaches to a valid element
      const parent = chartEl.parentElement;
      const newChart = document.createElement('div');
      newChart.id = 'chartMonte';
      newChart.className = 'chart';
      newChart.style.height = chartEl.style.height;
      parent.replaceChild(newChart, chartEl);
      Plotly.react('chartMonte', [band, obs], layout, {displayModeBar:false, responsive:true});
      document.getElementById('monteVerdict').parentElement.style.display = '';
    } else {
      // Show only the statistical summary
      const megaRate = typeof window.megaVerifRate === 'number' ? window.megaVerifRate : 0.42;
      const powerRate = typeof window.powerVerifRate === 'number' ? window.powerVerifRate : 0.18;
      const megaCount = typeof window.megaCandidateCount === 'number' ? window.megaCandidateCount : 7;
      const powerCount = typeof window.powerCandidateCount === 'number' ? window.powerCandidateCount : 3;

      // Use candidate counts for percentage calculation (more intuitive)
      const safeMegaCount = megaCount || 1;
      const safePowerCount = powerCount || 1;

      // Calculate percentage difference based on candidate counts
      const megaMorePct = (((safeMegaCount - safePowerCount) / safePowerCount) * 100).toFixed(0);
      const powerMorePct = (((safePowerCount - safeMegaCount) / safeMegaCount) * 100).toFixed(0);

      console.log('Rendering summary with:', {megaRate, powerRate, megaCount, powerCount, megaMorePct, powerMorePct});

      chartEl.innerHTML = `
        <div style="display:flex;justify-content:center;align-items:center;height:220px;width:100%;background:linear-gradient(135deg,#222 0%,#333 100%);border-radius:12px;">
          <div style="text-align:center;width:100%;">
            <div style="font-size:15px; color:#00ff88; font-weight:900; margin-bottom:6px;">🔎 Statistical Anomaly Summary</div>
            <div style="font-size:12px; color:#fff; margin-bottom:6px;">${
              safeMegaCount > safePowerCount
                ? `Mega Millions finds candidate patterns <span style=\"color:#00ff88;font-weight:900;\">${megaMorePct}%</span> more often than Powerball.`
                : `Powerball finds candidate patterns <span style=\"color:#00ff88;font-weight:900;\">${powerMorePct}%</span> more often than Mega Millions.`
            }</div>
            <div style="font-size:11px; color:#aaa;">Recent candidates: <span style="color:#ffd700; font-weight:700;">${megaCount}</span> (Mega), <span style="color:#ffd700; font-weight:700;">${powerCount}</span> (Powerball)</div>
            <div style="font-size:10px; color:#888; margin-top:8px; font-style:italic;">Panel will revert to verification chart shortly...</div>
          </div>
        </div>
      `;
      verdictBox.style.display = 'none';
    }
  }
  // Always clear and restart the timer to avoid multiple timers and ensure correct alternation
  if (window._montePanelRotateTimer) {
    clearInterval(window._montePanelRotateTimer);
    window._montePanelRotateTimer = null;
  }
  let choice = Math.floor(Math.random() * 2);
  renderMontePanel(choice);
  window._montePanelRotateTimer = setInterval(() => {
    choice = (choice + 1) % 2;
    renderMontePanel(choice);
  }, 30000);
}

async function refreshAll(){
  $('statusLine').textContent = '🔬 AI Research in progress...';
  $('updatedAt').textContent = '';

  try {
    // Check if Plotly is loaded
    if (typeof Plotly === 'undefined') {
      throw new Error('Plotly library not loaded');
    }

    // Get dual research-auto results
    const research = await jget(`/api/research-auto`);
    const feeds = research.feeds || [];
    const games = research.games || [];
    const researchArr = research.research || [];
    const histories = research.recent_histories || [];

    if (!feeds.length || !games.length || !researchArr.length) {
      let errMsg = 'No research data available. Please try again later.';
      if (research && research.error) {
        errMsg = `Backend error: ${research.error}`;
        if (research.traceback) {
          errMsg += `<br><pre style='font-size:10px; color:#aaa; max-height:120px; overflow:auto;'>${research.traceback}</pre>`;
        }
      }
      $('statusLine').textContent = 'Error: Research data not available.';
      $('researchOut').innerHTML = `<div style="color:#ff6b35;">${errMsg}</div>`;
      return;
    }


    // Use the first feed for heat/bait, but fetch monte for BOTH feeds
    const feed = feeds[0];
    const [heat, bait, monte, montePower, monteMega] = await Promise.all([
      jget(`/api/heat/${feed}`),
      jget(`/api/bait/${feed}`),
      jget(`/api/monte/${feed}`),
      jget(`/api/monte/powerball`),
      jget(`/api/monte/megamillions`)
    ]);

    // Update global summary numbers from backend (fetch both feeds for complete stats)
    console.log('Monte API responses:', {montePower, monteMega});
    if (montePower) {
      if (typeof montePower.powerVerifRate !== 'undefined') window.powerVerifRate = montePower.powerVerifRate;
      if (typeof montePower.powerCandidateCount !== 'undefined') window.powerCandidateCount = montePower.powerCandidateCount;
    }
    if (monteMega) {
      if (typeof monteMega.megaVerifRate !== 'undefined') window.megaVerifRate = monteMega.megaVerifRate;
      if (typeof monteMega.megaCandidateCount !== 'undefined') window.megaCandidateCount = monteMega.megaCandidateCount;
    }
    console.log('Window stats after update:', {
      megaVerifRate: window.megaVerifRate,
      powerVerifRate: window.powerVerifRate,
      megaCandidateCount: window.megaCandidateCount,
      powerCandidateCount: window.powerCandidateCount
    });

    console.log('Data received:', {heat: heat.matrix?.length, bait: bait.hot?.length, monte: monte.x?.length, research: researchArr.map(r=>r.iteration)});

    renderBait(bait);
    plotMonte(monte);
    renderDualResearch(games, researchArr, histories); // Show both AI research results

    // Cycle main view
    const view = VIEWS[viewIndex % VIEWS.length];
    $('panelTitle').textContent = view.title;

    if(view.key === 'heat'){
      plotHeat(heat);
    } else {
      plotHistogram(bait);
    }

    viewIndex += 1;

    // Update dropdown to reflect the first game being tested
    $('feedSelect').value = feed;

    // Update dynamic interval based on the first AI decision
    if (researchArr[0]?.next_interval_seconds) {
      // Cap minimum interval at 60s to prevent runaway costs (AI can still control slowdown for efficiency)
      DYNAMIC_INTERVAL = Math.max(60, researchArr[0].next_interval_seconds);
      if (researchArr[0].next_interval_seconds < 60) {
        console.warn(`AI requested ${researchArr[0].next_interval_seconds}s but capped at 60s minimum for cost control`);
      } else {
        console.log(`AI set next interval: ${DYNAMIC_INTERVAL}s - ${researchArr[0].interval_reasoning}`);
      }
    }

    // Set up real-time countdown for next research interval
    let countdown = DYNAMIC_INTERVAL;
    function updateCountdown() {
      $('statusLine').textContent = `🤖 AI Research Active | ${researchArr[0]?.iteration || 0} & ${researchArr[1]?.iteration || 0} | Next in ${countdown}s`;
      countdown--;
      if (countdown >= 0) {
        setTimeout(updateCountdown, 1000);
      }
    }
    updateCountdown();
    $('updatedAt').textContent = `Updated: ${fmtTime()}`;

    // === REFRESH PREDICTIONS & STATS (only on data updates) ===
    // These only change when new draws occur (4x per week), so we check if
    // the latest draw date has changed before refreshing
    try {
      const currentDrawDate = bait?.latest_draw_date;
      if (currentDrawDate && currentDrawDate !== window.lastDrawDateSeen) {
        window.lastDrawDateSeen = currentDrawDate;
        console.log('[PREDICTIONS] New draw detected:', currentDrawDate, '- updating predictions');
        displayLotteryPredictions();
        loadPredictionStats();
      }
    } catch(predErr) {
      console.warn('[PREDICTIONS] Error updating predictions:', predErr);
    }
  } catch(err) {
    console.error('Error in refreshAll:', err);
    $('statusLine').textContent = `Error: ${err.message}`;
    $('researchOut').innerHTML = `<div style='color:#ff6b35;'>${err.message}</div>`;
  }
}

// Carousel refresh - updates research display every 2 minutes with rotated results
// This keeps the UI dynamic without increasing computation costs
async function refreshCarousel() {
  try {
    // Get carousel data (rotates through recent research every 2 minutes)
    const carousel = await jget(`/api/research-carousel`);

    if (!carousel || carousel.error) {
      console.log('[CAROUSEL] No carousel data:', carousel?.error || 'unknown');
      return; // Don't update if carousel fails
    }

    const feeds = carousel.feeds || [];
    const games = carousel.games || [];
    const researchArr = carousel.research || [];
    const histories = carousel.recent_histories || [];

    if (!feeds.length || !games.length || !researchArr.length) {
      console.log('[CAROUSEL] Incomplete carousel data');
      return;
    }

    // Update the research display with rotated results
    renderDualResearch(games, researchArr, histories);

    // Show which iteration we're viewing
    const nextIn = carousel.next_rotation_in || 120;
    const rotationNum = carousel.rotation_cycle || 0;
    console.log(`[CAROUSEL] Updated (rotation ${rotationNum}, next in ${nextIn}s)`);

  } catch(err) {
    console.warn('[CAROUSEL] Error updating carousel:', err.message);
    // Don't show error - carousel is just for eye candy
  }
}

// Start carousel refresh (every 2 minutes for visual variety)
setInterval(refreshCarousel, 120000);

// Render two research results stacked in the same panel
function renderDualResearch(games, researchArr, histories) {
  if (!researchArr || researchArr.length < 2) {
    $('researchOut').innerHTML = '<div style="color:#888;">🔬 AI Research initializing...</div>';
    return;
  }
  // Compose a single panel with two test blocks for the same game, labeled Test A and Test B
  if (!researchArr || researchArr.length < 2) {
    $('researchOut').innerHTML = '<div style="color:#888;">🔬 AI Research initializing...</div>';
    return;
  }
  // Show both games in a single panel, Test A (Powerball) and Test B (Mega Millions)
  if (!researchArr || researchArr.length < 2) {
    $('researchOut').innerHTML = '<div style="color:#888;">🔬 AI Research initializing...</div>';
    return;
  }
  let html = '';
  html += `<div style="background: linear-gradient(135deg, #222634 0%, #1a1a1a 100%); border: 2px solid #6496ff; padding: 12px 8px 8px 8px; border-radius: 12px; margin-bottom: 18px; max-width: 1100px; margin-left:auto; margin-right:auto;">`;
  html += `<div style="font-size:18px;font-weight:900;color:#6496ff;margin-bottom:12px;text-align:center;">AI AUTONOMOUS RESEARCH LAB</div>`;
  // Responsive: stack vertically on small screens, side by side on wide screens
  html += `<div class="dual-research-flex" style="display: flex; flex-wrap: wrap; flex-direction: row; gap: 16px; justify-content: space-between; align-items: flex-start; width: 100%;">`;
  for (let i = 0; i < 2; i++) {
    const feed = (i === 0) ? 'powerball' : 'megamillions';
    const gameLabel = (i === 0) ? 'Powerball' : 'Mega Millions';
    const logo = feed === 'powerball'
      ? '<img src="/powerball.png" alt="Powerball" style="height:44px; aspect-ratio:2.5/1; vertical-align:middle; object-fit:contain; margin-right:6px;">'
      : '<img src="/megamillions.png" alt="Mega Millions" style="height:44px; aspect-ratio:2.5/1; vertical-align:middle; object-fit:contain; margin-right:6px;">';
    const r = {
      ...researchArr[i]?.full_result || {},
      feed: feed,
      current_research: researchArr[i],
      recent_history: histories[i] || []
    };
    html += `<div class="dual-research-block" data-dual-index="${i}" style="flex:1 1 350px; min-width:320px; max-width:520px; background:rgba(17,19,26,0.92); border-radius:10px; border:1px solid #222634; box-shadow:0 2px 8px #0004; padding:14px 10px 14px 10px; margin-bottom:0; margin-left:auto; margin-right:auto;">`;
    html += `<div style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:700;color:#6496ff;margin-bottom:8px;">${logo}${i === 0 ? 'Test A (Powerball)' : 'Test B (Mega Millions)'}</div>`;
    // Use a fake div to capture renderResearch output as string
    const tempDiv = document.createElement('div');
    const origResearchOut = document.getElementById('researchOut');
    const fakeId = `researchOut_fake_${i}`;
    tempDiv.id = fakeId;
    document.body.appendChild(tempDiv);
    const old = origResearchOut;
    window.$ = function(id) { return id === 'researchOut' ? tempDiv : old; };
    renderResearch(r);
    html += tempDiv.innerHTML;
    tempDiv.remove();
    html += `</div>`;
  }
  html += `</div></div>`;
  // Responsive CSS for stacking on small screens
  if (!document.getElementById('dual-research-style')) {
    const style = document.createElement('style');
    style.id = 'dual-research-style';
    style.textContent = `
      @media (max-width: 1100px) {
        .dual-research-flex { flex-direction: column !important; gap: 12px !important; }
      }
      @media (max-width: 700px) {
        .dual-research-flex > div { max-width: 98vw !important; min-width: 0 !important; }
      }
    `;
    document.head.appendChild(style);
  }
  // Restore window.$
  window.$ = function(id) { return document.getElementById(id); };
  document.getElementById('researchOut').innerHTML = html;

  // After rendering, start streaming animation in the real DOM for both panels
  for (let i = 0; i < 2; i++) {
    const block = document.querySelector(`.dual-research-block[data-dual-index="${i}"]`);
    if (!block) continue;
    const research = researchArr[i];
    const currPattern = research && research.hypothesis ? research.hypothesis : '';
    const currReasoning = research && research.reasoning ? research.reasoning : '';
    const reasoningEl = block.querySelector('#ai-reasoning-text');
    if (reasoningEl && currReasoning) {
      // Cancel any existing streaming loop for this panel
      const loopKey = `reasoning_${i}`;
      if (streamingLoops[loopKey]) {
        clearTimeout(streamingLoops[loopKey]);
        streamingLoops[loopKey] = null;
      }

      const fullText = currReasoning.replace(/'/g, "'").replace(/\"/g, '"').replace(/\\/g, '\\');
      let idx = 0;
      function streamReasoningLoop() {
        if (!reasoningEl.isConnected) {
          streamingLoops[loopKey] = null;
          return;
        }
        if (idx <= fullText.length) {
          reasoningEl.textContent = fullText.slice(0, idx);
          idx++;
          streamingLoops[loopKey] = setTimeout(streamReasoningLoop, 7 + Math.random()*18);
        } else {
          // Wait 1 minute before repeating the streaming animation
          streamingLoops[loopKey] = setTimeout(() => {
            idx = 0;
            streamReasoningLoop();
          }, 60000);
        }
      }
      streamReasoningLoop();
    }
  }
}

// Render a single research block (reuse most of renderResearch logic)
function renderResearchBlock(r, dualMode=false) {
  if (!r || !r.current_research) {
    return '<div style="color:#888;">🔬 AI Research initializing...</div>';
  }
  const curr = r.current_research;
  // Defensive: fallback for missing fields
  const hypothesis = curr.hypothesis || 'No hypothesis generated.';
  const reasoning = curr.reasoning || 'No reasoning provided.';
  const test_method = curr.test_method || 'Unknown';
  const viable = typeof curr.viable === 'boolean' ? curr.viable : false;
  const p_value = curr.p_value !== undefined ? curr.p_value : 1.0;
  const effect_size = curr.effect_size !== undefined ? curr.effect_size : 0;

  // Viable or not?
  const viableColor = viable ? '#00ff88' : '#888';
  const viableIcon = viable ? '✅' : '❌';
  const viableText = viable ? 'VIABLE PATTERN DETECTED' : 'NOT VIABLE - RANDOM VARIATION';

  // Build a simple HTML summary
  const cleanHypothesis = hypothesis.replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  const cleanReasoning = reasoning.replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\');

  return `
    <div style="font-size:12px; color:${viableColor}; margin-bottom:8px; font-weight:700;">
      ${viableIcon} ITERATION #${curr.iteration || 0}: ${viableText}
    </div>
    <div style="font-size:13px; line-height:1.6; margin-bottom:8px; color:#fff; font-weight:600;">
      ${cleanHypothesis}
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; font-size:10px; color:#aaa; margin-top:10px;">
      <div><strong>Test Method:</strong> ${test_method}</div>
      <div><strong>p-value:</strong> ${p_value.toFixed(6)}</div>
      <div><strong>Effect Size:</strong> ${effect_size.toFixed(4)}</div>
      <div><strong>Status:</strong> ${viable ? 'Statistically significant' : 'Random variation'}</div>
    </div>
  `;
}

function startLoop(){
  // Rotate myths immediately
  rotateMyths();
  // Load prediction stats and lottery predictions initially
  loadPredictionStats();
  displayLotteryPredictions();
  // Initialize draw date tracking
  window.lastDrawDateSeen = null;
  // Rotate between panels every 15 seconds
  setInterval(rotatePanels, 15000);
  // Load carousel data immediately on startup
  refreshCarousel();
  
  async function loop() {
    try {
      await refreshAll();
    } catch(err) {
      console.error(err);
      $('statusLine').textContent = 'Error fetching data (check backend)';
    }
    
    // Schedule next iteration based on AI's decision
    setTimeout(loop, DYNAMIC_INTERVAL * 1000);
  }
  
  loop();  // Start the loop
}

// ===== ANIMATION HELPER FUNCTIONS =====

// Typewriter streaming effect for text
function streamText(elementId, text, speed = 30) {
  const element = $(elementId);
  if (!element) return;

  element.textContent = '';
  let charIndex = 0;

  const typeInterval = setInterval(() => {
    if (charIndex < text.length) {
      element.textContent += text.charAt(charIndex);
      charIndex++;
    } else {
      clearInterval(typeInterval);
      // After streaming, add a fade-in effect
      element.classList.add('fade-in');
    }
  }, speed);
}

// Animate numbers appearing one by one
function animateNumbersSequentially(containerSelector, delayBetween = 200) {
  const container = document.querySelector(containerSelector);
  if (!container) return;

  const numberElements = container.querySelectorAll('.number-item');
  numberElements.forEach((el, index) => {
    setTimeout(() => {
      el.classList.add('appearing');
      // Flash after appearing
      setTimeout(() => {
        el.classList.add('flashing');
        setTimeout(() => el.classList.remove('flashing'), 1500);
      }, 800);
    }, index * delayBetween);
  });
}

// Cycle hot numbers with fade effect
function cycleHotNumbers() {
  const hotNumbers = document.querySelectorAll('.hot-number');
  hotNumbers.forEach((num, index) => {
    num.style.animationDelay = `${index * 0.3}s`;
  });
}

// Cycle cold numbers with fade effect
function cycleColdNumbers() {
  const coldNumbers = document.querySelectorAll('.cold-number');
  coldNumbers.forEach((num, index) => {
    num.style.animationDelay = `${index * 0.4}s`;
  });
}

// Add glitch effect to emphasize updates
function glitchEffect(elementId) {
  const element = $(elementId);
  if (!element) return;

  element.classList.add('glitch-effect');
  setTimeout(() => element.classList.remove('glitch-effect'), 300);
}

// Simulate AI thinking process by clearing and re-streaming reasoning
function simulateAIThinking(elementId, text) {
  const element = $(elementId);
  if (!element) return;

  // Clear with fade out
  element.style.opacity = '0';
  element.style.transition = 'opacity 0.5s';

  setTimeout(() => {
    element.textContent = '';
    element.style.opacity = '1';

    // Re-stream the text
    let charIndex = 0;
    const thinkInterval = setInterval(() => {
      if (charIndex < text.length) {
        element.textContent += text.charAt(charIndex);
        charIndex++;
      } else {
        clearInterval(thinkInterval);
      }
    }, 20);
  }, 500);
}

// Periodically refresh AI reasoning to make it feel alive
let lastReasoningText = '';
function periodicReasoningRefresh() {
  const reasoningEls = document.querySelectorAll('#ai-reasoning-text');
  reasoningEls.forEach(el => {
    if (el && el.textContent) {
      const currentText = el.textContent;
      // Always re-stream for animation, not just on change
      simulateAIThinking(el.id, currentText);
    }
  });
}
// Periodic reasoning refresh disabled - using streamReasoningLoop instead
// setInterval(() => {
//   periodicReasoningRefresh();
// }, 15000);

// Add pulsing status indicator
function addStatusIndicator(containerSelector) {
  const container = document.querySelector(containerSelector);
  if (!container) return;

  const indicator = document.createElement('span');
  indicator.className = 'status-dot active';
  indicator.style.cssText = 'position: absolute; top: 10px; right: 10px;';
  container.style.position = 'relative';
  container.appendChild(indicator);
}

// Animate chart updates with fade transitions
function animateChartUpdate(chartId) {
  const chart = $(chartId);
  if (!chart) return;

  chart.style.opacity = '0.3';
  chart.style.transition = 'opacity 0.5s ease-in-out';

  setTimeout(() => {
    chart.style.opacity = '1';
  }, 300);
}

// Make hot/cold numbers cycle their visibility
function startNumberCycling() {
  setInterval(() => {
    cycleHotNumbers();
    cycleColdNumbers();
  }, 5000);
}

// Randomly animate hot/cold/overdue numbers for a lively effect
function livelyNumberActivity() {
  ['.hot-number', '.cold-number', '.overdue-number'].forEach(selector => {
    const numbers = document.querySelectorAll(selector);
    numbers.forEach(num => {
      // Randomly decide to animate this number
      if (Math.random() < 0.4) {
        num.classList.add('flashing');
        // Optionally hide and show for a "disappear" effect
        if (Math.random() < 0.2) {
          num.style.visibility = 'hidden';
          setTimeout(() => num.style.visibility = 'visible', 1200 + Math.random() * 800);
        }
        setTimeout(() => num.classList.remove('flashing'), 1000 + Math.random() * 1000);
      }
    });
  });
}
setInterval(livelyNumberActivity, 2500);

// Periodic reasoning refresh disabled - using streamReasoningLoop instead
// setInterval(() => {
//   periodicReasoningRefresh();
// }, 15000);

// Start number cycling animations
startNumberCycling();

// Event listeners
$('btnNow').addEventListener('click', () => {
  glitchEffect('btnNow');
  refreshAll();
});

startLoop();