# ENTROPY Audit

Independent randomness audit system for lottery draws (Mega Millions and Powerball).

## Features

- **Auto-fetch draw data** from NY Open Data (Socrata API)
- **SQLite storage** for historical draw records
- **FastAPI backend** with statistical analysis endpoints
- **Interactive visualizations** using Plotly:
  - Rolling heatmap of number presence
  - Frequency histograms
  - Hot/Cold/Overdue analysis with statistical debunking
  - Monte Carlo baseline comparison (5-95% confidence bands)
- **Auto-refresh frontend** that cycles between views for 24/7 monitoring

## Data Sources

This project uses publicly available lottery draw data from:
- **Powerball** (2010-present): NY Open Data dataset `d6yy-54nr`
- **Mega Millions** (2002-present): NY Open Data dataset `5xaw-6ayf`

Note: While hosted by NY's portal, these datasets contain nationwide draw history.

## Project Structure

```
entropy-audit/
├── backend/          # Python FastAPI application
│   ├── app.py           # Main FastAPI server
│   ├── audit.py         # Statistical analysis functions
│   ├── data_sources.py  # Socrata API integration
│   ├── db.py            # SQLite database operations
│   ├── ingest.py        # Data ingestion pipeline
│   ├── parse.py         # Data parsing utilities
│   └── requirements.txt # Python dependencies
├── frontend/         # Single-page web application
│   ├── index.html       # Main HTML page
│   ├── styles.css       # Styling
│   └── app.js           # JavaScript with Plotly charts
├── data/             # SQLite database storage
├── .env              # Configuration (optional)
└── *.bat             # Windows setup scripts
```

## Quick Start

### 1. Setup Environment

Run the setup script to create a virtual environment and install dependencies:

```cmd
setup.bat
```

This will:
- Create a Python virtual environment in `.venv/`
- Install all required packages from `requirements.txt`
- Initialize the SQLite database

### 2. Ingest Data

Fetch historical draw data from NY Open Data:

```cmd
ingest.bat
```

This downloads and stores:
- ~1,700+ Powerball draws (2010-present)
- ~2,500+ Mega Millions draws (2002-present)

**Note**: First run takes 10-30 seconds depending on connection speed.

### 3. Run Server

Start the FastAPI development server:

```cmd
run.bat
```

The server starts at: **http://127.0.0.1:8000**

Open this URL in your browser to view the audit dashboard.

## Configuration

Edit `.env` to customize settings:

```env
SOCRATA_APP_TOKEN=      # Optional: improves API rate limits
DB_PATH=./data/entropy.sqlite
REFRESH_SECONDS=15      # Frontend auto-refresh interval
ROLLING_DRAWS=120       # Number of draws in rolling window
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main dashboard (HTML) |
| `GET /api/health` | Health check |
| `GET /api/recent/{feed_key}` | Recent draws data |
| `GET /api/bait/{feed_key}` | Hot/cold/overdue analysis |
| `GET /api/heat/{feed_key}` | Heatmap matrix data |
| `GET /api/monte/{feed_key}` | Monte Carlo baseline comparison |

Feed keys: `powerball`, `megamillions`

## Statistical Methodology

### Hot/Cold/Overdue Analysis
- Computes frequency counts over rolling window
- Tests against uniform distribution using χ² test
- **Debunks** pattern-based predictions by showing p-values
- Displays "overdue" numbers with disclaimer (no predictive value)

### Monte Carlo Baseline
- Simulates 1,500 random draw sequences
- Computes 5-95% confidence band for max deviation
- Compares observed deviation against simulated baseline
- **Purpose**: Verify observed patterns are within expected random variation

### Key Principle
**All patterns are stress-tested under the null hypothesis of uniform randomness.**  
No predictions, no picks, no gambling advice.

## Technology Stack

**Backend:**
- Python 3.10+
- FastAPI (web framework)
- SQLite (database)
- NumPy + SciPy (statistical analysis)
- HTTPX (async HTTP client)

**Frontend:**
- Vanilla JavaScript
- Plotly.js (interactive charts)
- CSS Grid layout
- Auto-refresh with view cycling

## Development

### Manual Commands

```cmd
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r backend\requirements.txt

# Initialize database
python -c "from backend.db import init_db; init_db()"

# Run ingestion
python -c "import asyncio; from backend.ingest import ingest_all; print(asyncio.run(ingest_all()))"

# Start server
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

## Disclaimer

This project uses publicly available lottery draw data for **identification and audit purposes only**. 

- No affiliation or endorsement by any lottery organization
- No predictions or gambling advice provided
- Past patterns have no bearing on future independent draws
- For educational and transparency purposes only

## License

This is an educational/transparency project. Use responsibly.
