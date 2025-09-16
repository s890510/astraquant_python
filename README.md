# AstraQuant Python FastAPI Project

A minimal FastAPI service skeleton for ML scoring, feature building, news fetching, and indices utilities.

## Project Structure
```
astraquant-py/
├─ app/
│  ├─ main.py              # FastAPI app entrypoint
│  ├─ routers/             # Routers by domain
│  │   ├─ __init__.py
│  │   ├─ ml.py            # /ml/score
│  │   ├─ features.py      # /features/build
│  │   ├─ news.py          # /news/latest
│  │   ├─ indices.py       # /indices/*
│  └─ core/
│      ├─ config.py        # Config loader (.env + config.yml)
│      ├─ __init__.py
├─ models/                 # Store ML model artifacts (e.g., model.pkl)
├─ data/                   # Parquet/CSV cache
├─ scripts/                # Training scripts and collectors
├─ requirements.txt
├─ config.yml              # External config (DATA_DIR, RSS sources)
└─ README.md
```

## Endpoints
- GET /healthz -> {"status":"ok"}
- POST /ml/score -> dummy model score from provided features
- POST /features/build -> creates a placeholder features file in data dir
- GET /news/latest -> fetch latest items from configured RSS feeds
- GET /indices/ and /indices/health -> example endpoints

## Run locally
1. Python 3.10+ recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Configure config.yml or .env. By default, DATA_DIR=data.
4. Start the server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Configuration
- .env: environment variables (e.g., DATA_DIR)
- config.yml: YAML file with keys like:
  ```yaml
  DATA_DIR: data
  rss_sources:
    - https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en
  ```

The application loads environment variables first, then falls back to config.yml values, then defaults. The DATA_DIR is created automatically if missing.

## Notes
- Replace the dummy scoring logic in app/routers/ml.py with your actual model inference and load artifacts from the models/ directory.
- Feature building in app/routers/features.py is a stub and should be extended to compute real features.
