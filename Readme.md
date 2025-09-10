<div align="center">

# YouTube Analysis (CommentSense)

Real‑time analysis of YouTube comments: quality labels, emotions (GoEmotions), relevance, trends, and a Streamlit dashboard — with an optional Chrome extension for on‑page insights.

</div>

---

## Features

- Comment quality labels: high‑quality, low‑quality, spam, toxic (MiniLM similarity).
- Emotions (GoEmotions): multi‑label emotion detection with top‑label summaries and trends.
- Relevance: keyword + semantic relevance vs. video title/description.
- Streamlit dashboard for exploration and CSV exports.
- Flask API for charts/JSON — used by the Chrome extension.
- Optional Chrome extension: fetches comments via your YouTube Data API key and calls the local API for insights.

## Quick Start

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Start the API (Flask)

```bash
python flask_app/app.py
```

The API runs on `http://localhost:5001` by default.

4) Start the dashboard (Streamlit)

```bash
streamlit run app.py
```

The dashboard opens on `http://localhost:8501`.

> First run downloads models (MiniLM, GoEmotions), which can take a few minutes.

## Chrome Extension (Optional)

Directory: `yt-chrome-plugin-frontend/`

- Open Chrome → Extensions → Enable Developer Mode → Load unpacked → select `yt-chrome-plugin-frontend/`.
- Click the extension → Settings:
  - API Host: `http://localhost`
  - API Port: `5001`
  - YouTube API Key: your key from Google Cloud (YouTube Data API v3).
- Visit a YouTube video, open the extension, and click Analyze.

The extension fetches comments using your API key and calls the local Flask API for emotion/quality charts and JSON summaries.

## API Endpoints

Base: `http://localhost:5001`

- `POST /quality_labels_chart`: PNG pie chart for quality labels.
- `POST /emotion_labels_chart`: PNG pie chart for top GoEmotions labels.
- `POST /hybrid_relevance_chart`: PNG relevant vs. not relevant using hybrid approach.
- `POST /goemotions_with_timestamps`: JSON of `{comment, emotion, score, timestamp}`.
- `POST /generate_emotion_trend`: PNG monthly trend of top emotions.

All endpoints accept a JSON payload with a `comments` array of strings or `{text, timestamp}` objects. See `flask_app/app.py` for details.

## Environment

This project reads simple key/value pairs from `.env.local` if present (no external dotenv dependency). Example:

```
# Flask server port
PORT=5001

# Optional MLflow local tracking (if used)
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT=youtube-sentiment
```

Notes:
- The Chrome extension stores your YouTube API key in Chrome storage; the backend does not need it.
- CPU is used by default. GPU is disabled via `CUDA_VISIBLE_DEVICES=""` for stability.

## Troubleshooting

- Slow first run: model weights download on first use; subsequent runs are fast.
- Transformers not installed: ensure `pip install -r requirements.txt` completed without errors.
- GoEmotions long text error: inputs are truncated/padded to model limits in code.
- Port in use: set a different `PORT` in `.env.local` and restart the API.

## Project Structure

- `app.py` — Streamlit dashboard (analysis, tables, charts, downloads).
- `flask_app/app.py` — Flask API for images/JSON used by the extension.
- `yt-chrome-plugin-frontend/` — Chrome extension (popup UI + fetch + calls API).
- `src/` — Supporting data/ingestion utilities.

