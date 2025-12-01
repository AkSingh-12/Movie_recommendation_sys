#!/usr/bin/env bash
set -e
# 1. create venv (optional)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Scrape (uncomment if you have TMDB key)
# python -c "from src.scraper import scrape_top_n_movies; scrape_top_n_movies(n=200)"

# 3. Start API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
