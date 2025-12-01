## Movie Recommender

Hybrid FastAPI + Streamlit project that lets you scrape TMDB, build a local
in-memory recommendation index, and explore results with a friendly UI.

### Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: install Playwright browsers if you intend to run UI tests
playwright install

# scrape seed data (or copy your own CSV to data/movies.csv)
python -c "from src.scraper import scrape_top_n_movies; scrape_top_n_movies(n=300)"

# start FastAPI backend (recommendation + CSV helpers)
uvicorn src.api:app --reload

# in another shell run the Streamlit UI
streamlit run web/app_streamlit.py
```

### Tests

Backend/unit tests currently consist of the Streamlit UI smoke test.
To run it, start the backend + frontend locally and execute:

```bash
RUN_UI_TESTS=1 pytest
```

By default `pytest` will skip the Playwright test so the suite succeeds
in CI environments where browsers are unavailable.
