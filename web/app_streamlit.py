import streamlit as st
import requests
from typing import Optional
from pathlib import Path
import json
import sys
import os
import time

# Ensure project root is on sys.path so `src` imports work when Streamlit
# runs with a different CWD. This inserts the repo root (one level up from
# this `web/` folder) at the front of sys.path.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import TMDB_API_KEY as DEFAULT_TMDB_KEY
from src.data__loader import load_movies_by_genre, load_movies, set_poster_for_title

# placeholder image used when no poster can be found
PLACEHOLDER_URL = "https://via.placeholder.com/200x300?text=No+Poster"

st.set_page_config(page_title="Movie Recommender", layout="wide")

# default TMDB key from config; UI only displays status
TMDB_API_KEY = (
    DEFAULT_TMDB_KEY if DEFAULT_TMDB_KEY and DEFAULT_TMDB_KEY != "YOUR_TMDB_API_KEY_HERE" else ""
)
if TMDB_API_KEY:
    st.sidebar.success("Using TMDB for posters.")
else:
    st.sidebar.warning("TMDB API key missing; posters may use placeholders.")
NUM = st.sidebar.slider("Number of recommendations", min_value=1, max_value=20, value=10)
AUTO_REFRESH = st.sidebar.checkbox("Enable auto-refresh (poll backend)", value=False)
POLL_INTERVAL = st.sidebar.slider("Poll interval (seconds)", min_value=5, max_value=600, value=60)


def _clean_poster_value(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


@st.cache_data(ttl=60 * 60)
def fetch_poster_url(
    title: Optional[str],
    poster_path: Optional[str],
    tmdb_api_key: Optional[str],
) -> str:
    """Try to fetch a poster URL from TMDB, fall back to placeholder."""

    # normalize title/poster inputs
    title_str = (title or "").strip()
    normalized_poster = _clean_poster_value(poster_path)

    # prefer existing poster reference
    if normalized_poster:
        if normalized_poster.startswith("/"):
            return f"https://image.tmdb.org/t/p/w200{poster_path}"
        return normalized_poster

    if tmdb_api_key and title_str:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            resp = requests.get(
                url,
                params={"api_key": tmdb_api_key, "query": title_str, "page": 1},
                timeout=5,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if results:
                tmdb_path = results[0].get("poster_path")
                if tmdb_path:
                    return f"https://image.tmdb.org/t/p/w200{tmdb_path}"
        except Exception:
            pass

    return PLACEHOLDER_URL


def show_movie_card(movie: dict, tmdb_api_key: Optional[str]):
    poster_value = movie.get('poster_path')
    poster = fetch_poster_url(
        movie.get("title", ""),
        poster_value if isinstance(poster_value, str) else None,
        tmdb_api_key,
    )
    if not isinstance(poster, str) or not poster.strip():
        poster = PLACEHOLDER_URL
    # simple card layout with image and details
    cols = st.columns([1, 3])
    # add lightweight CSS for card styling
    st.markdown(
        """
        <style>
        .movie-card {border:1px solid #e6e6e6; padding:12px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.04);}
        .movie-title {margin:0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with cols[0]:
        st.image(poster, width="stretch")
    with cols[1]:
        st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='movie-title'>{movie.get('title','Untitled')}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Score:** {movie.get('score', 0):.3f}  ")
        st.markdown(f"**Genres:** {movie.get('genres', '')}  ")
        st.markdown(f"**Director:** {movie.get('director', '')}  ")
        desc = movie.get('description', '') or movie.get('overview', '')
        if desc:
            st.write(desc)
        if st.button("Add to favorites", key=f"fav-{movie.get('title','')}"):
            favs = st.session_state.get("favorites", [])
            if movie.get('title') not in [m.get('title') for m in favs]:
                favs.append(movie)
                st.session_state["favorites"] = favs
                st.success("Added to favorites")
        st.markdown("</div>", unsafe_allow_html=True)
    # persist poster URL into CSV if we found a real image and CSV didn't have one
    try:
        title_str = str(movie.get('title') or "").strip()
        if title_str and (not movie.get('poster_path')) and poster and poster != PLACEHOLDER_URL:
            # update CSV safely; persist only if title exists and CSV lacked poster
            set_poster_for_title(title_str, poster)
    except Exception:
        # don't crash UI if persistence fails
        pass


st.title("🎬 Movie Recommender (Interactive)")
st.markdown("Use the sidebar to configure the (optional) TMDB key for posters or run the embedded scraper to update local data.")

title = st.text_input("Enter a movie title you like (optional):")
genre = st.text_input("Or enter a genre to get recommendations by genre:")

if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

if st.button("Recommend"):
    # If genre provided, use local CSV filtered by genre
    if genre:
        results = load_movies_by_genre(genre, top_n=NUM)
        if not results:
            st.info("No movies found for that genre in local data. Try running the scraper or check spelling.")
        else:
            per_row = 3
            rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for c, r in zip(cols, row):
                    with c:
                        show_movie_card(r, TMDB_API_KEY or None)
    elif title:
        # fallback: show movies with matching title substring from local data
        df = load_movies()
        matches = df[df['title'].str.contains(title, case=False, na=False)]
        if matches.empty:
            st.info("No local matches for that title. Try running the scraper to refresh data.")
        else:
            results = matches.head(NUM).to_dict(orient='records')
            per_row = 3
            rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for c, r in zip(cols, row):
                    with c:
                        show_movie_card(r, TMDB_API_KEY or None)
    else:
        st.warning("Type a title or a genre first")

st.markdown("---")
if st.session_state.get("favorites"):
    st.header("Favorites")
    for m in st.session_state["favorites"]:
        st.write(f"- {m.get('title')} ({m.get('score'):.3f})")
    # allow client-side download of favorites JSON
    if st.button("Export favorites to server"):
        # ensure data dir exists
        p = Path(__file__).resolve().parents[1] / "data"
        p.mkdir(exist_ok=True)
        out = p / "favorites.json"
        with out.open("w", encoding="utf8") as f:
            json.dump(st.session_state["favorites"], f, ensure_ascii=False, indent=2)
        st.success(f"Exported favorites to {out}")

    # download button (client-side)
    import io
    buf = io.BytesIO()
    buf.write(json.dumps(st.session_state["favorites"], ensure_ascii=False, indent=2).encode('utf-8'))
    buf.seek(0)
    st.download_button("Download favorites (JSON)", data=buf, file_name="favorites.json", mime="application/json")

# Manual refresh button
st.sidebar.markdown("---")
st.sidebar.write("No backend API configured — the app reads local CSV and can trigger the embedded scraper.")

# Auto-refresh loop: when enabled, periodically rerun the app which will cause the UI to reflect updated server data.
if AUTO_REFRESH:
    # store control flag in session state so user can uncheck to stop
    st.session_state.setdefault("_auto_refresh_on", True)
    st.session_state["_auto_refresh_on"] = True
    placeholder = st.empty()
    # Blocking loop that sleeps then triggers a rerun; Streamlit will re-run script after experimental_rerun
    # This is intentionally simple and user-controlled via the sidebar checkbox.
    try:
        time.sleep(POLL_INTERVAL)
        # simply rerun to refresh local data view
        st.experimental_rerun()
    except Exception:
        # on any interruption just continue (user may have unchecked)
        pass
