import streamlit as st
import requests
from typing import Optional
from pathlib import Path
import json
import io
from difflib import get_close_matches
import re
import sys
import os
import time
from urllib.parse import quote_plus
import pandas as pd

try:
    import sounddevice as _sd  # noqa: F401
    _HAS_SOUNDDEVICE = True
except Exception:
    _HAS_SOUNDDEVICE = False

# Ensure project root is on sys.path so `src` imports work when Streamlit
# runs with a different CWD. This inserts the repo root (one level up from
# this `web/` folder) at the front of sys.path.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import TMDB_API_KEY as DEFAULT_TMDB_KEY
from src.data__loader import load_movies, set_poster_for_title
from src.recomender import MOOD_TO_GENRES
from src.scraper import scrape_top_n_movies
from src.user_store import learning_progress, learning_summary, record_feedback, rerank_results_for_learning
from src.multimodal_mood import (
    detect_multimodal_mood_backend,
)

# placeholder image used when no poster can be found
PLACEHOLDER_URL = "https://via.placeholder.com/200x300?text=No+Poster"

# Configure default emotion model path only if a known local model file exists.
_MODEL_CANDIDATES = [
    repo_root / "models" / "emotion_model.h5",
    repo_root / "src" / "fer2013_mini_XCEPTION.102-0.66.hdf5",
]
for _p in _MODEL_CANDIDATES:
    if _p.exists():
        os.environ.setdefault("EMOTION_MODEL_PATH", str(_p))
        break

st.set_page_config(page_title="Flimi Duniya", layout="wide")

# cinematic dark UI styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Manrope:wght@400;500;700;800&display=swap');
    :root {
        --bg-1: #02070a;
        --bg-2: #061116;
        --accent: #47f5ff;
        --accent-soft: #1cc5d0;
        --card: #091219;
        --card-2: #0d1b24;
        --text: #e9f6ff;
        --muted: #9fb9c6;
        --line: #1b323e;
        --soft-glow: rgba(71,245,255,0.18);
    }
    .stApp {
        background:
            radial-gradient(1200px 800px at 85% 0%, rgba(71,245,255,0.16), transparent 60%),
            radial-gradient(900px 700px at 10% 20%, rgba(28,197,208,0.10), transparent 55%),
            linear-gradient(180deg, var(--bg-1), var(--bg-2));
        color: var(--text);
        font-family: "Manrope", system-ui, -apple-system, sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: "Bebas Neue", system-ui, -apple-system, sans-serif;
        letter-spacing: 0.08em;
    }
    .block-container {padding-top: 12px; padding-bottom: 80px;}
    .stMarkdown, .stTextInput label, .stSlider label, .stCheckbox label {color: var(--text);}
    .stTextInput > div > div {
        background:
            radial-gradient(900px 400px at 80% 0%, rgba(71,245,255,0.08), transparent 60%),
            linear-gradient(180deg, var(--bg-1), var(--bg-2));
        border: 1px solid var(--line);
        border-radius: 14px;
    }
    .stTextInput > div > div > input {
        background: transparent;
        border: 0;
        color: var(--text);
        border-radius: 14px;
        font-size: 17px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #18b3bd, #47f5ff);
        color: #001216;
        border: 1px solid rgba(71,245,255,0.6);
        border-radius: 999px;
        padding: 10px 16px;
        font-weight: 800;
        letter-spacing: 0.03em;
    }
    .stButton > button:hover {
        filter: brightness(1.08);
        box-shadow: 0 0 20px var(--soft-glow);
    }
    .app-shell {max-width: 1240px; margin: 0 auto;}
    .topbar-glass {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 14px 18px;
        background: linear-gradient(90deg, rgba(4,10,14,0.90), rgba(8,18,23,0.86));
        backdrop-filter: blur(6px);
        margin-bottom: 14px;
    }
    .brand {font-size: 30px; font-family: "Bebas Neue", sans-serif; letter-spacing: 0.05em;}
    .brand span {color: var(--accent);}
    .top-links {
        display:flex; gap:16px; justify-content:center; align-items:center;
        color: #d3f8ff; font-size:12px; letter-spacing:0.16em; text-transform:uppercase; font-weight:700;
    }
    .banner-panel {
        position: relative;
        min-height: 420px;
        border-radius: 24px;
        border: 1px solid rgba(173,240,255,0.25);
        overflow: hidden;
        margin-bottom: 18px;
        background:
          linear-gradient(90deg, rgba(0,10,14,0.78), rgba(0,10,14,0.25)),
          linear-gradient(180deg, rgba(0,6,10,0.15), rgba(0,3,8,0.8));
        box-shadow: 0 26px 70px rgba(0,0,0,0.48);
    }
    .banner-content {
        position:absolute;
        left: 24px;
        bottom: 26px;
        max-width: 58%;
    }
    .banner-kicker {
        color: #7cefff;
        letter-spacing: 0.20em;
        font-size: 11px;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 700;
    }
    .banner-title {
        color: #67f2ff;
        font-size: 62px;
        line-height: 0.9;
        text-transform: uppercase;
        margin: 0 0 8px 0;
        font-family: "Bebas Neue", sans-serif;
        text-shadow: 0 0 24px rgba(71,245,255,0.22);
    }
    .banner-sub {
        color: #d2e8f0;
        font-size: 14px;
        line-height: 1.45;
        max-width: 540px;
    }
    .banner-step {
        position:absolute;
        right:20px;
        top:20px;
        color: #d9fbff;
        font-size: 11px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        text-align:right;
        line-height:1.8;
        opacity: 0.8;
    }
    .hero {
        border:1px solid var(--line);
        background:
            radial-gradient(500px 400px at 85% 20%, rgba(71,245,255,0.28), transparent 60%),
            linear-gradient(120deg, rgba(8,15,21,0.92), rgba(4,10,14,0.92));
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.35);
        margin-bottom: 28px;
    }
    .hero-tag {
        display:inline-block; font-size:12px; font-weight:600; letter-spacing:0.08em;
        text-transform:uppercase; color:#d9fbff; background:#0a1c26;
        border:1px solid #245262; padding:4px 10px; border-radius:999px; margin-bottom:10px;
    }
    .hero-title {font-size: 44px; margin: 0 0 8px 0; line-height: 0.95;}
    .hero-meta {color: var(--muted); font-size: 13px; margin-bottom: 12px;}
    .movie-card {
        border:1px solid var(--line);
        background: linear-gradient(180deg, var(--card), var(--card-2));
        padding:12px; border-radius:16px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.30);
    }
    .movie-title {margin: 10px 0 6px 0; color: #dffcff;}
    .section-title {
        font-size: 34px;
        margin: 14px 0 8px 0;
        color: #73f4ff;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }
    .hero-dots {text-align: right; font-size: 14px; margin-bottom: 6px;}
    .hero-dot {color:#2a5565; margin-left:4px;}
    .hero-dot.active {color: var(--accent);}
    .rating-stars {display:flex; align-items:center; gap:8px; margin: 6px 0 8px 0;}
    .stars {letter-spacing:2px; font-size:16px;}
    .star-on {color:#ffb703;}
    .star-off {color:#245465;}
    .rating-badge {background:#07121a; border:1px solid var(--line); padding:6px 10px; border-radius:10px; font-weight:600;}
    .genre-strip {
        border: 1px solid var(--line);
        background:
            radial-gradient(1200px 800px at 85% 0%, rgba(71,245,255,0.16), transparent 60%),
            radial-gradient(900px 700px at 10% 20%, rgba(28,197,208,0.10), transparent 55%),
            linear-gradient(180deg, var(--bg-1), var(--bg-2));
        border-radius: 16px;
        padding: 7px 12px;
        margin: 10px 0 10px 0;
        max-height: 86px;
        overflow-y: auto;
    }
    .strip-title {
        font-size: 11px;
        color: #8defff;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin: 0 0 4px 0;
        font-weight: 700;
    }
    .genre-strip [data-testid="stMultiSelect"] {margin-bottom: 0;}
    .genre-strip [data-baseweb="tag"] {font-size: 11px;}
    .genre-strip [data-baseweb="select"] > div {
        background:
            radial-gradient(1200px 800px at 85% 0%, rgba(71,245,255,0.16), transparent 60%),
            radial-gradient(900px 700px at 10% 20%, rgba(28,197,208,0.10), transparent 55%),
            linear-gradient(180deg, var(--bg-1), var(--bg-2)) !important;
        border-color: var(--line) !important;
    }
    .genre-strip [data-baseweb="select"] input {
        color: var(--text) !important;
    }
    .genre-strip [data-baseweb="tag"] {
        background: rgba(22, 45, 56, 0.75) !important;
        color: var(--text) !important;
        border: 1px solid var(--line) !important;
    }
    .genre-note {
        color: var(--muted);
        font-size: 11px;
        margin-bottom: 4px;
    }
    @media (max-width: 900px) {
        .banner-content {max-width: 90%; left:16px; bottom:16px;}
        .banner-title {font-size: 44px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# TMDB key: prefer Streamlit secrets, then env/config, otherwise empty
secrets_key = "785c5f1bd5e3e823f06abdfe6168588e"
try:
    secrets_key = str(st.secrets.get("TMDB_API_KEY", "785c5f1bd5e3e823f06abdfe6168588e")).strip()
except Exception:
    secrets_key = ""
default_key = secrets_key or (
    DEFAULT_TMDB_KEY if DEFAULT_TMDB_KEY and DEFAULT_TMDB_KEY != "785c5f1bd5e3e823f06abdfe6168588e" else ""
)

tmdb_input = st.sidebar.text_input("TMDB API key", type="password", value=default_key)
TMDB_API_KEY = (tmdb_input or "").strip()
if TMDB_API_KEY:
    st.sidebar.success("Using TMDB for posters and ratings.")
else:
    st.sidebar.warning("TMDB API key missing; posters/ratings may use placeholders.")
st.sidebar.header("Recommendation Mode")
mode = "Mood Based"
st.sidebar.caption("Mood-based recommendation is always enabled.")
NUM = st.sidebar.slider("Number of recommendations", min_value=1, max_value=50, value=30)
AUTO_REFRESH = st.sidebar.checkbox("Enable auto-refresh (poll backend)", value=True)
POLL_INTERVAL = 7
st.sidebar.caption("Poll interval is fixed at 7 seconds.")
content_types = st.sidebar.multiselect(
    "Content type",
    options=["Movies", "TV Shows"],
    default=["Movies", "TV Shows"],
)
st.sidebar.markdown("[Open Netflix India](https://www.netflix.com/in/)")


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
    size: str = "w500",
) -> str:
    """Try to fetch a poster URL from TMDB, fall back to placeholder."""

    # normalize title/poster inputs
    title_str = (title or "").strip()
    normalized_poster = _clean_poster_value(poster_path)

    # prefer existing poster reference
    if normalized_poster:
        if normalized_poster.startswith("/"):
            return f"https://image.tmdb.org/t/p/{size}{normalized_poster}"
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
                    return f"https://image.tmdb.org/t/p/{size}{tmdb_path}"
        except Exception:
            pass

    return PLACEHOLDER_URL


@st.cache_data(ttl=60 * 60)
def fetch_watch_url(
    title: Optional[str],
    tmdb_api_key: Optional[str],
    media_type: Optional[str] = None,
) -> str:
    """Return a Netflix search URL for a title."""
    title_str = (title or "").strip()
    if not title_str:
        return "https://www.netflix.com"
    return f"https://www.netflix.com/search?q={quote_plus(title_str)}"


@st.cache_data(ttl=60 * 60)
def fetch_banner_image_url(
    title: Optional[str],
    poster_path: Optional[str],
    banner_path: Optional[str],
    movie_id: Optional[object],
    tmdb_api_key: Optional[str],
    size: str = "w1280",
) -> str:
    """Prefer TMDB backdrop via movie_id, then title search, then poster."""
    banner_str = (banner_path or "").strip()
    if banner_str:
        # Accept only movie-image sources (TMDB path or TMDB image CDN URL).
        if banner_str.startswith("/"):
            return f"https://image.tmdb.org/t/p/{size}{banner_str}"
        if "image.tmdb.org" in banner_str:
            return banner_str

    title_str = (title or "").strip()

    if tmdb_api_key and title_str:
        # Prefer exact movie by TMDB id to avoid wrong backdrop matches.
        try:
            mid = int(float(movie_id)) if movie_id is not None and str(movie_id).strip() else None
        except (TypeError, ValueError):
            mid = None
        if mid:
            try:
                details_resp = requests.get(
                    f"https://api.themoviedb.org/3/movie/{mid}",
                    params={"api_key": tmdb_api_key},
                    timeout=6,
                )
                details_resp.raise_for_status()
                details = details_resp.json() or {}
                backdrop = details.get("backdrop_path")
                if backdrop:
                    return f"https://image.tmdb.org/t/p/{size}{backdrop}"
            except Exception:
                pass

        try:
            url = "https://api.themoviedb.org/3/search/movie"
            resp = requests.get(
                url,
                params={"api_key": tmdb_api_key, "query": title_str, "page": 1},
                timeout=6,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if results:
                backdrop_path = results[0].get("backdrop_path")
                if backdrop_path:
                    return f"https://image.tmdb.org/t/p/{size}{backdrop_path}"
        except Exception:
            pass
    return fetch_poster_url(title_str, poster_path, tmdb_api_key, size="w780")


def show_movie_card(movie: dict[str, object], tmdb_api_key: Optional[str]):
    poster_value = movie.get('poster_path')
    title_str = str(movie.get("title", ""))
    media_type = str(movie.get("media_type", "movie")).strip().lower()
    poster = fetch_poster_url(
        title_str,
        poster_value if isinstance(poster_value, str) else None,
        tmdb_api_key,
        size="w780",
    )
    watch_url = fetch_watch_url(title_str, tmdb_api_key, media_type=media_type)
    if not poster.strip():
        poster = PLACEHOLDER_URL
    st.markdown(
        f"<a href='{watch_url}' target='_blank'><img src='{poster}' style='width:230px;border-radius:12px;'></a>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='movie-title'>{movie.get('title','Untitled')}</h3>", unsafe_allow_html=True)
    tmdb_info = fetch_tmdb_info(str(movie.get("title", "")), tmdb_api_key)
    rating_val = tmdb_info.get("rating") if tmdb_info else None
    if rating_val is None or rating_val == "":
        rating_val = movie.get("rating", 0)
    st.markdown(render_star_rating(rating_val), unsafe_allow_html=True)
    st.markdown(f"**Genres:** {movie.get('genres', '')}  ")
    st.markdown(f"**Director:** {movie.get('director', '')}  ")
    if st.button("Add to favorites", key=f"fav-{movie.get('title','')}"):
        favs = st.session_state.get("favorites", [])
        if movie.get('title') not in [m.get('title') for m in favs]:
            favs.append(movie)
            st.session_state["favorites"] = favs
            record_feedback(
                mood=st.session_state.get("detected_mood"),
                movie=movie,
                favorite=True,
            )
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
    

@st.cache_data(ttl=10 * 60)
def _get_all_genres() -> list[str]:
    df = _apply_content_filter(load_movies())
    if "genres" not in df.columns or df.empty:
        return []
    genres = set()
    for raw in df["genres"].fillna("").astype(str).tolist():
        if not raw.strip():
            continue
        # support common separators
        for g in raw.replace(",", "|").replace("/", "|").split("|"):
            g = g.strip()
            if g:
                genres.add(g)
    return sorted(genres, key=lambda x: x.lower())


def _apply_content_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset by sidebar content type selection when media_type exists."""
    if df.empty:
        return df
    selected = set(content_types or [])
    if selected == {"Movies", "TV Shows"} or not selected:
        return df
    if "media_type" not in df.columns:
        return df
    allowed = set()
    if "Movies" in selected:
        allowed.add("movie")
    if "TV Shows" in selected:
        allowed.add("tv")
    if not allowed:
        return df.iloc[0:0]
    mt = df["media_type"].fillna("").astype(str).str.lower()
    return df[mt.isin(allowed)]


def _resolve_voice_title(spoken_text: str, df: pd.DataFrame) -> Optional[str]:
    text = str(spoken_text or "").strip()
    if not text or df.empty or "title" not in df.columns:
        return None

    lowered = text.lower()
    for noise in [
        "play", "show", "search", "movie", "film", "recommend", "please",
        "for me", "watch", "find", "suggest",
    ]:
        lowered = lowered.replace(noise, " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        lowered = text.lower()

    titles = df["title"].fillna("").astype(str).tolist()
    contains = [t for t in titles if lowered in t.lower() or t.lower() in lowered]
    if contains:
        return contains[0]
    close = get_close_matches(lowered, [t.lower() for t in titles], n=1, cutoff=0.42)
    if close:
        key = close[0]
        for t in titles:
            if t.lower() == key:
                return t
    close = get_close_matches(text, titles, n=1, cutoff=0.42)
    return close[0] if close else None


st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
st.markdown("<div class='topbar-glass'>", unsafe_allow_html=True)
nav_left, nav_mid, nav_right = st.columns([2, 5, 3])
with nav_left:
    st.markdown("<div class='brand'>Filmi <span>Duniya</span></div>", unsafe_allow_html=True)
with nav_mid:
    st.markdown(
        "<div class='top-links'>Movies <span>•</span> Series <span>•</span> TV Shows</div>",
        unsafe_allow_html=True,
    )
with nav_right:
    typed_title = st.text_input("Search", placeholder="Search movies", label_visibility="collapsed", key="search_title")
st.markdown("</div>", unsafe_allow_html=True)

all_genres = _get_all_genres()
if all_genres:
    st.markdown("<div class='genre-strip'>", unsafe_allow_html=True)
    st.markdown("<div class='strip-title'>Filter Lane</div>", unsafe_allow_html=True)
    st.markdown("<div class='genre-note'>Choose one or more genres to focus the stream.</div>", unsafe_allow_html=True)
    selected_genres = st.multiselect(
        "Genres",
        options=all_genres,
        default=[],
        label_visibility="collapsed",
        placeholder="Filter recommendations by genres...",
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    selected_genres = []

banner_df = _apply_content_filter(load_movies())
banner_movies: list[dict] = []
if not banner_df.empty:
    ranked = banner_df.copy()
    if "popularity" in ranked.columns:
        ranked["popularity"] = pd.to_numeric(ranked["popularity"], errors="coerce")
        ranked = ranked.sort_values(by="popularity", ascending=False)
    elif "rating" in ranked.columns:
        ranked["rating"] = pd.to_numeric(ranked["rating"], errors="coerce")
        ranked = ranked.sort_values(by="rating", ascending=False)
    banner_movies = ranked.head(6).to_dict(orient="records")

st.session_state.setdefault("hero_index", 0)
banner_index = st.session_state["hero_index"] % max(1, len(banner_movies))
banner_main = banner_movies[banner_index] if banner_movies else {"title": "Featured Movies", "genres": "Movies|Series"}
banner_poster = fetch_banner_image_url(
    str(banner_main.get("title", "")),
    banner_main.get("poster_path") if isinstance(banner_main.get("poster_path"), str) else None,
    banner_main.get("banner_path") if isinstance(banner_main.get("banner_path"), str) else None,
    banner_main.get("movie_id"),
    TMDB_API_KEY or None,
    size="original",
)
banner_subtitles = " • ".join(
    [
        str(m.get("title", ""))
        for i, m in enumerate(banner_movies)
        if i != banner_index and str(m.get("title", "")).strip()
    ][:2]
)
if not banner_subtitles:
    banner_subtitles = "Live recommendations based on mood, voice, and your learning profile."
banner_desc = str(banner_main.get("description", "") or banner_main.get("overview", "")).strip()
if banner_desc:
    banner_desc = banner_desc[:170] + ("..." if len(banner_desc) > 170 else "")
else:
    banner_desc = "Live recommendations based on mood, voice, and your learning profile."
banner_director = str(banner_main.get("director", "") or "Unknown")
banner_genres = str(banner_main.get("genres", "") or "Movies|Series")
try:
    banner_rating = f"{float(banner_main.get('rating', 0.0)):.1f}"
except (TypeError, ValueError):
    banner_rating = "NA"
banner_watch_url = fetch_watch_url(str(banner_main.get("title", "")), TMDB_API_KEY or None)

st.markdown(
    f"""
    <a href="{banner_watch_url}" target="_blank" style="text-decoration:none;">
    <div class='banner-panel' style="
      background:
        linear-gradient(90deg, rgba(2,14,20,0.56), rgba(2,14,20,0.10)),
        linear-gradient(180deg, rgba(120,235,255,0.14), rgba(0,3,8,0.56)),
        url('{banner_poster}') center/contain no-repeat;
    ">
      <div class='banner-step'>01<br>02<br><strong>03</strong><br>04</div>
      <div class='banner-content'>
        <div class='banner-kicker'>Cinematic Mood Journey</div>
        <div class='banner-title'>{str(banner_main.get('title','Featured Movies')).upper()}</div>
        <div class='banner-sub'>
          {banner_desc}<br>
          Genres: {banner_genres} • Director: {banner_director} • Rating: {banner_rating}/10<br>
          {banner_subtitles}
        </div>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("page", "home")
st.session_state.setdefault("user_ratings", {})


def render_detail_page(movie: dict[str, object], tmdb_api_key: Optional[str]):
    if st.button("Back"):
        st.session_state["page"] = "home"
        st.experimental_rerun()

    tmdb_info = fetch_tmdb_info(str(movie.get("title", "")), tmdb_api_key)
    poster = fetch_poster_url(
        str(movie.get("title", "")),
        movie.get("poster_path") if isinstance(movie.get("poster_path"), str) else None,
        tmdb_api_key,
        size="w1280",
    )
    media_type = str(movie.get("media_type", "movie")).strip().lower()
    watch_url = fetch_watch_url(str(movie.get("title", "")), tmdb_api_key, media_type=media_type)
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    cols = st.columns([2, 3])
    with cols[0]:
        st.markdown(
            f"<a href='{watch_url}' target='_blank'><img src='{poster}' style='width:100%;border-radius:12px;'></a>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(f"<div class='hero-title'>{movie.get('title','Untitled')}</div>", unsafe_allow_html=True)
        rating_val = tmdb_info.get("rating") if tmdb_info else movie.get("rating", 0)
        st.markdown(render_star_rating(rating_val), unsafe_allow_html=True)
        desc = tmdb_info.get("overview") if tmdb_info else ""
        if not desc:
            desc = movie.get("description", "") or movie.get("overview", "")
        if desc:
            st.write(desc)
        st.markdown(f"**Genres:** {movie.get('genres', '')}")
        st.markdown(f"**Director:** {movie.get('director', '')}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Your Rating</div>", unsafe_allow_html=True)
    current = st.session_state["user_ratings"].get(movie.get("title", ""), 0.0)
    user_rating = st.slider("Rate this movie", 0.0, 5.0, float(current), 0.5)
    if st.button("Save Rating"):
        st.session_state["user_ratings"][movie.get("title", "")] = user_rating
        record_feedback(
            mood=st.session_state.get("detected_mood"),
            movie=movie,
            rating=float(user_rating),
        )
        st.success(f"Saved your rating: {user_rating:.1f}")


if st.session_state.get("page") == "detail" and st.session_state.get("selected_movie"):
    render_detail_page(st.session_state["selected_movie"], TMDB_API_KEY or None)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

@st.cache_data(ttl=10 * 60)
def _get_featured_new_release():
    df = _apply_content_filter(load_movies())
    if df.empty:
        return None
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
        if df["popularity"].notna().any():
            return df.sort_values(by="popularity", ascending=False).iloc[0].to_dict()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        if df["rating"].notna().any():
            return df.sort_values(by="rating", ascending=False).iloc[0].to_dict()
    return df.iloc[0].to_dict()


@st.cache_data(ttl=10 * 60)
def _get_featured_new_releases(limit: int = 5) -> list[dict]:
    df = _apply_content_filter(load_movies())
    if df.empty:
        return []
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
        df = df.sort_values(by="popularity", ascending=False)
    elif "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.sort_values(by="rating", ascending=False)
    return df.head(limit).to_dict(orient="records")


def _maybe_run_daily_scraper(tmdb_api_key: Optional[str]) -> None:
    """Run scraper once every 24h (movies + TV) to refresh local DB/posters."""
    if not tmdb_api_key:
        return
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = data_dir / ".last_daily_scrape_ts"
    lock_path = data_dir / ".daily_scrape.lock"
    now = time.time()
    interval_seconds = 86400

    # if a recent lock exists, avoid concurrent runs
    if lock_path.exists():
        try:
            if now - lock_path.stat().st_mtime < 2 * 3600:
                return
        except Exception:
            return

    last_ts = 0.0
    if stamp_path.exists():
        try:
            last_ts = float(stamp_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            last_ts = 0.0

    if now - last_ts < interval_seconds:
        return

    try:
        lock_path.write_text(str(now), encoding="utf-8")
        scrape_top_n_movies(
            n=400,
            append=True,
            max_per_run=200,
            include_tv=True,
            force=False,
        )
        stamp_path.write_text(str(now), encoding="utf-8")
    except Exception:
        pass
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


_maybe_run_daily_scraper(TMDB_API_KEY or None)

@st.cache_data(ttl=60 * 60)
def fetch_tmdb_info(title: str, tmdb_api_key: Optional[str]) -> dict:
    if not tmdb_api_key or not title:
        return {}
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        resp = requests.get(
            url,
            params={"api_key": tmdb_api_key, "query": title, "page": 1},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if results:
            top = results[0]
            return {
                "rating": top.get("vote_average"),
                "vote_count": top.get("vote_count"),
                "overview": top.get("overview"),
                "poster_path": top.get("poster_path"),
            }
    except Exception:
        pass
    return {}


def render_star_rating(rating_out_of_10: float) -> str:
    try:
        r10 = float(rating_out_of_10)
    except (TypeError, ValueError):
        r10 = 0.0
    r5 = max(0.0, min(5.0, r10 / 2))
    full = int(round(r5))
    stars = ""
    for i in range(5):
        cls = "star-on" if i < full else "star-off"
        stars += f"<span class='{cls}'>★</span>"
    return f"""
    <div class='rating-stars'>
        <div class='rating-badge'>Rating {r5:.1f}</div>
        <div class='stars'>{stars}</div>
    </div>
    """

if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

st.markdown("<div class='section-title'>Recommendations</div>", unsafe_allow_html=True)

# Mood-based flow using backend-only private sensing
if mode == "Mood Based":
    st.session_state.setdefault("recent_detected_moods", [])
    st.session_state.setdefault("private_scan_allowed", False)
    st.session_state.setdefault("last_private_scan_ts", 0.0)
    st.session_state.setdefault("scan_interval_sec", 6.0)

    if not st.session_state["private_scan_allowed"]:
        st.subheader("Private Mood Scan")
        st.caption("Allow once. After that, mood scanning runs automatically in backend.")
        if st.button("Allow and start", use_container_width=True):
            st.session_state["private_scan_allowed"] = True 
            st.rerun()
    else:
        now = time.time()
        should_scan = (
            ("detected_mood" not in st.session_state)
            or (
                now - float(st.session_state.get("last_private_scan_ts", 0.0))
                >= float(st.session_state.get("scan_interval_sec", 6.0))
            )
        )
        if should_scan:
            st.session_state["last_private_scan_ts"] = now
            try:
                final_details = detect_multimodal_mood_backend(voice_duration_sec=3.0)
            except RuntimeError:
                final_details = None

            if final_details:
                spoken_text = str(final_details.get("spoken_text", "")).strip()
                if spoken_text:
                    st.session_state["last_voice_transcript"] = spoken_text
                    matched = _resolve_voice_title(spoken_text, _apply_content_filter(load_movies()))
                    if matched:
                        st.session_state["voice_title_query"] = matched
                    else:
                        st.session_state["voice_title_query"] = spoken_text

            mood = final_details["mood"] if final_details else None
            if mood:
                st.session_state["last_mood_source"] = str(final_details.get("source", "unknown"))
                top_moods = final_details.get("top_moods", [])
                recent = st.session_state.get("recent_detected_moods", [])
                if len(recent) >= 3 and all(m == mood for m in recent[-3:]) and len(top_moods) > 1:
                    first_name, first_score = top_moods[0]
                    second_name, second_score = top_moods[1]
                    if second_name != first_name and second_score >= (0.75 * first_score):
                        mood = second_name
                recent.append(mood)
                st.session_state["recent_detected_moods"] = recent[-6:]
                st.session_state["detected_mood"] = mood

        if st.session_state.get("detected_mood"):
            source = str(st.session_state.get("last_mood_source", "face")).upper()
            st.caption(f"Detected mood: {str(st.session_state.get('detected_mood')).upper()} | Source: {source}")
        if not _HAS_SOUNDDEVICE:
            st.caption("Voice backend unavailable in this runtime (install `sounddevice` + PortAudio). Using face-only.")
    st.markdown("---")

# auto-search on input/genre selection (no button)
detected_mood = st.session_state.get("detected_mood")
voice_title = str(st.session_state.get("voice_title_query", "")).strip()
title = str(typed_title or "").strip() or voice_title

if voice_title and not str(typed_title or "").strip():
    st.caption(f"Voice movie request: **{voice_title}**")

if title:
    base = _apply_content_filter(load_movies())
    exact = base[base["title"].fillna("").str.lower() == title.lower()]
    if exact.empty:
        matches = base[base["title"].fillna("").str.contains(title, case=False, na=False)]
    else:
        matches = exact
    if matches.empty:
        results = []
    else:
        if "rating" in matches.columns:
            matches["rating"] = pd.to_numeric(matches["rating"], errors="coerce")
            matches = matches.sort_values(by="rating", ascending=False)
        results = matches.head(NUM).to_dict(orient="records")
        if detected_mood:
            results = rerank_results_for_learning(results, detected_mood)
elif selected_genres:
    df = _apply_content_filter(load_movies())
    mask = df["genres"].fillna("").str.lower().apply(
        lambda x: any(g.lower() in x for g in selected_genres)
    )
    matches = df[mask]
    if matches.empty:
        results = []
    else:
        if "rating" in matches.columns:
            matches["rating"] = pd.to_numeric(matches["rating"], errors="coerce")
            matches = matches.sort_values(by="rating", ascending=False)
        results = matches.head(NUM).to_dict(orient="records")
elif mode == "Mood Based" and detected_mood:
    base = _apply_content_filter(load_movies())
    genres = MOOD_TO_GENRES.get(detected_mood, [])
    if genres:
        mask = base["genres"].fillna("").str.lower().apply(
            lambda x: any(g.lower() in x for g in genres)
        )
        matches = base[mask]
        if matches.empty:
            results = []
        else:
            if "rating" in matches.columns:
                matches["rating"] = pd.to_numeric(matches["rating"], errors="coerce")
                matches = matches.sort_values(by="rating", ascending=False)
            results = matches.head(NUM).to_dict(orient="records")
            results = rerank_results_for_learning(results, detected_mood)
    else:
        results = []
else:
    results = []

if title:
    if not results:
        st.info("No local matches for that title. Try another name.")
    else:
        per_row = 3
        rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for c, r in zip(cols, row):
                with c:
                    show_movie_card(r, TMDB_API_KEY or None)
elif mode == "Mood Based" and detected_mood:
    if not results:
        st.info("No movies found for the detected mood. Try a different title or re-scan.")
    else:
        per_row = 3
        rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for c, r in zip(cols, row):
                with c:
                    show_movie_card(r, TMDB_API_KEY or None)
elif selected_genres:
    if not results:
        st.info("No movies found for the selected genres. Try different genres.")
    else:
        per_row = 3
        rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for c, r in zip(cols, row):
                with c:
                    show_movie_card(r, TMDB_API_KEY or None)
else:
    if mode == "Mood Based":
        if not st.session_state.get("private_scan_allowed", False):
            st.warning("Allow to start mood-based recommendations.")  
        else:
            st.warning("Mood scan is running in backend. Recommendations will appear shortly.")
    else:
        st.warning("Type a title or pick genres first.")

st.markdown("---")
if st.session_state.get("favorites"):
    st.header("Favorites")
    for m in st.session_state["favorites"]:
        try:
            score_txt = f"{float(m.get('score', 0.0)):.3f}"
        except (TypeError, ValueError):
            score_txt = "0.000"
        st.write(f"- {m.get('title')} ({score_txt})")
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
    buf = io.BytesIO()
    buf.write(json.dumps(st.session_state["favorites"], ensure_ascii=False, indent=2).encode('utf-8'))
    buf.seek(0)
    st.download_button("Download favorites (JSON)", data=buf, file_name="favorites.json", mime="application/json")

if mode == "Mood Based":
    summary = learning_summary()
    progress = learning_progress(target_events=200)
    st.caption(
        f"Learning events: {summary.get('feedback_events', 0)} | Mood history: {summary.get('mood_counts', {})}"
    )
    st.caption(
        f"Model training progress: {progress.get('progress_pct', 0.0):.1f}% ({progress.get('events', 0)}/{progress.get('target_events', 200)}) • Stage: {progress.get('stage', 'cold-start')}"
    )
    st.progress(min(1.0, float(progress.get("progress_pct", 0.0)) / 100.0))

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

# auto-rotate hero every 5 seconds without blocking the rest of the page
if st.session_state.get("page") == "home":
    featured_list = _get_featured_new_releases(limit=6)
    if featured_list:
        last_tick = st.session_state.get("hero_last_tick", 0.0)
        now = time.time()
        if now - last_tick >= 5:
            st.session_state["hero_last_tick"] = now
            hero_index = st.session_state.get("hero_index", 0)
            st.session_state["hero_index"] = (hero_index + 1) % len(featured_list)
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
