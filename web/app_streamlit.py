import streamlit as st
import requests
from typing import Optional
from pathlib import Path
import json
import io
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
from src.user_store import learning_summary, record_feedback, rerank_results_for_learning
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
    @import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');
    :root {
        --bg-1: #0b0b0f;
        --bg-2: #14141b;
        --accent: #ff2d55;
        --accent-soft: #ff4d6d;
        --card: #16161d;
        --card-2: #1c1c24;
        --text: #f2f2f3;
        --muted: #b0b3c2;
        --line: #232330;
    }
    .stApp {
        background:
            radial-gradient(1200px 800px at 85% 0%, rgba(255,45,85,0.20), transparent 60%),
            radial-gradient(900px 700px at 10% 20%, rgba(255,77,109,0.12), transparent 55%),
            linear-gradient(180deg, var(--bg-1), var(--bg-2));
        color: var(--text);
        font-family: "Space Grotesk", system-ui, -apple-system, sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: "Unbounded", system-ui, -apple-system, sans-serif;
        letter-spacing: 0.02em;
    }
    .block-container {padding-top: 24px; padding-bottom: 80px;}
    .stMarkdown, .stTextInput label, .stSlider label, .stCheckbox label {color: var(--text);}
    .stTextInput > div > div > input {
        background: var(--card-2);
        border: 1px solid var(--line);
        color: var(--text);
        border-radius: 12px;
    }
    .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-soft));
        color: white;
        border: 0;
        border-radius: 12px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stButton > button:hover {filter: brightness(1.05);}
    .app-shell {max-width: 1200px; margin: 0 auto;}
    .topbar {
        display:flex; align-items:center; justify-content:space-between;
        background: rgba(10,10,14,0.6);
        border:1px solid var(--line);
        padding: 14px 20px; border-radius: 18px; margin-bottom: 24px;
        backdrop-filter: blur(8px);
    }
    .brand {font-size: 20px; font-weight: 700;}
    .brand span {color: var(--accent);}
    .top-links {display:flex; gap:18px; color: var(--muted); font-weight: 500;}
    .hero {
        border:1px solid var(--line);
        background:
            radial-gradient(500px 400px at 85% 20%, rgba(255,45,85,0.35), transparent 60%),
            linear-gradient(120deg, rgba(20,20,27,0.9), rgba(12,12,16,0.9));
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.35);
        margin-bottom: 28px;
    }
    .hero-tag {
        display:inline-block; font-size:12px; font-weight:600; letter-spacing:0.08em;
        text-transform:uppercase; color:#ffe2e8; background:#2a121a;
        border:1px solid #5a1d2b; padding:4px 10px; border-radius:999px; margin-bottom:10px;
    }
    .hero-title {font-size: 32px; margin: 0 0 8px 0;}
    .hero-meta {color: var(--muted); font-size: 13px; margin-bottom: 12px;}
    .movie-card {
        border:1px solid var(--line);
        background: linear-gradient(180deg, var(--card), var(--card-2));
        padding:12px; border-radius:16px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.30);
    }
    .movie-title {margin: 10px 0 6px 0;}
    .section-title {font-size: 22px; margin: 10px 0 8px 0;}
    .hero-dots {text-align: right; font-size: 14px; margin-bottom: 6px;}
    .hero-dot {color:#4a4a57; margin-left:4px;}
    .hero-dot.active {color: var(--accent);}
    .rating-stars {display:flex; align-items:center; gap:8px; margin: 6px 0 8px 0;}
    .stars {letter-spacing:2px; font-size:16px;}
    .star-on {color:#ffb703;}
    .star-off {color:#3a3a45;}
    .rating-badge {background:#101018; border:1px solid var(--line); padding:6px 10px; border-radius:10px; font-weight:600;}
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


st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
topbar_left, topbar_mid, topbar_right = st.columns([2, 5, 3])
with topbar_left:
    st.markdown("<div class='brand'>Filmi <span>Duniya</span></div>", unsafe_allow_html=True)
with topbar_mid:
    st.markdown(
        "<div class='top-links'>Movies <span>•</span> Series <span>•</span> TV Shows</div>",
        unsafe_allow_html=True,
    )
with topbar_right:
    title = st.text_input("Search", placeholder="Search movies", label_visibility="collapsed")

st.markdown("Discover new releases and personalized recommendations.")

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

featured_list = _get_featured_new_releases(limit=6)
if featured_list:
    st.session_state.setdefault("hero_index", 0)
    hero_index = st.session_state["hero_index"] % len(featured_list)
    featured = featured_list[hero_index]
    featured_poster = fetch_poster_url(
        str(featured.get("title", "")),
        featured.get("poster_path") if isinstance(featured.get("poster_path"), str) else None,
        TMDB_API_KEY or None,
        size="w1280",
    )
    dot_html = "<div class='hero-dots'>" + "".join(
        f"<span class='hero-dot {'active' if i == hero_index else ''}'>●</span>"
        for i in range(len(featured_list))
    ) + "</div>"
    st.markdown(dot_html, unsafe_allow_html=True)
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    hcols = st.columns([3, 2])
    with hcols[0]:
        st.markdown("<div class='hero-tag'>New Release</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hero-title'>{featured.get('title','Untitled')}</div>", unsafe_allow_html=True)
        tmdb_info = fetch_tmdb_info(str(featured.get("title", "")), TMDB_API_KEY or None)
        rating_val = tmdb_info.get("rating") if tmdb_info else featured.get("rating", 0)
        st.markdown(render_star_rating(rating_val), unsafe_allow_html=True)
        st.write(featured.get("description", "") or featured.get("overview", ""))
        st.markdown(f"**Genres:** {featured.get('genres', '')}")
        st.markdown(f"**Director:** {featured.get('director', '')}")
    with hcols[1]:
        featured_watch_url = fetch_watch_url(str(featured.get("title", "")), TMDB_API_KEY or None)
        st.markdown(
            f"<a href='{featured_watch_url}' target='_blank'><img src='{featured_poster}' style='width:100%;border-radius:12px;'></a>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Filter by Genres</div>", unsafe_allow_html=True)
all_genres = _get_all_genres()
selected_genres = []
if all_genres:
    with st.expander("Choose genres", expanded=True):
        cols = st.columns(3)
        for i, g in enumerate(all_genres):
            with cols[i % 3]:
                if st.checkbox(g, key=f"genre-{g}"):
                    selected_genres.append(g)
else:
    st.info("No genres found in local data.")

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
        if st.button("Allow and start ", use_container_width=True):
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
if mode == "Mood Based" and detected_mood:
    base = _apply_content_filter(load_movies())
    genres = MOOD_TO_GENRES.get(detected_mood, [])
    if genres:
        if title:
            title_matches = base[base["title"].str.contains(title, case=False, na=False)]
            mask = title_matches["genres"].fillna("").str.lower().apply(
                lambda x: any(g.lower() in x for g in genres)
            )
            matches = title_matches[mask]
        else:
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
elif title:
    df = _apply_content_filter(load_movies())
    matches = df[df['title'].str.contains(title, case=False, na=False)]
    results = [] if matches.empty else matches.head(NUM).to_dict(orient='records')
else:
    results = []

if mode == "Mood Based" and detected_mood:
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
elif title:
    if not results:
        st.info("No local matches for that title. Try running the scraper to refresh data.")
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
    st.caption(
        f"Learning events: {summary.get('feedback_events', 0)} | Mood history: {summary.get('mood_counts', {})}"
    )

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
