import os
import re
import hashlib
import sqlite3
import altair as alt
from datetime import datetime
import time
import base64
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl

st.set_page_config(
    page_title="Movie Portal",
    page_icon="site_icon.png",
    layout="wide",
)

DB_PATH = "movie_recommendation_system.db"
SIM_CACHE_FILE = "item_similarity_cache.npz"
USER_ITEM_CACHE_FILE = "user_item_cache.npz"
GENRE_CACHE_FILE = "genre_cache.npz"


def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()


def get_poster_path(movie_id: int, folder: str = "posters"):
    """Шукає постер posters/<movie_id>.(png/jpg/jpeg/webp)."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = os.path.join(folder, f"{movie_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def inject_global_styles():
    import os, base64

    # --------- ПІДТЯГУЄМО ФОНИ --------- #
    dark_bg_b64 = ""
    for name in ("login_bg.jpg", "login_bg.png", "bg.jpg", "background.jpg", "space_bg.jpg"):
        if os.path.isfile(name):
            try:
                with open(name, "rb") as f:
                    dark_bg_b64 = base64.b64encode(f.read()).decode("utf-8")
                break
            except Exception:
                pass

    light_bg_b64 = ""
    for name in ("login_bg2.jpg", "login_bg2.png"):
        if os.path.isfile(name):
            try:
                with open(name, "rb") as f:
                    light_bg_b64 = base64.b64encode(f.read()).decode("utf-8")
                break
            except Exception:
                pass

    # ❗ Темну тему майже не чіпаю – лишаю той самий космос
    if dark_bg_b64:
        dark_bg_style = f"""
        background-image:
            radial-gradient(circle at top, rgba(15,23,42,0.55), rgba(3,7,18,0.95)),
            url("data:image/jpeg;base64,{dark_bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        """
    else:
        dark_bg_style = """
        background: radial-gradient(circle at top, #020617 0, #020617 40%, #000000 100%);
        """

    # Світла – космос, але з сіруватим тюнінгом
    if light_bg_b64:
        light_bg_style = f"""
        background-image:
            radial-gradient(circle at top, rgba(112,128,144,0.35), rgba(209,213,219,0.65)),
            url("data:image/jpeg;base64,{light_bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        """
    else:
        light_bg_style = """
        background: radial-gradient(circle at top, #f3f4f6 0, #e5e7eb 40%, #d1d5db 100%);
        """

    theme = st.session_state.get("ui_theme", "dark")

    if theme == "dark":
        # темний набір змінних – залишаємо як було
        theme_vars = """
        :root {
            --accent: #e50914;
            --accent-hover: #f6121d;
            --radius-lg: 0.9rem;

            --bg-main: #020617;
            --bg-card: rgba(15,23,42,0.97);
            --bg-card-soft: rgba(15,23,42,0.94);
            --border-subtle: #1f2937;
            --text-main: #708090;
            --text-muted: #9ca3af;
            --chip-bg: #111827;
            --card-shadow: 0 18px 45px rgba(0,0,0,0.85);
            --login-hero-bg: radial-gradient(circle at top left,
                                rgba(37,99,235,0.9) 0,
                                #020617 55%,
                                #000000 100%);
            --button-bg: #d1d5db;        /* сірий фон кнопки */
            --button-fg: #000000;        /* ЧОРНИЙ текст */
            --button-border: #9ca3af;
            --button-bg-hover: #9ca3af;  /* трохи темніший при наведенні */
            --button-fg-hover: #000000;  /* все одно чорний текст */

        }
        """
        app_bg_style = dark_bg_style
    else:
        # 🌫 більш сіра світла тема + темно-сіра ліва панель
        theme_vars = """
        :root {
            --accent: #e50914;
            --accent-hover: #f6121d;
            --radius-lg: 0.9rem;

            /* ліва панель – темно-сіра */
            --bg-main: #9ca3af;

            /* картки світло-сірі, не білі */
            --bg-card: rgba(243,244,246,0.98);
            --bg-card-soft: rgba(112,128,144,0.98);
            --border-subtle: #000000;
            --text-main: #111827;
            --text-muted: #000000;
            --chip-bg: #d4d4d8;
            --card-shadow: 0 18px 40px rgba(15,23,42,0.18);
            --login-hero-bg: radial-gradient(circle at top left,
                                rgba(148,163,184,0.4) 0,
                                #e5e7eb 45%,
                                #d4d4d8 100%);
            --button-bg: #d1d5db;
            --button-fg: #000000;
            --button-border: #9ca3af;
            --button-bg-hover: #9ca3af;
            --button-fg-hover: #000000;
        }
        """
        app_bg_style = light_bg_style

    css = f"""
    <style>
    {theme_vars}

        /* ФОН ДОДАТКУ */
    [data-testid="stAppViewContainer"] {{
        {app_bg_style}
        color: var(--text-main);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* ЛІВА ПАНЕЛЬ (sidebar) – примусово сірий */
    section[data-testid="stSidebar"] {{
        border-right: 1px solid var(--border-subtle);
    }}

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {{
        background-color: var(--bg-main) !important;   /* беремо колір із змінної */
        background-image: none !important;
    }}

    .block-container {{
        padding-top: 0rem;
        padding-bottom: 2.5rem;
        background: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }}

    /* Текст */
    h1, h2, h3, h4, h5, h6,
    p, li, label,
    .stMarkdown, .stRadio > label, .stCheckbox > label {{
        color: var(--text-main) !important;
    }}

    /* === ТАБЛИЦІ / DATAFRAME === */
    /* обгортка таблиць */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        background: var(--bg-card);
        border-radius: 1rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--card-shadow);
        padding: 0.35rem 0.4rem 0.45rem;
    }}

    /* внутрішній grid */
    [data-testid="stDataFrame"] div[role="grid"],
    [data-testid="stDataFrame"] div[role="presentation"],
    [data-testid="stTable"] table {{
        background-color: var(--bg-card-soft) !important;
        color: var(--text-main) !important;
    }}

    [data-testid="stDataFrame"] table,
    [data-testid="stTable"] table {{
        border-collapse: collapse !important;
    }}

    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {{
        background-color: var(--bg-card-soft) !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-main) !important;
    }}

    /* прибираємо білу рамку навколо таблиць */
    [data-testid="stDataFrame"] > div > div {{
        border-color: var(--border-subtle) !important;
    }}

    /* === КАРТКИ ФІЛЬМІВ === */
    div[data-testid="column"]:has(.movie-title) {{
        position: relative;
        background: var(--bg-card);
        border-radius: 0.9rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--card-shadow);
        padding: 0.9rem 0.9rem 0.85rem;
        margin-bottom: 1.4rem;
        overflow: hidden;
    }}

    div[data-testid="column"]:has(.movie-title) > div[data-testid="stVerticalBlock"] {{
        background: transparent !重要;
        padding: 0 !important;
    }}

    .movie-card-bgbox {{
        position: relative;
        width: 96%;
        height: 479px;
        border-radius: 1rem;
        background: var(--bg-card-soft);
        border: 2px solid var(--border-subtle);
        box-shadow: var(--card-shadow);
        margin-top: 1rem;
        margin-bottom: -492px;
        margin-left: -0.4%;
        z-index: 0;
    }}

    .movie-card-bgbox-catalog {{
        height: 422px;
        margin-bottom: -435px;
    }}

    .movie-poster {{
        width: 95%;
        aspect-ratio: 16 / 9;
        border-radius: 0.75rem;
        overflow: hidden;
        margin-bottom: 0.75rem;
    }}

    .movie-poster img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }}

    .movie-title {{
        font-size: 1.0rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        margin-left: 2.1%;
    }}

    .movie-title-catalog {{
        display: block;
        max-width: 55%;
        white-space: normal;
        word-wrap: break-word;
        line-height: 1.15;
        height: 2.4em;
        overflow: hidden;
        margin-left: 3.1%;
    }}

    .movie-rating-pill {{
        display: inline-block;
        padding: 0.12rem 0.65rem;
        border-radius: 999px;
        background: #f59e0b26;
        color: #facc15;
        font-size: 0.75rem;
        font-weight: 600;
        position: relative;
        top: -33px;
        margin-left: 67%;
    }}

    .movie-rating-pill-catalog {{
        margin-left: 56%;
        top: -30px;
    }}

    .genre-chip {{
        display: inline-block;
        padding: 0.12rem 0.6rem;
        margin-right: 0.3rem;
        margin-top: 0;
        border-radius: 999px;
        background: var(--chip-bg);
        font-size: 0.8rem;
        margin-left: 2.1%;
        position: relative;
        top: 50px;
    }}

    .genre-chip-catalog {{
        margin-left: 2.1%;
        position: relative;
        top: 50px;
    }}

    .movie-description {{
        font-size: 0.8rem;
        color: var(--text-main);
        margin-top: 0.4rem;
        position: relative;
        top: -77px;
        margin-left: 2.1%;
        max-width: 90%;
        min-height: 3em;
        max-height: 3em;
    }}

    .movie-description-catalog {{
        max-width: 88%;
        margin-left: 3.1%;
        line-height: 1.3;
        min-height: 3em;
        max-height: 3em;
    }}

    .movie-card-footer,
    div[data-testid="stButton"] {{
        position: relative;
        z-index: 1;
    }}

    div[data-testid="column"]:has(.movie-title) div[data-testid="stButton"] {{
        margin-top: -1.8rem !important;
    }}

    .movie-actions-row > div {{
        flex: 1 1 0;
    }}

    @media (max-width: 1400px) {{
        .movie-title {{
            font-size: 0.9rem;
        }}
        .movie-description {{
            font-size: 0.75rem;
        }}
    }}

    /* === ЛОГІН-БЛОК і решта – без змін, як у тебе було === */

    .login-hero {{
        border-radius: 1.4rem;
        padding: 2.1rem 2.4rem;
        min-height: 340px;
        background: var(--login-hero-bg);
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }}

    .login-hero-label {{
        font-size: 0.75rem;
        letter-spacing: .22em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }}

    .login-hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}

    .login-hero-sub {{
        color: var(--text-muted);
        max-width: 360px;
        font-size: 0.95rem;
        line-height: 1.5;
    }}

    .login-hero-mascot {{
        position: absolute;
        right: 2.4rem;
        bottom: 1.2rem;
        width: 150px;
        opacity: 0.95;
    }}

    .login-right-header {{
        display:flex;
        align-items:center;
        gap:0.75rem;
        margin-bottom:1.5rem;
    }}

    .login-right-logo {{
        width:40px;
        height:40px;
        border-radius:12px;
        object-fit:cover;
        box-shadow:0 0 0 1px rgba(105,105,105,0.2);
    }}

    .login-right-title {{
        font-size:1.5rem;
        font-weight:700;
        color:var(--text-main);
    }}

    .login-right-caption {{
        font-size:0.8rem;
        color:var(--text-muted);
    }}

    form[data-testid="stForm"] {{
        background-color: var(--bg-card);
        padding: 2.0rem 2.1rem 2.1rem;
        border-radius: 1.3rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(148,163,184,0.2);
    }}

    .top-nav-title {{
        display:flex;
        align-items:center;
        font-size:1.8rem;
        font-weight:800;
        letter-spacing:0.02em;
        color: var(--text-main);
    }}

    .dashboard-card {{
        background: var(--bg-card);
        border-radius: 1rem;
        padding: 1rem 1rem 1.1rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--card-shadow);
    }}

    .filters-card {{
        padding-top: 0.1rem;
        padding-bottom: 0.01rem;
        margin-top: 1.8rem;
        margin-bottom: 0.1rem;
        width: 100%;
        max-width: 110px;
        height: 38px;
    }}

    .dashboard-card h4 {{
        margin-top:0;
        margin-bottom:0.3rem;
        font-size:0.95rem;
        text-transform:uppercase;
        letter-spacing:0.08em;
        color:var(--text-muted);
    }}

        /* === Кнопки — завжди сірі з чорним текстом === */

    /* ВСІ кнопки Streamlit */
    .stButton > button {{
        width: 100% !important;
        max-width: 100% !important;
        border-radius: 999px !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 0.75rem !important;
        line-height: 1.15 !important;

        background-color: #ADD8E6 !important;  /* світло-сірий */
        color: #000000 !important;             /* ЧОРНИЙ текст */
        border: 1px solid #9ca3af !important;

        transition: all 0.16s ease-out !important;
    }}

    /* Hover стан */
    .stButton > button:hover {{
        background-color: #9ca3af !important;  /* трохи темніший сірий */
        color: #000000 !important;
        border-color: #6b7280 !important;
        transform: translateY(-1px);
    }}

    /* Фокус (по табу) */
    .stButton > button:focus {{
        outline: 2px solid #6b7280 !important;
        outline-offset: 1px;
        box-shadow: 0 0 0 1px #6b7280 !important;
    }}

    /* Disabled, але текст все одно чорний */
    .stButton > button:disabled {{
        background-color: #e5e7eb !important;
        color: #000000 !important;
        opacity: 0.6 !important;
    }}

        /* === SELECTBOX / DROPDOWN (щоб було видно роки) === */

    /* сам інпут */
    [data-testid="stSelectbox"] > div > div {{
        background-color: var(--bg-card-soft) !important;
        color: var(--text-main) !important;
        border-radius: 0.6rem !important;
        border: 1px solid var(--border-subtle) !important;
    }}

    /* список варіантів */
    [data-testid="stSelectbox"] div[role="listbox"],
    div[data-baseweb="menu"] {{
        background-color: var(--bg-card-soft) !important;
        color: var(--text-main) !important;
        border-radius: 0.6rem !important;
        border: 1px solid var(--border-subtle) !important;
    }}

    /* самі опції (роки) */
    [data-testid="stSelectbox"] div[role="option"],
    div[data-baseweb="menu"] div[role="option"] {{
        color: var(--text-main) !important;
    }}

    /* підсвітка наведеної опції */
    [data-testid="stSelectbox"] div[role="option"]:hover,
    div[data-baseweb="menu"] div[role="option"]:hover {{
        background-color: rgba(148,163,184,0.25) !important;
    }}


    a {{
        color: var(--accent);
    }}

        /* 🔒 Фікс: текст у всіх кнопках завжди чорний */
    section[data-testid="stSidebar"] .stButton > button,
    [data-testid="stAppViewContainer"] .stButton > button,
    .stButton button[kind] {{
        color: #000000 !important;   /* Чорний текст завжди */
    }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

def ensure_extra_tables(db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            comment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            movie_id     INTEGER NOT NULL,
            comment_text TEXT    NOT NULL,
            created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            user_id    INTEGER NOT NULL,
            movie_id   INTEGER NOT NULL,
            created_at TEXT    DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, movie_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id    INTEGER NOT NULL,
            movie_id   INTEGER NOT NULL,
            status     TEXT    DEFAULT 'planned',
            created_at TEXT    DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, movie_id)
        )
        """
    )

    conn.commit()
    conn.close()

def extract_year_from_title(title: str):
    if not isinstance(title, str):
        return None
    m = re.search(r"\((\d{4})\)", title)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


@st.cache_data
def load_base_data(db_path=DB_PATH):
    if not os.path.exists(db_path):
        st.error(f"База даних '{db_path}' не знайдена в поточній папці.")
        return None, None, None

    ensure_extra_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        movies = pd.read_sql_query(
            "SELECT movie_id, title, genres, release_year AS year, description FROM movies",
            conn,
        )
        ratings = pd.read_sql_query(
            "SELECT user_id, movie_id, rating, date FROM ratings",
            conn,
        )
        users = pd.read_sql_query(
            "SELECT user_id, first_name, last_name FROM users",
            conn,
        )
    except Exception as e:
        conn.close()
        st.error(f"Помилка читання таблиць з бази: {e}")
        return None, None, None

    conn.close()

    movies["movie_id"] = movies["movie_id"].astype(int)
    if "genres" not in movies.columns:
        movies["genres"] = "Unknown"
    if "year" not in movies.columns:
        movies["year"] = movies["title"].apply(extract_year_from_title)
    movies["description"] = movies["description"].fillna("Опис фільму відсутній у базі даних.")

    if "duration" not in movies.columns:
        np.random.seed(42)
        movies["duration"] = np.random.randint(80, 140, size=len(movies))

    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce")

    users["user_id"] = users["user_id"].astype(int)

    return movies.reset_index(drop=True), ratings.reset_index(drop=True), users.reset_index(drop=True)


def load_comments(db_path=DB_PATH):
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["comment_id", "user_id", "movie_id", "comment_text", "created_at"])
    ensure_extra_tables(db_path)
    conn = sqlite3.connect(db_path)
    try:
        comments = pd.read_sql_query(
            "SELECT comment_id, user_id, movie_id, comment_text, created_at FROM comments",
            conn,
        )
    except Exception:
        comments = pd.DataFrame(columns=["comment_id", "user_id", "movie_id", "comment_text", "created_at"])
    conn.close()
    return comments


def add_comment_to_db(user_id, movie_id, text, db_path=DB_PATH):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO comments (user_id, movie_id, comment_text, created_at) VALUES (?, ?, ?, ?)",
            (int(user_id), int(movie_id), text, created_at),
        )
        conn.commit()
    finally:
        conn.close()

def load_user_favorites(user_id, db_path=DB_PATH):
    if user_id is None or not os.path.exists(db_path):
        return pd.DataFrame(columns=["user_id", "movie_id", "created_at"])
    conn = sqlite3.connect(db_path)
    fav = pd.read_sql_query(
        "SELECT user_id, movie_id, created_at FROM favorites WHERE user_id = ?",
        conn,
        params=(int(user_id),),
    )
    conn.close()
    return fav


def load_user_watchlist(user_id, db_path=DB_PATH):
    if user_id is None or not os.path.exists(db_path):
        return pd.DataFrame(columns=["user_id", "movie_id", "status", "created_at"])
    conn = sqlite3.connect(db_path)
    wl = pd.read_sql_query(
        "SELECT user_id, movie_id, status, created_at FROM watchlist WHERE user_id = ?",
        conn,
        params=(int(user_id),),
    )
    conn.close()
    return wl


def toggle_favorite(user_id, movie_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM favorites WHERE user_id = ? AND movie_id = ?",
        (int(user_id), int(movie_id)),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "DELETE FROM favorites WHERE user_id = ? AND movie_id = ?",
            (int(user_id), int(movie_id)),
        )
        action = "removed"
    else:
        cur.execute(
            "INSERT INTO favorites (user_id, movie_id) VALUES (?, ?)",
            (int(user_id), int(movie_id)),
        )
        action = "added"
    conn.commit()
    conn.close()
    return action


def toggle_watchlist(user_id, movie_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM watchlist WHERE user_id = ? AND movie_id = ?",
        (int(user_id), int(movie_id)),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?",
            (int(user_id), int(movie_id)),
        )
        action = "removed"
    else:
        cur.execute(
            "INSERT INTO watchlist (user_id, movie_id, status) VALUES (?, ?, 'planned')",
            (int(user_id), int(movie_id)),
        )
        action = "added"
    conn.commit()
    conn.close()
    return action

def md5_of_array(arr: np.ndarray) -> str:
    m = hashlib.md5()
    m.update(arr.view(np.uint8))
    return m.hexdigest()


def save_npz(filename, **kwargs):
    np.savez_compressed(filename, **kwargs)


def load_npz(filename):
    return np.load(filename, allow_pickle=True)


@st.cache_data
def build_user_item_matrix(ratings: pd.DataFrame):
    ui = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
    ui = ui.sort_index(axis=1)
    return ui


@st.cache_data
def build_genre_matrix(movies: pd.DataFrame):
    if "genres" not in movies.columns:
        return pd.DataFrame(index=movies["movie_id"].values)

    genres_split = (
        movies["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.split("|")
        .apply(lambda x: [g.strip() for g in x])
    )
    mlb = MultiLabelBinarizer(sparse_output=False)
    try:
        encoded = mlb.fit_transform(genres_split)
        gm = pd.DataFrame(encoded, index=movies["movie_id"], columns=mlb.classes_)
    except Exception:
        gm = pd.DataFrame(index=movies["movie_id"])
    return gm


@st.cache_data
def build_movie_catalog(movies: pd.DataFrame, ratings: pd.DataFrame):
    agg = (
        ratings.groupby("movie_id")["rating"]
        .agg(avg_rating="mean", n_ratings="count")
        .reset_index()
    )
    catalog = movies.merge(agg, on="movie_id", how="left")
    catalog["avg_rating"] = catalog["avg_rating"].fillna(0.0)
    catalog["n_ratings"] = catalog["n_ratings"].fillna(0).astype(int)
    return catalog


def compute_item_similarity_diskcached(
    user_item_df: pd.DataFrame, metric="cosine", cache_file=SIM_CACHE_FILE
):
    item_matrix = user_item_df.T.values.astype(np.float32)
    item_ids = user_item_df.columns.values.astype(int)

    arr_hash = md5_of_array(item_matrix)
    combined_hash = hashlib.md5((arr_hash + str(item_matrix.shape)).encode("utf-8")).hexdigest()

    if os.path.exists(cache_file):
        try:
            cache = load_npz(cache_file)
            cached_hash = cache["hash"].item() if "hash" in cache else None
            if cached_hash == combined_hash and "sim" in cache:
                sim = cache["sim"]
                return sim, item_ids
        except Exception:
            pass

    if metric == "cosine":
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = item_matrix / norms
        sim = normalized @ normalized.T
    else:
        raise ValueError("Only 'cosine' metric implemented.")

    try:
        save_npz(cache_file, sim=sim, hash=np.array(combined_hash, dtype=object))
    except Exception as e:
        st.warning(f"Не вдалося зберегти кеш схожості: {e}")

    return sim, item_ids


def handle_cold_start_user(user_id, user_item_df, movies_df, top_k=10):
    item_counts = (user_item_df > 0).sum(axis=0)
    item_means = user_item_df.replace(0, np.nan).mean(axis=0).fillna(0)
    pop_score = item_means * np.log1p(item_counts)
    pop_sorted = pop_score.sort_values(ascending=False).head(top_k)
    top_ids = pop_sorted.index.values
    res = movies_df.set_index("movie_id").loc[top_ids].reset_index()
    res["predicted_score"] = pop_sorted.values
    return res[["movie_id", "title", "genres", "duration", "predicted_score"]]


def predict_item_based_for_user(user_id, user_item_df, movies_df, item_sim, item_ids, top_k=10):
    if user_id not in user_item_df.index:
        return handle_cold_start_user(user_id, user_item_df, movies_df, top_k=top_k)

    user_vec = user_item_df.loc[user_id].values.astype(np.float32)
    rated_mask = user_vec > 0
    if rated_mask.sum() == 0:
        return handle_cold_start_user(user_id, user_item_df, movies_df, top_k=top_k)

    scores = item_sim.dot(user_vec)
    denom = np.abs(item_sim).sum(axis=1)
    denom[denom == 0] = 1.0
    preds = scores / denom

    preds[rated_mask] = -np.inf

    top_idx = np.argsort(preds)[-top_k:][::-1]
    top_item_ids = item_ids[top_idx]
    preds_top = preds[top_idx]

    res = movies_df.set_index("movie_id").loc[top_item_ids].reset_index()
    res["predicted_score"] = preds_top

    cols = ["movie_id", "title", "genres", "year", "duration", "description", "predicted_score"]
    cols_existing = [c for c in cols if c in res.columns]
    return res[cols_existing]


def handle_cold_start_item(target_movie_id, genre_matrix, movies_df, user_item_df, top_k=10):
    if target_movie_id not in genre_matrix.index:
        return handle_cold_start_user(None, user_item_df, movies_df, top_k=top_k)

    gvec = genre_matrix.loc[target_movie_id].values.reshape(1, -1)
    all_g = genre_matrix.values
    sim = cosine_similarity(gvec, all_g).flatten()
    idxs = np.argsort(sim)[-top_k:][::-1]
    item_ids = genre_matrix.index.values[idxs]
    res = movies_df.set_index("movie_id").loc[item_ids].reset_index()
    res["genre_similarity"] = sim[idxs]
    return res[["movie_id", "title", "genres", "duration", "genre_similarity"]]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) else None


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) if len(y_true) else None


def precision_at_k(recommended_ids, true_ids, k):
    recommended_topk = recommended_ids[:k]
    hits = sum([1 for r in recommended_topk if r in true_ids])
    return hits / k


def train_test_split_ratings(ratings_df, test_size=0.2, seed=42):
    train, test = train_test_split(ratings_df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_item_cf_on_split(train_df, test_df, movies_df, k=10):
    ui_train = build_user_item_matrix(train_df)
    item_sim, item_ids = compute_item_similarity_diskcached(ui_train)
    users_in_test = test_df["user_id"].unique()
    precisions = []
    y_trues = []
    y_preds = []

    for uid in users_in_test:
        if uid in ui_train.index:
            recs = predict_item_based_for_user(uid, ui_train, movies_df, item_sim, item_ids, top_k=k)
            rec_ids = recs["movie_id"].tolist()
        else:
            recs = handle_cold_start_user(uid, ui_train, movies_df, top_k=k)
            rec_ids = recs["movie_id"].tolist()

        true_items = set(test_df[test_df["user_id"] == uid]["movie_id"].tolist())
        if len(true_items) == 0:
            continue
        precisions.append(precision_at_k(rec_ids, true_items, k))

        pred_map = dict(
            zip(
                recs["movie_id"].tolist(),
                recs.get("predicted_score", recs.get("genre_similarity", [])),
            )
        )
        for _, row in test_df[test_df["user_id"] == uid].iterrows():
            mid = row["movie_id"]
            true_r = row["rating"]
            if mid in pred_map:
                y_trues.append(true_r)
                y_preds.append(pred_map[mid])

    metrics = {
        f"precision_at_{k}": np.mean(precisions) if len(precisions) else 0.0,
        "rmse": rmse(y_trues, y_preds),
        "mae": mae(y_trues, y_preds),
    }
    return metrics


@st.cache_data
def cluster_users(users_df, ratings_df, n_clusters=3):
    user_stats = (
        ratings_df.groupby("user_id")
        .agg(avg_rating=("rating", "mean"), count=("rating", "count"), std_rating=("rating", "std"))
        .fillna(0)
    )
    data = users_df.set_index("user_id").join(user_stats, how="left").fillna(0)
    if "gender" in data.columns:
        le = LabelEncoder()
        data["gender_code"] = le.fit_transform(data["gender"].fillna("U"))
    else:
        data["gender_code"] = 0
    if "age" not in data.columns:
        data["age"] = 0
    features = data[["age", "gender_code", "avg_rating", "count", "std_rating"]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    data["cluster"] = labels
    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(X_scaled)
    data["pca_x"] = comp[:, 0]
    data["pca_y"] = comp[:, 1]
    return data

def solve_knapsack(candidates, time_budget):
    n = len(candidates)
    W = int(time_budget)
    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    take = [[False] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wi = int(candidates[i - 1]["duration"])
        vi = float(candidates[i - 1]["score"])
        for w in range(W + 1):
            no_take = dp[i - 1][w]
            take_val = -1.0
            if wi <= w:
                take_val = dp[i - 1][w - wi] + vi
            if take_val > no_take:
                dp[i][w] = take_val
                take[i][w] = True
            else:
                dp[i][w] = no_take

    w = W
    picked = []
    for i in range(n, 0, -1):
        if take[i][w]:
            picked.append(candidates[i - 1])
            w -= int(candidates[i - 1]["duration"])
    picked.reverse()
    return picked


def forecast_random_walk(current_rating, steps=12):
    trend = 0.02
    volatility = 0.15
    path = [current_rating]
    upper = [current_rating]
    lower = [current_rating]
    for _ in range(steps):
        next_val = path[-1] + trend + np.random.normal(0, volatility)
        next_val = max(0.0, min(5.0, next_val))
        path.append(next_val)
        upper.append(min(5.0, next_val + volatility))
        lower.append(max(0.0, next_val - volatility))
    return path, upper, lower

def render_cache_controls():
    st.sidebar.markdown("---")
    st.sidebar.write("🧹 Кеш рекомендацій")
    if st.sidebar.button("Очистити кеш схожості фільмів"):
        if os.path.exists(SIM_CACHE_FILE):
            os.remove(SIM_CACHE_FILE)
            st.sidebar.success("Файл кешу схожості видалено.")
        else:
            st.sidebar.info("Файл кешу ще не створювався.")
    if st.sidebar.button("Очистити всі кеші на диску"):
        removed_any = False
        for f in [SIM_CACHE_FILE, USER_ITEM_CACHE_FILE, GENRE_CACHE_FILE]:
            if os.path.exists(f):
                os.remove(f)
                removed_any = True
        if removed_any:
            st.sidebar.success("Усі кеш-файли видалено.")
        else:
            st.sidebar.info("Кеш-файлів поки немає.")


def process_login(username: str, password: str, remember: bool) -> bool:
    u = (username or "").strip().lower()
    p = (password or "").strip()

    role = None
    if u == "admin" and p == "admin":
        role = "admin"
    elif u == "user" and p == "user":
        role = "user"

    if role is None:
        return False

    st.session_state["auth_role"] = role
    st.session_state["username"] = u
    st.session_state["remember_me"] = remember
    return True


def show_login():
    st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        mascot_img = ""
        for fname in ("mascot.png", "mascot_no_bg.png"):
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                mascot_img = f"<img class='login-hero-mascot' src='data:image/png;base64,{b64}' />"
                break

        st.markdown(
            f"""
            <div class="login-hero">
                {mascot_img}
                <div class="login-hero-label">MOVIE PORTAL</div>
                <div class="login-hero-title">Фільми, серіали й рекомендації.</div>
                <div class="login-hero-sub">
                    Навчальний портал з рекомендаціями фільмів з двома ролями — <b>admin</b> і <b>user</b>.
                    Увійдіть, щоб побачити персональні підказки, популярні фільми та план перегляду.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        logo_html = ""
        for name in ("site_icon_no_bg.png", "site_icon.png"):
            if os.path.exists(name):
                with open(name, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                logo_html = f"<img class='login-right-logo' src='data:image/png;base64,{b64}' alt='logo' />"
                break

        st.markdown(
            f"""
            <div class="login-right-header">
                {logo_html}
                <div>
                    <div class="login-right-title">Увійти</div>
                    <div class="login-right-caption">
                        Використайте <b>admin / admin</b> або <b>user / user</b>.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Логін", key="login_username")
            password = st.text_input("Пароль", type="password", key="login_password")
            remember = st.checkbox("Запам'ятати мене", value=True, key="login_remember")
            submit = st.form_submit_button("Увійти", use_container_width=True)

        if submit:
            if process_login(username, password, remember):
                safe_rerun()
            else:
                st.error("Невірний логін або пароль.")


def show_movie_cards(
    df: pd.DataFrame,
    max_items: int = 30,
    current_user_id: int | None = None,
    favorites_ids: set | None = None,
    watchlist_ids: set | None = None,
    prefix: str = "",
):
    if df is None or df.empty:
        st.info("Немає фільмів за заданими умовами.")
        return

    df = df.head(max_items)

    cols_per_row = 3
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row)

        col = cols[idx % cols_per_row]

        with col:
            card = st.container()
            with card:

                bg_classes = "movie-card-bgbox"
                if prefix == "catalog":
                    bg_classes += " movie-card-bgbox-catalog"

                st.markdown(
                    f'<div class="{bg_classes}"></div>',
                    unsafe_allow_html=True,
                )

                movie_id = int(row.get("movie_id"))
                title = row.get("title", "Без назви")
                genres = row.get("genres", "")
                year = row.get("year", None)
                duration = row.get("duration", None)
                rating = row.get("avg_rating", row.get("predicted_score", None))
                n_ratings = row.get("n_ratings", None)
                description = row.get("description", "")

                if isinstance(description, str) and len(description) > 220:
                    description = description[:220] + "..."

                poster_path = get_poster_path(movie_id)
                if poster_path:
                    poster_b64 = encode_image_base64(poster_path)
                    st.markdown(
                        f"""
                        <div class="movie-poster">
                            <img src="data:image/jpeg;base64,{poster_b64}" alt="poster">
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                title_classes = "movie-title"
                if prefix == "catalog":
                    title_classes += " movie-title-catalog"

                st.markdown(
                    f'<div class="{title_classes}">{title}</div>',
                    unsafe_allow_html=True,
                )

                if rating is not None and not pd.isna(rating):
                    rating_str = f"{float(rating):.2f}"
                    extra = (
                        f" • {int(n_ratings)} оцінок"
                        if n_ratings not in (None, 0, np.nan)
                        else ""
                    )

                    rating_classes = "movie-rating-pill"
                    if prefix == "catalog":
                        rating_classes += " movie-rating-pill-catalog"

                    st.markdown(
                        f'<div class="{rating_classes}">⭐ {rating_str}{extra}</div>',
                        unsafe_allow_html=True,
                    )

                if genres:
                    extra_genre_class = " genre-chip-catalog" if prefix == "catalog" else ""
                    chips = "".join(
                        f'<span class="genre-chip{extra_genre_class}">{g.strip()}</span>'
                        for g in str(genres).split("|")
                        if g.strip()
                    )
                    st.markdown(chips, unsafe_allow_html=True)

                if description:
                    desc_classes = "movie-description"
                    if prefix == "catalog":
                        desc_classes += " movie-description-catalog"

                    st.markdown(
                        f'<div class="{desc_classes}">{description}</div>',
                        unsafe_allow_html=True,
                    )

                if current_user_id is not None:
                    is_fav = favorites_ids is not None and movie_id in favorites_ids
                    is_planned = watchlist_ids is not None and movie_id in watchlist_ids

                    pad_left, col_fav, gap_mid, col_plan, pad_right = st.columns(
                        [0.01, 2, 3, 8, 0.30]
                    )

                    with col_fav:
                        label = "❤️" if is_fav else "♡"
                        if st.button(label, key=f"{prefix}_fav_{movie_id}"):
                            action = toggle_favorite(current_user_id, movie_id)
                            if action == "added":
                                st.toast("Фільм додано до 'Обраного'.")
                            else:
                                st.toast("Фільм видалено з 'Обраного'.")
                            safe_rerun()

                    with col_plan:
                        label_w = "📋 У плані" if is_planned else "➕ У план перегляду"
                        if st.button(label_w, key=f"{prefix}_plan_{movie_id}"):
                            action = toggle_watchlist(current_user_id, movie_id)
                            if action == "added":
                                st.toast("Фільм додано до плану перегляду.")
                            else:
                                st.toast("Фільм видалено з плану перегляду.")
                            safe_rerun()

def enable_altair_theme(theme: str):
    """Кольори для Altair + фон графіків."""

    def dark_theme():
        return {
            "config": {
                "background": "transparent",          # прозорий фон
                "view": {"fill": "transparent"},
                "axis": {
                    "labelColor": "#e5e7eb",
                    "titleColor": "#e5e7eb",
                    "gridColor": "#1f2937",
                },
                "legend": {
                    "labelColor": "#e5e7eb",
                    "titleColor": "#e5e7eb",
                },
                "title": {"color": "#e5e7eb"},
            }
        }

    def light_theme():
        return {
            "config": {
                "background": "transparent",          # теж прозорий
                "view": {"fill": "transparent"},
                "axis": {
                    "labelColor": "#111827",
                    "titleColor": "#111827",
                    "gridColor": "#9ca3af",
                },
                "legend": {
                    "labelColor": "#111827",
                    "titleColor": "#111827",
                },
                "title": {"color": "#111827"},
            }
        }

    if theme == "dark":
        if "mp_dark" not in alt.themes.names():
            alt.themes.register("mp_dark", dark_theme)
        alt.themes.enable("mp_dark")

        # matplotlib – темний фон, без білих прямокутників
        mpl.rcParams.update({
            "figure.facecolor": "none",
            "axes.facecolor": "#020617",
            "savefig.facecolor": "none",
            "axes.edgecolor": "#e5e7eb",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#e5e7eb",
            "ytick.color": "#e5e7eb",
            "text.color": "#e5e7eb",
        })
    else:
        if "mp_light" not in alt.themes.names():
            alt.themes.register("mp_light", light_theme)
        alt.themes.enable("mp_light")

        # matplotlib – світло-сірий, а не білий
        mpl.rcParams.update({
            "figure.facecolor": "none",
            "axes.facecolor": "#e5e7eb",
            "savefig.facecolor": "none",
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "text.color": "#111827",
        })

def admin_dashboard(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    user_item: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    item_sim: np.ndarray,
    item_ids: np.ndarray,
    comments: pd.DataFrame,
):
    st.title("📱 Адмін-панель")

    st.caption("Огляд активності, часові тренди та якість моделі рекомендацій.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Фільмів у каталозі", f"{len(movies):,}".replace(",", " "))
    c2.metric("Користувачів", f"{len(users):,}".replace(",", " "))
    c3.metric("Оцінок", f"{len(ratings):,}".replace(",", " "))
    global_avg = float(ratings["rating"].mean()) if not ratings.empty else 0.0
    c4.metric("Середній рейтинг", f"{global_avg:.2f}")

    tab_overview, tab_time, tab_recs, tab_cluster = st.tabs(
        ["Загальна картина", "Часові тренди", "Рекомендації", "Кластери та модель"]
    )

    with tab_overview:
        if not ratings.empty:
            dist = (
                ratings["rating"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            dist.columns = ["rating", "count"]

            st.subheader("Розподіл оцінок")

            chart = (
                alt.Chart(dist)
                .mark_bar()
                .encode(
                    x=alt.X("rating:O", title="Рейтинг"),
                    y=alt.Y("count:Q", title="Кількість оцінок"),
                    tooltip=["rating", "count"],
                )
                .properties(height=360)
            )

            st.altair_chart(chart, use_container_width=True)



            top_users = (
                ratings.groupby("user_id")["rating"]
                .agg(count="count", avg="mean")
                .reset_index()
                .sort_values(["count", "avg"], ascending=[False, False])
                .head(15)
            )
            st.subheader("Топ активних користувачів")
            st.dataframe(
                top_users.rename(
                    columns={"user_id": "user_id", "count": "кількість", "avg": "середній"}
                ).style.format({"середній": "{:.2f}"}),
                use_container_width=True,
            )
        else:
            st.info("Поки що немає оцінок для статистики.")

        st.markdown("---")
        st.subheader("ТОП фільмів за весь час (за кількістю оцінок)")

        catalog = build_movie_catalog(movies, ratings)
        top_movies = catalog.sort_values(
            ["n_ratings", "avg_rating"], ascending=[False, False]
        ).head(20)
        display_cols = ["title", "avg_rating", "n_ratings", "genres", "year", "duration"]
        existing = [c for c in display_cols if c in top_movies.columns]
        st.dataframe(
            top_movies[existing]
            .rename(
                columns={
                    "title": "Фільм",
                    "avg_rating": "Середній рейтинг",
                    "n_ratings": "К-сть оцінок",
                    "genres": "Жанри",
                    "year": "Рік",
                    "duration": "Тривалість (хв)",
                }
            )
            .style.format({"Середній рейтинг": "{:.2f}"}),
            use_container_width=True,
        )

    with tab_time:
        if "date" in ratings.columns and ratings["date"].notna().any():
            ratings_date = ratings.dropna(subset=["date"]).copy()
            if not np.issubdtype(ratings_date["date"].dtype, np.datetime64):
                ratings_date["date"] = pd.to_datetime(
                    ratings_date["date"], errors="coerce"
                )
            ratings_date = ratings_date.dropna(subset=["date"])

            if ratings_date.empty:
                st.info("Недостатньо даних з датами для побудови часових трендів.")
            else:
                st.subheader("Місячна динаміка рейтингу")

                window = st.slider(
                    "Вікно ковзного середнього (місяців)",
                    1, 12, 3,
                    key="monthly_sma_window",
                )

                ratings_monthly = ratings_date.copy()
                ratings_monthly["year_month"] = ratings_monthly["date"].dt.to_period("M")

                monthly = (
                    ratings_monthly
                    .groupby("year_month")["rating"]
                    .agg(avg_rating="mean", n_ratings="count")
                    .reset_index()
                )

                monthly["month_dt"] = monthly["year_month"].dt.to_timestamp()

                monthly["smooth"] = (
                    monthly["avg_rating"]
                    .rolling(window=window, min_periods=1)
                    .mean()
                )

                x = np.arange(len(monthly))
                y = monthly["avg_rating"].values
                coef = np.polyfit(x, y, 1)
                slope = coef[0]             
                trend = np.polyval(coef, x)

                monthly["trend"] = trend  

                chart_df = monthly[["month_dt", "avg_rating", "smooth", "trend"]].rename(
                    columns={
                        "month_dt": "Місяць",
                        "avg_rating": "Середній рейтинг",
                        "smooth": f"Згладжений ({window} міс.)",
                        "trend": "Лінійний тренд",
                    }
                )

                chart_long = chart_df.melt(
                    id_vars=["Місяць"],
                    var_name="Показник",
                    value_name="Рейтинг",
                )

                chart = (
                    alt.Chart(chart_long)
                    .mark_line()
                    .encode(
                        x="Місяць:T",
                        y=alt.Y("Рейтинг:Q", title="Рейтинг"),
                        color=alt.Color("Показник:N", title=""),
                        tooltip=["Місяць:T", "Показник:N", "Рейтинг:Q"],
                    )
                    .properties(
                        height=400,
                        title="Місячна динаміка рейтингу",
                    )
                    .interactive()
                    .configure_view(fill="rgba(0,0,0,0)")
                    .configure(background="rgba(0,0,0,0)")
                )


                st.altair_chart(chart, use_container_width=True)

                st.subheader("Автоматичні висновки")

                global_mean = float(monthly["avg_rating"].mean())
                last_12 = monthly.tail(12)
                recent_mean = float(last_12["avg_rating"].mean()) if not last_12.empty else global_mean
                diff_recent = recent_mean - global_mean

                if slope > 0.002:
                    trend_text = "загальний тренд ⬆️ (рейтинги зростають з часом)"
                elif slope < -0.002:
                    trend_text = "загальний тренд ⬇️ (рейтинги знижуються з часом)"
                else:
                    trend_text = "загальний тренд приблизно стабільний"

                if abs(diff_recent) < 0.05:
                    recent_text = "За останній рік середній рейтинг майже не відрізняється від загального."
                elif diff_recent > 0:
                    recent_text = (
                        f"За останній рік середній рейтинг **вищий** за історичний "
                        f"на {diff_recent:.2f}."
                    )
                else:
                    recent_text = (
                        f"За останній рік середній рейтинг **нижчий** за історичний "
                        f"на {abs(diff_recent):.2f}."
                    )

                anomalies = monthly.copy()
                anomalies["delta"] = anomalies["avg_rating"] - global_mean
                high_spikes = (
                    anomalies[anomalies["delta"] > 0.4]
                    .sort_values("delta", ascending=False)
                    .head(3)
                )
                low_spikes = (
                    anomalies[anomalies["delta"] < -0.4]
                    .sort_values("delta")
                    .head(3)
                )

                st.markdown(
                    f"""
                - **Загальний середній рейтинг:** {global_mean:.2f}  
                - **Середній рейтинг за останні 12 місяців:** {recent_mean:.2f}  
                - **Тренд:** {trend_text}  
                - {recent_text}
                    """
                )

                if not high_spikes.empty or not low_spikes.empty:
                    st.markdown("**Виділені періоди:**")
                if not high_spikes.empty:
                    st.write("Місяці з нетипово ВИСОКИМ рейтингом:")
                    for _, row in high_spikes.iterrows():
                        st.write(
                            f"- {row['year_month']} · {row['avg_rating']:.2f} "
                            f"(вище середнього на {row['delta']:.2f})"
                        )
                if not low_spikes.empty:
                    st.write("Місяці з нетипово НИЗЬКИМ рейтингом:")
                    for _, row in low_spikes.iterrows():
                        st.write(
                            f"- {row['year_month']} · {row['avg_rating']:.2f} "
                            f"(нижче середнього на {abs(row['delta']):.2f})"
                        )

                st.markdown("---")

                ratings_date["year"] = ratings_date["date"].dt.year
                yearly = (
                    ratings_date.groupby("year")
                    .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
                    .reset_index()
                )

                st.subheader("Річна динаміка середнього рейтингу")
                left, right = st.columns([3, 2])
                with left:
                    yearly_chart = (
                        alt.Chart(yearly)
                        .mark_bar()
                        .encode(
                            x=alt.X("year:O", title="Рік"),
                            y=alt.Y("avg_rating:Q", title="Середній рейтинг"),
                            tooltip=["year", "avg_rating", "n_ratings"],
                        )
                        .properties(height=320)
                        .configure_view(fill="rgba(0,0,0,0)")
                        .configure(background="rgba(0,0,0,0)")
                    )
                    st.altair_chart(yearly_chart, use_container_width=True)

                with right:
                    st.dataframe(
                        yearly.rename(
                            columns={
                                "year": "Рік",
                                "avg_rating": "Середній рейтинг",
                                "n_ratings": "Кількість оцінок",
                            }
                        ).style.format({"Середній рейтинг": "{:.2f}"}),
                        use_container_width=True,
                    )

                available_years = sorted(yearly["year"].unique())
                selected_year = st.selectbox("Рік для перегляду кварталів", available_years)
                subset = ratings_date[ratings_date["year"] == selected_year].copy()
                subset["quarter"] = subset["date"].dt.to_period("Q").astype(str)
                quarterly = (
                    subset.groupby("quarter")
                    .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
                    .reset_index()
                )

                st.subheader(f"Рейтинг по кварталах — {selected_year}")
                c1_, c2_ = st.columns([3, 2])
                with c1_:
                    quarter_chart = (
                        alt.Chart(quarterly)
                        .mark_bar()
                        .encode(
                            x=alt.X("quarter:O", title="Квартал"),
                            y=alt.Y("avg_rating:Q", title="Середній рейтинг"),
                            tooltip=["quarter", "avg_rating", "n_ratings"],
                        )
                        .properties(height=320)
                        .configure_view(fill="rgba(0,0,0,0)")
                        .configure(background="rgba(0,0,0,0)")
                    )
                    st.altair_chart(quarter_chart, use_container_width=True)

                with c2_:
                    st.dataframe(
                        quarterly.rename(
                            columns={
                                "quarter": "Квартал",
                                "avg_rating": "Середній рейтинг",
                                "n_ratings": "Кількість оцінок",
                            }
                        ).style.format({"Середній рейтинг": "{:.2f}"}),
                        use_container_width=True,
                    )
        else:
            st.info("У наборі даних немає коректного стовпця 'date' для часових графіків.")

    with tab_recs:
        st.subheader("Швидкі рекомендації для вибраного користувача")
        if user_item.shape[0]:
            all_user_ids = sorted(list(user_item.index))
            uid = st.selectbox("ID користувача", all_user_ids)
            k = st.slider("Кількість рекомендованих фільмів", 5, 30, 10)
            recs = predict_item_based_for_user(uid, user_item, movies, item_sim, item_ids, top_k=k)
            st.dataframe(
                recs.rename(
                    columns={
                        "title": "Фільм",
                        "genres": "Жанри",
                        "duration": "Тривалість (хв)",
                        "predicted_score": "Прогнозований рейтинг",
                    }
                ).style.format({"Прогнозований рейтинг": "{:.3f}"}),
                use_container_width=True,
            )
        else:
            st.info("Немає користувачів для побудови рекомендацій.")

    with tab_cluster:
        sub1, sub2 = st.tabs(["Кластери користувачів", "Оцінка моделі"])
        with sub1:
            n_clusters = st.slider("Кількість кластерів", 2, 8, 3)
            if st.button("Запустити кластеризацію"):
                with st.spinner("Кластеризуємо користувачів..."):
                    clustered = cluster_users(users, ratings, n_clusters=n_clusters)
                    theme = st.session_state.get("ui_theme", "dark")

                fig, ax = plt.subplots(figsize=(8, 6))

                # фон під тему
                if theme == "dark":
                    fig.patch.set_facecolor("#020617")
                    ax.set_facecolor("#020617")
                    tick_color = "#e5e7eb"
                else:
                    fig.patch.set_facecolor("#e5e7eb")   # світло-сірий
                    ax.set_facecolor("#e5e7eb")
                    tick_color = "#111827"

                ax.scatter(
                    clustered["pca_x"],
                    clustered["pca_y"],
                    c=clustered["cluster"],
                    s=80,
                    alpha=0.8,
                )

                ax.set_title("Карта користувачів (PCA проекція)", color=tick_color)
                ax.tick_params(colors=tick_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(tick_color)

                st.pyplot(fig)

                st.write("Середні значення по кластерах:")
                st.dataframe(
                    clustered.groupby("cluster")[["avg_rating", "count", "age"]]
                    .mean()
                    .round(2),
                    use_container_width=True,
                )
        with sub2:
            test_size = st.slider("Частка тестової вибірки", 0.05, 0.5, 0.2)
            k_eval = st.slider("K у Precision@K", 1, 20, 10)
            if st.button("Оцінити модель"):
                with st.spinner("Обчислюємо метрики..."):
                    train_df, test_df = train_test_split_ratings(
                        ratings, test_size=test_size
                    )
                    metrics = evaluate_item_cf_on_split(train_df, test_df, movies, k=k_eval)
                st.metric(f"Precision@{k_eval}", f"{metrics[f'precision_at_{k_eval}']:.3f}")
                st.write(
                    {
                        "RMSE (перекриті прогнози)": metrics["rmse"],
                        "MAE (перекриті прогнози)": metrics["mae"],
                    }
                )

def render_user_top_nav():
    catalog_link = "Каталог"
    options = ["Головна", catalog_link, "План перегляду", "Коментарі", "Профіль"]
    current = st.session_state.get("user_nav", "Головна")

    col_logo, col_nav, col_icons = st.columns([2, 6, 2.5])

    with col_logo:
        logo_html = ""
        for name in ["site_icon_no_bg.png", "site_icon.png"]:
            if os.path.exists(name):
                with open(name, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                logo_html = (
                    f"<img src='data:image/png;base64,{b64}' "
                    f"width='26' style='margin-right:6px;border-radius:6px;'>"
                )
                break

        st.markdown(
            f"""
            <div class="top-nav-title">
                {logo_html}Movie<span style="color:#e50914;">Portal</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_nav:
        idx = options.index(current) if current in options else 0
        selection = st.radio(
            "Розділи порталу",
            options,
            index=idx,
            horizontal=True,
            label_visibility="collapsed",
            key="user_nav_radio",
        )
        st.session_state["user_nav"] = selection

    with col_icons:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🎲", help="Випадковий фільм"):
                st.session_state["random_movie_trigger"] = time.time()
        with c2:
            if st.button("🔍", help="Перейти до каталогу"):
                st.session_state["user_nav"] = catalog_link
                safe_rerun()
        with c3:
            if st.button("⚙️", help="Профіль користувача"):
                st.session_state["user_nav"] = "Профіль"
                safe_rerun()

    st.markdown("---")
    return st.session_state["user_nav"]


def render_user_profile(active_user_id, users_df, catalog_df):
    st.subheader("Профіль користувача")

    user_row = users_df[users_df["user_id"] == active_user_id]
    if not user_row.empty:
        rec = user_row.iloc[0]
        full_name = f"{rec.get('first_name','') or ''} {rec.get('last_name','') or ''}".strip()
    else:
        full_name = f"User {active_user_id}"

    st.markdown(f"**{full_name}**  \nID: `{active_user_id}`")

    tab_fav, tab_plan, tab_settings, tab_exit = st.tabs(
        ["Обране", "План перегляду", "Налаштування", "Вихід"]
    )

    favorites_df = load_user_favorites(active_user_id)
    watchlist_df = load_user_watchlist(active_user_id)

    with tab_fav:
        if favorites_df.empty:
            st.write("Поки що немає обраних фільмів.")
        else:
            fav_movies = favorites_df.merge(catalog_df, on="movie_id", how="left")
            show_movie_cards(
                fav_movies,
                max_items=30,
                current_user_id=active_user_id,
                favorites_ids=set(favorites_df["movie_id"]),
                watchlist_ids=set(watchlist_df["movie_id"]),
                prefix="favprofile",
            )

    with tab_plan:
        if watchlist_df.empty:
            st.write("У плані перегляду поки порожньо.")
        else:
            wl_movies = watchlist_df.merge(catalog_df, on="movie_id", how="left")
            show_movie_cards(
                wl_movies,
                max_items=30,
                current_user_id=active_user_id,
                favorites_ids=set(favorites_df["movie_id"]),
                watchlist_ids=set(watchlist_df["movie_id"]),
                prefix="planprofile",
            )

    with tab_settings:
        st.write("Налаштування профілю (демо):")
        st.checkbox("Надсилати email-сповіщення про нові рекомендації", value=True)
        st.checkbox("Показувати тільки фільми з рейтингом вище 3.0", value=True)
        st.text_input("Псевдонім (відображення в коментарях)", value=full_name or "")

    with tab_exit:
        st.write("Вийти з поточного акаунта.")
        if st.button("Вийти з порталу"):
            for key in ["auth_role", "username", "selected_user_id", "user_nav"]:
                st.session_state.pop(key, None)
            safe_rerun()


def user_dashboard(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    user_item: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    item_sim: np.ndarray,
    item_ids: np.ndarray,
    comments: pd.DataFrame,
):
    st.title("🍿 Користувацький портал")

    catalog = build_movie_catalog(movies, ratings)

    users_local = users.copy()
    users_local["first_name"] = users_local.get("first_name", "").fillna("")
    users_local["last_name"] = users_local.get("last_name", "").fillna("")
    users_local["full_name"] = (users_local["first_name"] + " " + users_local["last_name"]).str.strip()
    user_map = users_local.set_index("user_id")

    st.sidebar.markdown("---")
    st.sidebar.header("Користувач (для рекомендацій і коментарів)")

    if not user_map.empty:
        def format_user(uid):
            name = user_map.loc[uid, "full_name"]
            if not isinstance(name, str) or not name.strip():
                name = "Без імені"
            return f"{uid}: {name}"

        active_user_id = st.sidebar.selectbox(
            "Оберіть користувача",
            user_map.index.tolist(),
            format_func=format_user,
            key="selected_user_id",
        )
    else:
        active_user_id = None

    if active_user_id is None:
        st.info("Оберіть користувача у лівій панелі, щоб побачити персональні дані.")
        return

    favorites_df = load_user_favorites(active_user_id)
    watchlist_df = load_user_watchlist(active_user_id)
    fav_ids = set(favorites_df["movie_id"])
    watch_ids = set(watchlist_df["movie_id"])

    current_section = render_user_top_nav()

    if "random_movie_trigger" in st.session_state:
        rnd = catalog.sample(1, random_state=int(st.session_state["random_movie_trigger"]))
        st.info("🎲 Випадковий фільм:")
        show_movie_cards(
            rnd,
            max_items=1,
            current_user_id=active_user_id,
            favorites_ids=fav_ids,
            watchlist_ids=watch_ids,
            prefix="random",
        )

    if current_section == "Головна":
        st.subheader("Персональні рекомендації")
        if user_item.shape[0] > 0 and active_user_id is not None:
            recs = predict_item_based_for_user(active_user_id, user_item, movies, item_sim, item_ids, top_k=9)
            if recs is not None and not recs.empty:
                recs_full = recs.merge(
                    catalog[["movie_id", "avg_rating", "n_ratings"]],
                    on="movie_id",
                    how="left",
                )
                show_movie_cards(
                    recs_full,
                    max_items=9,
                    current_user_id=active_user_id,
                    favorites_ids=fav_ids,
                    watchlist_ids=watch_ids,
                    prefix="home",
                )
            else:
                st.info("Поки що немає персональних рекомендацій.")
        else:
            st.info("У наборі даних немає користувачів для побудови рекомендацій.")

        st.subheader("Популярні зараз")
        popular = catalog.sort_values(["n_ratings", "avg_rating"], ascending=[False, False]).head(12)
        show_movie_cards(
            popular,
            max_items=12,
            current_user_id=active_user_id,
            favorites_ids=fav_ids,
            watchlist_ids=watch_ids,
            prefix="popular",
        )

    elif current_section == "Каталог":
        st.subheader("Каталог фільмів")

        col_movies, col_filters = st.columns([3, 1])

        catalog_local = catalog.copy()

        with col_filters:
            st.markdown(
                '<div class="dashboard-card filters-card"><h4>ФІЛЬТРИ</h4>',
                unsafe_allow_html=True,
            )

            search = st.text_input(
                "Пошук за назвою",
                key="catalog_search",
            )

            all_genres = sorted(
                {
                    g.strip()
                    for row in catalog_local["genres"].dropna()
                    for g in str(row).split("|")
                    if g.strip()
                }
            )
            selected_genres = st.multiselect(
                "Жанри",
                all_genres,
                key="catalog_genres",
            )

            min_rating = st.slider(
                "Мін. середній рейтинг",
                0.0, 5.0, 3.0, 0.1,
                key="catalog_min_rating",
            )

            sort_by = st.selectbox(
                "Сортувати за",
                ["Найпопулярніші", "Найкращий рейтинг", "Новіші спочатку"],
                key="catalog_sort_by",
            )

            st.markdown("</div>", unsafe_allow_html=True)

        df = catalog_local

        if search:
            df = df[df["title"].str.contains(search, case=False, na=False)]

        if selected_genres:
            mask = df["genres"].fillna("").apply(
                lambda s: any(g in s for g in selected_genres)
            )
            df = df[mask]

        df = df[df["avg_rating"] >= min_rating]

        if sort_by == "Найпопулярніші":
            df = df.sort_values(["n_ratings", "avg_rating"], ascending=[False, False])
        elif sort_by == "Найкращий рейтинг":
            df = df.sort_values(["avg_rating", "n_ratings"], ascending=[False, False])
        else:
            if "year" in df.columns:
                df = df.sort_values(["year", "avg_rating"], ascending=[False, False])
            else:
                df = df.sort_values(["avg_rating", "n_ratings"], ascending=[False, False])

        with col_movies:
            st.markdown('<div class="catalog-section">', unsafe_allow_html=True)

            show_movie_cards(
                df,
                max_items=36,
                current_user_id=active_user_id,
                favorites_ids=fav_ids,
                watchlist_ids=watch_ids,
                prefix="catalog",
            )

            st.markdown('</div>', unsafe_allow_html=True)

    elif current_section == "План перегляду":
        st.subheader("План перегляду")

        watchlist_df = load_user_watchlist(active_user_id)
        if watchlist_df.empty:
            st.info("План перегляду поки порожній. Додайте фільми з каталогу або рекомендацій.")
        else:
            wl_movies = watchlist_df.merge(
                catalog,
                on="movie_id",
                how="left",
            )
            show_movie_cards(
                wl_movies,
                max_items=30,
                current_user_id=active_user_id,
                favorites_ids=fav_ids,
                watchlist_ids=watchlist_df["movie_id"].pipe(set),
                prefix="plan",
            )

    elif current_section == "Коментарі":
        st.subheader("Коментарі до фільмів")

        comments_merged = comments.merge(
            users_local[["user_id", "full_name"]],
            on="user_id",
            how="left",
        )

        df_for_comments = catalog.sort_values("title")
        comment_movie_title = st.selectbox(
            "Оберіть фільм для коментаря", df_for_comments["title"].tolist()
        )
        selected_movie = df_for_comments[df_for_comments["title"] == comment_movie_title].iloc[0]
        movie_id = int(selected_movie["movie_id"])
        comment_text = st.text_area("Ваш коментар")

        if st.button("Зберегти коментар"):
            if comment_text.strip():
                add_comment_to_db(active_user_id, movie_id, comment_text.strip())
                st.success("Коментар збережено.")
                safe_rerun()
            else:
                st.warning("Коментар порожній.")

        movie_comments = comments_merged[comments_merged["movie_id"] == movie_id].copy()
        movie_comments = movie_comments.sort_values("created_at", ascending=False).head(50)

        if not movie_comments.empty:
            st.write("Коментарі:")
            for _, row in movie_comments.iterrows():
                name = row.get("full_name")
                if not isinstance(name, str) or not name.strip():
                    name = f"User {row['user_id']}"
                created = row.get("created_at", "")
                text = row.get("comment_text", "")
                st.markdown(f"**{name}** · _{created}_  \n{text}")
        else:
            st.info("До цього фільму ще немає коментарів.")

    else:
        render_user_profile(active_user_id, users, catalog)


def main():
    # ---- Власний перемикач теми (dark / light) ----
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "dark"

    with st.sidebar:
        theme_choice = st.radio(
            "Тема інтерфейсу",
            ["Темна", "Світла"],
            index=0 if st.session_state["ui_theme"] == "dark" else 1,
        )
    st.session_state["ui_theme"] = "dark" if theme_choice == "Темна" else "light"

    # Altair під ту ж тему
    enable_altair_theme(st.session_state["ui_theme"])

    # стилі під вибрану тему
    inject_global_styles()

    # ---- решта main як було ----
    if "auth_role" not in st.session_state:
        st.session_state["auth_role"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None

    if st.session_state["auth_role"] is None:
        show_login()
        return

    movies, ratings, users = load_base_data()
    if movies is None:
        st.stop()

    comments = load_comments()

    user_item = build_user_item_matrix(ratings)
    genre_matrix = build_genre_matrix(movies)
    with st.spinner("Обчислюємо / завантажуємо схожість фільмів..."):
        item_sim, item_ids = compute_item_similarity_diskcached(user_item)

    if os.path.exists("site_icon.png"):
        st.sidebar.image("site_icon.png", width=40)
    elif os.path.exists("site_icon_no_bg.png"):
        st.sidebar.image("site_icon_no_bg.png", width=40)

    role_label = "Адміністратор" if st.session_state["auth_role"] == "admin" else "Користувач"
    st.sidebar.markdown(f"**Ви увійшли як:** {role_label}")
    if st.sidebar.button("Вийти"):
        for key in ["auth_role", "username", "selected_user_id", "user_nav"]:
            st.session_state.pop(key, None)
        safe_rerun()

    st.sidebar.markdown("---")
    st.sidebar.write("📊 Коротка статистика")
    st.sidebar.write(f"Фільмів: **{len(movies)}**")
    st.sidebar.write(f"Користувачів: **{len(users)}**")
    st.sidebar.write(f"Оцінок: **{len(ratings)}**")

    if st.session_state["auth_role"] == "admin":
        render_cache_controls()
        admin_dashboard(movies, ratings, users, user_item, genre_matrix, item_sim, item_ids, comments)
    else:
        user_dashboard(movies, ratings, users, user_item, genre_matrix, item_sim, item_ids, comments)

if __name__ == "__main__":
    main()