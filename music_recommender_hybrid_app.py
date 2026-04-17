from html import escape
from pathlib import Path

import streamlit as st

from music_recommender.log import get_logger
from music_recommender.service import MusicRecommenderService

LOGGER = get_logger()
BASE_DIR = Path(__file__).resolve().parent
MOOD_OPTIONS = ["None", "Study", "Dance", "Happy", "Sad", "Relax", "Party", "Workout"]
SEARCH_LIMIT = 12
QUICK_TILE_COUNT = 6
FEATURED_CARD_COUNT = 5
QUEUE_LIMIT = 6
LIBRARY_LIMIT = 8
PALETTES = [
    ("#165d46", "#1db954"),
    ("#b45309", "#f59e0b"),
    ("#1d4ed8", "#38bdf8"),
    ("#0f766e", "#14b8a6"),
    ("#be123c", "#fb7185"),
]


@st.cache_resource
def load_service() -> MusicRecommenderService:
    return MusicRecommenderService.from_base_dir(BASE_DIR)


def inject_spotify_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --spotify-black: #000000;
            --spotify-canvas: #070707;
            --spotify-panel: #121212;
            --spotify-panel-elevated: #181818;
            --spotify-panel-hover: #242424;
            --spotify-text: #ffffff;
            --spotify-muted: #b3b3b3;
            --spotify-green: #1db954;
            --spotify-green-strong: #1ed760;
            --spotify-line: rgba(255, 255, 255, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top center, rgba(29, 185, 84, 0.11), transparent 22%),
                linear-gradient(180deg, #101010 0%, #080808 42%, #050505 100%);
            color: var(--spotify-text);
            font-family: "Circular Std", "Circular", "Helvetica Neue", "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1600px;
            padding-top: 1rem;
            padding-bottom: 1.2rem;
        }

        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stAppDeployButton,
        footer,
        #MainMenu {
            display: none !important;
        }

        .spotify-shell-title {
            display: none;
        }

        .spotify-panel {
            background: linear-gradient(180deg, rgba(18, 18, 18, 0.98), rgba(13, 13, 13, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
            margin-bottom: 0.85rem;
        }

        .chrome-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.85rem;
        }

        .chrome-buttons {
            display: flex;
            gap: 0.45rem;
        }

        .chrome-pill,
        .icon-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
        }

        .icon-pill {
            width: 2rem;
            height: 2rem;
            background: rgba(0, 0, 0, 0.72);
            color: var(--spotify-text);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .chrome-pill {
            padding: 0.55rem 0.95rem;
            background: rgba(255, 255, 255, 0.08);
            color: var(--spotify-text);
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 0.8rem;
        }

        .brand-logo {
            width: 1.85rem;
            height: 1.85rem;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, var(--spotify-green-strong), var(--spotify-green));
            box-shadow: 0 0 24px rgba(29, 185, 84, 0.35);
        }

        .brand-copy .title {
            color: var(--spotify-text);
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .brand-copy .subtitle,
        .section-copy p,
        .muted-copy,
        .track-meta .subtitle,
        .stat-copy .hint,
        .queue-meta,
        .library-meta {
            color: var(--spotify-muted);
            line-height: 1.45;
        }

        .nav-stack {
            display: grid;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.72rem 0.8rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.03);
            color: var(--spotify-muted);
            font-weight: 700;
            font-size: 0.94rem;
        }

        .nav-item.active {
            background: rgba(255, 255, 255, 0.09);
            color: var(--spotify-text);
        }

        .section-label {
            color: #8f8f8f;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .section-copy .eyebrow,
        .hero-meta .eyebrow,
        .track-meta .eyebrow,
        .player-copy .eyebrow {
            color: #a3e635;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .section-copy h2,
        .section-copy h3,
        .hero-copy h1,
        .track-meta h4,
        .player-copy h4 {
            margin: 0;
            color: var(--spotify-text);
            letter-spacing: -0.03em;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 1.2rem;
            border-radius: 16px;
            box-shadow: 0 28px 40px rgba(0, 0, 0, 0.26);
            margin-bottom: 0.85rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(0, 0, 0, 0.02), rgba(0, 0, 0, 0.56));
            pointer-events: none;
        }

        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 190px 1fr;
            gap: 1.15rem;
            align-items: end;
        }

        .cover-art,
        .card-art,
        .mini-art,
        .queue-art,
        .player-art,
        .library-art {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 800;
            box-shadow: 0 18px 30px rgba(0, 0, 0, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .cover-art {
            width: 190px;
            height: 190px;
            border-radius: 10px;
            font-size: 3rem;
            letter-spacing: 0.08em;
        }

        .card-art {
            width: 100%;
            aspect-ratio: 1 / 1;
            border-radius: 10px;
            font-size: 2rem;
            letter-spacing: 0.06em;
        }

        .mini-art,
        .library-art {
            width: 52px;
            height: 52px;
            border-radius: 8px;
            font-size: 1rem;
        }

        .queue-art,
        .player-art {
            width: 68px;
            height: 68px;
            border-radius: 10px;
            font-size: 1.2rem;
        }

        .hero-meta {
            color: rgba(255, 255, 255, 0.95);
        }

        .hero-copy h1 {
            font-size: clamp(2.2rem, 4vw, 5rem);
            line-height: 0.92;
            margin: 0.22rem 0 0.35rem;
            letter-spacing: -0.06em;
        }

        .hero-copy .subtitle {
            margin: 0 0 0.85rem;
            font-size: 0.98rem;
        }

        .pill-row,
        .metric-row,
        .tile-grid,
        .card-grid,
        .score-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .pill,
        .score-pill,
        .tag-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: rgba(0, 0, 0, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: var(--spotify-text);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.83rem;
            color: rgba(255, 255, 255, 0.92);
        }

        .quick-tile,
        .recommendation-card,
        .track-row-shell,
        .queue-row-shell,
        .stat-card,
        .library-row-shell {
            background: var(--spotify-panel-elevated);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }

        .quick-tile,
        .track-row-shell,
        .queue-row-shell,
        .library-row-shell {
            padding: 0.7rem;
        }

        .recommendation-card {
            padding: 0.9rem;
            min-height: 255px;
        }

        .section-copy {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            margin: 0.2rem 0 0.75rem;
        }

        .section-copy h2,
        .section-copy h3 {
            font-size: 1.45rem;
        }

        .show-all {
            color: var(--spotify-muted);
            font-size: 0.84rem;
            font-weight: 700;
        }

        .track-meta,
        .player-copy,
        .library-copy {
            min-width: 0;
        }

        .track-meta h4,
        .player-copy h4,
        .library-copy h4 {
            font-size: 0.98rem;
            margin: 0.12rem 0;
        }

        .player-copy h4 {
            font-size: 1.02rem;
        }

        .topbar-input label,
        .taste-control label {
            display: none !important;
        }

        div[data-baseweb="input"] > div {
            background: #2a2a2a;
            border-radius: 999px;
            border: 1px solid transparent;
            min-height: 2.85rem;
        }

        div[data-baseweb="select"] > div {
            background: #2a2a2a;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] * {
            color: var(--spotify-text) !important;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 0.4rem;
        }

        div.stButton > button {
            min-height: 2.5rem;
            border-radius: 999px;
            font-weight: 800;
            letter-spacing: 0.01em;
            border: none;
            transition: transform 0.15s ease, background-color 0.15s ease;
        }

        div.stButton > button[kind="primary"] {
            background: var(--spotify-green);
            color: #041209;
        }

        div.stButton > button[kind="primary"]:hover {
            background: var(--spotify-green-strong);
            transform: translateY(-1px);
        }

        div.stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.08);
            color: var(--spotify-text);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        div.stButton > button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.12);
            transform: translateY(-1px);
        }

        div.stButton > button:disabled {
            opacity: 0.38;
        }

        [data-testid="stAudio"] audio {
            width: 100%;
            border-radius: 999px;
            background: #0e0e0e;
        }

        .track-head {
            display: grid;
            grid-template-columns: 0.5fr 3fr 1.2fr 1fr;
            gap: 0.6rem;
            padding: 0 0.15rem 0.45rem;
            color: var(--spotify-muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .stat-row {
            display: grid;
            gap: 0.55rem;
        }

        .stat-card {
            padding: 0.8rem 0.85rem;
        }

        .stat-copy .label {
            color: var(--spotify-muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
        }

        .stat-copy .value {
            margin-top: 0.45rem;
            color: var(--spotify-text);
            font-size: 1.2rem;
            font-weight: 800;
        }

        @media (max-width: 1200px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .cover-art {
                width: 150px;
                height: 150px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_text(value, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def song_label(row) -> str:
    return f"{safe_text(row.get('name'), 'Unknown Track')} - {safe_text(row.get('artist'), 'Unknown Artist')}"


def format_year(value) -> str:
    text = safe_text(value, "")
    if not text:
        return "Unknown year"
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def format_duration_ms(value) -> str:
    text = safe_text(value, "")
    if not text:
        return "--:--"
    try:
        total_seconds = int(float(text) / 1000)
    except ValueError:
        return "--:--"

    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def preview_url_for(row) -> str | None:
    preview_url = row.get("spotify_preview_url", "")
    if isinstance(preview_url, str) and preview_url.startswith("http"):
        return preview_url
    return None


def has_preview(row) -> bool:
    return preview_url_for(row) is not None


def initials_for(row) -> str:
    artist = safe_text(row.get("artist"), "")
    tokens = [token[0] for token in artist.split() if token][:2]
    if tokens:
        return "".join(tokens).upper()
    return safe_text(row.get("name"), "M")[:2].upper()


def palette_for(row) -> tuple[str, str]:
    seed = sum(ord(character) for character in f"{row.get('name', '')}{row.get('artist', '')}")
    return PALETTES[seed % len(PALETTES)]


def art_markup(row, class_name: str) -> str:
    color_a, color_b = palette_for(row)
    return (
        f"<div class='{class_name}' style='background: linear-gradient(135deg, {color_a}, {color_b});'>"
        f"{escape(initials_for(row))}</div>"
    )


def pill_row(values: list[str], class_name: str = "pill") -> str:
    pills = "".join(
        f"<span class='{class_name}'>{escape(value)}</span>"
        for value in values
        if value
    )
    return f"<div class='pill-row'>{pills}</div>"


def unique_rows(rows: list, limit: int) -> list:
    unique = []
    seen: set[int] = set()
    for row in rows:
        row_index = int(row.name)
        if row_index in seen:
            continue
        seen.add(row_index)
        unique.append(row)
        if len(unique) == limit:
            break
    return unique


def select_track(song_index: int) -> None:
    st.session_state.selected_song_index = int(song_index)


def play_track(row) -> None:
    preview_url = preview_url_for(row)
    if preview_url:
        st.session_state.play_url = preview_url
        st.session_state.now_playing_title = song_label(row)
        st.session_state.playing_song_index = int(row.name)


def stop_audio() -> None:
    st.session_state.play_url = None
    st.session_state.now_playing_title = None
    st.session_state.playing_song_index = None


def render_left_rail(summary: dict[str, int], library_rows: list) -> None:
    st.markdown(
        """
        <div class="spotify-panel">
          <div class="brand-row">
            <div class="brand-logo"></div>
            <div class="brand-copy">
              <div class="title">Spotify</div>
              <div class="subtitle">Clone shell for your recommender</div>
            </div>
          </div>
          <div class="nav-stack">
            <div class="nav-item active">Home</div>
            <div class="nav-item">Search</div>
            <div class="nav-item">Your Library</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='spotify-panel'><div class='section-label'>Your Library</div></div>", unsafe_allow_html=True)
    for row in library_rows:
        row_cols = st.columns([0.9, 2.7, 1.1], gap="small")
        with row_cols[0]:
            st.markdown(art_markup(row, "library-art"), unsafe_allow_html=True)
        with row_cols[1]:
            st.markdown(
                f"""
                <div class="library-copy">
                  <h4>{escape(safe_text(row.get("name"), "Unknown Track"))}</h4>
                  <div class="library-meta">{escape(safe_text(row.get("artist"), "Unknown Artist"))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row_cols[2]:
            if st.button("Open", key=f"library_open_{int(row.name)}", type="secondary", use_container_width=True):
                select_track(int(row.name))
                st.rerun()

    st.markdown(
        f"""
        <div class="spotify-panel">
          <div class="section-label">Catalog Stats</div>
          <div class="stat-row">
            <div class="stat-card"><div class="stat-copy"><div class="label">Songs</div><div class="value">{summary['song_count']:,}</div><div class="hint">Full catalog loaded into the app.</div></div></div>
            <div class="stat-card"><div class="stat-copy"><div class="label">Features</div><div class="value">{summary['feature_count']:,}</div><div class="hint">Content vectors per song.</div></div></div>
            <div class="stat-card"><div class="stat-copy"><div class="label">Signals</div><div class="value">{summary['collaborative_item_count']:,}</div><div class="hint">Collaborative history available.</div></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar() -> str:
    chrome_cols = st.columns([1.0, 2.4, 1.3], gap="small")
    with chrome_cols[0]:
        st.markdown(
            """
            <div class="spotify-panel">
              <div class="chrome-buttons">
                <div class="icon-pill">&#8592;</div>
                <div class="icon-pill">&#8594;</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with chrome_cols[1]:
        st.markdown("<div class='spotify-panel'><div class='section-label'>Search</div></div>", unsafe_allow_html=True)
        query = st.text_input(
            "Search the catalog",
            placeholder="What do you want to play?",
            label_visibility="collapsed",
            key="topbar_query",
        )
    with chrome_cols[2]:
        st.markdown(
            """
            <div class="spotify-panel">
              <div class="chrome-row">
                <div class="chrome-pill">Explore Premium</div>
                <div class="chrome-pill">Codex User</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return query


def render_hero(selected_row, mood_choice: str, alpha: float, top_k: int) -> None:
    color_a, color_b = palette_for(selected_row)
    st.markdown(
        f"""
        <div class="hero-shell" style="background: linear-gradient(180deg, {color_b}, {color_a});">
          <div class="hero-grid">
            {art_markup(selected_row, "cover-art")}
            <div class="hero-copy hero-meta">
              <div class="eyebrow">Playlist</div>
              <h1>{escape(safe_text(selected_row.get("name"), "Unknown Track"))}</h1>
              <p class="subtitle">{escape(safe_text(selected_row.get("artist"), "Unknown Artist"))} • {escape(safe_text(selected_row.get("genre"), "Unknown genre").title())}</p>
              {pill_row([
                  format_year(selected_row.get("year")),
                  format_duration_ms(selected_row.get("duration_ms")),
                  "Preview ready" if has_preview(selected_row) else "Preview unavailable",
                  "Home" if mood_choice == "None" else f"{mood_choice} mix",
              ])}
              <div style="height: .7rem;"></div>
              <div class="metric-row">
                <span class="metric-pill">Built from {top_k} recommendations</span>
                <span class="metric-pill">{alpha:.0%} collaborative blend</span>
                <span class="metric-pill">{1.0 - alpha:.0%} content similarity</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_access(rows: list) -> None:
    if not rows:
        return
    st.markdown(
        """
        <div class="section-copy">
          <div>
            <div class="eyebrow">Good evening</div>
            <h3>Jump back in</h3>
            <p>Fast tiles, like Spotify home.</p>
          </div>
          <div class="show-all">6 quick picks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns(2, gap="small")
    for position, row in enumerate(rows):
        target = left_col if position % 2 == 0 else right_col
        with target:
            tile_cols = st.columns([0.82, 2.5, 1.0, 1.0], gap="small")
            with tile_cols[0]:
                st.markdown(art_markup(row, "mini-art"), unsafe_allow_html=True)
            with tile_cols[1]:
                st.markdown(
                    f"""
                    <div class="track-meta">
                      <h4>{escape(safe_text(row.get("name"), "Unknown Track"))}</h4>
                      <div class="subtitle">{escape(safe_text(row.get("artist"), "Unknown Artist"))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with tile_cols[2]:
                if st.button("Open", key=f"quick_open_{int(row.name)}", type="secondary", use_container_width=True):
                    select_track(int(row.name))
                    st.rerun()
            with tile_cols[3]:
                if has_preview(row):
                    if st.button("Play", key=f"quick_play_{int(row.name)}", type="primary", use_container_width=True):
                        play_track(row)
                        st.rerun()
                else:
                    st.button("Play", key=f"quick_disabled_{int(row.name)}", type="secondary", use_container_width=True, disabled=True)


def render_featured_cards(rows: list) -> None:
    if not rows:
        return
    st.markdown(
        """
        <div class="section-copy">
          <div>
            <div class="eyebrow">Made for you</div>
            <h3>Daily mixes</h3>
            <p>Recommendation cards that feel closer to Spotify’s home shelves.</p>
          </div>
          <div class="show-all">Personalized</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_columns = st.columns(len(rows), gap="small")
    for column, row in zip(card_columns, rows):
        with column:
            st.markdown(
                f"""
                <div class="recommendation-card">
                  {art_markup(row, "card-art")}
                  <div style="height: .85rem;"></div>
                  <div class="track-meta">
                    <div class="eyebrow">Daily Mix</div>
                    <h4>{escape(safe_text(row.get("name"), "Unknown Track"))}</h4>
                    <div class="subtitle">{escape(safe_text(row.get("artist"), "Unknown Artist"))}</div>
                    <div style="height: .55rem;"></div>
                    {pill_row([safe_text(row.get("genre"), "Unknown genre").title(), format_year(row.get("year"))], "tag-pill")}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            button_cols = st.columns(2, gap="small")
            with button_cols[0]:
                if st.button("Open", key=f"card_open_{int(row.name)}", type="secondary", use_container_width=True):
                    select_track(int(row.name))
                    st.rerun()
            with button_cols[1]:
                if has_preview(row):
                    if st.button("Play", key=f"card_play_{int(row.name)}", type="primary", use_container_width=True):
                        play_track(row)
                        st.rerun()
                else:
                    st.button("Play", key=f"card_disabled_{int(row.name)}", type="secondary", use_container_width=True, disabled=True)


def render_track_rows(recommendations, catalog) -> None:
    if not recommendations:
        return
    st.markdown(
        """
        <div class="section-copy">
          <div>
            <div class="eyebrow">Inspired by this track</div>
            <h3>Recommended songs</h3>
            <p>A dense, Spotify-like track list with hybrid ranking signals.</p>
          </div>
          <div class="show-all">Auto queued</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='track-head'><div>#</div><div>Title</div><div>Why it fits</div><div>Actions</div></div>", unsafe_allow_html=True)
    for position, recommendation in enumerate(recommendations, start=1):
        row = catalog.loc[recommendation.song_index]
        row_cols = st.columns([0.5, 3.0, 1.35, 1.25], gap="small")
        with row_cols[0]:
            st.markdown(f"<div class='track-row-shell'><div class='show-all'>#{position}</div></div>", unsafe_allow_html=True)
        with row_cols[1]:
            st.markdown(
                f"""
                <div class="track-row-shell">
                  <div style="display:flex; gap:.75rem; align-items:center;">
                    {art_markup(row, "mini-art")}
                    <div class="track-meta">
                      <h4>{escape(safe_text(row.get("name"), "Unknown Track"))}</h4>
                      <div class="subtitle">{escape(safe_text(row.get("artist"), "Unknown Artist"))}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row_cols[2]:
            st.markdown(
                f"""
                <div class="track-row-shell">
                  <div class="score-row">
                    <span class="score-pill">Hybrid {recommendation.hybrid_score:.3f}</span>
                    <span class="score-pill">Content {recommendation.content_score:.3f}</span>
                    <span class="score-pill">CF {recommendation.collaborative_score:.3f}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row_cols[3]:
            action_cols = st.columns(2, gap="small")
            with action_cols[0]:
                if st.button("Open", key=f"row_open_{recommendation.song_index}", type="secondary", use_container_width=True):
                    select_track(int(recommendation.song_index))
                    st.rerun()
            with action_cols[1]:
                if has_preview(row):
                    if st.button("Play", key=f"row_play_{recommendation.song_index}", type="primary", use_container_width=True):
                        play_track(row)
                        st.rerun()
                else:
                    st.button("Play", key=f"row_disabled_{recommendation.song_index}", type="secondary", use_container_width=True, disabled=True)


def render_right_rail(selected_row, current_row, summary: dict[str, int], queue_rows: list) -> tuple[str, int, float]:
    st.markdown(
        """
        <div class="spotify-panel">
          <div class="section-copy">
            <div>
              <div class="eyebrow">Now playing view</div>
              <h3>Track focus</h3>
              <p>Right rail details like Spotify’s side context.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="spotify-panel">
          <div style="display:flex; gap:.9rem; align-items:center;">
            {art_markup(current_row, "player-art")}
            <div class="player-copy">
              <div class="eyebrow">Current track</div>
              <h4>{escape(safe_text(current_row.get("name"), "Unknown Track"))}</h4>
              <p>{escape(safe_text(current_row.get("artist"), "Unknown Artist"))}</p>
              <div style="height:.45rem;"></div>
              {pill_row([safe_text(current_row.get("genre"), "Unknown genre").title(), format_year(current_row.get("year"))], "tag-pill")}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='spotify-panel'><div class='section-label'>Taste Profile</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Mood Lane</div>", unsafe_allow_html=True)
    mood_choice = st.selectbox(
        "Mood lane",
        MOOD_OPTIONS,
        index=MOOD_OPTIONS.index(st.session_state.get("mood_choice", "None")),
        key="mood_choice",
        label_visibility="collapsed",
    )
    st.markdown("<div class='section-label'>Recommendation Depth</div>", unsafe_allow_html=True)
    top_k = st.slider(
        "Recommendation depth",
        4,
        20,
        int(st.session_state.get("top_k", 10)),
        key="top_k",
        label_visibility="collapsed",
    )
    st.markdown("<div class='section-label'>Collaborative Blend</div>", unsafe_allow_html=True)
    alpha = st.slider(
        "Collaborative blend",
        0.0,
        1.0,
        float(st.session_state.get("alpha", 0.3)),
        0.05,
        key="alpha",
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="spotify-panel">
          <div class="section-copy">
            <div>
              <div class="eyebrow">Queue</div>
              <h3>Up next</h3>
              <p>{len(queue_rows)} tracks staged around your current search and recommendations.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for row in queue_rows:
        queue_cols = st.columns([0.85, 2.65, 1.0], gap="small")
        with queue_cols[0]:
            st.markdown(art_markup(row, "queue-art"), unsafe_allow_html=True)
        with queue_cols[1]:
            st.markdown(
                f"""
                <div class="track-meta">
                  <div class="eyebrow">Queue</div>
                  <h4>{escape(safe_text(row.get("name"), "Unknown Track"))}</h4>
                  <div class="queue-meta">{escape(safe_text(row.get("artist"), "Unknown Artist"))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with queue_cols[2]:
            if st.button("Open", key=f"queue_open_{int(row.name)}", type="secondary", use_container_width=True):
                select_track(int(row.name))
                st.rerun()

    st.markdown(
        f"""
        <div class="spotify-panel">
          <div class="section-label">Session Stats</div>
          <div class="stat-row">
            <div class="stat-card"><div class="stat-copy"><div class="label">Songs</div><div class="value">{summary['song_count']:,}</div><div class="hint">Catalog footprint.</div></div></div>
            <div class="stat-card"><div class="stat-copy"><div class="label">Selected</div><div class="value">{escape(safe_text(selected_row.get('genre'), 'Unknown').title())}</div><div class="hint">Current track mood anchor.</div></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return mood_choice, top_k, alpha


def render_bottom_player(current_row, selected_row, mood_choice: str, top_k: int) -> None:
    active_row = current_row if st.session_state.play_url else selected_row
    player_cols = st.columns([1.15, 2.0, 1.0], gap="medium")
    with player_cols[0]:
        st.markdown(
            f"""
            <div class="spotify-panel">
              <div style="display:flex; gap:.85rem; align-items:center;">
                {art_markup(active_row, "player-art")}
                <div class="player-copy">
                  <div class="eyebrow">Player</div>
                  <h4>{escape(safe_text(active_row.get("name"), "Unknown Track"))}</h4>
                  <p>{escape(safe_text(active_row.get("artist"), "Unknown Artist"))}</p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with player_cols[1]:
        st.markdown(
            """
            <div class="spotify-panel">
              <div class="section-copy">
                <div>
                  <div class="eyebrow">Playback controls</div>
                  <h3>Preview player</h3>
                  <p>Styled like Spotify’s bottom bar, using the track preview URL under the hood.</p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        controls = st.columns([0.95, 0.95, 3.1], gap="small")
        with controls[0]:
            if has_preview(active_row):
                if st.button("Play", key="bottom_play", type="primary", use_container_width=True):
                    play_track(active_row)
                    st.rerun()
            else:
                st.button("Play", key="bottom_disabled", type="secondary", use_container_width=True, disabled=True)
        with controls[1]:
            if st.button("Stop", key="bottom_stop", type="secondary", use_container_width=True):
                stop_audio()
                st.rerun()
        with controls[2]:
            if st.session_state.play_url:
                st.audio(st.session_state.play_url)
            else:
                st.markdown("<p class='muted-copy'>No preview playing. Use any green play button to start.</p>", unsafe_allow_html=True)
    with player_cols[2]:
        st.markdown(
            f"""
            <div class="spotify-panel">
              <div class="section-label">Listening Context</div>
              <div class="stat-row">
                <div class="stat-card"><div class="stat-copy"><div class="label">Mood</div><div class="value">{escape(mood_choice if mood_choice != 'None' else 'Home')}</div><div class="hint">Active recommendation lane.</div></div></div>
                <div class="stat-card"><div class="stat-copy"><div class="label">Queue</div><div class="value">{top_k}</div><div class="hint">Tracks generated for the session.</div></div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.set_page_config(
    page_title="Spotify Clone | Music Recommender",
    page_icon=":material/music_note:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_spotify_css()

try:
    service = load_service()
except Exception as exc:
    LOGGER.exception("Failed to load application artifacts.")
    st.error(f"Unable to start the recommender: {exc}")
    st.stop()

catalog = service.catalog
default_song_index = int(catalog.index[0])

if "selected_song_index" not in st.session_state:
    st.session_state.selected_song_index = default_song_index

if "play_url" not in st.session_state:
    st.session_state.play_url = None

if "now_playing_title" not in st.session_state:
    st.session_state.now_playing_title = None

if "playing_song_index" not in st.session_state:
    st.session_state.playing_song_index = None

if "mood_choice" not in st.session_state:
    st.session_state.mood_choice = "None"

if "top_k" not in st.session_state:
    st.session_state.top_k = 10

if "alpha" not in st.session_state:
    st.session_state.alpha = 0.3

if st.session_state.selected_song_index not in catalog.index:
    st.session_state.selected_song_index = default_song_index

summary = service.health_summary()
mood_choice = str(st.session_state.get("mood_choice", "None"))
top_k = int(st.session_state.get("top_k", 10))
alpha = float(st.session_state.get("alpha", 0.3))
query = str(st.session_state.get("topbar_query", "")).strip()

selected_row = service.get_song(int(st.session_state.selected_song_index))
selected_idx = int(selected_row.name)
search_results = service.search_songs(query, max_results=SEARCH_LIMIT)
recommendations = service.recommend(selected_idx, alpha=alpha, top_k=top_k, mood=mood_choice)

recommended_rows = [catalog.loc[recommendation.song_index] for recommendation in recommendations]
search_rows = [row for _, row in search_results.iterrows()]
quick_rows = unique_rows([selected_row] + search_rows + recommended_rows, QUICK_TILE_COUNT)
featured_rows = unique_rows(recommended_rows, FEATURED_CARD_COUNT)
library_source_rows = search_rows if search_rows else [selected_row] + recommended_rows
library_rows = unique_rows(library_source_rows, LIBRARY_LIMIT)
queue_source_rows = search_rows[:QUEUE_LIMIT] if query else recommended_rows[1:]
queue_rows = unique_rows(queue_source_rows, QUEUE_LIMIT)

playing_index = st.session_state.get("playing_song_index")
if playing_index in catalog.index:
    current_row = service.get_song(int(playing_index))
else:
    current_row = selected_row

left_col, main_col, right_col = st.columns([0.95, 2.4, 1.1], gap="medium")

with left_col:
    render_left_rail(summary, library_rows)

with main_col:
    render_top_bar()
    render_hero(selected_row, mood_choice, alpha, top_k)

    action_cols = st.columns([1.05, 0.95, 2.4], gap="small")
    with action_cols[0]:
        if has_preview(selected_row):
            if st.button("Play", key="hero_play", type="primary", use_container_width=True):
                play_track(selected_row)
                st.rerun()
        else:
            st.button("Play", key="hero_play_disabled", type="secondary", use_container_width=True, disabled=True)
    with action_cols[1]:
        if st.button("Stop", key="hero_stop", type="secondary", use_container_width=True):
            stop_audio()
            st.rerun()
    with action_cols[2]:
        st.markdown(
            "<p class='muted-copy'>The selected song acts like the active playlist header. "
            "Everything below mirrors a Spotify-style home feed while still using your hybrid recommender.</p>",
            unsafe_allow_html=True,
        )

    render_quick_access(quick_rows)
    render_featured_cards(featured_rows)
    render_track_rows(recommendations, catalog)

with right_col:
    mood_choice, top_k, alpha = render_right_rail(selected_row, current_row, summary, queue_rows)

render_bottom_player(current_row, selected_row, mood_choice, top_k)
