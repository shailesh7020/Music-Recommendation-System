from html import escape
from pathlib import Path

import streamlit as st

from music_recommender.log import get_logger
from music_recommender.service import MusicRecommenderService

LOGGER = get_logger()
BASE_DIR = Path(__file__).resolve().parent
MOOD_OPTIONS = ["None", "Study", "Dance", "Happy", "Sad", "Relax", "Party", "Workout"]
SEARCH_LIMIT = 8
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
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 185, 84, 0.18), transparent 28%),
                linear-gradient(180deg, #191414 0%, #0a0a0a 25%, #090909 100%);
            color: #f5f5f5;
            font-family: "Circular Std", "Helvetica Neue", "Segoe UI", sans-serif;
        }
        .block-container {max-width: 1480px; padding-top: 1.2rem; padding-bottom: 2.4rem;}
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(24,24,24,.98), rgba(10,10,10,.98));
            border-right: 1px solid rgba(255,255,255,.06);
        }
        [data-testid="stHeader"] {
            background: rgba(9,9,9,.72);
            border-bottom: 1px solid rgba(255,255,255,.04);
            backdrop-filter: blur(14px);
        }
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background: rgba(255,255,255,.05);
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.08);
        }
        div.stButton > button {
            min-height: 2.7rem;
            border-radius: 999px;
            font-weight: 800;
            border: none;
        }
        div.stButton > button[kind="primary"] {background: #1db954; color: #03150a;}
        div.stButton > button[kind="secondary"] {
            background: rgba(255,255,255,.08);
            color: #f5f5f5;
            border: 1px solid rgba(255,255,255,.08);
        }
        .page-title .eyebrow, .track-copy .eyebrow, .section-copy .eyebrow, .player-copy .eyebrow {
            color: #9ae6b4; font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
        }
        .page-title h1 {
            margin: .18rem 0 .4rem; font-size: clamp(2.3rem, 4vw, 4.5rem); line-height: .94; letter-spacing: -.05em;
        }
        .page-title p, .muted-copy {color: #b3b3b3; line-height: 1.55;}
        .hero-card, .glass-card, .player-card, .mini-card {
            background: linear-gradient(180deg, rgba(24,24,24,.97), rgba(18,18,18,.98));
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 24px;
            box-shadow: 0 20px 45px rgba(0,0,0,.22);
        }
        .hero-card {padding: 1.4rem; overflow: hidden; margin-bottom: 1rem;}
        .hero-grid {display: grid; grid-template-columns: 170px 1fr; gap: 1.2rem; align-items: end;}
        .hero-cover, .album-mini, .rank-chip {
            display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 800;
            border: 1px solid rgba(255,255,255,.12);
        }
        .hero-cover {
            width: 170px; height: 170px; border-radius: 22px; font-size: 2.6rem; letter-spacing: .08em;
            background: rgba(0,0,0,.24); box-shadow: 0 18px 35px rgba(0,0,0,.3);
        }
        .hero-copy h2 {margin: .18rem 0 .24rem; font-size: clamp(2rem, 4vw, 4.7rem); line-height: .96; letter-spacing: -.05em;}
        .hero-copy .artist {margin: 0 0 .7rem; color: rgba(255,255,255,.88);}
        .chip-row, .score-row {display: flex; flex-wrap: wrap; gap: .45rem;}
        .chip, .score-pill {
            padding: .36rem .7rem; border-radius: 999px; background: rgba(0,0,0,.22); border: 1px solid rgba(255,255,255,.12);
            font-size: .8rem; font-weight: 700;
        }
        .section-copy {margin: .2rem 0 .9rem;}
        .section-copy h3 {margin: 0; font-size: 1.35rem; letter-spacing: -.03em;}
        .section-copy p {margin: .18rem 0 0; color: #b3b3b3;}
        .mini-card {padding: 1rem; min-height: 100px;}
        .mini-card .label {color: #b3b3b3; font-size: .75rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;}
        .mini-card .value {margin-top: .5rem; font-size: 1.2rem; font-weight: 800;}
        .mini-card .hint {margin-top: .3rem; color: #b3b3b3; font-size: .85rem;}
        .sidebar-brand {display: flex; gap: .75rem; align-items: center; margin-bottom: 1rem;}
        .sidebar-dot {
            width: 18px; height: 18px; border-radius: 50%; background: linear-gradient(135deg, #1db954, #1ed760);
            box-shadow: 0 0 20px rgba(29,185,84,.45);
        }
        .sidebar-brand .title {font-size: 1.14rem; font-weight: 800; letter-spacing: -.03em;}
        .sidebar-brand .subtitle, .sidebar-note {color: #b3b3b3; font-size: .86rem;}
        .sidebar-nav {display: grid; gap: .55rem; margin: 1rem 0 1.2rem;}
        .sidebar-nav .nav-item {
            padding: .72rem .85rem; border-radius: 16px; background: rgba(255,255,255,.03); color: #b3b3b3; font-weight: 700;
        }
        .sidebar-nav .nav-item.active {background: rgba(255,255,255,.08); color: #fff;}
        .sidebar-label {margin: 1rem 0 .45rem; color: #9f9f9f; font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;}
        .sidebar-row {
            display: flex; justify-content: space-between; align-items: baseline; gap: .75rem; padding: .74rem .84rem;
            border-radius: 16px; background: rgba(255,255,255,.04); margin-bottom: .65rem;
        }
        .sidebar-row span {color: #b3b3b3; font-size: .85rem;}
        .sidebar-row strong {font-size: .96rem;}
        .rank-chip {
            width: 46px; height: 46px; border-radius: 16px; background: rgba(255,255,255,.05); margin-bottom: .45rem;
        }
        .album-mini {
            width: 56px; height: 56px; border-radius: 18px; font-size: 1rem; letter-spacing: .06em;
            box-shadow: 0 16px 25px rgba(0,0,0,.2);
        }
        .track-copy h4 {margin: .12rem 0 .12rem; font-size: 1rem; letter-spacing: -.03em;}
        .track-copy .artist {color: #b3b3b3; margin-bottom: .45rem;}
        .player-card {padding: 1rem 1.05rem;}
        .player-card h4 {margin: .2rem 0 .16rem; font-size: 1.02rem; letter-spacing: -.03em;}
        .player-card p {margin: 0; color: #b3b3b3;}
        @media (max-width: 1100px) {
            .hero-grid {grid-template-columns: 1fr;}
            .hero-cover {width: 132px; height: 132px;}
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


def format_tempo(value) -> str:
    text = safe_text(value, "")
    if not text:
        return "Tempo unavailable"
    try:
        return f"{float(text):.0f} BPM"
    except ValueError:
        return "Tempo unavailable"


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


def chip_row(values: list[str], class_name: str = "chip") -> str:
    pills = "".join(
        f"<span class='{class_name}'>{escape(value)}</span>"
        for value in values
        if value
    )
    return f"<div class='chip-row'>{pills}</div>"


def album_square(row) -> str:
    color_a, color_b = palette_for(row)
    return (
        f"<div class='album-mini' style='background: linear-gradient(135deg, {color_a}, {color_b});'>"
        f"{escape(initials_for(row))}</div>"
    )


def play_track(row) -> None:
    preview_url = preview_url_for(row)
    if preview_url:
        st.session_state.play_url = preview_url
        st.session_state.now_playing_title = song_label(row)


def stop_audio() -> None:
    st.session_state.play_url = None
    st.session_state.now_playing_title = None


def render_sidebar(summary: dict[str, int]) -> tuple[str, int, float]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="sidebar-dot"></div>
              <div>
                <div class="title">Spotify Mirror</div>
                <div class="subtitle">Hybrid recommendations wrapped like a streaming app.</div>
              </div>
            </div>
            <div class="sidebar-nav">
              <div class="nav-item active">Home</div>
              <div class="nav-item">Search</div>
              <div class="nav-item">Your Library</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-label'>Tune The Mix</div>", unsafe_allow_html=True)
        mood_choice = st.selectbox("Mood lane", MOOD_OPTIONS, index=0)
        top_k = st.slider("Recommendation depth", 3, 20, 10)
        alpha = st.slider("Collaborative blend", 0.0, 1.0, 0.3, 0.05)
        st.markdown("<div class='sidebar-label'>Catalog Health</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-row"><span>Songs</span><strong>{summary['song_count']:,}</strong></div>
            <div class="sidebar-row"><span>Features</span><strong>{summary['feature_count']:,}</strong></div>
            <div class="sidebar-row"><span>Collaborative Items</span><strong>{summary['collaborative_item_count']:,}</strong></div>
            <div class="sidebar-note">Run <code>py -3 -m music_recommender.healthcheck --base-dir .</code> before deployment.</div>
            """,
            unsafe_allow_html=True,
        )
    return mood_choice, top_k, alpha


st.set_page_config(
    page_title="Spotify Mirror | Music Recommender",
    page_icon=":material/music_note:",
    layout="wide",
    initial_sidebar_state="expanded",
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

if st.session_state.selected_song_index not in catalog.index:
    st.session_state.selected_song_index = default_song_index

summary = service.health_summary()
mood_choice, top_k, alpha = render_sidebar(summary)

st.markdown(
    """
    <div class="page-title">
      <div class="eyebrow">Spotify inspired interface</div>
      <h1>Your soundtrack, rebuilt in Streamlit.</h1>
      <p>
        Search the catalog, tune the mood lane, and browse recommendations in a layout designed
        to feel like a streaming dashboard instead of a prototype form.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

query = st.text_input(
    "Search the catalog",
    placeholder="What do you want to play?",
    label_visibility="collapsed",
)

search_results = service.search_songs(query, max_results=SEARCH_LIMIT)
selected_row = service.get_song(int(st.session_state.selected_song_index))
selected_idx = int(selected_row.name)

main_col, rail_col = st.columns([1.8, 1.05], gap="large")

with main_col:
    color_a, color_b = palette_for(selected_row)
    st.markdown(
        f"""
        <div class="hero-card" style="background: linear-gradient(135deg, {color_a}, {color_b});">
          <div class="hero-grid">
            <div class="hero-cover">{escape(initials_for(selected_row))}</div>
            <div class="hero-copy">
              <div class="eyebrow">Spotify mirror</div>
              <h2>{escape(safe_text(selected_row.get("name"), "Unknown Track"))}</h2>
              <div class="artist">{escape(safe_text(selected_row.get("artist"), "Unknown Artist"))}</div>
              {chip_row([
                  safe_text(selected_row.get("genre"), "Unknown genre").title(),
                  format_year(selected_row.get("year")),
                  "Preview ready" if has_preview(selected_row) else "Preview unavailable",
                  "All moods" if mood_choice == "None" else f"{mood_choice} lane",
              ])}
              <p class="muted-copy">Selected track driving a {top_k}-song blend with {1.0 - alpha:.0%}
              content similarity and {alpha:.0%} collaborative weight.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_cols = st.columns([1.05, 0.9, 2.8], gap="small")
    with action_cols[0]:
        if has_preview(selected_row):
            if st.button("Play Track", key="play_selected", type="primary", use_container_width=True):
                play_track(selected_row)
                st.rerun()
        else:
            st.button("No Preview", key="selected_no_preview", type="secondary", use_container_width=True, disabled=True)
    with action_cols[1]:
        if st.button("Stop Audio", key="stop_audio", type="secondary", use_container_width=True):
            stop_audio()
            st.rerun()
    with action_cols[2]:
        st.markdown(
            "<p class='muted-copy'>Preview playback uses the stored "
            "<code>spotify_preview_url</code> field when the track has one.</p>",
            unsafe_allow_html=True,
        )

    stat_cols = st.columns(4, gap="small")
    stat_cards = [
        ("Genre", safe_text(selected_row.get("genre"), "Unknown genre").title(), "Primary flavor in the current selection."),
        ("Year", format_year(selected_row.get("year")), "Useful for era-aware browsing."),
        ("Tempo", format_tempo(selected_row.get("tempo")), "Approximate pace from the artifact data."),
        ("Blend", f"{alpha:.0%} collaborative", f"{1.0 - alpha:.0%} content still in play."),
    ]
    for column, (label, value, hint) in zip(stat_cols, stat_cards):
        with column:
            st.markdown(
                f"<div class='mini-card'><div class='label'>{escape(label)}</div><div class='value'>{escape(value)}</div><div class='hint'>{escape(hint)}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-copy">
          <div class="eyebrow">Made for this track</div>
          <h3>Recommendations</h3>
          <p>Ranked by your selected mood lane and the hybrid scoring blend.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommendations = service.recommend(selected_idx, alpha=alpha, top_k=top_k, mood=mood_choice)
    for position, recommendation in enumerate(recommendations, start=1):
        row = catalog.loc[recommendation.song_index]
        row_cols = st.columns([0.8, 4.8, 1.1, 1.1], gap="small")
        with row_cols[0]:
            st.markdown(f"<div class='rank-chip'>#{position}</div>{album_square(row)}", unsafe_allow_html=True)
        with row_cols[1]:
            st.markdown(
                f"""
                <div class="track-copy">
                  <div class="eyebrow">Because you listened</div>
                  <h4>{escape(safe_text(row.get("name"), "Unknown Track"))}</h4>
                  <div class="artist">{escape(safe_text(row.get("artist"), "Unknown Artist"))}</div>
                  <div class="score-row">
                    <span class="score-pill">Hybrid {recommendation.hybrid_score:.3f}</span>
                    <span class="score-pill">Content {recommendation.content_score:.3f}</span>
                    <span class="score-pill">CF {recommendation.collaborative_score:.3f}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row_cols[2]:
            if st.button("Open", key=f"open_rec_{recommendation.song_index}", type="secondary", use_container_width=True):
                st.session_state.selected_song_index = int(recommendation.song_index)
                st.rerun()
        with row_cols[3]:
            if has_preview(row):
                if st.button("Play", key=f"play_rec_{recommendation.song_index}", type="primary", use_container_width=True):
                    play_track(row)
                    st.rerun()
            else:
                st.button("No Preview", key=f"rec_no_preview_{recommendation.song_index}", type="secondary", use_container_width=True, disabled=True)

with rail_col:
    st.markdown(
        """
        <div class="section-copy">
          <div class="eyebrow">Jump back in</div>
          <h3>Search hits</h3>
          <p>Quick picks rendered as a discovery rail on the right.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if search_results.empty:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-copy" style="margin: 0;">
                <h3>No direct matches</h3>
                <p>Try another artist, genre, or tag. Your current selection stays live on the left.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for position, (song_index, row) in enumerate(search_results.iterrows(), start=1):
            row_cols = st.columns([0.85, 3.2, 1.05, 1.05], gap="small")
            with row_cols[0]:
                st.markdown(album_square(row), unsafe_allow_html=True)
            with row_cols[1]:
                st.markdown(
                    f"""
                    <div class="track-copy">
                      <div class="eyebrow">Search hit #{position}</div>
                      <h4>{escape(safe_text(row.get("name"), "Unknown Track"))}</h4>
                      <div class="artist">{escape(safe_text(row.get("artist"), "Unknown Artist"))}</div>
                      {chip_row([safe_text(row.get("genre"), "Unknown genre").title(), format_year(row.get("year"))], "score-pill")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with row_cols[2]:
                if st.button("Open", key=f"open_search_{song_index}", type="secondary", use_container_width=True):
                    st.session_state.selected_song_index = int(song_index)
                    st.rerun()
            with row_cols[3]:
                if has_preview(row):
                    if st.button("Play", key=f"play_search_{song_index}", type="primary", use_container_width=True):
                        play_track(row)
                        st.rerun()
                else:
                    st.button("No Preview", key=f"search_no_preview_{song_index}", type="secondary", use_container_width=True, disabled=True)

    st.markdown(
        """
        <div class="section-copy" style="margin-top: 1.2rem;">
          <div class="eyebrow">Now playing</div>
          <h3>Player</h3>
          <p>The bottom-player feel, condensed into a Streamlit card.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.play_url:
        st.markdown(
            f"""
            <div class="player-card">
              <div class="eyebrow">Playback live</div>
              <h4>{escape(st.session_state.now_playing_title or 'Unknown Track')}</h4>
              <p>Audio preview is being streamed from the stored track URL.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.audio(st.session_state.play_url)
    else:
        st.markdown(
            """
            <div class="player-card">
              <div class="eyebrow">Standby</div>
              <h4>No track playing</h4>
              <p>Use any green play button to start a preview and light up the player.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
