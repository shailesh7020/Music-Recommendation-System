from pathlib import Path

import streamlit as st

from music_recommender.log import get_logger
from music_recommender.service import MusicRecommenderService

LOGGER = get_logger()
BASE_DIR = Path(__file__).resolve().parent
MOOD_OPTIONS = ["None", "Study", "Dance", "Happy", "Sad", "Relax", "Party", "Workout"]


@st.cache_resource
def load_service() -> MusicRecommenderService:
    return MusicRecommenderService.from_base_dir(BASE_DIR)


def song_label(row) -> str:
    return f"{row['name']} - {row['artist']}"


def set_now_playing(url: str, title: str) -> None:
    st.session_state.play_url = url
    st.session_state.now_playing_title = title


st.set_page_config(page_title="Hybrid Music Recommender", layout="wide")
st.title("Hybrid Music Recommendation System")
st.write("Hybrid = content similarity + collaborative filtering + mood filter")

if "play_url" not in st.session_state:
    st.session_state.play_url = None

if "now_playing_title" not in st.session_state:
    st.session_state.now_playing_title = None

try:
    service = load_service()
except Exception as exc:
    LOGGER.exception("Failed to load application artifacts.")
    st.error(f"Unable to start the recommender: {exc}")
    st.stop()

catalog = service.catalog
summary = service.health_summary()

with st.sidebar:
    st.subheader("Catalog Health")
    st.metric("Songs", f"{summary['song_count']:,}")
    st.metric("Features", f"{summary['feature_count']:,}")
    st.metric("Collaborative Items", f"{summary['collaborative_item_count']:,}")
    st.caption("Use the health check CLI before deployment for artifact validation.")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    query = st.text_input("Search Song")

with col2:
    mood_choice = st.selectbox("Mood", MOOD_OPTIONS)

with col3:
    top_k = st.slider("Top-K", 1, 20, 10)

alpha = st.slider("Hybrid Weight", 0.0, 1.0, 0.3, 0.05)

results = service.search_songs(query)
st.write(f"Found {len(results)} songs")

if results.empty:
    st.warning("No songs found.")
    st.stop()

option_labels = results.apply(song_label, axis=1).tolist()
selected_label = st.selectbox("Select Song", option_labels)
selected_row = results.iloc[option_labels.index(selected_label)]
selected_idx = int(selected_row.name)

st.subheader("Selected Song")
details_col, actions_col = st.columns(2)

with details_col:
    st.write("**Name:**", selected_row["name"])
    st.write("**Artist:**", selected_row["artist"])
    st.write("**Genre:**", selected_row.get("genre", "-"))
    st.write("**Year:**", selected_row.get("year", "-"))

with actions_col:
    preview_url = selected_row.get("spotify_preview_url", "")
    if isinstance(preview_url, str) and preview_url.startswith("http"):
        if st.button("Play Selected"):
            set_now_playing(preview_url, song_label(selected_row))

st.subheader("Recommendations")
recommendations = service.recommend(
    selected_idx,
    alpha=alpha,
    top_k=top_k,
    mood=mood_choice,
)

for position, recommendation in enumerate(recommendations, start=1):
    row = catalog.loc[recommendation.song_index]
    st.markdown(f"### {position}. {song_label(row)}")
    st.write(
        "Hybrid: "
        f"{recommendation.hybrid_score:.4f} | "
        f"Content: {recommendation.content_score:.4f} | "
        f"CF: {recommendation.collaborative_score:.4f}"
    )

    preview_url = row.get("spotify_preview_url", "")
    if isinstance(preview_url, str) and preview_url.startswith("http"):
        if st.button(f"Play {position}", key=f"play_{position}"):
            set_now_playing(preview_url, song_label(row))

    st.divider()

st.subheader("Now Playing")
if st.session_state.play_url:
    st.write(st.session_state.now_playing_title)
    st.audio(st.session_state.play_url)
    if st.button("Stop"):
        st.session_state.play_url = None
        st.session_state.now_playing_title = None
        st.rerun()
else:
    st.info("Click Play to listen.")

st.success("Application loaded successfully.")
