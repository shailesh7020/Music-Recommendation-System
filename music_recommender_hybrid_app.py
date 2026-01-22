import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(page_title="Hybrid Music Recommender (Mood-Based)", layout="wide")
st.title("🎧 Hybrid Music Recommendation System (Mood-Based)")
st.write("Hybrid = Content Similarity + Collaborative Filtering + Mood Filter")

# ---------------------------
# PATH CONFIG
# ---------------------------
SAVE_DIR = r"E:\Music Recommendation\Hybrid model"

SONGS_PATH = os.path.join(SAVE_DIR, "songs_df.joblib")
FEATURE_PATH = os.path.join(SAVE_DIR, "feature_matrix.joblib")
ITEM_USERS_PATH = os.path.join(SAVE_DIR, "item_users_dict.joblib")

# ---------------------------
# LOAD ARTIFACTS
# ---------------------------
@st.cache_resource
def load_artifacts():
    df = joblib.load(SONGS_PATH)
    feature_matrix = joblib.load(FEATURE_PATH)
    item_users = joblib.load(ITEM_USERS_PATH)
    return df, feature_matrix, item_users

df, feature_matrix, item_users = load_artifacts()

# ---------------------------
# BASIC CLEANUP
# ---------------------------
for col in ["name", "artist", "genre", "tags"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

if "spotify_preview_url" not in df.columns:
    df["spotify_preview_url"] = ""

# Mood numeric columns
required_mood_cols = [
    "danceability", "energy", "valence", "tempo",
    "speechiness", "instrumentalness", "acousticness", "liveness"
]
for c in required_mood_cols:
    if c not in df.columns:
        df[c] = 0
df[required_mood_cols] = df[required_mood_cols].fillna(0)

# ---------------------------
# SESSION STATE (Single Audio Player)
# ---------------------------
if "play_url" not in st.session_state:
    st.session_state.play_url = None

if "now_playing_title" not in st.session_state:
    st.session_state.now_playing_title = None

# ---------------------------
# MOOD FILTER FUNCTION
# ---------------------------
def apply_mood_filter(df_in: pd.DataFrame, mood: str) -> pd.DataFrame:
    mood = mood.lower().strip()

    if mood == "study":
        return df_in[
            (df_in["energy"] < 0.55) &
            (df_in["speechiness"] < 0.10) &
            (df_in["instrumentalness"] > 0.25)
        ]

    elif mood == "dance":
        return df_in[
            (df_in["danceability"] > 0.70) &
            (df_in["energy"] > 0.65) &
            (df_in["tempo"] > 110)
        ]

    elif mood == "happy":
        return df_in[
            (df_in["valence"] > 0.65) &
            (df_in["energy"] > 0.55)
        ]

    elif mood == "sad":
        return df_in[
            (df_in["valence"] < 0.35) &
            (df_in["energy"] < 0.55) &
            (df_in["acousticness"] > 0.30)
        ]

    elif mood == "relax":
        return df_in[
            (df_in["energy"] < 0.50) &
            (df_in["acousticness"] > 0.40) &
            (df_in["tempo"] < 110)
        ]

    elif mood == "party":
        return df_in[
            (df_in["energy"] > 0.75) &
            (df_in["valence"] > 0.55) &
            (df_in["danceability"] > 0.65)
        ]

    elif mood == "workout":
        return df_in[
            (df_in["energy"] > 0.80) &
            (df_in["tempo"] > 120)
        ]

    else:
        return df_in

# ---------------------------
# COLLABORATIVE SIMILARITY (JACCARD)
# ---------------------------
def jaccard_similarity(item_a: int, item_b: int, item_users_dict: dict) -> float:
    users_a = item_users_dict.get(item_a, set())
    users_b = item_users_dict.get(item_b, set())

    if not users_a or not users_b:
        return 0.0

    inter = len(users_a & users_b)
    union = len(users_a | users_b)
    return inter / union if union != 0 else 0.0

# ---------------------------
# SEARCH FUNCTION (NO RAPIDFUZZ)
# ---------------------------
def search_songs(df_in: pd.DataFrame, query: str, max_results=50) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return df_in.head(max_results)

    mask = (
        df_in["name"].str.lower().str.contains(q, regex=False) |
        df_in["artist"].str.lower().str.contains(q, regex=False) |
        df_in["genre"].str.lower().str.contains(q, regex=False)
    )
    return df_in[mask].head(max_results)

# ---------------------------
# HYBRID RECOMMENDER
# ---------------------------
def get_hybrid_recommendations(
    selected_idx: int,
    df_full: pd.DataFrame,
    feature_matrix,
    item_users_dict: dict,
    alpha: float = 0.3,
    top_k: int = 10,
    mood: str = "None"
):
    # Mood filtered candidates
    df_candidates = df_full.copy()
    if mood != "None":
        df_candidates = apply_mood_filter(df_candidates, mood)

    candidate_indices = df_candidates.index.tolist()

    # Fallback if too few candidates
    if len(candidate_indices) < top_k + 1:
        candidate_indices = df_full.index.tolist()

    # Content similarity
    base_vec = feature_matrix[selected_idx]
    cand_mat = feature_matrix[candidate_indices]
    content_scores = cosine_similarity(base_vec, cand_mat).flatten()

    # Collaborative similarity
    collab_scores = np.array([
        jaccard_similarity(selected_idx, cand_idx, item_users_dict)
        for cand_idx in candidate_indices
    ], dtype=np.float32)

    # Hybrid score
    hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores

    ranked = sorted(
        zip(candidate_indices, hybrid_scores, content_scores, collab_scores),
        key=lambda x: x[1],
        reverse=True
    )

    ranked = [r for r in ranked if r[0] != selected_idx]
    return ranked[:top_k]

# ---------------------------
# UI (FRONT PAGE)
# ---------------------------
colA, colB, colC = st.columns([2, 1, 1])

with colA:
    query = st.text_input("🔎 Search song (name / artist / genre)", "")

with colB:
    mood_choice = st.selectbox(
        "🎭 Select Mood",
        ["None", "Study", "Dance", "Happy", "Sad", "Relax", "Party", "Workout"]
    )

with colC:
    top_k = st.slider("Top-K", 1, 20, 10)

alpha = st.slider("⚖ Hybrid Weight α (Collaborative vs Content)", 0.0, 1.0, 0.3, 0.05)

results = search_songs(df, query)
st.write(f"✅ Found **{len(results)}** matching songs (showing up to 50)")

if len(results) == 0:
    st.warning("No matching songs found. Try a different search.")
    st.stop()

# Song selection
options = results.apply(lambda r: f"{r['name']} — {r['artist']} ({r.get('genre','')})", axis=1).tolist()
selected_option = st.selectbox("🎵 Select a song", options)

selected_row = results.iloc[options.index(selected_option)]
selected_idx = int(selected_row.name)

# ---------------------------
# SELECTED SONG DISPLAY (NO AUTO PLAY)
# ---------------------------
st.subheader("🎶 Selected Song")
c1, c2 = st.columns([2, 2])

with c1:
    st.markdown(f"### **{selected_row['name']}**")
    st.write(f"**Artist:** {selected_row['artist']}")
    st.write(f"**Genre:** {selected_row.get('genre','-')}")
    st.write(f"**Year:** {selected_row.get('year','-')}")
    st.write(f"**Track ID:** {selected_row.get('track_id','-')}")

with c2:
    selected_preview = selected_row.get("spotify_preview_url", "")
    if isinstance(selected_preview, str) and selected_preview.startswith("http"):
        if st.button("▶ Play Selected Song Preview", key="play_selected"):
            st.session_state.play_url = selected_preview
            st.session_state.now_playing_title = f"{selected_row['name']} — {selected_row['artist']}"
    else:
        st.info("No Spotify preview available for this song.")

# ---------------------------
# RECOMMENDATIONS
# ---------------------------
st.subheader(f"🔥 Top {top_k} Recommendations (Mood: {mood_choice})")

ranked = get_hybrid_recommendations(
    selected_idx=selected_idx,
    df_full=df,
    feature_matrix=feature_matrix,
    item_users_dict=item_users,
    alpha=alpha,
    top_k=top_k,
    mood=mood_choice
)

for i, (idx, hybrid_s, content_s, collab_s) in enumerate(ranked, start=1):
    row = df.loc[idx]

    st.markdown(f"### {i}. {row['name']} — {row['artist']}")
    st.write(f"Genre: {row.get('genre','-')} | Year: {row.get('year','-')}")
    st.write(f"Hybrid Score: **{hybrid_s:.4f}** | Content: {content_s:.4f} | Collaborative: {collab_s:.4f}")

    preview = row.get("spotify_preview_url", "")
    if isinstance(preview, str) and preview.startswith("http"):
        if st.button(f"▶ Play Preview {i}", key=f"play_{idx}"):
            st.session_state.play_url = preview
            st.session_state.now_playing_title = f"{row['name']} — {row['artist']}"
    else:
        st.caption("No preview available.")

    st.divider()

# ---------------------------
# SINGLE NOW PLAYING PLAYER
# ---------------------------
st.subheader("🎵 Now Playing")

if st.session_state.play_url:
    st.write(f"**Now Playing:** {st.session_state.now_playing_title}")
    st.audio(st.session_state.play_url)
    if st.button("⏹ Stop", key="stop_audio"):
        st.session_state.play_url = None
        st.session_state.now_playing_title = None
        st.rerun()
else:
    st.info("Click ▶ Play on any song preview to start playback.")

st.success("✅ Recommendations generated successfully!")
