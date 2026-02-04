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
# PATH CONFIG (CLOUD SAFE)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SONGS_PATH = os.path.join(BASE_DIR, "songs_df.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_matrix.joblib")
ITEM_USERS_PATH = os.path.join(BASE_DIR, "item_users_dict.joblib")


# ---------------------------
# LOAD ARTIFACTS
# ---------------------------
@st.cache_resource
def load_artifacts():
    df = joblib.load(SONGS_PATH)
    feature_matrix = joblib.load(FEATURE_PATH)
    item_users = joblib.load(ITEM_USERS_PATH)
    return df, feature_matrix, item_users


# ✅ IMPORTANT: Call function
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
# MOOD FILTER
# ---------------------------
def apply_mood_filter(df_in, mood):

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
# COLLABORATIVE (JACCARD)
# ---------------------------
def jaccard_similarity(a, b, users_dict):

    users_a = users_dict.get(a, set())
    users_b = users_dict.get(b, set())

    if not users_a or not users_b:
        return 0.0

    return len(users_a & users_b) / len(users_a | users_b)


# ---------------------------
# SEARCH
# ---------------------------
def search_songs(df_in, query, max_results=50):

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
    selected_idx,
    df_full,
    feature_matrix,
    item_users_dict,
    alpha=0.3,
    top_k=10,
    mood="None"
):

    df_candidates = df_full.copy()

    if mood != "None":
        df_candidates = apply_mood_filter(df_candidates, mood)

    candidate_indices = df_candidates.index.tolist()

    if len(candidate_indices) < top_k + 1:
        candidate_indices = df_full.index.tolist()

    base_vec = feature_matrix[selected_idx]
    cand_mat = feature_matrix[candidate_indices]

    content_scores = cosine_similarity(base_vec, cand_mat).flatten()

    collab_scores = np.array([
        jaccard_similarity(selected_idx, i, item_users_dict)
        for i in candidate_indices
    ])

    hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores

    ranked = sorted(
        zip(candidate_indices, hybrid_scores, content_scores, collab_scores),
        key=lambda x: x[1],
        reverse=True
    )

    ranked = [r for r in ranked if r[0] != selected_idx]

    return ranked[:top_k]


# ---------------------------
# UI
# ---------------------------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    query = st.text_input("🔎 Search Song")

with col2:
    mood_choice = st.selectbox(
        "🎭 Mood",
        ["None", "Study", "Dance", "Happy", "Sad", "Relax", "Party", "Workout"]
    )

with col3:
    top_k = st.slider("Top-K", 1, 20, 10)

alpha = st.slider("⚖ Hybrid Weight", 0.0, 1.0, 0.3, 0.05)


results = search_songs(df, query)

st.write(f"Found {len(results)} songs")

if len(results) == 0:
    st.warning("No songs found")
    st.stop()


options = results.apply(
    lambda r: f"{r['name']} — {r['artist']}",
    axis=1
).tolist()

selected = st.selectbox("🎵 Select Song", options)

selected_row = results.iloc[options.index(selected)]
selected_idx = int(selected_row.name)


# ---------------------------
# SELECTED SONG
# ---------------------------
st.subheader("Selected Song")

c1, c2 = st.columns(2)

with c1:
    st.write("**Name:**", selected_row["name"])
    st.write("**Artist:**", selected_row["artist"])
    st.write("**Genre:**", selected_row.get("genre", "-"))
    st.write("**Year:**", selected_row.get("year", "-"))

with c2:
    url = selected_row.get("spotify_preview_url", "")

    if isinstance(url, str) and url.startswith("http"):

        if st.button("▶ Play Selected"):
            st.session_state.play_url = url
            st.session_state.now_playing_title = f"{selected_row['name']} — {selected_row['artist']}"


# ---------------------------
# RECOMMENDATIONS
# ---------------------------
st.subheader("Recommendations")

ranked = get_hybrid_recommendations(
    selected_idx,
    df,
    feature_matrix,
    item_users,
    alpha,
    top_k,
    mood_choice
)

for i, (idx, h, c, cf) in enumerate(ranked, 1):

    row = df.loc[idx]

    st.markdown(f"### {i}. {row['name']} — {row['artist']}")
    st.write(f"Hybrid: {h:.4f} | Content: {c:.4f} | CF: {cf:.4f}")

    url = row.get("spotify_preview_url", "")

    if isinstance(url, str) and url.startswith("http"):

        if st.button(f"▶ Play {i}", key=f"p{i}"):

            st.session_state.play_url = url
            st.session_state.now_playing_title = f"{row['name']} — {row['artist']}"

    st.divider()


# ---------------------------
# AUDIO PLAYER
# ---------------------------
st.subheader("🎵 Now Playing")

if st.session_state.play_url:

    st.write(st.session_state.now_playing_title)
    st.audio(st.session_state.play_url)

    if st.button("⏹ Stop"):
        st.session_state.play_url = None
        st.session_state.now_playing_title = None
        st.rerun()

else:
    st.info("Click Play to listen")


st.success("✅ App Running Successfully")
