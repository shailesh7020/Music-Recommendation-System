from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RecommendationResult:
    song_index: int
    hybrid_score: float
    content_score: float
    collaborative_score: float


def normalize_mood(mood: str | None) -> str:
    if mood is None:
        return "none"
    return mood.strip().lower()


def apply_mood_filter(df_in: pd.DataFrame, mood: str | None) -> pd.DataFrame:
    normalized_mood = normalize_mood(mood)

    if normalized_mood == "study":
        return df_in[
            (df_in["energy"] < 0.55)
            & (df_in["speechiness"] < 0.10)
            & (df_in["instrumentalness"] > 0.25)
        ]

    if normalized_mood == "dance":
        return df_in[
            (df_in["danceability"] > 0.70)
            & (df_in["energy"] > 0.65)
            & (df_in["tempo"] > 110)
        ]

    if normalized_mood == "happy":
        return df_in[(df_in["valence"] > 0.65) & (df_in["energy"] > 0.55)]

    if normalized_mood == "sad":
        return df_in[
            (df_in["valence"] < 0.35)
            & (df_in["energy"] < 0.55)
            & (df_in["acousticness"] > 0.30)
        ]

    if normalized_mood == "relax":
        return df_in[
            (df_in["energy"] < 0.50)
            & (df_in["acousticness"] > 0.40)
            & (df_in["tempo"] < 110)
        ]

    if normalized_mood == "party":
        return df_in[
            (df_in["energy"] > 0.75)
            & (df_in["valence"] > 0.55)
            & (df_in["danceability"] > 0.65)
        ]

    if normalized_mood == "workout":
        return df_in[(df_in["energy"] > 0.80) & (df_in["tempo"] > 120)]

    return df_in


def jaccard_similarity(item_a: int, item_b: int, users_dict: dict[int, set[Any]]) -> float:
    users_a = users_dict.get(item_a, set())
    users_b = users_dict.get(item_b, set())

    if not users_a or not users_b:
        return 0.0

    return len(users_a & users_b) / len(users_a | users_b)


def search_songs(df_in: pd.DataFrame, query: str, max_results: int = 50) -> pd.DataFrame:
    if max_results <= 0:
        raise ValueError("max_results must be greater than zero.")

    normalized_query = query.strip().lower()
    if not normalized_query:
        return df_in.head(max_results)

    mask = (
        df_in["name"].str.lower().str.contains(normalized_query, regex=False)
        | df_in["artist"].str.lower().str.contains(normalized_query, regex=False)
        | df_in["genre"].str.lower().str.contains(normalized_query, regex=False)
        | df_in["tags"].str.lower().str.contains(normalized_query, regex=False)
    )

    return df_in.loc[mask].head(max_results)


def get_hybrid_recommendations(
    selected_idx: int,
    df_full: pd.DataFrame,
    feature_matrix: Any,
    item_users_dict: dict[int, set[Any]],
    alpha: float = 0.3,
    top_k: int = 10,
    mood: str | None = "None",
) -> list[RecommendationResult]:
    if selected_idx < 0 or selected_idx >= len(df_full):
        raise IndexError(f"Selected song index {selected_idx} is out of range.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    df_candidates = apply_mood_filter(df_full, mood)
    candidate_indices = df_candidates.index.tolist()

    if len(candidate_indices) < top_k + 1:
        candidate_indices = df_full.index.tolist()

    base_vector = feature_matrix[selected_idx]
    if getattr(base_vector, "ndim", 0) == 1:
        base_vector = base_vector.reshape(1, -1)

    candidate_matrix = feature_matrix[candidate_indices]

    content_scores = cosine_similarity(base_vector, candidate_matrix).flatten()
    collaborative_scores = np.array(
        [jaccard_similarity(selected_idx, idx, item_users_dict) for idx in candidate_indices]
    )
    hybrid_scores = alpha * collaborative_scores + (1.0 - alpha) * content_scores

    ranked = sorted(
        zip(candidate_indices, hybrid_scores, content_scores, collaborative_scores),
        key=lambda result: result[1],
        reverse=True,
    )

    recommendations: list[RecommendationResult] = []
    for candidate_idx, hybrid_score, content_score, collaborative_score in ranked:
        if candidate_idx == selected_idx:
            continue

        recommendations.append(
            RecommendationResult(
                song_index=int(candidate_idx),
                hybrid_score=float(hybrid_score),
                content_score=float(content_score),
                collaborative_score=float(collaborative_score),
            )
        )

        if len(recommendations) == top_k:
            break

    return recommendations
