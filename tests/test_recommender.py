from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from music_recommender.recommender import (
    apply_mood_filter,
    get_hybrid_recommendations,
    jaccard_similarity,
    search_songs,
)


def build_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Focus Flow",
                "artist": "Alice",
                "genre": "ambient",
                "tags": "study calm",
                "danceability": 0.20,
                "energy": 0.30,
                "valence": 0.40,
                "tempo": 90.0,
                "speechiness": 0.05,
                "instrumentalness": 0.80,
                "acousticness": 0.50,
                "liveness": 0.10,
            },
            {
                "name": "Night Focus",
                "artist": "Bob",
                "genre": "lofi",
                "tags": "focus beats",
                "danceability": 0.35,
                "energy": 0.40,
                "valence": 0.45,
                "tempo": 92.0,
                "speechiness": 0.04,
                "instrumentalness": 0.30,
                "acousticness": 0.40,
                "liveness": 0.12,
            },
            {
                "name": "Dance Sparks",
                "artist": "Carol",
                "genre": "dance",
                "tags": "party energy",
                "danceability": 0.90,
                "energy": 0.92,
                "valence": 0.88,
                "tempo": 128.0,
                "speechiness": 0.07,
                "instrumentalness": 0.01,
                "acousticness": 0.05,
                "liveness": 0.22,
            },
            {
                "name": "Rain Window",
                "artist": "Dana",
                "genre": "acoustic",
                "tags": "sad relax",
                "danceability": 0.25,
                "energy": 0.22,
                "valence": 0.20,
                "tempo": 82.0,
                "speechiness": 0.03,
                "instrumentalness": 0.40,
                "acousticness": 0.80,
                "liveness": 0.08,
            },
        ]
    )


class RecommendationLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = build_catalog()
        self.feature_matrix = np.array(
            [
                [1.00, 0.00, 0.00],
                [0.95, 0.05, 0.00],
                [0.00, 1.00, 0.00],
                [0.80, 0.00, 0.20],
            ]
        )
        self.item_users = {
            0: {"u1", "u2"},
            1: {"u2", "u3"},
            2: {"u9"},
            3: {"u4", "u5"},
        }

    def test_search_songs_matches_tags(self) -> None:
        results = search_songs(self.df, "beats")
        self.assertEqual(list(results["name"]), ["Night Focus"])

    def test_apply_mood_filter_study_is_selective(self) -> None:
        filtered = apply_mood_filter(self.df, "Study")
        self.assertEqual(list(filtered["name"]), ["Focus Flow", "Night Focus", "Rain Window"])

    def test_jaccard_similarity_returns_zero_for_missing_item(self) -> None:
        self.assertEqual(jaccard_similarity(0, 99, self.item_users), 0.0)

    def test_recommendations_exclude_selected_song_and_fallback_when_needed(self) -> None:
        recommendations = get_hybrid_recommendations(
            selected_idx=0,
            df_full=self.df,
            feature_matrix=self.feature_matrix,
            item_users_dict=self.item_users,
            alpha=0.3,
            top_k=3,
            mood="Study",
        )

        recommendation_indices = [recommendation.song_index for recommendation in recommendations]
        self.assertEqual(len(recommendation_indices), 3)
        self.assertNotIn(0, recommendation_indices)
        self.assertEqual(recommendation_indices[0], 1)
