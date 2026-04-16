from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifacts import ModelArtifacts, load_artifacts, resolve_paths
from .recommender import RecommendationResult, get_hybrid_recommendations, search_songs


class MusicRecommenderService:
    """Application service that coordinates artifact-backed recommendations."""

    def __init__(self, artifacts: ModelArtifacts):
        self._artifacts = artifacts

    @classmethod
    def from_base_dir(cls, base_dir: str | Path) -> "MusicRecommenderService":
        return cls(load_artifacts(resolve_paths(base_dir)))

    @property
    def catalog(self) -> pd.DataFrame:
        return self._artifacts.songs

    def get_song(self, song_index: int) -> pd.Series:
        return self.catalog.loc[song_index]

    def search_songs(self, query: str, max_results: int = 50) -> pd.DataFrame:
        return search_songs(self.catalog, query, max_results=max_results)

    def recommend(
        self,
        selected_idx: int,
        *,
        alpha: float = 0.3,
        top_k: int = 10,
        mood: str | None = "None",
    ) -> list[RecommendationResult]:
        return get_hybrid_recommendations(
            selected_idx=selected_idx,
            df_full=self.catalog,
            feature_matrix=self._artifacts.feature_matrix,
            item_users_dict=self._artifacts.item_users,
            alpha=alpha,
            top_k=top_k,
            mood=mood,
        )

    def health_summary(self) -> dict[str, int]:
        return {
            "song_count": self._artifacts.song_count,
            "feature_count": self._artifacts.feature_count,
            "collaborative_item_count": self._artifacts.collaborative_item_count,
        }
