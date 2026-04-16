from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from music_recommender.artifacts import ArtifactPaths, load_artifacts


class ArtifactLoadingTests(unittest.TestCase):
    @staticmethod
    def build_paths() -> ArtifactPaths:
        base_dir = Path("E:/Music-Recommendation-System")
        return ArtifactPaths(
            base_dir=base_dir,
            songs_path=base_dir / "songs_df.joblib",
            feature_matrix_path=base_dir / "feature_matrix.joblib",
            item_users_path=base_dir / "item_users_dict.joblib",
        )

    def test_load_artifacts_normalizes_missing_columns(self) -> None:
        df = pd.DataFrame(
            {
                "track_id": ["song-1", "song-2"],
                "name": ["One", "Two"],
                "artist": ["Alice", "Bob"],
            }
        )
        feature_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        item_users = {0: {"u1"}, 1: {"u2"}}

        with patch("music_recommender.artifacts.Path.exists", return_value=True):
            with patch(
                "music_recommender.artifacts.joblib.load",
                side_effect=[df, feature_matrix, item_users],
            ):
                artifacts = load_artifacts(self.build_paths())

        self.assertIn("genre", artifacts.songs.columns)
        self.assertIn("spotify_preview_url", artifacts.songs.columns)
        self.assertEqual(artifacts.song_count, 2)

    def test_load_artifacts_rejects_misaligned_feature_matrix(self) -> None:
        df = pd.DataFrame(
            {
                "track_id": ["song-1"],
                "name": ["One"],
                "artist": ["Alice"],
            }
        )
        feature_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        item_users = {0: {"u1"}}

        with patch("music_recommender.artifacts.Path.exists", return_value=True):
            with patch(
                "music_recommender.artifacts.joblib.load",
                side_effect=[df, feature_matrix, item_users],
            ):
                with self.assertRaises(ValueError):
                    load_artifacts(self.build_paths())
