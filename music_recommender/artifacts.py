from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

TEXT_COLUMNS = ("name", "artist", "genre", "tags")
MOOD_COLUMNS = (
    "danceability",
    "energy",
    "valence",
    "tempo",
    "speechiness",
    "instrumentalness",
    "acousticness",
    "liveness",
)


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path
    songs_path: Path
    feature_matrix_path: Path
    item_users_path: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (self.songs_path, self.feature_matrix_path, self.item_users_path)


@dataclass(frozen=True)
class ModelArtifacts:
    songs: pd.DataFrame
    feature_matrix: Any
    item_users: dict[int, set[Any]]

    @property
    def song_count(self) -> int:
        return len(self.songs)

    @property
    def feature_count(self) -> int:
        return int(self.feature_matrix.shape[1])

    @property
    def collaborative_item_count(self) -> int:
        return len(self.item_users)


def resolve_paths(base_dir: str | Path) -> ArtifactPaths:
    resolved_base_dir = Path(base_dir).resolve()
    return ArtifactPaths(
        base_dir=resolved_base_dir,
        songs_path=resolved_base_dir / "songs_df.joblib",
        feature_matrix_path=resolved_base_dir / "feature_matrix.joblib",
        item_users_path=resolved_base_dir / "item_users_dict.joblib",
    )


def normalize_song_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("songs_df.joblib must contain a pandas DataFrame.")

    normalized = df.copy().reset_index(drop=True)

    for column in TEXT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
        normalized[column] = normalized[column].fillna("").astype(str)

    if "spotify_preview_url" not in normalized.columns:
        normalized["spotify_preview_url"] = ""
    normalized["spotify_preview_url"] = (
        normalized["spotify_preview_url"].fillna("").astype(str)
    )

    for column in MOOD_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = 0.0

    normalized[list(MOOD_COLUMNS)] = normalized[list(MOOD_COLUMNS)].fillna(0.0)
    return normalized


def normalize_item_users(raw_item_users: Mapping[Any, Any]) -> dict[int, set[Any]]:
    if not isinstance(raw_item_users, Mapping):
        raise TypeError("item_users_dict.joblib must contain a mapping.")

    normalized: dict[int, set[Any]] = {}
    for key, users in raw_item_users.items():
        if not isinstance(key, Integral):
            raise TypeError(f"Collaborative item key {key!r} is not an integer index.")

        if users is None:
            normalized[int(key)] = set()
            continue

        if isinstance(users, (str, bytes)) or not isinstance(users, Iterable):
            raise TypeError(
                f"Collaborative user collection for item {key!r} must be iterable."
            )

        normalized[int(key)] = set(users)

    return normalized


def validate_artifacts(
    songs: pd.DataFrame,
    feature_matrix: Any,
    item_users: Mapping[int, set[Any]],
) -> None:
    if not hasattr(feature_matrix, "shape"):
        raise TypeError("feature_matrix.joblib must contain an object with a shape.")

    if songs.empty:
        raise ValueError("Songs dataframe is empty.")

    if len(feature_matrix.shape) != 2:
        raise ValueError("Feature matrix must be two-dimensional.")

    if feature_matrix.shape[0] != len(songs):
        raise ValueError(
            "Feature matrix row count does not match the number of songs. "
            f"matrix_rows={feature_matrix.shape[0]} songs={len(songs)}"
        )

    invalid_keys = [key for key in item_users if key < 0 or key >= len(songs)]
    if invalid_keys:
        preview = ", ".join(str(key) for key in invalid_keys[:5])
        raise ValueError(
            "Collaborative item indices must map to valid song rows. "
            f"Invalid keys: {preview}"
        )


def load_artifacts(paths: ArtifactPaths) -> ModelArtifacts:
    for path in paths.all_paths():
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    songs = normalize_song_dataframe(joblib.load(paths.songs_path))
    feature_matrix = joblib.load(paths.feature_matrix_path)
    item_users = normalize_item_users(joblib.load(paths.item_users_path))

    validate_artifacts(songs, feature_matrix, item_users)
    return ModelArtifacts(songs=songs, feature_matrix=feature_matrix, item_users=item_users)
