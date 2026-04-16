"""Reusable application package for the music recommender."""

from .artifacts import ArtifactPaths, ModelArtifacts, load_artifacts, resolve_paths
from .service import MusicRecommenderService

__all__ = [
    "ArtifactPaths",
    "ModelArtifacts",
    "MusicRecommenderService",
    "load_artifacts",
    "resolve_paths",
]
