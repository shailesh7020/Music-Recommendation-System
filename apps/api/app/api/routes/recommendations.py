from __future__ import annotations

from fastapi import APIRouter, Query

from apps.api.app.core.config import get_catalog_service, get_recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("")
def recommendation_page(user_id: str | None = Query(default=None)) -> dict:
    return get_catalog_service().get_recommendation_page(user_id=user_id)


@router.get("/songs/{song_id}/similar")
def similar_songs(song_id: str, user_id: str | None = Query(default=None), limit: int = Query(default=8, ge=1, le=20)) -> list[dict]:
    return get_recommendation_service().because_you_listened(song_id, user_id=user_id, limit=limit)


@router.get("/moods/{mood}")
def by_mood(mood: str, limit: int = Query(default=8, ge=1, le=20)) -> list[dict]:
    return get_recommendation_service().mood_based(mood, limit=limit)


@router.get("/genres/{genre}")
def by_genre(genre: str, limit: int = Query(default=8, ge=1, le=20)) -> list[dict]:
    return get_recommendation_service().genre_based(genre, limit=limit)


@router.get("/areas/{region}")
def trending_in_area(region: str, limit: int = Query(default=6, ge=1, le=20)) -> list[dict]:
    return get_recommendation_service().trending_in_area(region, limit=limit)
