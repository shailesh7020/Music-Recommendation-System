from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from apps.api.app.api.deps import get_optional_user
from apps.api.app.core.config import get_catalog_service

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/home")
def home(user: dict | None = Depends(get_optional_user)) -> dict:
    return get_catalog_service().get_home(user_id=user["id"] if user else None)


@router.get("/search")
def search(
    q: str = Query(default=""),
    category: str | None = Query(default=None),
    limit: int = Query(default=8, ge=1, le=20),
) -> dict:
    return get_catalog_service().search(query=q, category=category, limit=limit)


@router.get("/songs/{song_id}")
def song_detail(song_id: str, user: dict | None = Depends(get_optional_user)) -> dict:
    return get_catalog_service().get_song_detail(song_id, user_id=user["id"] if user else None)


@router.get("/artists/{artist_id}")
def artist_detail(artist_id: str) -> dict:
    return get_catalog_service().get_artist_detail(artist_id)


@router.get("/albums/{album_id}")
def album_detail(album_id: str) -> dict:
    return get_catalog_service().get_album_detail(album_id)


@router.get("/playlists/{playlist_id}")
def playlist_detail(playlist_id: str) -> dict:
    return get_catalog_service().get_playlist_detail(playlist_id)


@router.get("/library/{user_id}")
def user_library(user_id: str) -> dict:
    return get_catalog_service().get_user_library(user_id)
