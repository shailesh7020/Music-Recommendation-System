from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.app.api.deps import get_current_user
from apps.api.app.core.config import get_playlist_service
from apps.api.app.models.schemas import PlaylistMutationRequest, PlaylistReorderRequest

router = APIRouter(prefix="/playlists", tags=["playlists"])


@router.post("")
def create_playlist(payload: PlaylistMutationRequest, user: dict = Depends(get_current_user)) -> dict:
    return get_playlist_service().create_playlist(
        user_id=user["id"],
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility,
        songs=payload.songs,
        cover_image=payload.cover_image,
    )


@router.patch("/{playlist_id}")
def update_playlist(
    playlist_id: str,
    payload: PlaylistMutationRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    return get_playlist_service().update_playlist(
        playlist_id,
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility,
        cover_image=payload.cover_image,
        user_id=user["id"],
    )


@router.delete("/{playlist_id}")
def delete_playlist(playlist_id: str, user: dict = Depends(get_current_user)) -> dict:
    get_playlist_service().delete(playlist_id)
    return {"status": "deleted", "playlist_id": playlist_id, "user_id": user["id"]}


@router.post("/{playlist_id}/songs/{song_id}")
def add_song(playlist_id: str, song_id: str, user: dict = Depends(get_current_user)) -> dict:
    return get_playlist_service().add_song(playlist_id, song_id)


@router.delete("/{playlist_id}/songs/{song_id}")
def remove_song(playlist_id: str, song_id: str, user: dict = Depends(get_current_user)) -> dict:
    return get_playlist_service().remove_song(playlist_id, song_id)


@router.post("/{playlist_id}/reorder")
def reorder_playlist(
    playlist_id: str,
    payload: PlaylistReorderRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    return get_playlist_service().reorder(playlist_id, payload.song_ids)


@router.post("/{playlist_id}/duplicate")
def duplicate_playlist(playlist_id: str, user: dict = Depends(get_current_user)) -> dict:
    return get_playlist_service().duplicate(playlist_id, user["id"])


@router.get("/user/{user_id}")
def user_playlists(user_id: str) -> list[dict]:
    return get_playlist_service().list_for_user(user_id)
