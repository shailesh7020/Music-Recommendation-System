"""Playlist CRUD operations for the demo backend."""

from __future__ import annotations

from apps.api.app.services.data_store import DemoDataStore


class PlaylistService:
    def __init__(self, store: DemoDataStore):
        self.store = store

    def list_for_user(self, user_id: str) -> list[dict]:
        return list(self.store.user_playlists.get(user_id, []))

    def create_playlist(
        self,
        *,
        user_id: str,
        name: str,
        description: str,
        visibility: str = "public",
        songs: list[str] | None = None,
        cover_image: str = "",
    ) -> dict:
        return self.store.create_playlist(
            user_id=user_id,
            name=name,
            description=description,
            visibility=visibility,
            songs=songs,
            cover_image=cover_image,
        )

    def update_playlist(self, playlist_id: str, **updates) -> dict:
        return self.store.update_playlist(playlist_id, **updates)

    def add_song(self, playlist_id: str, song_id: str) -> dict:
        return self.store.add_song_to_playlist(playlist_id, song_id)

    def remove_song(self, playlist_id: str, song_id: str) -> dict:
        return self.store.remove_song_from_playlist(playlist_id, song_id)

    def reorder(self, playlist_id: str, song_ids: list[str]) -> dict:
        return self.store.reorder_playlist(playlist_id, song_ids)

    def duplicate(self, playlist_id: str, user_id: str) -> dict:
        return self.store.duplicate_playlist(playlist_id, user_id)

    def delete(self, playlist_id: str) -> None:
        self.store.delete_playlist(playlist_id)
