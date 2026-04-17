"""In-memory data store and indexes for the demo backend."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from uuid import uuid4


class DemoDataStore:
    """Mutable in-memory store that mimics a small application database."""

    def __init__(self, seed: dict[str, list | dict]):
        self.users = deepcopy(seed["users"])
        self.artists = deepcopy(seed["artists"])
        self.albums = deepcopy(seed["albums"])
        self.songs = deepcopy(seed["songs"])
        self.playlists = deepcopy(seed["playlists"])
        self.listening_history = deepcopy(seed["listening_history"])
        self.song_comments = deepcopy(seed["song_comments"])
        self.trending_areas = deepcopy(seed["trending_areas"])
        self.mood_taglines = deepcopy(seed["mood_taglines"])
        self.refresh_indexes()

    def refresh_indexes(self) -> None:
        self.users_by_id = {user["id"]: user for user in self.users}
        self.users_by_email = {user["email"]: user for user in self.users}
        self.artists_by_id = {artist["id"]: artist for artist in self.artists}
        self.albums_by_id = {album["id"]: album for album in self.albums}
        self.songs_by_id = {song["id"]: song for song in self.songs}
        self.playlists_by_id = {playlist["id"]: playlist for playlist in self.playlists}

        self.artist_songs = defaultdict(list)
        self.album_songs = defaultdict(list)
        for song in self.songs:
            self.artist_songs[song["artist_id"]].append(song)
            self.album_songs[song["album_id"]].append(song)

        self.user_playlists = defaultdict(list)
        for playlist in self.playlists:
            self.user_playlists[playlist["user_id"]].append(playlist)

    def list_recent_song_ids(self, user_id: str) -> list[str]:
        user = self.users_by_id.get(user_id)
        if not user:
            return []
        return list(user.get("recently_played", []))

    def list_liked_song_ids(self, user_id: str) -> list[str]:
        user = self.users_by_id.get(user_id)
        if not user:
            return []
        return list(user.get("liked_songs", []))

    def add_user(self, username: str, email: str, password_hash: str) -> dict:
        user = {
            "id": f"user-{uuid4().hex[:10]}",
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "profile_image": "",
            "bio": "",
            "liked_songs": [],
            "recently_played": [],
        }
        self.users.append(user)
        self.refresh_indexes()
        return user

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
        playlist = {
            "id": f"playlist-{uuid4().hex[:10]}",
            "user_id": user_id,
            "name": name,
            "description": description,
            "cover_image": cover_image,
            "visibility": visibility,
            "songs": list(songs or []),
        }
        self.playlists.append(playlist)
        self.refresh_indexes()
        return playlist

    def update_playlist(self, playlist_id: str, **updates) -> dict:
        playlist = self.playlists_by_id[playlist_id]
        playlist.update({key: value for key, value in updates.items() if value is not None})
        self.refresh_indexes()
        return playlist

    def delete_playlist(self, playlist_id: str) -> None:
        self.playlists = [playlist for playlist in self.playlists if playlist["id"] != playlist_id]
        self.refresh_indexes()

    def duplicate_playlist(self, playlist_id: str, user_id: str) -> dict:
        source = deepcopy(self.playlists_by_id[playlist_id])
        return self.create_playlist(
            user_id=user_id,
            name=f"{source['name']} Copy",
            description=source["description"],
            visibility=source["visibility"],
            songs=source["songs"],
            cover_image=source["cover_image"],
        )

    def add_song_to_playlist(self, playlist_id: str, song_id: str) -> dict:
        playlist = self.playlists_by_id[playlist_id]
        if song_id not in playlist["songs"]:
            playlist["songs"].append(song_id)
        self.refresh_indexes()
        return playlist

    def remove_song_from_playlist(self, playlist_id: str, song_id: str) -> dict:
        playlist = self.playlists_by_id[playlist_id]
        playlist["songs"] = [track_id for track_id in playlist["songs"] if track_id != song_id]
        self.refresh_indexes()
        return playlist

    def reorder_playlist(self, playlist_id: str, song_ids: list[str]) -> dict:
        playlist = self.playlists_by_id[playlist_id]
        playlist["songs"] = [song_id for song_id in song_ids if song_id in self.songs_by_id]
        self.refresh_indexes()
        return playlist
