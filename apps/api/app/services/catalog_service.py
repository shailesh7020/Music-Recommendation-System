"""Catalog aggregation service powering the app pages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apps.api.app.services.data_store import DemoDataStore
from apps.api.app.services.recommendation_service import RecommendationService

if TYPE_CHECKING:
    from apps.api.app.core.config import Settings


class CatalogService:
    def __init__(
        self,
        store: DemoDataStore,
        recommendation_service: RecommendationService,
        settings: Settings,
    ):
        self.store = store
        self.recommendation_service = recommendation_service
        self.settings = settings

    def get_home(self, user_id: str | None = None) -> dict:
        target_user_id = user_id or self.settings.default_user_id
        recent_ids = self.store.list_recent_song_ids(target_user_id)
        liked_ids = self.store.list_liked_song_ids(target_user_id)
        mood_seed = self.store.songs_by_id[recent_ids[0]]["mood"] if recent_ids else "Focus"

        recent_songs = [self.store.songs_by_id[song_id] for song_id in recent_ids[:6]]
        made_for_you = self.recommendation_service.hybrid(
            user_id=target_user_id,
            seed_song_id=recent_ids[0] if recent_ids else None,
            mood=mood_seed,
            limit=6,
        )
        trending_now = sorted(self.store.songs, key=lambda song: song["popularity"], reverse=True)[:6]
        top_artists = sorted(
            self.store.artists, key=lambda artist: artist["monthly_listeners"], reverse=True
        )[:5]
        new_releases = sorted(self.store.albums, key=lambda album: album["release_date"], reverse=True)[:5]
        based_on_your_mood = self.recommendation_service.mood_based(mood_seed, limit=6)
        recommended_playlists = self.store.playlists[:4]

        return {
            "brand": "Pulsewave",
            "recently_played": recent_songs,
            "made_for_you": made_for_you,
            "trending_now": trending_now,
            "top_artists": top_artists,
            "new_releases": new_releases,
            "based_on_your_mood": based_on_your_mood,
            "recommended_playlists": recommended_playlists,
            "liked_song_ids": liked_ids,
        }

    def search(self, query: str, category: str | None = None, limit: int = 8) -> dict:
        normalized = query.strip().lower()
        if not normalized:
            return {
                "songs": [],
                "artists": [],
                "albums": [],
                "genres": sorted({song["genre"] for song in self.store.songs})[:8],
                "history": ["Neon Tide", "Focus After Dark", "Workout"],
            }

        song_matches = [
            song
            for song in self.store.songs
            if normalized in song["title"].lower()
            or normalized in song["artist"].lower()
            or normalized in song["genre"].lower()
            or normalized in song["mood"].lower()
        ]
        artist_matches = [
            artist
            for artist in self.store.artists
            if normalized in artist["name"].lower()
            or any(normalized in genre.lower() for genre in artist.get("genres", []))
        ]
        album_matches = [
            album
            for album in self.store.albums
            if normalized in album["title"].lower() or normalized in album["genre"].lower()
        ]
        genres = sorted(
            {
                song["genre"]
                for song in self.store.songs
                if normalized in song["genre"].lower() or normalized in song["mood"].lower()
            }
        )

        if category == "songs":
            artist_matches = []
            album_matches = []
        elif category == "artists":
            song_matches = []
            album_matches = []
        elif category == "albums":
            song_matches = []
            artist_matches = []

        return {
            "songs": song_matches[:limit],
            "artists": artist_matches[:limit],
            "albums": album_matches[:limit],
            "genres": genres[:limit],
            "history": ["Neon Tide", "Luna Harbor", "Workout"],
        }

    def get_song_detail(self, song_id: str, user_id: str | None = None) -> dict:
        song = self.store.songs_by_id[song_id]
        artist = self.store.artists_by_id[song["artist_id"]]
        album = self.store.albums_by_id[song["album_id"]]
        comments = self.store.song_comments.get(song_id, [])
        similar = self.recommendation_service.because_you_listened(song_id, user_id=user_id, limit=6)
        return {
            "song": song,
            "artist": artist,
            "album": album,
            "lyrics": [
                "Streetlights bloom across the glass tonight",
                "We move in color, coded in neon light",
                "Every skyline pulse becomes a melody",
            ],
            "comments": comments,
            "similar_songs": similar,
            "recommended_songs": self.recommendation_service.hybrid(
                user_id=user_id or self.settings.default_user_id,
                seed_song_id=song_id,
                mood=song["mood"],
                genre=song["genre"],
                limit=6,
            ),
        }

    def get_artist_detail(self, artist_id: str) -> dict:
        artist = self.store.artists_by_id[artist_id]
        songs = sorted(
            self.store.artist_songs[artist_id],
            key=lambda song: song["popularity"],
            reverse=True,
        )
        albums = [album for album in self.store.albums if album["artist_id"] == artist_id]
        return {
            "artist": artist,
            "popular_songs": songs[:6],
            "albums": albums,
            "similar_artists": self.recommendation_service.similar_artists(artist_id),
        }

    def get_album_detail(self, album_id: str) -> dict:
        album = self.store.albums_by_id[album_id]
        artist = self.store.artists_by_id[album["artist_id"]]
        songs = self.store.album_songs[album_id]
        total_duration = sum(song["duration_ms"] for song in songs)
        return {
            "album": album,
            "artist": artist,
            "songs": songs,
            "total_duration_ms": total_duration,
        }

    def get_playlist_detail(self, playlist_id: str) -> dict:
        playlist = self.store.playlists_by_id[playlist_id]
        songs = [self.store.songs_by_id[song_id] for song_id in playlist["songs"]]
        total_duration = sum(song["duration_ms"] for song in songs)
        return {
            "playlist": playlist,
            "songs": songs,
            "total_duration_ms": total_duration,
        }

    def get_recommendation_page(self, user_id: str | None = None) -> dict:
        target_user_id = user_id or self.settings.default_user_id
        recent_song_id = self.store.list_recent_song_ids(target_user_id)[0]
        recent_song = self.store.songs_by_id[recent_song_id]
        return {
            "because_you_listened_to": {
                "song": recent_song,
                "results": self.recommendation_service.because_you_listened(
                    recent_song_id, user_id=target_user_id, limit=8
                ),
            },
            "similar_users_like": self.recommendation_service.collaborative_for_user(target_user_id, limit=8),
            "mood_based": {
                mood: self.recommendation_service.mood_based(mood, limit=5)
                for mood in ["Happy", "Sad", "Chill", "Workout", "Party", "Romantic", "Focus"]
            },
            "genre_based": {
                genre: self.recommendation_service.genre_based(genre, limit=5)
                for genre in ["Synthwave", "Dance-pop", "Dream-pop", "Lo-fi", "House"]
            },
            "trending_in_area": self.recommendation_service.trending_in_area(),
            "mood_taglines": self.store.mood_taglines,
        }

    def get_user_library(self, user_id: str) -> dict:
        liked_songs = [self.store.songs_by_id[song_id] for song_id in self.store.list_liked_song_ids(user_id)]
        playlists = self.store.user_playlists.get(user_id, [])
        return {"liked_songs": liked_songs, "playlists": playlists}
