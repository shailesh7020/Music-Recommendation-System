"""Hybrid recommendation logic for the full-stack demo backend."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

from apps.api.app.services.data_store import DemoDataStore

FEATURE_KEYS = ("danceability", "energy", "valence", "acousticness", "tempo")


class RecommendationService:
    def __init__(self, store: DemoDataStore, default_region: str):
        self.store = store
        self.default_region = default_region

    @staticmethod
    def _normalized_feature_vector(song: dict) -> list[float]:
        return [
            float(song["danceability"]),
            float(song["energy"]),
            float(song["valence"]),
            float(song["acousticness"]),
            float(song["tempo"]) / 160.0,
        ]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _content_score(self, base_song: dict, candidate_song: dict) -> float:
        score = self._cosine_similarity(
            self._normalized_feature_vector(base_song),
            self._normalized_feature_vector(candidate_song),
        )
        if base_song["genre"] == candidate_song["genre"]:
            score += 0.18
        if base_song["artist_id"] == candidate_song["artist_id"]:
            score += 0.12
        if base_song["mood"] == candidate_song["mood"]:
            score += 0.1
        return score

    def _user_profile(self, user_id: str) -> set[str]:
        liked = set(self.store.list_liked_song_ids(user_id))
        recent = set(self.store.list_recent_song_ids(user_id))
        return liked | recent

    def content_based(self, seed_song_id: str, limit: int = 8, exclude: set[str] | None = None) -> list[dict]:
        exclude_ids = set(exclude or set())
        exclude_ids.add(seed_song_id)
        base_song = self.store.songs_by_id[seed_song_id]
        scored: list[tuple[float, dict]] = []
        for song in self.store.songs:
            if song["id"] in exclude_ids:
                continue
            scored.append((self._content_score(base_song, song), song))
        scored.sort(key=lambda item: (item[0], item[1]["popularity"]), reverse=True)
        return [song for _, song in scored[:limit]]

    def collaborative_for_user(
        self, user_id: str, limit: int = 8, exclude: set[str] | None = None
    ) -> list[dict]:
        exclude_ids = set(exclude or set())
        target_profile = self._user_profile(user_id)
        scored = defaultdict(float)
        if not target_profile:
            return []

        for other_user in self.store.users:
            if other_user["id"] == user_id:
                continue
            other_profile = self._user_profile(other_user["id"])
            intersection = len(target_profile & other_profile)
            union = len(target_profile | other_profile)
            similarity = intersection / union if union else 0.0
            if similarity <= 0:
                continue
            for song_id in other_profile - target_profile:
                if song_id in exclude_ids:
                    continue
                scored[song_id] += similarity

        ranked_ids = sorted(
            scored,
            key=lambda song_id: (scored[song_id], self.store.songs_by_id[song_id]["popularity"]),
            reverse=True,
        )
        return [self.store.songs_by_id[song_id] for song_id in ranked_ids[:limit]]

    def mood_based(self, mood: str, limit: int = 8, exclude: set[str] | None = None) -> list[dict]:
        exclude_ids = set(exclude or set())
        mood_lower = mood.lower()
        filtered = [
            song
            for song in self.store.songs
            if song["id"] not in exclude_ids and song["mood"].lower() == mood_lower
        ]
        filtered.sort(key=lambda song: song["popularity"], reverse=True)
        return filtered[:limit]

    def genre_based(self, genre: str, limit: int = 8, exclude: set[str] | None = None) -> list[dict]:
        exclude_ids = set(exclude or set())
        genre_lower = genre.lower()
        filtered = [
            song
            for song in self.store.songs
            if song["id"] not in exclude_ids and genre_lower in song["genre"].lower()
        ]
        filtered.sort(key=lambda song: song["popularity"], reverse=True)
        return filtered[:limit]

    def hybrid(
        self,
        *,
        user_id: str | None = None,
        seed_song_id: str | None = None,
        mood: str | None = None,
        genre: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        scored = defaultdict(float)
        reasons = defaultdict(list)
        exclude_ids = set()

        if user_id:
            exclude_ids |= self._user_profile(user_id)
            for rank, song in enumerate(self.collaborative_for_user(user_id, limit=limit * 2), start=1):
                scored[song["id"]] += 0.6 / rank
                reasons[song["id"]].append("Similar listeners saved this.")

        if seed_song_id:
            exclude_ids.add(seed_song_id)
            for rank, song in enumerate(
                self.content_based(seed_song_id, limit=limit * 2, exclude=exclude_ids), start=1
            ):
                scored[song["id"]] += 0.8 / rank
                reasons[song["id"]].append(
                    f"Matches the vibe of {self.store.songs_by_id[seed_song_id]['title']}."
                )

        if mood:
            for rank, song in enumerate(self.mood_based(mood, limit=limit * 2, exclude=exclude_ids), start=1):
                scored[song["id"]] += 0.5 / rank
                reasons[song["id"]].append(f"Fits the {mood.lower()} lane.")

        if genre:
            for rank, song in enumerate(
                self.genre_based(genre, limit=limit * 2, exclude=exclude_ids), start=1
            ):
                scored[song["id"]] += 0.4 / rank
                reasons[song["id"]].append(f"Shares the {genre} palette.")

        if not scored:
            for song in self.store.songs:
                scored[song["id"]] = song["popularity"] / 100.0
                reasons[song["id"]].append("Trending now on Pulsewave.")

        ranked_ids = sorted(
            scored,
            key=lambda song_id: (scored[song_id], self.store.songs_by_id[song_id]["popularity"]),
            reverse=True,
        )[:limit]

        return [
            {
                **self.store.songs_by_id[song_id],
                "score": round(scored[song_id], 4),
                "reason": reasons[song_id][0],
            }
            for song_id in ranked_ids
        ]

    def because_you_listened(self, seed_song_id: str, user_id: str | None = None, limit: int = 8) -> list[dict]:
        base_song = self.store.songs_by_id[seed_song_id]
        return self.hybrid(
            user_id=user_id,
            seed_song_id=seed_song_id,
            mood=base_song["mood"],
            genre=base_song["genre"],
            limit=limit,
        )

    def similar_artists(self, artist_id: str, limit: int = 4) -> list[dict]:
        base_artist = self.store.artists_by_id[artist_id]
        base_genres = set(genre.lower() for genre in base_artist.get("genres", []))
        scored = []
        for artist in self.store.artists:
            if artist["id"] == artist_id:
                continue
            overlap = len(base_genres & set(genre.lower() for genre in artist.get("genres", [])))
            score = overlap + artist["monthly_listeners"] / 3_000_000
            scored.append((score, artist))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [artist for _, artist in scored[:limit]]

    def trending_in_area(self, region: str | None = None, limit: int = 6) -> list[dict]:
        area = region or self.default_region
        song_ids = self.store.trending_areas.get(area, [])
        if not song_ids:
            return sorted(self.store.songs, key=lambda song: song["popularity"], reverse=True)[:limit]
        return [self.store.songs_by_id[song_id] for song_id in song_ids[:limit]]
