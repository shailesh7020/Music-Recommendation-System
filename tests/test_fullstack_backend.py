from __future__ import annotations

import unittest

from apps.api.app.core.security import create_access_token, decode_access_token
from apps.api.app.data.sample_seed import clone_seed
from apps.api.app.services.data_store import DemoDataStore
from apps.api.app.services.recommendation_service import RecommendationService


class FullStackBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = DemoDataStore(clone_seed())
        self.recommendation_service = RecommendationService(
            self.store,
            default_region="Bengaluru",
        )

    def test_content_based_recommendations_exclude_seed_song(self) -> None:
        results = self.recommendation_service.content_based("song-neon-tide", limit=4)
        result_ids = [song["id"] for song in results]
        self.assertEqual(len(result_ids), 4)
        self.assertNotIn("song-neon-tide", result_ids)

    def test_hybrid_recommendations_include_reason(self) -> None:
        results = self.recommendation_service.hybrid(
            user_id="user-ava",
            seed_song_id="song-neon-tide",
            mood="Chill",
            genre="Synthwave",
            limit=5,
        )
        self.assertEqual(len(results), 5)
        self.assertTrue(all("reason" in song for song in results))

    def test_collaborative_recommendations_use_other_listener_profiles(self) -> None:
        results = self.recommendation_service.collaborative_for_user("user-ava", limit=5)
        result_ids = {song["id"] for song in results}
        self.assertIn("song-lowlight-code", result_ids)

    def test_jwt_round_trip_preserves_subject(self) -> None:
        token = create_access_token(
            subject="user-ava",
            secret="test-secret",
            expires_minutes=5,
            extra_claims={"username": "ava"},
        )
        payload = decode_access_token(token, "test-secret")
        self.assertEqual(payload["sub"], "user-ava")
        self.assertEqual(payload["username"], "ava")
