"""Configuration and singleton service access for the FastAPI app."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from apps.api.app.data.sample_seed import BRAND_NAME, clone_seed
from apps.api.app.services.auth_service import AuthService
from apps.api.app.services.catalog_service import CatalogService
from apps.api.app.services.data_store import DemoDataStore
from apps.api.app.services.playlist_service import PlaylistService
from apps.api.app.services.recommendation_service import RecommendationService


@dataclass(frozen=True)
class Settings:
    app_name: str = f"{BRAND_NAME} API"
    api_prefix: str = "/api"
    jwt_secret: str = "pulsewave-demo-secret"
    access_token_expire_minutes: int = 60 * 24
    default_user_id: str = "user-ava"
    default_region: str = "Bengaluru"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_data_store() -> DemoDataStore:
    return DemoDataStore(clone_seed())


@lru_cache
def get_recommendation_service() -> RecommendationService:
    return RecommendationService(get_data_store(), get_settings().default_region)


@lru_cache
def get_catalog_service() -> CatalogService:
    return CatalogService(get_data_store(), get_recommendation_service(), get_settings())


@lru_cache
def get_auth_service() -> AuthService:
    return AuthService(get_data_store(), get_settings())


@lru_cache
def get_playlist_service() -> PlaylistService:
    return PlaylistService(get_data_store())
