"""Authentication service with lightweight JWT access tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apps.api.app.core.security import create_access_token, hash_password, verify_password
from apps.api.app.services.data_store import DemoDataStore

if TYPE_CHECKING:
    from apps.api.app.core.config import Settings


class AuthService:
    def __init__(self, store: DemoDataStore, settings: Settings):
        self.store = store
        self.settings = settings

    def signup(self, username: str, email: str, password: str) -> dict:
        if email in self.store.users_by_email:
            raise ValueError("An account with that email already exists.")

        user = self.store.add_user(
            username=username,
            email=email,
            password_hash=hash_password(password, self.settings.jwt_secret),
        )
        return self._auth_payload(user)

    def login(self, email: str, password: str) -> dict:
        user = self.store.users_by_email.get(email)
        if not user:
            raise ValueError("Invalid email or password.")

        if not verify_password(password, user["password_hash"], self.settings.jwt_secret):
            raise ValueError("Invalid email or password.")

        return self._auth_payload(user)

    def get_user(self, user_id: str) -> dict:
        return self.store.users_by_id[user_id]

    def _auth_payload(self, user: dict) -> dict:
        token = create_access_token(
            subject=user["id"],
            secret=self.settings.jwt_secret,
            expires_minutes=self.settings.access_token_expire_minutes,
            extra_claims={"username": user["username"], "email": user["email"]},
        )
        return {"access_token": token, "token_type": "bearer", "user": user}
