"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from apps.api.app.core.config import get_auth_service, get_settings
from apps.api.app.core.security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict | None:
    if credentials is None:
        return None

    try:
        payload = decode_access_token(credentials.credentials, get_settings().jwt_secret)
        return get_auth_service().get_user(payload["sub"])
    except Exception as exc:  # pragma: no cover - thin HTTP wrapper
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def get_current_user(
    user: dict | None = Depends(get_optional_user),
) -> dict:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return user
