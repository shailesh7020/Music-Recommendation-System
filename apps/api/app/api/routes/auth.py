from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.app.api.deps import get_current_user
from apps.api.app.core.config import get_auth_service
from apps.api.app.models.schemas import AuthLoginRequest, AuthSignupRequest, TokenResponse, UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse)
def signup(payload: AuthSignupRequest) -> TokenResponse:
    try:
        result = get_auth_service().signup(payload.username, payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return TokenResponse(
        access_token=result["access_token"],
        token_type=result["token_type"],
        user=UserPublic.model_validate(result["user"]),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: AuthLoginRequest) -> TokenResponse:
    try:
        result = get_auth_service().login(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    return TokenResponse(
        access_token=result["access_token"],
        token_type=result["token_type"],
        user=UserPublic.model_validate(result["user"]),
    )


@router.get("/me", response_model=UserPublic)
def me(user: dict = Depends(get_current_user)) -> UserPublic:
    return UserPublic.model_validate(user)
