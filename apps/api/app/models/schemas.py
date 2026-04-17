"""Pydantic schemas used by the FastAPI routes."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class UserPublic(BaseModel):
    id: str
    username: str
    email: EmailStr
    profile_image: str = ""
    bio: str = ""


class AuthSignupRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class AuthLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class PlaylistMutationRequest(BaseModel):
    name: str
    description: str = ""
    visibility: str = "public"
    cover_image: str = ""
    songs: list[str] = []


class PlaylistReorderRequest(BaseModel):
    song_ids: list[str]
