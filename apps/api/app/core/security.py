"""Small standard-library JWT and password helpers for the demo backend."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from typing import Any


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("utf-8")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(f"{raw}{padding}")


def hash_password(password: str, secret: str) -> str:
    digest = hashlib.sha256(f"{secret}:{password}".encode("utf-8")).hexdigest()
    return f"sha256${digest}"


def verify_password(password: str, stored_hash: str, secret: str) -> bool:
    if stored_hash.startswith("demo:"):
        return stored_hash.split(":", maxsplit=1)[1] == password

    if not stored_hash.startswith("sha256$"):
        return False

    return hmac.compare_digest(stored_hash, hash_password(password, secret))


def create_access_token(
    subject: str,
    secret: str,
    expires_minutes: int,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "iat": int(datetime.now(tz=UTC).timestamp()),
        "exp": int((datetime.now(tz=UTC) + timedelta(minutes=expires_minutes)).timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)

    encoded_header = _b64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    encoded_payload = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    encoded_signature = _b64url_encode(signature)
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
    except ValueError as exc:
        raise ValueError("Invalid token structure.") from exc

    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    expected_signature = hmac.new(
        secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()

    if not hmac.compare_digest(_b64url_encode(expected_signature), encoded_signature):
        raise ValueError("Invalid token signature.")

    payload = json.loads(_b64url_decode(encoded_payload).decode("utf-8"))
    if int(payload["exp"]) < int(datetime.now(tz=UTC).timestamp()):
        raise ValueError("Token has expired.")
    return payload
