from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache

from fastapi import HTTPException, Request, Response, WebSocket, status


@dataclass(frozen=True)
class AuthSettings:
    login: str
    password: str
    cookie_name: str
    cookie_secure: bool


@lru_cache(maxsize=1)
def get_settings() -> AuthSettings:
    login = os.getenv("ADMIN_LOGIN", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin123")
    cookie_name = os.getenv("SESSION_COOKIE_NAME", "support_session")
    cookie_secure = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    return AuthSettings(login=login, password=password, cookie_name=cookie_name, cookie_secure=cookie_secure)


def _expected_token(settings: AuthSettings | None = None) -> str:
    settings = settings or get_settings()
    payload = f"{settings.login}:{settings.password}".encode()
    return hashlib.sha256(payload).hexdigest()


def validate_credentials(username: str, password: str) -> bool:
    settings = get_settings()
    return username == settings.login and password == settings.password


def issue_session_cookie(response: Response) -> None:
    settings = get_settings()
    response.set_cookie(
        settings.cookie_name,
        _expected_token(settings),
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=7 * 24 * 60 * 60,
    )


def clear_session_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(settings.cookie_name)


def is_authenticated_request(request: Request) -> bool:
    settings = get_settings()
    expected = _expected_token(settings)
    return request.cookies.get(settings.cookie_name) == expected


def ensure_api_auth(request: Request) -> None:
    if not is_authenticated_request(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def is_authenticated_websocket(websocket: WebSocket) -> bool:
    settings = get_settings()
    expected = _expected_token(settings)
    return websocket.cookies.get(settings.cookie_name) == expected


WEBSOCKET_UNAUTHORIZED_CLOSE_CODE = 4401

