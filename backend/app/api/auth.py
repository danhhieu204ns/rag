from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from ..core.security import create_access_token, decode_access_token, verify_password
from ..db import get_db
from ..models import AdminUser
from ..schemas import AdminUserRead, LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])

_bearer_scheme = HTTPBearer(auto_error=False)


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> AdminUser:
    """FastAPI dependency — validates Bearer JWT and returns the active admin user."""

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không có quyền truy cập. Vui lòng đăng nhập.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    try:
        payload = decode_access_token(credentials.credentials)
        username: str | None = payload.get("sub")
        if not username:
            raise unauthorized
    except JWTError:
        raise unauthorized

    user = db.query(AdminUser).filter(
        AdminUser.username == username,
        AdminUser.is_active == True,  # noqa: E712
    ).first()

    if user is None:
        raise unauthorized

    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Authenticate admin and return a JWT access token."""

    user = db.query(AdminUser).filter(AdminUser.username == payload.username).first()

    if user is None or not user.is_active or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không đúng.",
        )

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, token_type="bearer")


@router.get("/me", response_model=AdminUserRead)
def get_me(current_admin: AdminUser = Depends(require_admin)) -> AdminUser:
    """Return info about the currently authenticated admin."""

    return current_admin
