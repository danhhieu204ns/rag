from __future__ import annotations
import logging
from pathlib import Path
from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_BACKEND_ROOT / ".env", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .api.users import router as users_router
from .core.settings import settings
from .db import init_db



logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://10.20.2.60:5173",
        "http://10.20.2.60:3000",
        "*"
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10\.20\.2\.\d+)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database tables and seed default admin at app startup."""

    init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    """Basic health endpoint for backend service."""

    return {"status": "ok"}


app.include_router(auth_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(users_router, prefix="/api")
