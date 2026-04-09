from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .core.settings import settings
from .db import init_db


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.20.0.195:5173/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database tables at app startup."""

    init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    """Basic health endpoint for backend service."""

    return {"status": "ok"}


app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
