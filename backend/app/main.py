from __future__ import annotations
import logging
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .api.users import router as users_router
from .core.settings import settings
from .db import init_db
from .services.chunk_metadata import warmup_metadata_model
from .services.rag_runtime import warmup_chat_model, warmup_embedding_model


logger = logging.getLogger(__name__)


def _warmup_orchestrated_models() -> None:
    if not settings.model_warmup_on_startup:
        return

    if settings.model_warmup_embedding:
        try:
            warmup_embedding_model()
            logger.info("Embedding model warmup completed.")
        except Exception as exc:
            logger.warning("Embedding model warmup failed: %s", exc)

    if settings.model_warmup_metadata:
        try:
            warmup_metadata_model()
            logger.info("Metadata model warmup completed.")
        except Exception as exc:
            logger.warning("Metadata model warmup failed: %s", exc)

    if settings.model_warmup_chat:
        try:
            warmup_chat_model()
            logger.info("Chat model warmup completed.")
        except Exception as exc:
            logger.warning("Chat model warmup failed: %s", exc)


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
    _warmup_orchestrated_models()


@app.get("/api/health")
def health() -> dict[str, str]:
    """Basic health endpoint for backend service."""

    return {"status": "ok"}


app.include_router(auth_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(users_router, prefix="/api")
