"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.logging_config import configure_logging, get_logger
from backend.app.db import init_db
from backend.app.core.errors import AppError

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Quantum-Centric Federated Learning API...")
    init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quantum-Centric Federated Learning API",
        description=(
            "Experimental research prototype for screen-time behavioural risk assessment. "
            "NOT a validated clinical tool. All predictions are experimental."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    from src.config.settings import get_settings
    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            settings.frontend_url,
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "error_type": type(exc).__name__},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )

    # ── Routers ───────────────────────────────────────────────────────────
    from backend.app.api.v1.health import router as health_router
    from backend.app.api.v1.videos import router as videos_router
    from backend.app.api.v1.predict import router as predict_router
    from backend.app.api.v1.experiments import router as experiments_router
    from backend.app.api.v1.datasets import router as datasets_router
    from backend.app.api.v1.models_registry import router as models_router
    from backend.app.api.v1.models_registry import federated_router

    app.include_router(health_router)
    app.include_router(videos_router)
    app.include_router(predict_router)
    app.include_router(experiments_router)
    app.include_router(datasets_router)
    app.include_router(models_router)
    app.include_router(federated_router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Quantum-Centric Federated Learning API",
            "version": "1.0.0",
            "docs": "/api/docs",
            "health": "/api/v1/health",
            "ready": "/api/v1/ready",
            "disclaimer": (
                "Experimental research prototype. "
                "NOT a validated clinical or diagnostic tool."
            ),
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    from src.config.settings import get_settings
    s = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=s.api_host,
        port=s.api_port,
        reload=s.is_development,
        log_level=s.log_level.lower(),
    )
