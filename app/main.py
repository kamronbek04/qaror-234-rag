"""FastAPI application factory. Run with ``uvicorn app.main:app``."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.api import routes_ask, routes_search, routes_system
from app.core.config import Settings, load_settings
from app.core.container import Container, build_container
from app.core.errors import AppError
from app.core.logging import configure_logging, request_id_var

logger = logging.getLogger(__name__)

ContainerFactory = Callable[[Settings], Awaitable[Container]]

DESCRIPTION = """\
Vazirlar Mahkamasining 2026-yil 11-maydagi **234-son qarori** matni asosida savollarga
**faqat hujjatdan** javob beradigan, to'liq lokal (Ollama) ishlaydigan RAG API.

Hujjatda javob bo'lmasa, javob aynan shunday bo'ladi: `Hujjatda bu haqida ma'lumot yo'q`.
"""


def create_app(
    settings: Settings | None = None,
    *,
    container: Container | None = None,
    container_factory: ContainerFactory = build_container,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or load_settings()
        configure_logging(resolved.log_level)
        if getattr(app.state, "container", None) is None:
            app.state.container = await container_factory(resolved)
        logger.info("servis tayyor", extra={"index_ready": app.state.container.index.ready})
        try:
            yield
        finally:
            app.state.container.close()

    app = FastAPI(
        title="234-son qaror RAG API",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    if container is not None:
        app.state.container = container

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        value = request.headers.get("X-Request-ID") or uuid4().hex[:16]
        token = request_id_var.set(value)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = value
        return response

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.warning("so'rov bajarilmadi", extra={"code": exc.code, "error": exc.message})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    app.include_router(routes_ask.router)
    app.include_router(routes_search.router)
    app.include_router(routes_system.router)
    return app


app = create_app()
