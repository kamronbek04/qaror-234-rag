"""FastAPI dependencies: everything a route needs comes from the application container."""

from fastapi import Request

from app.core.container import Container
from app.core.errors import IndexNotReadyError


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise IndexNotReadyError("Servis hali ishga tushmoqda.")
    return container
