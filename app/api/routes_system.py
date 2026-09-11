"""GET /health and the demo page."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from fastapi.responses import HTMLResponse

from app.api.deps import get_container
from app.core.container import Container
from app.services.health import HealthReport, check_health

router = APIRouter(tags=["Tizim"])

_PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"


@router.get(
    "/health",
    response_model=HealthReport,
    summary="Ollama, modellar va indeks holati (200 — tayyor, 503 — muammo bor)",
    responses={503: {"model": HealthReport}},
)
async def health(
    response: Response, container: Annotated[Container, Depends(get_container)]
) -> HealthReport:
    report = await check_health(container)
    response.status_code = 200 if report.status == "ok" else 503
    return report


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def demo_page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))
