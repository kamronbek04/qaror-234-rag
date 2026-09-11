# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Dependencies first: this layer is rebuilt only when pyproject.toml changes.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY app ./app
COPY eval ./eval
COPY data/raw ./data/raw

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/index \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=900s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
