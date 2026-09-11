import json
import logging

from app.core.errors import (
    ChunkNotFoundError,
    EmbeddingUnavailableError,
    IndexNotReadyError,
    LLMUnavailableError,
)
from app.core.logging import JsonFormatter, request_id_var


def test_backend_errors_map_to_service_unavailable():
    for error_type, code in [
        (LLMUnavailableError, "llm_unavailable"),
        (EmbeddingUnavailableError, "embedding_unavailable"),
        (IndexNotReadyError, "index_not_ready"),
    ]:
        error = error_type()
        assert error.status_code == 503
        assert error.code == code
        assert error.message


def test_missing_chunk_maps_to_not_found():
    error = ChunkNotFoundError("a2-b99")

    assert error.status_code == 404
    assert error.code == "chunk_not_found"
    assert "a2-b99" in error.message


def test_json_log_record_carries_request_id_and_extra_fields():
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "answered", None, None)
    record.timings_ms = {"total": 12}
    token = request_id_var.set("demo-1")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert payload["request_id"] == "demo-1"
    assert payload["message"] == "answered"
    assert payload["level"] == "INFO"
    assert payload["timings_ms"] == {"total": 12}
