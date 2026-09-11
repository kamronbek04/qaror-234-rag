"""Domain exceptions with stable error codes and Uzbek user-facing messages."""


class AppError(Exception):
    code = "internal_error"
    status_code = 500
    default_message = "Ichki xatolik yuz berdi."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class ConfigError(AppError):
    code = "invalid_config"
    default_message = "Sozlamalarda xatolik bor."


class LLMUnavailableError(AppError):
    code = "llm_unavailable"
    status_code = 503
    default_message = "Til modeli (Ollama) mavjud emas yoki javob bermayapti."


class EmbeddingUnavailableError(AppError):
    code = "embedding_unavailable"
    status_code = 503
    default_message = "Embedding modeli (Ollama) mavjud emas yoki javob bermayapti."


class IndexNotReadyError(AppError):
    code = "index_not_ready"
    status_code = 503
    default_message = "Qidiruv indeksi hali tayyor emas yoki eskirgan."


class ChunkNotFoundError(AppError):
    code = "chunk_not_found"
    status_code = 404

    def __init__(self, chunk_id: str) -> None:
        super().__init__(f"'{chunk_id}' identifikatorli bo'lak topilmadi.")


class IngestionError(AppError):
    code = "ingestion_failed"
    default_message = "Hujjatni indekslashda xatolik yuz berdi."


class SourceFetchError(IngestionError):
    code = "source_fetch_failed"
    default_message = "Hujjatni lex.uz'dan yuklab bo'lmadi."


class ParseIntegrityError(IngestionError):
    code = "parse_integrity_failed"
    default_message = "Hujjat tuzilmasi kutilganidek emas."
