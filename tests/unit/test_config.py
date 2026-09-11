import pytest

from app.core.config import Settings, load_settings
from app.core.errors import ConfigError


def test_defaults_match_documented_values():
    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_model == "qwen2.5:7b"
    assert settings.embed_model == "bge-m3"
    assert settings.llm_num_ctx == 8192
    assert settings.llm_temperature == 0.0
    assert settings.retrieval_top_k == 6
    assert settings.chunk_strategy == "structural"
    assert settings.llm_think is None


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("REFUSAL_THRESHOLD", "0.5")
    monkeypatch.setenv("LLM_THINK", "false")

    settings = load_settings(env_file=None)

    assert settings.llm_model == "qwen3.5:4b"
    assert settings.refusal_threshold == 0.5
    assert settings.llm_think is False


def test_empty_think_value_means_not_sent(monkeypatch):
    monkeypatch.setenv("LLM_THINK", "")

    assert load_settings(env_file=None).llm_think is None


def test_invalid_threshold_names_the_variable(monkeypatch):
    monkeypatch.setenv("REFUSAL_THRESHOLD", "abc")

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    assert "REFUSAL_THRESHOLD" in str(excinfo.value)


def test_threshold_must_be_between_zero_and_one(monkeypatch):
    monkeypatch.setenv("REFUSAL_THRESHOLD", "1.5")

    with pytest.raises(ConfigError, match="REFUSAL_THRESHOLD"):
        load_settings(env_file=None)


def test_unknown_chunk_strategy_is_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_STRATEGY", "semantic")

    with pytest.raises(ConfigError, match="CHUNK_STRATEGY"):
        load_settings(env_file=None)
