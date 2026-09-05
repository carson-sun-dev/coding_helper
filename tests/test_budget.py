import pytest
from pydantic import ValidationError

from coding_helper.config import Settings
from coding_helper.context.budget import (
    compact_target_tokens,
    compact_trigger_tokens,
    context_window_tokens,
    estimate_tokens,
    should_compact,
)
from coding_helper.models import ModelTarget


def make_settings(**overrides) -> Settings:
    values = {
        "ark_api_key": "test-secret",
        "ark_deepseek_model": "deepseek-endpoint",
        "ark_glm_model": "glm-endpoint",
        "ark_auxiliary_model": "doubao-endpoint",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_windows_are_per_primary_model_and_not_hardcoded_256k() -> None:
    settings = make_settings(
        ark_deepseek_context_window=64_000,
        ark_glm_context_window=256_000,
    )

    assert context_window_tokens(settings, ModelTarget.DEEPSEEK) == 64_000
    assert context_window_tokens(settings, ModelTarget.GLM) == 256_000
    assert compact_trigger_tokens(settings, ModelTarget.DEEPSEEK) == 44_800
    assert compact_target_tokens(settings, ModelTarget.DEEPSEEK) == 25_600


def test_should_compact_uses_threshold_and_ignores_auxiliary() -> None:
    settings = make_settings()

    assert should_compact(89_599, settings, ModelTarget.DEEPSEEK) is False
    assert should_compact(89_600, settings, ModelTarget.DEEPSEEK) is True
    assert should_compact(200_000, settings, ModelTarget.AUXILIARY) is False


def test_estimate_tokens_treats_cjk_as_heavier_than_ascii() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("中文任务") == 2


def test_compact_target_must_be_below_threshold() -> None:
    with pytest.raises(ValidationError):
        make_settings(context_compact_target=0.80, context_compact_threshold=0.70)
