import pytest

from coding_helper.config import Settings
from coding_helper.models import (
    ModelConfigurationError,
    ModelTarget,
    create_chat_model,
    model_id_for,
    other_primary,
    primary_order,
)


def make_settings(**overrides) -> Settings:
    """构造不读取开发者 .env 的测试配置。"""

    values = {
        "ark_api_key": "test-secret",
        "ark_deepseek_model": "deepseek-endpoint",
        "ark_glm_model": "glm-endpoint",
        "ark_auxiliary_model": "doubao-endpoint",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize(
    ("default_primary", "expected"),
    [
        ("deepseek", (ModelTarget.DEEPSEEK, ModelTarget.GLM)),
        ("glm", (ModelTarget.GLM, ModelTarget.DEEPSEEK)),
    ],
)
def test_primary_order_uses_other_main_model_as_fallback(default_primary, expected) -> None:
    settings = make_settings(ark_default_primary=default_primary)

    assert primary_order(settings) == expected
    assert other_primary(expected[0]) is expected[1]


def test_other_primary_rejects_auxiliary() -> None:
    with pytest.raises(ValueError, match="辅助模型"):
        other_primary(ModelTarget.AUXILIARY)


def test_model_id_for_rejects_unconfigured_role() -> None:
    settings = make_settings(ark_glm_model=None)

    with pytest.raises(ModelConfigurationError, match="glm"):
        model_id_for(settings, ModelTarget.GLM)


def test_create_chat_model_configures_langchain_adapter(monkeypatch) -> None:
    """模型工厂应关闭 SDK 隐式重试，并且不能改变用户选择的角色。"""

    captured = {}
    sentinel = object()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("coding_helper.models.ChatOpenAI", fake_chat_openai)

    model = create_chat_model(make_settings(), ModelTarget.AUXILIARY)

    assert model is sentinel
    assert captured["model"] == "doubao-endpoint"
    assert captured["max_retries"] == 0
    assert captured["temperature"] == 0
