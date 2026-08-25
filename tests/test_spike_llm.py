"""LLM 客户端测试：真实/fake 共用契约、惰性导入、配置读取、无 key 行为."""

import pytest

from spike.llm import (
    AnthropicClient,
    FakeLLM,
    LLMClient,
    LLMConfig,
    SpikeConfigError,
)


class TestLLMConfig:
    def test_default_model_from_env(self, monkeypatch):
        monkeypatch.setenv("COGMIRROR_SPIKE_MODEL", "claude-opus-4-7")
        assert LLMConfig().model == "claude-opus-4-7"

    def test_default_model_fallback(self, monkeypatch):
        monkeypatch.delenv("COGMIRROR_SPIKE_MODEL", raising=False)
        assert LLMConfig().model == "claude-sonnet-4-6"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert LLMConfig().api_key == "sk-test"


class TestFakeLLM:
    def test_shared_contract_and_calls_recorded(self):
        seen = []

        def responder(system, user):
            seen.append((system, user))
            return '{"ok": true}'

        llm = FakeLLM(responder)
        out = llm.complete("sys", "user", cache_breakpoint=True,
                           json_schema={"type": "object"})
        assert out == '{"ok": true}'
        assert seen == [("sys", "user")]
        assert len(llm.calls) == 1
        llm.close()

    def test_is_llm_client(self):
        assert issubclass(FakeLLM, LLMClient)
        assert issubclass(AnthropicClient, LLMClient)

    def test_json_schema_ignored(self):
        """fake 直接返回注入文本，保证真实/fake 两条路径测试一致."""
        llm = FakeLLM(lambda s, u: "raw text")
        assert llm.complete("s", "u", json_schema={...}) == "raw text"


class TestAnthropicClientLazyImport:
    def test_spike_llm_imports_without_anthropic(self):
        """anthropic 惰性导入：未装依赖时 spike.llm 模块仍可导入."""
        import sys
        import spike.llm
        assert spike.llm
        assert "anthropic" not in sys.modules

    def test_missing_api_key_raises_before_sdk_import(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(SpikeConfigError):
            AnthropicClient(LLMConfig(api_key=""))
