"""LLM 客户端测试：真实/fake 共用契约、惰性导入、.env 加载、配置读取、无 key 行为."""

import sys
import types

import pytest

from spike.llm import (
    FakeLLM,
    LLMClient,
    LLMConfig,
    OpenAICompatClient,
    SpikeConfigError,
    SpikeLLMError,
)


@pytest.fixture
def no_env_file(monkeypatch, tmp_path):
    """隔离仓库根真实 .env：指向不存在的路径，避免污染缺省值测试."""
    monkeypatch.setattr("spike.llm._ENV_PATH", tmp_path / "missing.env")


class TestLLMConfig:
    def test_default_model_from_env(self, monkeypatch, no_env_file):
        monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M3-7B")
        assert LLMConfig().model == "MiniMax-M3-7B"

    def test_default_model_fallback(self, monkeypatch, no_env_file):
        monkeypatch.delenv("MINIMAX_MODEL", raising=False)
        assert LLMConfig().model == "MiniMax-M3"

    def test_base_url_from_env(self, monkeypatch, no_env_file):
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
        assert LLMConfig().base_url == "https://api.minimaxi.com/v1"

    def test_base_url_fallback(self, monkeypatch, no_env_file):
        monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
        assert LLMConfig().base_url == "https://api.minimaxi.com/v1"

    def test_api_key_from_env(self, monkeypatch, no_env_file):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
        assert LLMConfig().api_key == "sk-test"


class TestDotenvLoader:
    def test_loads_env_file(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# 注释行\n"
            "MINIMAX_API_KEY=sk-dotenv\n"
            "MINIMAX_MODEL=MiniMax-M3\n"
            "MINIMAX_BASE_URL = https://api.minimaxi.com/v1\n",
            encoding="utf-8")
        for k in ("MINIMAX_API_KEY", "MINIMAX_MODEL", "MINIMAX_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setattr("spike.llm._ENV_PATH", env_file)
        assert LLMConfig().api_key == "sk-dotenv"
        assert LLMConfig().model == "MiniMax-M3"
        assert LLMConfig().base_url == "https://api.minimaxi.com/v1"

    def test_existing_env_wins_over_dotenv(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MINIMAX_API_KEY=sk-dotenv\n", encoding="utf-8")
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-env")
        monkeypatch.setattr("spike.llm._ENV_PATH", env_file)
        assert LLMConfig().api_key == "sk-env"

    def test_missing_env_file_no_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setattr("spike.llm._ENV_PATH", tmp_path / "absent.env")
        assert LLMConfig().api_key == ""


class TestOpenAICompatClient:
    def _install_fake_openai(self, monkeypatch):
        """注入假 openai 模块：记录 create 调用，可注入错误，返回确定性响应."""
        calls: list[dict] = []
        errors: list[Exception] = []

        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Resp:
            def __init__(self, content):
                self.choices = [type("C", (), {"message": _Msg(content)})()]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if errors:
                    raise errors.pop(0)
                return _Resp('{"ok": true}')

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.chat = FakeChat()

            def close(self):
                pass

        class APIError(Exception):
            pass

        mod = types.ModuleType("openai")
        mod.OpenAI = FakeOpenAI
        mod.APIError = APIError
        monkeypatch.setitem(sys.modules, "openai", mod)
        return calls, errors, APIError

    def test_lazy_import(self):
        """openai 惰性导入：导入 spike.llm 不触发 openai 包加载."""
        import spike.llm
        assert spike.llm
        assert "openai" not in sys.modules

    def test_missing_api_key_raises_before_sdk(self, monkeypatch, no_env_file):
        monkeypatch.setenv("MINIMAX_API_KEY", "")
        with pytest.raises(SpikeConfigError):
            OpenAICompatClient(LLMConfig(api_key=""))

    def test_sends_messages_and_returns_text(self, monkeypatch, no_env_file):
        calls, _, _ = self._install_fake_openai(monkeypatch)
        client = OpenAICompatClient(LLMConfig(api_key="sk-test", model="MiniMax-M3"))
        out = client.complete("sys", "user")
        assert out == '{"ok": true}'
        assert calls[0]["model"] == "MiniMax-M3"
        assert calls[0]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
        assert client._client.kwargs["api_key"] == "sk-test"
        client.close()

    def test_json_schema_accepted_but_no_response_format(self, monkeypatch, no_env_file):
        """MiniMax 不支持 json_schema response_format——参数接受但单次调用，不发送."""
        calls, _, _ = self._install_fake_openai(monkeypatch)
        client = OpenAICompatClient(LLMConfig(api_key="sk-test"))
        out = client.complete("s", "u", json_schema={"type": "object"})
        assert out == '{"ok": true}'
        assert len(calls) == 1
        assert "response_format" not in calls[0]

    def test_api_error_wrapped(self, monkeypatch, no_env_file):
        calls, errors, APIError = self._install_fake_openai(monkeypatch)
        errors.append(APIError("boom"))
        client = OpenAICompatClient(LLMConfig(api_key="sk-test"))
        with pytest.raises(SpikeLLMError):
            client.complete("s", "u")

    def test_is_llm_client(self):
        assert issubclass(OpenAICompatClient, LLMClient)
        assert issubclass(FakeLLM, LLMClient)


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

    def test_json_schema_ignored(self):
        """fake 直接返回注入文本，保证真实/fake 两条路径测试一致."""
        llm = FakeLLM(lambda s, u: "raw text")
        assert llm.complete("s", "u", json_schema={...}) == "raw text"
