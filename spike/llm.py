"""LLM 客户端封装——spike 唯一的 LLM 出口.

约定：
- 项目标配 LLM = MiniMax-M3，走 OpenAI 兼容协议（base_url 指向 MiniMax v1）。
- openai SDK 惰性导入（函数内 import）：未安装 openai 依赖时，
  `python -m spike smoke` 与全部测试（用 FakeLLM）仍能正常导入运行。
- API key 只从环境变量 MINIMAX_API_KEY 读取，绝不写进代码或 prompt；
  仓库根 .env（已 gitignore）可用 _load_dotenv 自动加载，仅 setdefault 不覆盖已设环境变量。
- 结构化约束：MiniMax-M3 不支持 OpenAI 式 json_schema response_format（400），
  json_object 也去不掉 content 里的思维链前导——依赖调用方宽松 JSON 解析兜底
  （dialogue.extract_json_object），client 不发送 response_format。
- cache_breakpoint 是 Anthropic 特有（prompt caching），OpenAI 兼容路径忽略。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
_DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# 测试可 monkeypatch：指向不存在的路径即可隔离真实 .env 对缺省值的干扰
_ENV_PATH: Path | None = _DEFAULT_ENV_PATH
_loaded_env_paths: set[str] = set()


def _load_dotenv(path: Path | None = None) -> None:
    """加载仓库根 .env（KEY=VALUE 行），只 setdefault 不覆盖已有环境变量."""
    path = path or _ENV_PATH
    if not path or not path.is_file() or str(path) in _loaded_env_paths:
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    finally:
        _loaded_env_paths.add(str(path))


def _getenv(key: str, default: str) -> str:
    _load_dotenv()
    return os.environ.get(key, default)


class SpikeLLMError(Exception):
    """LLM 调用的中文封装异常."""


class SpikeConfigError(SpikeLLMError):
    """配置缺失（如未设置 API key）."""


@dataclass
class LLMConfig:
    model: str = field(
        default_factory=lambda: _getenv("MINIMAX_MODEL", DEFAULT_MODEL))
    base_url: str = field(
        default_factory=lambda: _getenv("MINIMAX_BASE_URL", DEFAULT_BASE_URL))
    max_tokens: int = 4096
    temperature: float | None = 0.2
    api_key: str = field(default_factory=lambda: _getenv("MINIMAX_API_KEY", ""))


class LLMClient(ABC):
    """LLM 客户端抽象接口（真实 / fake 两条路径共用契约）."""

    @abstractmethod
    def complete(self, system: str, user: str, *,
                 cache_breakpoint: bool = False,
                 json_schema: dict | None = None) -> str:
        """单轮补全。

        Args:
            system: 稳定 system 指令（放前）
            user: 易变对话内容（放后）
            cache_breakpoint: 是否要求缓存断点（Anthropic prompt caching，OpenAI 兼容路径忽略）
            json_schema: 非空时要求结构化输出（json_schema 的 JSON Schema）
        Returns:
            assistant 文本
        """

    @abstractmethod
    def close(self) -> None:
        """释放资源."""


class OpenAICompatClient(LLMClient):
    """MiniMax-M3 等 OpenAI 兼容端点的实现（惰性导入 openai SDK）."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        if not self.config.api_key:
            raise SpikeConfigError(
                "未设置 MINIMAX_API_KEY（可从仓库根 .env 或环境变量提供）")
        import openai  # 惰性导入：未装依赖时模块仍可导入
        self._openai = openai
        self._client = openai.OpenAI(api_key=self.config.api_key,
                                     base_url=self.config.base_url)

    def complete(self, system: str, user: str, *,
                 cache_breakpoint: bool = False,
                 json_schema: dict | None = None) -> str:
        # 注：MiniMax-M3 不支持 OpenAI 式 json_schema response_format（400），
        # json_object 也去不掉 content 里的思维链前导——结构化约束改由调用方
        # 的宽松 JSON 解析兜底（extract_json_object），故不发送 response_format。
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except self._openai.APIError as e:
            raise SpikeLLMError(f"LLM API 错误: {e}") from e
        return self._extract_text(resp)

    @staticmethod
    def _extract_text(resp) -> str:
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            raise SpikeLLMError("LLM 返回内容中没有文本")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - 关闭失败不阻塞流程
            pass


class FakeLLM(LLMClient):
    """确定性 fake：构造时注入 responder 回调，json_schema 参数直接忽略.

    保证真实 / fake 两条路径的调用契约一致；用于测试与 smoke（无需 API key）。
    """

    def __init__(self, responder: Callable[[str, str], str]) -> None:
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *,
                 cache_breakpoint: bool = False,
                 json_schema: dict | None = None) -> str:
        self.calls.append((system, user))
        return self._responder(system, user)

    def close(self) -> None:
        pass
