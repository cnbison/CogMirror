"""LLM 客户端封装——spike 唯一的 LLM 出口.

约定：
- anthropic 必须惰性导入（函数内 import）：未安装 anthropic 依赖时，
  `python -m spike smoke` 与全部测试（用 FakeLLM）仍能正常导入运行。
- API key 只从环境变量 ANTHROPIC_API_KEY 读取，绝不写进代码或 prompt。
- 稳定 system 块放前 + cache_control ephemeral 做 prompt caching；
  volatile 对话内容在后（缓存是前缀匹配，任何前缀字节变化都会失效）。
- 结构化输出：json_schema 非空时走 output_config.format
  （claude-api skill 确认的当前 SDK 用法；Sonnet 4.6 支持）。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_MODEL = "claude-sonnet-4-6"


class SpikeLLMError(Exception):
    """LLM 调用的中文封装异常."""


class SpikeConfigError(SpikeLLMError):
    """配置缺失（如未设置 API key）."""


@dataclass
class LLMConfig:
    model: str = field(
        default_factory=lambda: os.environ.get("COGMIRROR_SPIKE_MODEL", DEFAULT_MODEL))
    max_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))


class LLMClient(ABC):
    """LLM 客户端抽象接口（真实 / fake 两条路径共用契约）."""

    @abstractmethod
    def complete(self, system: str, user: str, *,
                 cache_breakpoint: bool = False,
                 json_schema: dict | None = None) -> str:
        """单轮补全。

        Args:
            system: 稳定 system 指令（放前，可缓存）
            user: 易变对话内容（放后）
            cache_breakpoint: 在 system 块上加 cache_control ephemeral
            json_schema: 非空时要求结构化输出（json_schema 的 JSON Schema）
        Returns:
            assistant 文本
        """

    @abstractmethod
    def close(self) -> None:
        """释放资源."""


class AnthropicClient(LLMClient):
    """官方 Anthropic SDK 实现（惰性导入 anthropic）."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        if not self.config.api_key:
            raise SpikeConfigError(
                "未设置 ANTHROPIC_API_KEY 环境变量（API key 只从环境变量读取）")
        import anthropic  # 惰性导入：未装依赖时模块仍可导入
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=self.config.api_key)

    def complete(self, system: str, user: str, *,
                 cache_breakpoint: bool = False,
                 json_schema: dict | None = None) -> str:
        system_param: str | list[dict]
        if cache_breakpoint:
            system_param = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_param = system

        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system_param,
            "messages": [{"role": "user", "content": user}],
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if json_schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}

        try:
            resp = self._client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as e:
            raise SpikeLLMError(f"LLM 速率限制: {e}") from e
        except self._anthropic.APIError as e:
            raise SpikeLLMError(f"LLM API 错误: {e}") from e

        for block in resp.content:
            if block.type == "text":
                return block.text
        raise SpikeLLMError("LLM 返回内容中没有文本块")

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
