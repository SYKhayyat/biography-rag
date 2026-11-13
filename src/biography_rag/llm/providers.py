"""OpenAI, Qwen and DeepSeek all speak the OpenAI chat-completions protocol, so one
adapter covers them; a provider is just a base URL, a key and a default model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biography_rag.config import Settings
from biography_rag.llm.base import ChatModel, Message


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    api_key: str
    model: str
    base_url: str | None  # None -> the OpenAI SDK default


def provider_specs(settings: Settings) -> dict[str, ProviderSpec]:
    specs = [
        ProviderSpec("openai", "OpenAI", settings.openai_api_key, settings.openai_model, None),
        ProviderSpec("qwen", "Qwen (DashScope)", settings.qwen_api_key, settings.qwen_model, settings.qwen_base_url),
        ProviderSpec(
            "deepseek", "DeepSeek", settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url
        ),
    ]
    return {s.name: s for s in specs}


class OpenAICompatibleChat:
    def __init__(self, provider: str, model: str, client: Any) -> None:
        self._provider = provider
        self._model = model
        self._client = client

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: list[Message], *, max_tokens: int = 1200) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class ChatModelFactory:
    """Builds a chat model for any configured provider. Only providers with a key are offered."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._specs = provider_specs(settings)
        self._cache: dict[tuple[str, str], ChatModel] = {}

    def available(self) -> list[ProviderSpec]:
        return [s for s in self._specs.values() if s.api_key]

    @property
    def default_provider(self) -> str:
        names = [s.name for s in self.available()]
        if self._settings.llm_provider in names:
            return self._settings.llm_provider
        if names:
            return names[0]
        raise RuntimeError("no LLM provider has an API key configured")

    def create(self, provider: str | None = None, model: str | None = None) -> ChatModel:
        name = provider or self.default_provider
        spec = self._specs.get(name)
        if spec is None:
            raise ValueError(f"unknown provider {name!r}; choose from {sorted(self._specs)}")
        if not spec.api_key:
            raise ValueError(f"provider {name!r} has no API key configured")
        key = (name, model or spec.model)
        if key not in self._cache:
            from openai import OpenAI

            client = OpenAI(api_key=spec.api_key, base_url=spec.base_url, timeout=self._settings.llm_timeout_seconds)
            self._cache[key] = OpenAICompatibleChat(name, key[1], client)
        return self._cache[key]
