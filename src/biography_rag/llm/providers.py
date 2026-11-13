"""Every provider here speaks the OpenAI chat-completions protocol, so one adapter covers
them all. A provider is a base URL, a key and a fixed list of models users may pick from;
the public API never accepts a model name that is not on that list."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from biography_rag.config import Settings
from biography_rag.llm.base import ChatModel, Message

log = logging.getLogger(__name__)

# Reasoning models served through vLLM/NIM templates honour this switch; without it some
# spend the whole token budget thinking and return an empty answer.
NO_THINKING: Mapping[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class ModelOption:
    id: str
    slow: bool = False
    fallbacks: tuple[str, ...] = ()  # tried in order when this model fails
    extra_body: Mapping[str, Any] | None = None  # provider-specific request fields

    @property
    def label(self) -> str:
        return f"{self.id} (slow)" if self.slow else self.id


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    api_key: str
    base_url: str | None  # None -> the OpenAI SDK default
    models: tuple[ModelOption, ...]
    timeout: float = 120.0  # seconds to wait for the next streamed bytes
    max_retries: int = 3  # SDK retries with backoff on 429/5xx, honouring Retry-After
    concurrency: int = 4  # calls in flight per provider, across all runs

    @property
    def model(self) -> str:
        return self.models[0].id

    def option(self, model: str | None) -> ModelOption:
        if model is None:
            return self.models[0]
        for option in self.models:
            if option.id == model:
                return option
        raise ValueError(f"model {model!r} is not offered for {self.name}; choose from {[m.id for m in self.models]}")


def provider_specs(settings: Settings) -> dict[str, ProviderSpec]:
    s = settings
    slow_timeout = max(s.llm_timeout_seconds, 420.0)
    specs = [
        ProviderSpec("openai", "OpenAI", s.openai_api_key, None, (ModelOption(s.openai_model),), s.llm_timeout_seconds),
        ProviderSpec(
            "groq",
            "Groq",
            s.groq_api_key,
            "https://api.groq.com/openai/v1",
            (
                ModelOption("openai/gpt-oss-120b", extra_body={"reasoning_effort": "low"}),
                ModelOption("openai/gpt-oss-20b", extra_body={"reasoning_effort": "low"}),
                ModelOption("qwen/qwen3.8-27b", extra_body={"reasoning_effort": "none"}),
            ),
            s.llm_timeout_seconds,
            max_retries=6,  # tokens-per-minute limits are tight; wait them out
            concurrency=2,
        ),
        ProviderSpec(
            "gemini",
            "Google Gemini",
            s.gemini_api_key,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            (ModelOption("gemini-3.5-flash-lite", fallbacks=("gemini-3.1-flash-lite",)),),
            s.llm_timeout_seconds,
            concurrency=2,
        ),
        ProviderSpec(
            "nvidia",
            "NVIDIA NIM",
            s.nvidia_api_key,
            "https://integrate.api.nvidia.com/v1",
            (
                ModelOption("nvidia/nemotron-3-ultra-550b-a55b", extra_body=NO_THINKING),
                ModelOption("nvidia/nemotron-3.5-lightning-30b-a3b", extra_body=NO_THINKING),
                ModelOption("poolside/laguna-xs-2.1"),
                ModelOption("openai/gpt-oss-20b"),
                ModelOption("google/gemma-4-31b-it"),
                ModelOption("meta/muse-glimmer-30b", extra_body=NO_THINKING),
                ModelOption("mistralai/mistral-large-4-1t-a52b"),
                ModelOption("deepseek-ai/deepseek-v4.1-flash", slow=True),
                ModelOption("z-ai/glm-5.3-flash", slow=True, extra_body=NO_THINKING),
            ),
            slow_timeout,  # NIM queues requests; minutes before the first token are normal
            concurrency=3,
        ),
        ProviderSpec(
            "cheapllm",
            "cheapllm.shop (DeepSeek)",
            s.cheapllm_api_key,
            "https://cheapllm.shop/v1",
            (ModelOption("deepseek-v4.1-flash-1017", slow=True),),
            slow_timeout,
            concurrency=2,
        ),
        ProviderSpec(
            "qwen",
            "Qwen (DashScope)",
            s.qwen_api_key,
            s.qwen_base_url,
            (ModelOption(s.qwen_model),),
            s.llm_timeout_seconds,
        ),
        ProviderSpec(
            "deepseek",
            "DeepSeek",
            s.deepseek_api_key,
            s.deepseek_base_url,
            (ModelOption(s.deepseek_model),),
            s.llm_timeout_seconds,
        ),
    ]
    return {spec.name: spec for spec in specs}


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    """Some models inline their chain of thought; only the answer after it is wanted."""
    text = _THINK.sub("", text)
    if "</think>" in text:  # opening tag was in the prompt template, not the output
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


class OpenAICompatibleChat:
    """Streams every completion: gateways (Cloudflare and others) cut idle connections after
    ~100 s, and a slow model that is streaming keeps the connection alive."""

    def __init__(
        self,
        provider: str,
        option: ModelOption,
        client: Any,
        limiter: threading.BoundedSemaphore | None = None,
    ) -> None:
        self._provider = provider
        self._option = option
        self._client = client
        self._limiter = limiter or threading.BoundedSemaphore(64)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._option.id

    def complete(self, messages: list[Message], *, max_tokens: int = 1200) -> str:
        with self._limiter:
            stream = self._client.chat.completions.create(
                model=self._option.id,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                extra_body=dict(self._option.extra_body) if self._option.extra_body else None,
            )
            parts = [
                chunk.choices[0].delta.content
                for chunk in stream
                if chunk.choices and getattr(chunk.choices[0].delta, "content", None)
            ]
        return strip_reasoning("".join(parts))


@dataclass
class FallbackChat:
    """Tries each model in order. A model that fails is skipped for `cooldown` seconds, so a
    spent daily quota costs one failed call, not one per question."""

    chain: Sequence[ChatModel]
    cooldown: float = 600.0
    clock: Callable[[], float] = time.monotonic
    _benched_until: dict[int, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def provider(self) -> str:
        return self.chain[0].provider

    @property
    def model(self) -> str:
        return " → ".join(c.model for c in self.chain)

    def complete(self, messages: list[Message], *, max_tokens: int = 1200) -> str:
        last_error: Exception | None = None
        for i, chat in enumerate(self.chain):
            is_last = i == len(self.chain) - 1
            with self._lock:
                benched = self._benched_until.get(i, 0.0) > self.clock()
            if benched and not is_last:
                continue
            try:
                return chat.complete(messages, max_tokens=max_tokens)
            except Exception as exc:  # any provider failure means "try the next model"
                last_error = exc
                if not is_last:
                    log.warning("%s failed (%s); falling back", chat.model, type(exc).__name__)
                    with self._lock:
                        self._benched_until[i] = self.clock() + self.cooldown
        assert last_error is not None
        raise last_error


ClientFactory = Callable[[ProviderSpec], Any]


def _openai_client(spec: ProviderSpec) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=spec.api_key, base_url=spec.base_url, timeout=spec.timeout, max_retries=spec.max_retries)


class ChatModelFactory:
    """Builds a chat model for any configured provider. Only providers with a key are offered,
    and only models on the provider's list can be built."""

    def __init__(self, settings: Settings, client_factory: ClientFactory = _openai_client) -> None:
        self._settings = settings
        self._specs = provider_specs(settings)
        self._client_factory = client_factory
        self._clients: dict[str, Any] = {}
        self._limiters = {name: threading.BoundedSemaphore(spec.concurrency) for name, spec in self._specs.items()}
        self._cache: dict[tuple[str, str], ChatModel] = {}
        self._lock = threading.Lock()

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

    def _single(self, spec: ProviderSpec, option: ModelOption) -> ChatModel:
        if spec.name not in self._clients:
            self._clients[spec.name] = self._client_factory(spec)
        return OpenAICompatibleChat(spec.name, option, self._clients[spec.name], self._limiters[spec.name])

    def create(self, provider: str | None = None, model: str | None = None) -> ChatModel:
        name = provider or self.default_provider
        spec = self._specs.get(name)
        if spec is None:
            raise ValueError(f"unknown provider {name!r}; choose from {sorted(self._specs)}")
        if not spec.api_key:
            raise ValueError(f"provider {name!r} has no API key configured")
        option = spec.option(model)
        with self._lock:
            key = (name, option.id)
            if key not in self._cache:
                primary = self._single(spec, option)
                if option.fallbacks:
                    chain = [primary] + [self._single(spec, ModelOption(m)) for m in option.fallbacks]
                    self._cache[key] = FallbackChat(chain)
                else:
                    self._cache[key] = primary
            return self._cache[key]
