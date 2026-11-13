from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from biography_rag.config import Settings
from biography_rag.llm.providers import (
    ChatModelFactory,
    FallbackChat,
    ModelOption,
    OpenAICompatibleChat,
    provider_specs,
    strip_reasoning,
)


def _chunk(text: str | None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeClient:
    """Records each request and streams back `reply` in a few pieces."""

    def __init__(self, reply: str = "Hello there", error: Exception | None = None) -> None:
        self.reply, self.error, self.requests = reply, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        pieces = [self.reply[i : i + 4] for i in range(0, len(self.reply), 4)]
        return iter([_chunk(None), *(_chunk(p) for p in pieces), SimpleNamespace(choices=[])])


def settings(**keys) -> Settings:
    return Settings(_env_file=None, **keys)


class TestCatalog:
    def test_only_providers_with_keys_are_offered(self):
        factory = ChatModelFactory(
            settings(groq_api_key="g", nvidia_api_key="n"), client_factory=lambda s: FakeClient()
        )
        assert [p.name for p in factory.available()] == ["groq", "nvidia"]

    def test_default_model_is_first_on_the_list(self):
        spec = provider_specs(settings())["gemini"]
        assert spec.model == "gemini-3.5-flash-lite"
        assert spec.option(None).fallbacks == ("gemini-3.1-flash-lite",)

    def test_models_off_the_list_are_rejected(self):
        factory = ChatModelFactory(settings(nvidia_api_key="n"), client_factory=lambda s: FakeClient())
        with pytest.raises(ValueError, match="not offered"):
            factory.create("nvidia", "some/private-finetune")

    def test_slow_models_are_labelled(self):
        nvidia = provider_specs(settings())["nvidia"]
        slow = {m.id for m in nvidia.models if m.slow}
        assert slow == {"deepseek-ai/deepseek-v4.1-flash", "z-ai/glm-5.3-flash"}
        assert nvidia.option("z-ai/glm-5.3-flash").label == "z-ai/glm-5.3-flash (slow)"

    def test_slow_providers_wait_longer(self):
        specs = provider_specs(settings())
        assert specs["nvidia"].timeout >= 400 and specs["cheapllm"].timeout >= 400

    def test_missing_key_and_unknown_provider(self):
        factory = ChatModelFactory(settings(), client_factory=lambda s: FakeClient())
        with pytest.raises(ValueError, match="no API key"):
            factory.create("groq")
        with pytest.raises(ValueError, match="unknown provider"):
            factory.create("nope")
        with pytest.raises(RuntimeError):
            _ = factory.default_provider

    def test_models_are_cached_and_share_one_client(self):
        built = []
        factory = ChatModelFactory(settings(groq_api_key="g"), client_factory=lambda s: built.append(s) or FakeClient())
        a = factory.create("groq", "openai/gpt-oss-20b")
        assert factory.create("groq", "openai/gpt-oss-20b") is a
        factory.create("groq", "qwen/qwen3.8-27b")
        assert len(built) == 1

    def test_fallback_model_is_wrapped(self):
        factory = ChatModelFactory(settings(gemini_api_key="k"), client_factory=lambda s: FakeClient())
        chat = factory.create("gemini")
        assert isinstance(chat, FallbackChat)
        assert chat.model == "gemini-3.5-flash-lite"
        assert [c.model for c in chat.chain] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


class TestStreamingChat:
    def test_joins_streamed_deltas_and_sends_extra_body(self):
        client = FakeClient("The answer is 42.")
        option = ModelOption("m", extra_body={"reasoning_effort": "low"})
        chat = OpenAICompatibleChat("groq", option, client)
        assert chat.complete([{"role": "user", "content": "q"}], max_tokens=50) == "The answer is 42."
        req = client.requests[0]
        assert req["stream"] is True and req["max_tokens"] == 50 and req["temperature"] == 0
        assert req["extra_body"] == {"reasoning_effort": "low"}

    def test_no_extra_body_when_none(self):
        client = FakeClient()
        OpenAICompatibleChat("gemini", ModelOption("m"), client).complete([])
        assert client.requests[0]["extra_body"] is None

    def test_strips_inline_reasoning(self):
        client = FakeClient("<think>let me see…</think>\nANSWER: yes")
        assert OpenAICompatibleChat("p", ModelOption("m"), client).complete([]) == "ANSWER: yes"

    def test_limiter_caps_concurrent_calls(self):
        limiter = threading.BoundedSemaphore(1)
        limiter.acquire()  # someone else holds the only slot
        chat = OpenAICompatibleChat("p", ModelOption("m"), FakeClient(), limiter)
        t = threading.Thread(target=chat.complete, args=([],), daemon=True)
        t.start()
        t.join(0.2)
        assert t.is_alive()
        limiter.release()
        t.join(2)
        assert not t.is_alive()


@pytest.mark.parametrize(
    ("raw", "clean"),
    [("plain", "plain"), ("<think>a\nb</think> x", "x"), ("dangling reasoning</think>answer", "answer")],
)
def test_strip_reasoning(raw, clean):
    assert strip_reasoning(raw) == clean


class Flaky:
    def __init__(self, model: str, fail: bool) -> None:
        self.provider, self.model, self.fail, self.calls = "p", model, fail, 0

    def complete(self, messages, *, max_tokens=1200):
        self.calls += 1
        if self.fail:
            raise RuntimeError("quota")
        return self.model


class TestFallback:
    def test_uses_next_model_when_primary_fails(self):
        primary, backup = Flaky("a", fail=True), Flaky("b", fail=False)
        assert FallbackChat([primary, backup]).complete([]) == "b"

    def test_failed_primary_is_benched_for_the_cooldown(self):
        now = [0.0]
        primary, backup = Flaky("a", fail=True), Flaky("b", fail=False)
        chat = FallbackChat([primary, backup], cooldown=60, clock=lambda: now[0])
        chat.complete([])
        chat.complete([])
        assert primary.calls == 1  # second call skipped it
        now[0] = 61
        primary.fail = False
        assert chat.complete([]) == "a"

    def test_raises_when_every_model_fails(self):
        with pytest.raises(RuntimeError, match="quota"):
            FallbackChat([Flaky("a", True), Flaky("b", True)]).complete([])
