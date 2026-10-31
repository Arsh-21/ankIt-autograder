"""Model client. Every model call in the project goes through this module.

Ollama and OpenRouter both expose an OpenAI-compatible API, so the `openai`
package serves both; only the base URL changes. Agents call
`chat(role, messages)` or `chat_json(...)` and config.yaml decides the model.

The module also keeps:
  * a disk cache: the same prompt is answered from .cache/llm/ at no cost
  * usage and cost counters per role and per run (see Usage)
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import io
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import TYPE_CHECKING, Any

from PIL import Image

from .config import ROOT, load_config, model_for, price_for
from .utils import parse_json, strip_think

if TYPE_CHECKING:
    from openai import OpenAI

_PROVIDERS = {
    "ollama": {
        "base_url": lambda: os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
        "api_key": lambda: "ollama",  # Ollama ignores the key but the library needs one
    },
    # Ollama's cloud models, called directly over the internet with nothing installed.
    # Key from ollama.com/settings/keys, model names from scripts/list_ollama_models.py.
    "ollama_api": {
        "base_url": lambda: os.getenv("OLLAMA_API_BASE_URL") or "https://ollama.com/v1",
        "api_key": lambda: os.getenv("OLLAMA_API_KEY"),
    },
    "openrouter": {
        "base_url": lambda: os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        "api_key": lambda: os.getenv("OPENROUTER_API_KEY"),
    },
}

CACHE_DIR = ROOT / ".cache" / "llm"


class LLMError(RuntimeError):
    """Raised with a readable message when a model call fails."""


# ------------------------------------------------------------------ usage
class Usage:
    """Counts calls, tokens and cost for one run, split by role.

    Attributes:
        by_role (dict[str, dict]): Per role: "calls", "prompt_tokens", "completion_tokens".
        cache_hits (int): Calls answered from the disk cache.
        cost_usd (float): Total known cost in US dollars.
        cost_unknown_calls (int): Calls whose price could not be worked out.
        seconds (float): Total time spent waiting on models.
    """

    def __init__(self) -> None:
        """Create empty counters."""
        self._lock = threading.Lock()
        self.by_role: dict[str, dict] = {}
        self.cache_hits = 0
        self.cost_usd = 0.0
        self.cost_unknown_calls = 0
        self.seconds = 0.0

    def add(self, role: str, prompt: int | None = 0, completion: int | None = 0, cost: float | None = None,
            cached: bool = False, seconds: float = 0.0) -> None:
        """Record one model call.

        Args:
            role (str): Agent role that made the call.
            prompt (int | None): Prompt tokens used.
            completion (int | None): Completion tokens used.
            cost (float | None): Cost in US dollars, or None if unknown.
            cached (bool): True if the reply came from the disk cache; only the hit is counted.
            seconds (float): Time the call took.
        """
        with self._lock:
            if cached:
                self.cache_hits += 1
                return
            r = self.by_role.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            r["calls"] += 1
            r["prompt_tokens"] += prompt or 0
            r["completion_tokens"] += completion or 0
            self.seconds += seconds
            if cost is None:
                self.cost_unknown_calls += 1
            else:
                self.cost_usd += cost

    def as_dict(self) -> dict:
        """Summarise the counters for reports.

        Returns:
            dict: Keys "api_calls", "cache_hits", "prompt_tokens", "completion_tokens",
                "cost_usd", "cost_inr", "calls_with_unknown_cost", "model_seconds", "by_role".
        """
        with self._lock:
            calls = sum(r["calls"] for r in self.by_role.values())
            usd_to_inr = load_config()["llm"].get("usd_to_inr", 88)
            return {
                "api_calls": calls,
                "cache_hits": self.cache_hits,
                "prompt_tokens": sum(r["prompt_tokens"] for r in self.by_role.values()),
                "completion_tokens": sum(r["completion_tokens"] for r in self.by_role.values()),
                "cost_usd": round(self.cost_usd, 5),
                "cost_inr": round(self.cost_usd * usd_to_inr, 3),
                "calls_with_unknown_cost": self.cost_unknown_calls,
                "model_seconds": round(self.seconds, 1),
                "by_role": {k: dict(v) for k, v in self.by_role.items()},
            }


_current_usage: contextvars.ContextVar = contextvars.ContextVar("autograder_usage", default=None)
_fallback_usage = Usage()


def start_usage() -> Usage:
    """Start counting for a new run. Each app request and CLI run calls this.

    Returns:
        Usage: The new counter, also set as the current one for this context.
    """
    u = Usage()
    _current_usage.set(u)
    return u


def current_usage() -> Usage:
    """Get the counter of the current run.

    Returns:
        Usage: The run's counter, or a shared fallback if start_usage() was not called.
    """
    return _current_usage.get() or _fallback_usage


def submit(pool: Executor, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """Submit a job to a thread pool, keeping the current run's context.

    The usage counter and trace sink live in context variables, so the job
    runs inside a copy of the caller's context.

    Args:
        pool (Executor): Pool to submit to.
        fn (Callable[..., Any]): Function to run.
        *args (Any): Positional arguments for fn.
        **kwargs (Any): Keyword arguments for fn.

    Returns:
        Future: The pool's future for the job.
    """
    return pool.submit(contextvars.copy_context().run, fn, *args, **kwargs)


# ------------------------------------------------------------------ clients
_clients = {}
_client_lock = threading.Lock()


def _client(provider: str) -> OpenAI:
    """Get (and cache) the OpenAI-compatible client for a provider.

    Args:
        provider (str): Provider name from config.yaml, e.g. "ollama" or "openrouter".

    Returns:
        OpenAI: A client with the provider's base URL and key.

    Raises:
        LLMError: If the provider is unknown or its API key is empty.
    """
    from openai import OpenAI

    with _client_lock:
        if provider not in _clients:
            if provider not in _PROVIDERS:
                raise LLMError(f"Unknown provider '{provider}' in config.yaml")
            key = _PROVIDERS[provider]["api_key"]()
            if not key:
                name = "OLLAMA_API_KEY" if provider == "ollama_api" else "OPENROUTER_API_KEY"
                raise LLMError(f"{name} is empty. Copy .env.example to .env and paste your key.")
            headers = {"X-Title": "DA7016 Ankit"} if provider == "openrouter" else None
            _clients[provider] = OpenAI(
                base_url=_PROVIDERS[provider]["base_url"](),
                api_key=key,
                timeout=load_config()["llm"]["timeout_seconds"],
                max_retries=0,  # retries are handled in chat()
                default_headers=headers,
            )
        return _clients[provider]


# ------------------------------------------------------------------ images
def image_part(img: Image.Image, max_side: int | None = None) -> dict:
    """Convert a PIL image to the message part vision models expect.

    Args:
        img (Image.Image): Page or crop to send.
        max_side (int | None): Longest side in pixels; defaults to ingest.image_max_side.

    Returns:
        dict: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}.
    """
    max_side = max_side or load_config()["ingest"]["image_max_side"]
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


# ------------------------------------------------------------------ cache
def _cache_key(spec: dict, messages: list[dict], temperature: float, max_tokens: int, sample_idx: int) -> str:
    """Hash everything that affects a reply into a cache key.

    Args:
        spec (dict): Model spec with "provider", "model" and optional "extra_body".
        messages (list[dict]): Chat messages.
        temperature (float): Sampling temperature.
        max_tokens (int): Output token limit.
        sample_idx (int): Sample number, so repeated samples are cached separately.

    Returns:
        str: SHA-256 hex digest.
    """
    raw = json.dumps([spec["provider"], spec["model"], spec.get("extra_body"), messages,
                      temperature, max_tokens, sample_idx], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    """Read a cached reply.

    Args:
        key (str): Cache key from _cache_key().

    Returns:
        str | None: The cached text, or None if missing or unreadable.
    """
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["text"]
        except Exception:
            return None
    return None


def _cache_put(key: str, text: str) -> None:
    """Store a reply in the disk cache.

    Args:
        key (str): Cache key from _cache_key().
        text (str): Reply text.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({"text": text}), encoding="utf-8")


def _cost_of(spec: dict, u: Any) -> float | None:
    """Work out the cost of one call.

    Uses the cost the provider reports if there is one, otherwise the token
    counts times the price in config.yaml. Local and mock models are free.

    Args:
        spec (dict): Model spec with "provider" and "model".
        u (Any): The `usage` object of the API response, or None.

    Returns:
        float | None: Cost in US dollars, or None if unknown.
    """
    if spec["provider"] in ("ollama", "mock"):
        return 0.0
    reported = getattr(u, "cost", None)
    if reported is None and getattr(u, "model_extra", None):
        reported = u.model_extra.get("cost")
    if reported is not None:
        return float(reported)
    price = price_for(spec["model"])
    if price and u is not None:
        return (u.prompt_tokens * price[0] + u.completion_tokens * price[1]) / 1e6
    return None


# ------------------------------------------------------------------ main API
def chat(role: str, messages: list[dict], *, profile: str | None = None,
         temperature: float = 0.0, max_tokens: int = 4096, sample_idx: int = 0) -> str:
    """Send messages to the model configured for a role and return its text.

    Replies are cached on disk. Network errors and empty replies are retried
    with backoff; errors that retrying cannot fix (400-404) stop at once. A
    reply cut off at the token limit is asked for again with twice the limit,
    and a cut-off reply is never cached.

    Args:
        role (str): Agent role, e.g. "grader" or "ocr"; config.yaml maps it to a model.
        messages (list[dict]): Chat messages in OpenAI format.
        profile (str | None): Config profile; None uses the active one.
        temperature (float): Sampling temperature.
        max_tokens (int): Output token limit.
        sample_idx (int): Sample number. Only affects caching: samples 0, 1, 2 of the
            same prompt are cached separately, so 3 samples stay 3 different answers.

    Returns:
        str: The reply text, with any <think> block removed.

    Raises:
        LLMError: If the call fails after retries or the model returns no answer.
    """
    spec = model_for(role, profile)
    usage = current_usage()

    if spec["provider"] == "mock":
        from .mock import mock_response

        usage.add(role, cost=0.0)
        return mock_response(role, messages, sample_idx)

    cfg = load_config()["llm"]
    key = _cache_key(spec, messages, temperature, max_tokens, sample_idx)
    if cfg.get("cache", True):
        cached = _cache_get(key)
        if cached is not None:
            usage.add(role, cached=True)
            return cached

    extra = dict(spec.get("extra_body") or {})
    if spec["provider"] == "openrouter":
        extra.setdefault("usage", {"include": True})  # ask OpenRouter to report the cost

    client = _client(spec["provider"])
    last_err = None
    limit = max_tokens
    for attempt in range(cfg.get("max_retries", 3)):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=spec["model"], messages=messages, temperature=temperature,
                max_tokens=limit, extra_body=extra or None)
            choice = resp.choices[0]
            text = strip_think(choice.message.content or "")
            u = getattr(resp, "usage", None)
            usage.add(role, prompt=getattr(u, "prompt_tokens", 0) if u else 0,
                      completion=getattr(u, "completion_tokens", 0) if u else 0,
                      cost=_cost_of(spec, u), seconds=time.time() - t0)
            cut_off = getattr(choice, "finish_reason", "") == "length"
            if text and cut_off:  # the reply stops mid-way (e.g. a long JSON): ask again with more room
                last_err = LLMError(f"{spec['model']} stopped at the {limit}-token limit.")
                if limit < 32000:
                    limit *= 2
                    continue
                return text  # still cut off at the largest limit; the caller decides, nothing is cached
            if not text:
                if cut_off:
                    raise LLMError(
                        f"{spec['model']} used up all {limit} output tokens without an answer "
                        "(often hidden reasoning). Turn reasoning down via extra_body in config.yaml "
                        "or use another model.")
                last_err = LLMError(f"{spec['model']} returned an empty reply.")
                time.sleep(2 ** (attempt + 1))
                continue  # an empty reply is usually a passing glitch, so ask again
            if cfg.get("cache", True):
                _cache_put(key, text)
            return text
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001 - translated below
            last_err = e
            status = getattr(e, "status_code", None)
            if status in (400, 401, 402, 403, 404):
                break  # retrying will not help
            wait = 2 ** (attempt + 1)
            if status == 429:  # rate limit: wait as long as the server asks, at least 15 s, 30 s, 45 s
                wait = max(_retry_after(e), 15 * (attempt + 1))
            time.sleep(wait)
    if isinstance(last_err, LLMError):
        raise last_err
    raise LLMError(_friendly_error(spec, last_err))


def _retry_after(err: Exception) -> float:
    """Read the Retry-After header of a failed request.

    Args:
        err (Exception): Error raised by the openai client.

    Returns:
        float: Seconds the server asks the client to wait, or 0.0 if not given.
    """
    try:
        return float(err.response.headers.get("retry-after", 0))
    except Exception:  # noqa: BLE001 - header missing or not a number
        return 0.0


def _friendly_error(spec: dict, err: Exception | None) -> str:
    """Turn an API error into a message that says how to fix it.

    Args:
        spec (dict): Model spec with "provider" and "model".
        err (Exception | None): The last error raised by the client.

    Returns:
        str: A one-line message for the UI or CLI.
    """
    status = getattr(err, "status_code", None)
    name = type(err).__name__
    where = f"{spec['provider']} / {spec['model']}"
    if spec["provider"] == "ollama" and "Connection" in name:
        return f"Could not reach Ollama ({where}). Is the Ollama app running? Try 'ollama list'."
    if status == 401:
        return f"Wrong or missing API key for {where}. Check .env."
    if status == 402:
        return f"Out of credits on {where}."
    if status == 404:
        return (f"Model not found: {where}. For Ollama run 'ollama pull {spec['model']}'. "
                "For OpenRouter run scripts/list_openrouter_models.py and fix config.yaml.")
    if status == 429:
        return f"Rate limited by {where}. Wait a minute or lower max_parallel in config.yaml."
    if status == 400:
        return f"{where} rejected the request (400): {err}. Does this model accept images?"
    return f"Model call failed ({where}): {name}: {err}"


def chat_json(role: str, messages: list[dict], **kwargs: Any) -> dict | list:
    """Call chat() and parse the reply as JSON.

    If the reply is not valid JSON, the model is asked once more, from the
    same messages plus a reminder to reply with complete JSON only, with
    twice the token limit.

    Args:
        role (str): Agent role, as in chat().
        messages (list[dict]): Chat messages in OpenAI format.
        **kwargs (Any): Passed on to chat().

    Returns:
        dict | list: The parsed JSON.

    Raises:
        LLMError: If the second reply is still not valid JSON, or the call fails.
    """
    text = chat(role, messages, **kwargs)
    try:
        return parse_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        retry = messages + [{"role": "user", "content": (
            f"Your previous reply could not be read as JSON ({e}). Reply again with the complete JSON "
            "only: nothing before or after it, every string closed.")}]
        kwargs = kwargs | {"max_tokens": 2 * kwargs.get("max_tokens", 4096)}
        text2 = chat(role, retry, **kwargs)
        try:
            return parse_json(text2)
        except (ValueError, json.JSONDecodeError) as e2:
            raise LLMError(f"Model for '{role}' did not return valid JSON: {e2}. "
                           f"Reply started with: {text2[:300]!r}") from e2
