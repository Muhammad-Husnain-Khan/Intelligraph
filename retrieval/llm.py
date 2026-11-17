"""LLM provider layer: Gemini (primary) and any OpenAI-compatible endpoint (fallback).

* No hard-coded model: ``probe_gemini`` lists the models the key can really use.
* Keys are sent in headers only and never logged.
* Retry policy: 429/503, honour retryDelay / Retry-After, else exponential backoff with
  jitter, at most 2 retries, total cap 8 s.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Iterator, Protocol

import requests

from data_scripts import config

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
MAX_RETRIES = 2
TOTAL_CAP_S = 8.0


class LLMError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, system: str = "") -> str: ...
    def stream(self, prompt: str, system: str = "") -> Iterator[str]: ...


def _retry_delay(resp: requests.Response, attempt: int) -> float:
    ra = resp.headers.get("Retry-After", "")
    if ra.replace(".", "", 1).isdigit():
        return float(ra)
    try:
        for d in resp.json().get("error", {}).get("details", []):
            m = re.fullmatch(r"([\d.]+)s", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    return (2 ** attempt) + random.random()


def _describe(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("error", {}).get("message", ""))[:300] or f"HTTP {resp.status_code}"
    except Exception:  # noqa: BLE001
        return f"HTTP {resp.status_code}"


def _request(method: str, url: str, *, headers: dict, json_body: dict | None = None,
             stream: bool = False, timeout: float = 60.0) -> requests.Response:
    """HTTP call with the retry policy; raises LLMError on failure."""
    start = time.monotonic()
    attempt = 0
    while True:
        try:
            resp = requests.request(method, url, headers=headers, json=json_body,
                                    stream=stream, timeout=timeout)
        except requests.Timeout as exc:
            raise LLMError("llm_timeout", "The language model timed out.", True) from exc
        except requests.RequestException as exc:
            raise LLMError("llm_unreachable", "The language model could not be reached.", True) from exc
        if resp.status_code in (429, 503):
            delay = _retry_delay(resp, attempt)
            if attempt < MAX_RETRIES and time.monotonic() - start + delay <= TOTAL_CAP_S:
                resp.close()
                time.sleep(delay)
                attempt += 1
                continue
            code = "llm_rate_limited" if resp.status_code == 429 else "llm_unavailable"
            raise LLMError(code, _describe(resp), True)
        if resp.status_code in (401, 403):
            raise LLMError("llm_auth", f"The API key was rejected: {_describe(resp)}", False)
        if resp.status_code == 404:
            raise LLMError("llm_model_not_found", _describe(resp), False)
        if not resp.ok:
            raise LLMError("llm_error", _describe(resp), False)
        return resp


# --------------------------------------------------------------------------- Gemini
def _gemini_headers() -> dict:
    return {"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"}


def list_gemini_models() -> list[str]:
    """Names (without the ``models/`` prefix) that support generateContent."""
    if not config.GEMINI_API_KEY:
        raise LLMError("llm_no_key", "No Gemini API key is configured (INTELLIGRAPH_GEMINI_API_KEY).")
    resp = _request("GET", f"{GEMINI_BASE}/models?pageSize=200", headers=_gemini_headers(), timeout=20)
    return [m["name"].removeprefix("models/") for m in resp.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def choose_gemini_model(available: list[str]) -> str:
    if config.GEMINI_MODEL_PIN:
        pin = config.GEMINI_MODEL_PIN.removeprefix("models/")
        if pin in available:
            return pin
        raise LLMError("llm_model_not_found", f"Pinned model '{pin}' is not in the model list for this key.")
    if config.GEMINI_MODEL_PREFERENCE:
        prefs = [p.strip() for p in config.GEMINI_MODEL_PREFERENCE.split(",") if p.strip()]
    else:  # cheapest first: Flash-Lite, then Flash; newest version first within each group
        def ver(n: str) -> tuple:
            return tuple(-int(x) for x in re.findall(r"\d+", n)[:2])
        usable = [m for m in available if m.startswith("gemini-") and not re.search(
            r"preview|exp|image|tts|audio|live|thinking|embedding|vision|robotics|computer", m)]
        lite = sorted((m for m in usable if "flash-lite" in m), key=ver)
        flash = sorted((m for m in usable if "flash" in m and "lite" not in m), key=ver)
        prefs = lite + flash
    for p in prefs:
        if p in available:
            return p
    raise LLMError("llm_model_not_found", "No suitable Gemini Flash / Flash-Lite model is available to this key.")


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str):
        self.model = model

    def _body(self, prompt: str, system: str) -> dict:
        body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.0}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @staticmethod
    def _text(data: dict) -> str:
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            return ""

    def generate(self, prompt: str, system: str = "") -> str:
        resp = _request("POST", f"{GEMINI_BASE}/models/{self.model}:generateContent",
                        headers=_gemini_headers(), json_body=self._body(prompt, system))
        return self._text(resp.json()).strip()

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        resp = _request("POST", f"{GEMINI_BASE}/models/{self.model}:streamGenerateContent?alt=sse",
                        headers=_gemini_headers(), json_body=self._body(prompt, system), stream=True)
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if raw and raw.startswith("data:"):
                    try:
                        txt = self._text(json.loads(raw[5:].strip()))
                    except json.JSONDecodeError:
                        continue
                    if txt:
                        yield txt
        finally:
            resp.close()


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, model: str | None = None):
        self.model = model or config.OPENAI_COMPAT_MODEL
        self.base = config.OPENAI_COMPAT_BASE_URL.rstrip("/")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"}

    def _body(self, prompt: str, system: str, stream: bool) -> dict:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return {"model": self.model, "messages": msgs, "temperature": 0, "stream": stream}

    def generate(self, prompt: str, system: str = "") -> str:
        r = _request("POST", f"{self.base}/chat/completions", headers=self._headers(),
                     json_body=self._body(prompt, system, False))
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        r = _request("POST", f"{self.base}/chat/completions", headers=self._headers(),
                     json_body=self._body(prompt, system, True), stream=True)
        try:
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content") or ""
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
        finally:
            r.close()


# --------------------------------------------------------------------------- registry + probe
@dataclass
class ProbeResult:
    provider: str
    model: str
    ok: bool
    detail: str
    latency_ms: int | None = None

    def as_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model, "ok": self.ok, "detail": self.detail}


_state: dict = {"primary": None, "fallback": None, "probe": ProbeResult("none", "", False, "not probed yet")}


def _build_primary() -> tuple[LLMProvider | None, ProbeResult]:
    prov = config.LLM_PROVIDER
    if prov == "openai_compat":
        if not config.GROQ_API_KEY:
            return None, ProbeResult(prov, config.OPENAI_COMPAT_MODEL, False, "no API key configured")
        p: LLMProvider = OpenAICompatProvider()
    else:
        try:
            p = GeminiProvider(choose_gemini_model(list_gemini_models()))
        except LLMError as exc:
            return None, ProbeResult("gemini", config.GEMINI_MODEL_PIN, False, f"{exc.code}: {exc.message}")
    t0 = time.monotonic()
    try:
        p.generate("Reply with the single word: ok")
    except LLMError as exc:
        return None, ProbeResult(p.name, p.model, False, f"{exc.code}: {exc.message}")
    return p, ProbeResult(p.name, p.model, True, "round trip succeeded", int((time.monotonic() - t0) * 1000))


def probe() -> ProbeResult:
    """Pick a model that exists and make one tiny real generation call. Stores the result."""
    primary, result = _build_primary()
    _state["primary"], _state["probe"] = primary, result
    _state["fallback"] = None
    if config.LLM_PROVIDER != "openai_compat" and config.GROQ_API_KEY:
        _state["fallback"] = OpenAICompatProvider()
    return result


def get_providers() -> list[LLMProvider]:
    """Primary first (if the probe succeeded or has not run), then the fallback."""
    out = [p for p in (_state["primary"], _state["fallback"]) if p is not None]
    return out


def last_probe() -> ProbeResult:
    return _state["probe"]


def generate_default(prompt: str, system: str = "") -> str:
    """Used by the legacy gemini_client path; same model setting as the new layer."""
    provs = get_providers() or [GeminiProvider(choose_gemini_model(list_gemini_models()))]
    return provs[0].generate(prompt, system)
