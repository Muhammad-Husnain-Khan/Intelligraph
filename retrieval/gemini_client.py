"""Minimal Gemini REST client for the legacy /query/* endpoints. Key comes from config/.env.

Model: INTELLIGRAPH_GEMINI_MODEL if pinned, else the model chosen by the startup probe,
else the historical default.
"""
from __future__ import annotations

import requests

from data_scripts import config

MODEL = "gemini-2.0-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _model() -> str:
    if config.GEMINI_MODEL_PIN:
        return config.GEMINI_MODEL_PIN
    try:
        from retrieval import llm

        for p in llm.get_providers():
            if p.name == "gemini":
                return p.model
        if config.GEMINI_API_KEY:
            chosen = llm.choose_gemini_model(llm.list_gemini_models())
            llm._state["primary"] = llm.GeminiProvider(chosen)
            return chosen
    except Exception:  # noqa: BLE001
        pass
    return MODEL


def generate(prompt: str, model: str | None = None, timeout: float = 60.0) -> str:
    """Return generated text; raises RuntimeError if no key is configured."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("INTELLIGRAPH_GEMINI_API_KEY is not set (see .env.example)")
    resp = requests.post(
        URL.format(model=model or _model()),
        headers={"x-goog-api-key": config.GEMINI_API_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.0}},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return f"[no answer returned: {data.get('promptFeedback', data)}]"
