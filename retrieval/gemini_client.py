"""Minimal Gemini REST client (no SDK dependency). Key comes from config/.env."""
from __future__ import annotations

import requests

from data_scripts import config

MODEL = "gemini-2.0-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def generate(prompt: str, model: str = MODEL, timeout: float = 60.0) -> str:
    """Return generated text; raises RuntimeError if no key is configured."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("INTELLIGRAPH_GEMINI_API_KEY is not set (see .env.example)")
    resp = requests.post(
        URL.format(model=model),
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
