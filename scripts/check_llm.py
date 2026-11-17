"""Run before the demo: ``python -m scripts.check_llm``. Never prints the key."""
from __future__ import annotations

import sys

from data_scripts import config
from retrieval import llm

FIXES = {
    "llm_no_key": "Set INTELLIGRAPH_GEMINI_API_KEY (or GEMINI_API_KEY) in .env. Get one at https://aistudio.google.com/apikey",
    "llm_auth": "The key was rejected. Create a new key at https://aistudio.google.com/apikey and update .env.",
    "llm_rate_limited": "Quota exhausted or too many requests. Check your limits in the AI Studio console, wait a minute, "
                        "or set a backup (GROQ_API_KEY).",
    "llm_model_not_found": "Model not available to this key. Unset INTELLIGRAPH_GEMINI_MODEL to auto-pick, or choose one "
                           "from the list printed below.",
}


def main() -> int:
    print(f"provider: {config.LLM_PROVIDER}")
    if config.LLM_PROVIDER == "gemini":
        try:
            models = llm.list_gemini_models()
            print(f"models supporting generateContent: {len(models)}")
            print("chosen:", llm.choose_gemini_model(models))
        except llm.LLMError as exc:
            print(f"FAIL [{exc.code}] {exc.message}")
            print("fix:", FIXES.get(exc.code, "see message above"))
            return 1
    res = llm.probe()
    if res.ok:
        print(f"OK  provider={res.provider} model={res.model} round-trip={res.latency_ms} ms")
        print("fallback configured:", bool(llm._state["fallback"]))
        return 0
    print(f"FAIL provider={res.provider} model={res.model}: {res.detail}")
    code = res.detail.split(":")[0]
    print("fix:", FIXES.get(code, "see detail above"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
