"""
Multi-provider LLM fallback for the voice service — same chain and reasoning
as the main backend's services/llm_service.py, kept as a self-contained copy
here since this is a separately deployed Render service (no shared package).

If you change one, change the other — see the main backend repo for the
fuller write-up of why each fix exists (model deprecations, reasoning-token
behavior differences between providers, etc).
"""
import os
import time
import logging
import requests

logger = logging.getLogger("maitrilearn-voice")

GLOBAL_BUDGET_SECONDS = 40  # this service's gunicorn timeout should be >= this + a margin
MIN_USEFUL_TIMEOUT    = 5


class LLMError(Exception):
    def __init__(self, message, retryable=True):
        self.retryable = retryable
        super().__init__(message)


def _providers():
    return [
        {
            "name":    "groq",
            "api_key": os.getenv("GROQ_API_KEY", "").strip(),
            "url":     "https://api.groq.com/openai/v1/chat/completions",
            "model":   os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "retries": 1,
        },
        {
            "name":             "cerebras",
            "api_key":          os.getenv("CEREBRAS_API_KEY", "").strip(),
            "url":              "https://api.cerebras.ai/v1/chat/completions",
            "model":            os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"),
            "retries":          0,
            # Cerebras rejects reasoning_effort="none" (400) — "low" is its
            # lightest valid value. Confirmed against the live API.
            "reasoning_effort": os.getenv("CEREBRAS_REASONING_EFFORT", "low"),
        },
        {
            "name":             "gemini",
            "api_key":          os.getenv("GEMINI_API_KEY", "").strip(),
            "url":              "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "model":            os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
            "retries":          0,
            "reasoning_effort": os.getenv("GEMINI_REASONING_EFFORT", "none"),
        },
    ]


def _call_provider(provider, prompt, max_tokens, timeout):
    payload = {
        "model":       provider["model"],
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens":  max_tokens,
    }
    if provider.get("reasoning_effort"):
        payload["reasoning_effort"] = provider["reasoning_effort"]
        max_tokens = int(max_tokens * 1.5) + 150
        payload["max_tokens"] = max_tokens

    try:
        response = requests.post(
            provider["url"],
            headers={"Authorization": f"Bearer {provider['api_key']}",
                     "Content-Type":  "application/json"},
            json=payload,
            timeout=(5, timeout),
        )
    except requests.exceptions.Timeout:
        raise LLMError(f"{provider['name']} timed out", retryable=True)
    except requests.exceptions.RequestException as e:
        raise LLMError(f"{provider['name']} network error: {e}", retryable=True)

    if response.status_code == 429:
        raise LLMError(f"{provider['name']} rate limited (429)", retryable=True)
    if response.status_code == 401:
        raise LLMError(f"{provider['name']} rejected the API key (401)", retryable=False)
    if response.status_code == 404:
        raise LLMError(
            f"{provider['name']} model '{provider['model']}' not found (404) — "
            f"likely deprecated, override with {provider['name'].upper()}_MODEL env var",
            retryable=False,
        )
    if response.status_code != 200:
        raise LLMError(f"{provider['name']} error {response.status_code}: {response.text[:200]}", retryable=False)

    try:
        data = response.json()
        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
    except (ValueError, IndexError, KeyError):
        raise LLMError(f"{provider['name']} returned an unparseable response", retryable=False)

    if not content or not content.strip():
        finish_reason = choices[0].get("finish_reason", "?") if choices else "?"
        raise LLMError(
            f"{provider['name']} returned empty content (finish_reason={finish_reason})",
            retryable=False,
        )
    return content.strip()


def ask_ai(prompt: str, max_tokens: int = 400) -> str:
    """
    Try Groq, then Cerebras, then Gemini — skipping any provider whose API
    key isn't configured. Raises ValueError only if every configured
    provider failed (or none are configured).
    """
    providers = [p for p in _providers() if p["api_key"]]
    if not providers:
        raise ValueError(
            "No LLM provider configured. Set at least one of GROQ_API_KEY, "
            "CEREBRAS_API_KEY, GEMINI_API_KEY in the environment."
        )

    start  = time.time()
    errors = []

    for provider in providers:
        attempts = provider.get("retries", 0) + 1
        for attempt in range(attempts):
            remaining = GLOBAL_BUDGET_SECONDS - (time.time() - start)
            if remaining < MIN_USEFUL_TIMEOUT:
                logger.warning(f"[llm] time budget exhausted before trying {provider['name']}")
                raise ValueError("AI service is slow to respond right now. Please try again in a moment.")

            try:
                t0      = time.time()
                content = _call_provider(provider, prompt, max_tokens, remaining)
                elapsed = round((time.time() - t0) * 1000)
                logger.info(f"[llm] provider={provider['name']} attempt={attempt+1} time={elapsed}ms OK")
                return content
            except LLMError as e:
                logger.warning(f"[llm] {provider['name']} failed (attempt {attempt+1}/{attempts}): {e}")
                errors.append(str(e))
                remaining_after = GLOBAL_BUDGET_SECONDS - (time.time() - start)
                if e.retryable and attempt < attempts - 1 and remaining_after > MIN_USEFUL_TIMEOUT + 2:
                    time.sleep(2)
                    continue
                break

    logger.error(f"[llm] all providers exhausted: {errors}")
    combined = "; ".join(errors)
    if any("429" in e or "rate limited" in e for e in errors):
        raise ValueError("Too many requests right now — all AI providers are rate-limited. Please try again shortly.")
    raise ValueError(f"AI service unavailable right now. Please try again in a moment. ({combined})")
