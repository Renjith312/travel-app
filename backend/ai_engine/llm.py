"""
ai_engine/llm.py
================
OpenRouter LLM with robust retry + model rotation.

Fixes:
  - Removed reasoning/thinking models that waste tokens on internal thought
    and return content: null
  - 400 Bad Request now rotates model (prompt too large for that model)
  - 502 treated same as 429 (rotate + wait)
  - Proper cycling _model_index never resets to same model
  - Timeout retries same model once then rotates
"""
import os, re, json, time, requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL      = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free")

# Only instruction-following models — NO reasoning/thinking models
# Reasoning models (nemotron-3-nano, nemotron-3-super, etc.) spend all tokens
# on internal chain-of-thought and return content: null
FREE_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",   # best instruction following
    "meta-llama/llama-3.3-70b-instruct:free",           # very capable
    "google/gemma-3-12b-it:free",                       # good fallback
    "stepfun/step-3.5-flash:free",                      # fast
    "arcee-ai/trinity-large-preview:free",              # fallback
]

_model_index = 0


def _next_model() -> str:
    """Advance index and return next model — guaranteed to be different."""
    global _model_index
    _model_index = (_model_index + 1) % len(FREE_MODELS)
    return FREE_MODELS[_model_index]


def llm_chat(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Single OpenRouter call.
    Raises:
      requests.HTTPError  — HTTP error (includes 400, 429, 502…)
      ValueError          — HTTP 200 but bad/empty response body
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    chosen = model or DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "X-Title":       "TravelCopilot",
    }
    payload = {
        "model":       chosen,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    resp = requests.post(OPENROUTER_BASE, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()   # raises HTTPError for 4xx/5xx

    data = resp.json()

    # OpenRouter can return HTTP 200 with an error body
    if "error" in data:
        err  = data["error"]
        code = err.get("code", 0) if isinstance(err, dict) else 0
        msg  = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"[LLM] Response error body: code={code}  msg={msg[:120]}")
        if code in (429, 502, 503, 402) or \
                any(w in msg.lower() for w in ("quota","rate","network","connection","timeout")):
            fake = requests.Response(); fake.status_code = 429
            raise requests.HTTPError(msg, response=fake)
        raise ValueError(f"OpenRouter error: {msg}")

    if "choices" not in data or not data["choices"]:
        raise ValueError(f"No choices in response: {str(data)[:200]}")

    content = data["choices"][0].get("message", {}).get("content", "")

    # Reasoning models return content: null with finish_reason: length
    finish = data["choices"][0].get("finish_reason", "")
    if not content and finish == "length":
        raise ValueError(
            f"Model '{chosen}' hit token limit with no content output — "
            "likely a reasoning model. Rotating."
        )
    if not content:
        raise ValueError(f"Empty content in response (finish={finish}): {str(data)[:200]}")

    return content


def llm_chat_with_retry(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 5,
) -> str:
    """
    Call OpenRouter with automatic model rotation on any failure.

    Rotatable errors (rotate + wait):
      429 / 502 / 503 / 402  — quota / rate / network
      400                     — prompt too large for this model
      ValueError              — empty content / reasoning model timeout
      Timeout                 — network timeout

    Non-rotatable (raise immediately):
      401  — bad API key
    """
    chosen   = model or DEFAULT_MODEL
    last_err: Exception = RuntimeError("No attempts made")

    for attempt in range(max_retries):
        try:
            print(f"[LLM] Attempt {attempt+1}/{max_retries} — {chosen}")
            result = llm_chat(
                messages, model=chosen,
                temperature=temperature, max_tokens=max_tokens,
            )
            if attempt > 0:
                print(f"[LLM] ✅ Success on attempt {attempt+1} with {chosen}")
            return result

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0

            if status == 401:
                # Bad API key — no point retrying
                print(f"[LLM] 401 Unauthorized — check OPENROUTER_API_KEY")
                raise

            if status in (429, 502, 503, 402):
                old = chosen; chosen = _next_model()
                wait = 30 if status == 429 else 15
                print(f"[LLM] HTTP {status} — rotating {old} → {chosen}, waiting {wait}s")
                time.sleep(wait)
                last_err = e

            elif status == 400:
                # Prompt too large for this model — rotate immediately, no wait needed
                old = chosen; chosen = _next_model()
                print(f"[LLM] HTTP 400 (prompt too large?) — rotating {old} → {chosen}")
                last_err = e

            else:
                print(f"[LLM] HTTP {status} — non-retryable: {e}")
                raise

        except ValueError as e:
            # Empty content / reasoning model / bad body — rotate
            old = chosen; chosen = _next_model()
            wait = 10 * (attempt + 1)
            print(f"[LLM] Bad response ({str(e)[:80]}) — rotating {old} → {chosen}, waiting {wait}s")
            time.sleep(wait)
            last_err = e

        except requests.Timeout:
            wait = 15
            if attempt % 2 == 1:          # second consecutive timeout → rotate
                old = chosen; chosen = _next_model()
                print(f"[LLM] Timeout ×2 — rotating {old} → {chosen}, waiting {wait}s")
            else:
                print(f"[LLM] Timeout — retrying same model in {wait}s")
            time.sleep(wait)
            last_err = requests.Timeout()

        except Exception as e:
            print(f"[LLM] Unexpected error: {e}")
            raise

    raise RuntimeError(
        f"LLM call failed after {max_retries} retries. Last: {last_err}"
    )


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract first valid JSON object from LLM output.
    Handles: markdown fences, leading prose, nested braces,
             strings containing braces, trailing commas.
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip().replace("```", "").strip()
    start = text.find("{")
    if start == -1:
        return None

    depth = end = 0
    in_string = escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:          escape_next = False; continue
        if ch == "\\" and in_string: escape_next = True; continue
        if ch == '"':            in_string = not in_string; continue
        if in_string:            continue
        if ch == "{":            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:       end = i; break

    if not end:
        return None
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[LLM] JSON decode error: {e} — trying cleanup")
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", json_str))
        except Exception:
            return None