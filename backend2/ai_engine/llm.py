"""
ai_engine/llm.py  — Ollama edition
====================================
Uses a local Ollama instance as the primary LLM provider.
OpenRouter is kept as an optional fallback.

Configuration (via .env):
  OLLAMA_BASE_URL    — Ollama API base URL (default: http://localhost:11434)
  OLLAMA_MODEL       — model to use       (default: llama3.2)
  OPENROUTER_API_KEY — optional fallback  (leave blank to disable)
  OPENROUTER_MODEL   — optional fallback model
"""

import os, re, json, time, requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── OpenRouter fallback (optional) ────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1/chat/completions"

_or_primary = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
OPENROUTER_MODELS = list(dict.fromkeys([
    _or_primary,
    "openrouter/auto",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-3n-e4b-it:free",
]))

_openrouter_model_idx = 0


def _next_openrouter_model(current: str = "") -> str:
    global _openrouter_model_idx
    for _ in range(len(OPENROUTER_MODELS)):
        _openrouter_model_idx = (_openrouter_model_idx + 1) % len(OPENROUTER_MODELS)
        if OPENROUTER_MODELS[_openrouter_model_idx] != current:
            return OPENROUTER_MODELS[_openrouter_model_idx]
    return OPENROUTER_MODELS[0]


# ── Ollama API call ────────────────────────────────────────────────────────────
def _ollama_chat(messages: list, model: str,
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call local Ollama instance via its /api/chat endpoint."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code == 404:
        raise requests.HTTPError(
            f"404: model '{model}' not found in Ollama — run: ollama pull {model}",
            response=resp,
        )
    if resp.status_code != 200:
        raise requests.HTTPError(
            f"HTTP {resp.status_code}: {resp.text[:300]}", response=resp
        )

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Empty response from Ollama: {str(data)[:200]}")
    return content


# ── OpenRouter fallback ────────────────────────────────────────────────────────
def _openrouter_chat(messages: list, model: str,
                     temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    key = OPENROUTER_API_KEY.strip()
    if not key.startswith("sk-or-"):
        raise ValueError("Invalid OPENROUTER_API_KEY format")

    resp = requests.post(
        OPENROUTER_BASE,
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": os.getenv("FRONTEND_URL", "http://localhost:3000"),
                 "X-Title": "TravelCopilot"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        code = err.get("code", 0) if isinstance(err, dict) else 0
        if code in (429, 502, 503) or any(w in msg.lower() for w in ("quota", "rate")):
            fake = requests.Response(); fake.status_code = 429
            raise requests.HTTPError(msg, response=fake)
        raise ValueError(f"OpenRouter error: {msg}")

    content = data["choices"][0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("Empty response from OpenRouter")
    return content


# ── Main public function ───────────────────────────────────────────────────────
def llm_chat_with_retry(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    max_retries: int = 5,
) -> str:
    """
    Try Ollama first (local), fall back to OpenRouter if Ollama is unavailable.
    """
    ollama_model = model or OLLAMA_MODEL

    # ── Try Ollama ─────────────────────────────────────────────────────────────
    for attempt in range(3):
        try:
            print(f"[LLM] Ollama attempt {attempt + 1} — {ollama_model}")
            result = _ollama_chat(messages, ollama_model, temperature, max_tokens)
            print(f"[LLM] ✅ Ollama success ({ollama_model})")
            return result
        except requests.ConnectionError:
            print(f"[LLM] Ollama not reachable at {OLLAMA_BASE_URL} — is Ollama running?")
            break  # no point retrying a connection error; fall through to OpenRouter
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 404:
                print(f"[LLM] Ollama 404 — model '{ollama_model}' not found. "
                      f"Run: ollama pull {ollama_model}")
                break  # no point retrying a missing model
            print(f"[LLM] Ollama HTTP {status}: {e} — retrying in 3s")
            time.sleep(3)
        except requests.Timeout:
            print(f"[LLM] Ollama timeout on attempt {attempt + 1} — retrying in 3s")
            time.sleep(3)
        except Exception as e:
            print(f"[LLM] Ollama exception: {e} — retrying in 3s")
            time.sleep(3)

    # ── Fall back to OpenRouter ────────────────────────────────────────────────
    if OPENROUTER_API_KEY:
        print("[LLM] Falling back to OpenRouter...")
        or_model = OPENROUTER_MODELS[0]
        for attempt in range(max_retries):
            try:
                print(f"[LLM] OpenRouter attempt {attempt + 1} — {or_model}")
                time.sleep(3)
                result = _openrouter_chat(messages, or_model, temperature, max_tokens)
                print(f"[LLM] ✅ OpenRouter success ({or_model})")
                return result
            except requests.HTTPError as e:
                status = e.response.status_code if e.response else 0
                if status in (429, 502, 503):
                    old = or_model
                    or_model = _next_openrouter_model(or_model)
                    wait = 60 if status == 429 else 20
                    print(f"[LLM] OpenRouter {status} — rotating {old} -> {or_model}, waiting {wait}s")
                    time.sleep(wait)
                elif status in (400, 404):
                    old = or_model
                    or_model = _next_openrouter_model(or_model)
                    print(f"[LLM] OpenRouter {status} — rotating {old} -> {or_model}")
                    time.sleep(5)
                else:
                    print(f"[LLM] OpenRouter non-retryable {status}: {e}")
                    break
            except requests.Timeout:
                old = or_model
                or_model = _next_openrouter_model(or_model)
                print(f"[LLM] OpenRouter timeout — rotating {old} -> {or_model}, waiting 10s")
                time.sleep(10)
            except ValueError as e:
                old = or_model
                or_model = _next_openrouter_model(or_model)
                print(f"[LLM] OpenRouter bad response: {e} — rotating {old} -> {or_model}")
                time.sleep(10)
    else:
        print("[LLM] No OPENROUTER_API_KEY set — OpenRouter fallback disabled.")

    raise RuntimeError(
        "All LLM providers failed. "
        "Make sure Ollama is running (`ollama serve`) and the model is pulled "
        f"(`ollama pull {ollama_model}`)."
    )


# Keep llm_chat as alias for compatibility
def llm_chat(messages, model=None, temperature=0.3, max_tokens=2000):
    return llm_chat_with_retry(messages, model, temperature, max_tokens)


# ── JSON extractor ─────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0; end = 0; in_string = False; escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:              escape_next = False; continue
        if ch == "\\" and in_string: escape_next = True; continue
        if ch == '"':               in_string = not in_string; continue
        if in_string:                continue
        if ch == "{":                depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:           end = i; break

    if not end:
        json_str = text[start:]
        # Close any open string first (truncation inside a quoted value)
        # e.g. "2026-03-25  →  "2026-03-25"
        if json_str.count('"') % 2 != 0:
            json_str += '"'
        open_braces   = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")
        # Strip the last incomplete key-value pair or array element
        json_str = re.sub(r',?\s*"[^"]*"?\s*:\s*[^,{\["\n]*"?[^"\n]*$', "", json_str)
        json_str = re.sub(r',?\s*"[^"]*"?\s*$', "", json_str)
        json_str = json_str.rstrip(", \t\n")
        open_braces   = max(0, json_str.count("{") - json_str.count("}"))
        open_brackets = max(0, json_str.count("[") - json_str.count("]"))
        json_str += "]" * open_brackets + "}" * open_braces
        try:
            return json.loads(json_str)
        except Exception:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", json_str))
            except Exception:
                return None

    json_str = text[start:end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", json_str))
        except Exception:
            return None
