"""
ai_engine/llm.py  — v3
========================
Uses Google Gemini API directly (primary) with OpenRouter as fallback.

Free limits (March 2026):
  gemini-2.5-flash-lite  → 1000 req/day, 15 RPM  ← PRIMARY (most generous)
  gemini-2.5-flash       → 250 req/day,  10 RPM  ← FALLBACK
  gemini-2.5-pro         → 100 req/day,   5 RPM  ← FALLBACK

OpenRouter free tier: 50 req/day total (last resort)
"""
import os, re, json, time, requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1/chat/completions"

# Gemini REST endpoint
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Primary Gemini model from .env, with fallbacks
_gemini_primary = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODELS = list(dict.fromkeys([  # deduplicate while preserving order
    _gemini_primary,
    "gemini-2.5-flash",              # primary — confirmed working
    "gemini-2.0-flash",              # fallback
    "gemini-2.0-flash-lite",         # fallback
]))

# OpenRouter fallback model from .env
_or_primary = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
OPENROUTER_MODELS = list(dict.fromkeys([
    _or_primary,
    "openrouter/auto",                           # auto-picks best available
    "nvidia/nemotron-3-super-120b-a12b:free",   # ✅ confirmed working
    "stepfun/step-3.5-flash:free",               # ✅ confirmed working
    "arcee-ai/trinity-large-preview:free",       # ✅ confirmed working
    "nvidia/nemotron-nano-9b-v2:free",           # ✅ confirmed working
    "z-ai/glm-4.5-air:free",                    # ✅ confirmed working
    "google/gemma-3n-e4b-it:free",              # ✅ confirmed working
]))

_gemini_model_idx     = 0
_openrouter_model_idx = 0
_gemini_session_dead  = False   # True once ALL Gemini models hit 429 this session
_gemini_429_count     = 0       # consecutive 429s — after threshold, skip Gemini entirely


def _next_gemini_model(current: str = "") -> str:
    global _gemini_model_idx
    for _ in range(len(GEMINI_MODELS)):
        _gemini_model_idx = (_gemini_model_idx + 1) % len(GEMINI_MODELS)
        if GEMINI_MODELS[_gemini_model_idx] != current:
            return GEMINI_MODELS[_gemini_model_idx]
    return GEMINI_MODELS[0]


def _next_openrouter_model(current: str = "") -> str:
    global _openrouter_model_idx
    for _ in range(len(OPENROUTER_MODELS)):
        _openrouter_model_idx = (_openrouter_model_idx + 1) % len(OPENROUTER_MODELS)
        if OPENROUTER_MODELS[_openrouter_model_idx] != current:
            return OPENROUTER_MODELS[_openrouter_model_idx]
    return OPENROUTER_MODELS[0]


# ── Gemini API call ────────────────────────────────────────────────────────────
def _gemini_chat(messages: list, model: str,
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call Gemini REST API directly."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")

    # Convert OpenAI-style messages to Gemini format
    gemini_contents = []
    system_text = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            system_text = content  # Gemini handles system separately
        elif role == "user":
            gemini_contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            gemini_contents.append({"role": "model", "parts": [{"text": content}]})

    payload = {
        "contents": gemini_contents,
        "generationConfig": {
            "temperature":     temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
    resp = requests.post(url, json=payload, timeout=60)

    if resp.status_code == 429:
        raise requests.HTTPError("429 rate limit", response=resp)
    if resp.status_code == 404:
        raise requests.HTTPError(f"404 model not found: {model}", response=resp)
    if resp.status_code != 200:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp)

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates in Gemini response: {str(data)[:200]}")

    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    if not text:
        raise ValueError("Empty text in Gemini response")
    return text


# ── OpenRouter call ────────────────────────────────────────────────────────────
def _openrouter_chat(messages: list, model: str,
                     temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    key = OPENROUTER_API_KEY.strip()
    if not key.startswith("sk-or-"):
        raise ValueError(f"Invalid OPENROUTER_API_KEY format")

    resp = requests.post(
        OPENROUTER_BASE,
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": os.getenv("FRONTEND_URL", "http://localhost:3000"),
                 "X-Title": "TravelCopilot"},
        timeout=30,  # fail fast — rotate to next model sooner
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
    Try Gemini first (1000 free req/day), fall back to OpenRouter.
    """
    # ── Try Gemini ─────────────────────────────────────────────────────────────
    global _gemini_session_dead, _gemini_429_count
    if GEMINI_API_KEY and not _gemini_session_dead:
        gemini_model = GEMINI_MODELS[0]
        dead_models   = set()   # models that returned 404 — never retry these
        local_429s    = 0       # 429s in this call
        for attempt in range(3):   # max 3 attempts — don't waste time if 429ing
            # Skip dead models
            while gemini_model in dead_models:
                gemini_model = _next_gemini_model(gemini_model)
                if all(m in dead_models for m in GEMINI_MODELS):
                    print("[LLM] All Gemini models dead (404) — falling back to OpenRouter")
                    break
            else:
                try:
                    print(f"[LLM] Gemini attempt {attempt+1} — {gemini_model}")
                    result = _gemini_chat(messages, gemini_model, temperature, max_tokens)
                    print(f"[LLM] ✅ Gemini success ({gemini_model})")
                    # Reset session 429 counter on success
                    _gemini_429_count = 0
                    return result
                except requests.HTTPError as e:
                    status = 0
                    if e.response is not None:
                        status = e.response.status_code
                    else:
                        m = re.search(r"(\d{3})", str(e))
                        if m: status = int(m.group(1))
                    old_model = gemini_model
                    if status == 404:
                        dead_models.add(gemini_model)
                        gemini_model = _next_gemini_model(gemini_model)
                        print(f"[LLM] Gemini 404 — {old_model} is dead, trying {gemini_model}")
                        continue   # immediate retry with new model, no sleep
                    elif status == 429:
                        local_429s += 1
                        _gemini_429_count += 1
                        gemini_model = _next_gemini_model(gemini_model)
                        # If we've had 3+ consecutive 429s across all models in this call,
                        # or 6+ total this session — mark Gemini dead for the session
                        if local_429s >= 3 or _gemini_429_count >= 6:
                            _gemini_session_dead = True
                            print(f"[LLM] Gemini 429 threshold reached ({_gemini_429_count} total) "
                                  f"— skipping Gemini for rest of session, going straight to OpenRouter")
                            break
                        wait = 5   # short wait only — we bail fast now
                        print(f"[LLM] Gemini 429 — rotating {old_model} → {gemini_model}, waiting {wait}s")
                        time.sleep(wait)
                    else:
                        gemini_model = _next_gemini_model(gemini_model)
                        print(f"[LLM] Gemini HTTP {status} — rotating {old_model} → {gemini_model}")
                        time.sleep(2)
                except Exception as e:
                    print(f"[LLM] Gemini exception: {e}")
                    gemini_model = _next_gemini_model(gemini_model)
                    time.sleep(2)
                continue
            break  # all models dead
    elif _gemini_session_dead:
        print("[LLM] Gemini session-dead (429 limit) — going straight to OpenRouter")

    # ── Fall back to OpenRouter ────────────────────────────────────────────────
    if OPENROUTER_API_KEY:
        or_model = model or OPENROUTER_MODELS[0]
        for attempt in range(max_retries):
            try:
                print(f"[LLM] OpenRouter attempt {attempt+1} — {or_model}")
                time.sleep(3)  # respect free tier rate limit
                result = _openrouter_chat(messages, or_model, temperature, max_tokens)
                print(f"[LLM] ✅ OpenRouter success ({or_model})")
                return result
            except requests.HTTPError as e:
                status = e.response.status_code if e.response else 0
                if status in (429, 502, 503):
                    old = or_model
                    or_model = _next_openrouter_model(or_model)
                    wait = 60 if status == 429 else 20
                    print(f"[LLM] OpenRouter {status} — rotating {old} → {or_model}, waiting {wait}s")
                    time.sleep(wait)
                elif status in (400, 404):
                    old = or_model
                    or_model = _next_openrouter_model(or_model)
                    print(f"[LLM] OpenRouter {status} — rotating {old} → {or_model}")
                    time.sleep(5)
                elif status == 0:
                    # No response object — network error, retry with delay
                    old = or_model
                    or_model = _next_openrouter_model(or_model)
                    print(f"[LLM] OpenRouter network error — rotating {old} → {or_model}, waiting 10s")
                    time.sleep(10)
                else:
                    print(f"[LLM] OpenRouter non-retryable {status}: {e}")
                    break
            except requests.Timeout:
                old = or_model
                or_model = _next_openrouter_model(or_model)
                print(f"[LLM] OpenRouter timeout — rotating {old} → {or_model}, waiting 10s")
                time.sleep(10)
            except ValueError as e:
                old = or_model
                or_model = _next_openrouter_model(or_model)
                print(f"[LLM] OpenRouter bad response — rotating {old} → {or_model}")
                time.sleep(10)

    raise RuntimeError("All LLM providers failed. Check GEMINI_API_KEY and OPENROUTER_API_KEY.")


# Keep llm_chat as alias for compatibility
def llm_chat(messages, model=None, temperature=0.3, max_tokens=2000):
    return llm_chat_with_retry(messages, model, temperature, max_tokens)


# ── JSON extractor ─────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    # Aggressively strip ALL markdown fences and surrounding text
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Find first {
    start = text.find("{")
    if start == -1:
        return None

    # Walk to find matching closing }
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
        # JSON got truncated — try to auto-close it
        json_str = text[start:]
        open_braces  = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")
        # Remove trailing incomplete key/value
        json_str = re.sub(r',?\s*"[^"]*$', "", json_str)
        json_str = re.sub(r',?\s*"[^"]*"\s*:\s*[^,{\[]*$', "", json_str)
        # Close open structures
        json_str += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
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