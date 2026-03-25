import requests
import itertools
import time

API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://localhost",
    "X-Title": "Travel Guide",
}

# ── Auto-discover working free models ─────────────────────────────────────────

def get_free_models():
    r = requests.get(f"{BASE_URL}/models", headers=HEADERS)
    r.raise_for_status()
    return [
        m["id"] for m in r.json()["data"]
        if m.get("pricing", {}).get("prompt") in ("0", 0)
        and m.get("pricing", {}).get("completion") in ("0", 0)
    ]

def probe_model(model_id):
    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions", headers=HEADERS,
            json={"model": model_id, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
            timeout=15,
        )
        return r.status_code == 200
    except:
        return False

def discover_working_models():
    print("🔍 Discovering working free models...")
    all_free = get_free_models()
    working = []
    for m in all_free:
        ok = probe_model(m)
        print(f"  {'✅' if ok else '❌'} {m}")
        if ok:
            working.append(m)
        time.sleep(0.4)
    print(f"\n✅ {len(working)}/{len(all_free)} models available\n")
    return working

# ── Rotator class ──────────────────────────────────────────────────────────────

class OpenRouterRotator:
    def __init__(self, models: list):
        self.models = models
        self._cycle = itertools.cycle(models)
        self.stats = {m: {"success": 0, "fail": 0} for m in models}

    def chat(self, messages: list, max_retries: int = None) -> str:
        max_retries = max_retries or len(self.models)
        for _ in range(max_retries):
            model = next(self._cycle)
            try:
                r = requests.post(
                    f"{BASE_URL}/chat/completions", headers=HEADERS,
                    json={"model": model, "messages": messages, "max_tokens": 1000},
                    timeout=30,
                )
                if r.status_code == 200:
                    self.stats[model]["success"] += 1
                    content = r.json()["choices"][0]["message"]["content"]
                    print(f"[{model}] ✅")
                    return content
                else:
                    self.stats[model]["fail"] += 1
                    print(f"[{model}] ❌ {r.status_code} — rotating...")
            except Exception as e:
                self.stats[model]["fail"] += 1
                print(f"[{model}] 💥 {e} — rotating...")
        raise RuntimeError("All models failed.")

    def print_stats(self):
        print("\n📊 Model Stats:")
        for m, s in self.stats.items():
            print(f"  {m}: ✅{s['success']} ❌{s['fail']}")

# ── Usage ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    working_models = discover_working_models()
    rotator = OpenRouterRotator(working_models)

    # Example usage in your travel guide
    response = rotator.chat([
        {"role": "system", "content": "You are a helpful travel guide assistant."},
        {"role": "user", "content": "Suggest 3 must-visit places in Kerala, India."}
    ])
    print("\n", response)

    rotator.print_stats()