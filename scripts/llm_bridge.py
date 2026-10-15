#!/usr/bin/env python3
"""llm_bridge.py — provider-agnostic LLM bridge for the voice pipeline.

The voice agent posts the transcribed utterance and gets a spoken-length reply:

    POST /ask   {"text": "..."}  ->  {"reply": "..."}
    GET  /health                 ->  {"ok": true, "provider": ..., "model": ...}

Everything is configured by environment variables (or `docker run -e ...`),
so no API key ever lives in code or config files:

    LLM_PROVIDER       openai | anthropic | moonshot | mistral | lmstudio | openai_compatible
    LLM_API_KEY        your API key — NOT required for lmstudio / local endpoints
    LLM_BASE_URL       override the provider default endpoint
    LLM_MODEL          model name (per-provider default below)
    LLM_SYSTEM_PROMPT  system prompt (default: short spoken-answer assistant)
    LLM_MAX_TOKENS     default 300
    LLM_TEMPERATURE    default 0.7
    LLM_HISTORY        number of recent turns kept as context (default 10)
    LISTEN_PORT        default 8401

Stdlib only — no dependencies to install.
"""
import json
import os
import ssl
import http.client
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROVIDERS = {
    #            default base URL                          default model                style
    "openai":            ("https://api.openai.com/v1",        "gpt-4o-mini",              "openai"),
    "moonshot":          ("https://api.moonshot.ai/v1",       "moonshot-v1-8k",           "openai"),
    "mistral":           ("https://api.mistral.ai/v1",        "mistral-small-latest",     "openai"),
    "lmstudio":          ("http://localhost:1234/v1",         "local-model",              "openai"),
    "openai_compatible": ("http://localhost:1234/v1",         "local-model",              "openai"),
    "anthropic":         ("https://api.anthropic.com/v1",     "claude-3-5-haiku-latest",  "anthropic"),
}

PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai"
if PROVIDER not in PROVIDERS:
    raise SystemExit(f"LLM_PROVIDER must be one of: {', '.join(sorted(PROVIDERS))}")
DEFAULT_BASE, DEFAULT_MODEL, STYLE = PROVIDERS[PROVIDER]

# Empty-string env vars (e.g. from docker-compose defaults) count as unset.
def env_or(name, default):
    value = os.environ.get(name, "").strip()
    return value if value else default

BASE_URL = env_or("LLM_BASE_URL", DEFAULT_BASE).rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "").strip()
MODEL = env_or("LLM_MODEL", DEFAULT_MODEL)
SYSTEM_PROMPT = env_or(
    "LLM_SYSTEM_PROMPT",
    "You are a voice assistant. Answer in the user's language, in one to four short, "
    "naturally speakable sentences. No markdown, no lists, no code.")
MAX_TOKENS = int(env_or("LLM_MAX_TOKENS", "300"))
TEMPERATURE = float(env_or("LLM_TEMPERATURE", "0.7"))
HISTORY_TURNS = int(env_or("LLM_HISTORY", "10"))
LISTEN_PORT = int(env_or("LISTEN_PORT", "8401"))

if STYLE == "openai" and PROVIDER not in ("lmstudio", "openai_compatible") and not API_KEY:
    raise SystemExit(f"LLM_API_KEY is required for provider '{PROVIDER}' "
                     "(set it as an environment variable, never in code)")

history = []  # [(user, assistant), ...] — single-session, in memory


def _connection(url):
    u = urllib.parse.urlparse(url)
    if u.scheme == "https":
        return u, http.client.HTTPSConnection(u.hostname, u.port or 443,
                                              timeout=120,
                                              context=ssl.create_default_context())
    return u, http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)


def ask_openai_style(text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": text})
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    u, conn = _connection(BASE_URL)
    try:
        conn.request("POST", (u.path or "") + "/chat/completions", body=body,
                     headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"LLM HTTP {resp.status}: {data[:300]!r}")
        out = json.loads(data)
        return out["choices"][0]["message"]["content"].strip()
    finally:
        conn.close()


def ask_anthropic(text):
    messages = []
    for u, a in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": text})
    body = json.dumps({"model": MODEL, "max_tokens": MAX_TOKENS,
                       "system": SYSTEM_PROMPT, "messages": messages}).encode()
    headers = {"Content-Type": "application/json",
               "x-api-key": API_KEY,
               "anthropic-version": "2023-06-01"}
    u, conn = _connection(BASE_URL)
    try:
        conn.request("POST", (u.path or "") + "/messages", body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"Anthropic HTTP {resp.status}: {data[:300]!r}")
        out = json.loads(data)
        return "".join(b.get("text", "") for b in out.get("content", [])).strip()
    finally:
        conn.close()


def ask(text):
    reply = ask_anthropic(text) if STYLE == "anthropic" else ask_openai_style(text)
    history.append((text, reply))
    del history[:-HISTORY_TURNS]
    return reply


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == "/health":
            self._send(200, {"ok": True, "provider": PROVIDER, "model": MODEL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/ask":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = (req.get("text") or "").strip()
            if not text:
                self._send(400, {"error": "empty text"})
                return
            self._send(200, {"reply": ask(text)})
        except Exception as e:  # upstream errors surface as 502 with a snippet
            self._send(502, {"error": str(e)[:300]})


if __name__ == "__main__":
    print(f"[llm_bridge] provider={PROVIDER} model={MODEL} base={BASE_URL} "
          f"key={'set' if API_KEY else 'not-set'} port={LISTEN_PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler).serve_forever()
