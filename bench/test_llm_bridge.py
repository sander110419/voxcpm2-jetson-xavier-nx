#!/usr/bin/env python3
"""Mock-LLM test for llm_bridge.py: verifies request shape per provider."""
import json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAPTURED = {}

class MockLLM(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        CAPTURED["path"] = self.path
        CAPTURED["headers"] = dict(self.headers)
        CAPTURED["body"] = json.loads(body or b"{}")
        if self.path.endswith("/messages"):  # anthropic shape
            resp = {"content": [{"type": "text", "text": "mock anthropic reply"}]}
        else:
            resp = {"choices": [{"message": {"role": "assistant",
                                             "content": "mock openai-style reply"}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

mock = ThreadingHTTPServer(("127.0.0.1", 9_999), MockLLM)
threading.Thread(target=mock.serve_forever, daemon=True).start()

BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "llm_bridge.py")
fails = 0

def run_case(provider, extra_env, expect_path, expect_auth, expect_key_in_headers):
    global fails
    env = dict(os.environ)
    env.update({"LLM_PROVIDER": provider, "LLM_BASE_URL": "http://127.0.0.1:9999",
                    "LLM_MODEL": "test-model", "LISTEN_PORT": "8411"})
    env.update(extra_env)
    p = subprocess.Popen([sys.executable, BRIDGE], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import urllib.request
        ok = False
        for _ in range(50):
            try:
                urllib.request.urlopen("http://127.0.0.1:8411/health", timeout=1)
                ok = True; break
            except Exception:
                time.sleep(0.1)
        assert ok, "bridge did not start"
        req = urllib.request.Request("http://127.0.0.1:8411/ask",
                                     data=json.dumps({"text": "Hoi"}).encode(),
                                     headers={"Content-Type": "application/json"})
        reply = json.loads(urllib.request.urlopen(req, timeout=10).read())["reply"]
        path_ok = CAPTURED["path"] == expect_path
        auth = CAPTURED["headers"].get("Authorization", "")
        xkey = CAPTURED["headers"].get("X-Api-Key", CAPTURED["headers"].get("x-api-key", ""))
        auth_ok = (auth == expect_auth) if expect_auth is not None else True
        key_ok = (xkey == expect_key_in_headers) if expect_key_in_headers is not None else True
        model_ok = CAPTURED["body"].get("model") == "test-model"
        status = "OK " if (path_ok and auth_ok and key_ok and model_ok) else "FAIL"
        if status == "FAIL": fails += 1
        print(f"{status} {provider}: path={CAPTURED['path']} auth={auth or '-'} x-api-key={xkey or '-'} reply={reply[:30]!r}")
    finally:
        p.terminate(); p.wait()

run_case("openai", {"LLM_API_KEY": "test-placeholder-key"}, "/chat/completions", "Bearer test-placeholder-key", None)
run_case("moonshot", {"LLM_API_KEY": "test-placeholder-key"}, "/chat/completions", "Bearer test-placeholder-key", None)
run_case("mistral", {"LLM_API_KEY": "test-placeholder-key"}, "/chat/completions", "Bearer test-placeholder-key", None)
run_case("lmstudio", {}, "/chat/completions", "", None)  # no key -> no Authorization header expected
run_case("anthropic", {"LLM_API_KEY": "test-placeholder-key"}, "/messages", None, "test-placeholder-key")

# provider without key must refuse to start
env = dict(os.environ); env.update({"LLM_PROVIDER": "openai", "LLM_API_KEY": ""})
p = subprocess.run([sys.executable, BRIDGE], env=env, capture_output=True, timeout=10)
refused = p.returncode != 0 and b"LLM_API_KEY is required" in p.stderr
print("OK  openai-without-key refuses to start" if refused else "FAIL openai-without-key did not refuse")
fails += 0 if refused else 1

sys.exit(1 if fails else 0)
