#!/usr/bin/env python3
"""tts_adapter.py — voice-agent <-> audiocpp_server bridge, STREAMING.

Serves the voice-agent's existing contract (GET /v1/audio/stream?input=...)
on :8399 and proxies to audiocpp_server's streaming speech endpoint
(POST :8402/v1/audio/speech, stream_format=audio → chunked raw 48kHz s16le
mono PCM). Each upstream chunk is downsampled to the agent's expected 24kHz
with audioop (stateful) and forwarded immediately — playback starts on the
first engine chunk (~2.5s) instead of after full generation.

A synthetic WAV header with a max-size data chunk is emitted first; the
agent only scans for the 'data' marker and ignores the size field.

Stdlib only (l4t-pytorch py3.10 has audioop). No third-party deps.
"""
import audioop
import http.client
import json
import struct
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 8402
LISTEN_PORT = 8399


def wav_header_24k() -> bytes:
    """RIFF/WAV header, s16le mono 24kHz, with a (fake) max-size data chunk."""
    data_size = 0x7FFFF000
    byte_rate = 24000 * 2
    return (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, byte_rate, 2, 16)
        + b"data" + struct.pack("<I", data_size)
    )


def stream_pcm_chunks(text: str):
    """Yield (ok, chunk) — upstream 48kHz s16le mono PCM chunks, or error."""
    body = json.dumps({
        "model": "voxcpm2", "input": text,
        "stream": True, "stream_format": "audio", "response_format": "pcm",
    }).encode()
    conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=600)
    try:
        conn.request("POST", "/v1/audio/speech", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            yield False, resp.read()[:300]
            return
        yield True, None  # status OK marker — real chunks follow
        while True:
            chunk = resp.read1(32768)
            if not chunk:
                break
            yield True, chunk
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            try:
                with urllib.request.urlopen(
                        f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/health", timeout=3) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            self._send(200 if ok else 502, b'{"status":"ok"}' if ok else b'{"status":"down"}')
            return
        if u.path != "/v1/audio/stream":
            self._send(404, b'{"error":"not found"}')
            return
        text = (parse_qs(u.query).get("input") or [""])[0].strip()
        if not text:
            self._send(400, b'{"error":"empty input"}')
            return

        t0 = time.time()
        gen = stream_pcm_chunks(text)
        ok, first = next(gen)
        if not ok:
            self._send(502, first if first else b'{"error":"upstream"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        # no Content-Length: connection close terminates the body
        self.close_connection = True
        self.end_headers()
        self.wfile.write(wav_header_24k())
        self.wfile.flush()

        state = None  # audioop ratecv state
        total = 0
        first_audio_t = None

        def forward(chunk):
            nonlocal state, total, first_audio_t
            if len(chunk) % 2:
                chunk = chunk[:-1]  # never split a sample
            pcm24, state = audioop.ratecv(chunk, 2, 1, 48000, 24000, state)
            if pcm24:
                if first_audio_t is None:
                    first_audio_t = time.time() - t0
                self.wfile.write(pcm24)
                self.wfile.flush()
                total += len(pcm24)

        try:
            for ok, chunk in gen:  # first yield was the status marker
                if not ok:
                    break
                forward(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        print(f"[adapter] {len(text)} chars -> {total} bytes, "
              f"first-audio {first_audio_t and round(first_audio_t, 2)}s "
              f"total {time.time()-t0:.1f}s", flush=True)

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler).serve_forever()
