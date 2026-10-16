# VoxCPM2 GPU TTS on Jetson Xavier NX — full build & run guide

Everything needed to reproduce the fastest zero-shot voice-cloning TTS service that
runs on a Jetson Xavier NX (JetPack 5, CUDA 11.4, Volta sm_72): the
[audio.cpp](https://github.com/0xShug0/audio.cpp) engine, the patches in this repo, a
Q4_K GGUF of [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), and a thin
streaming HTTP adapter.

Measured reference numbers on a production Xavier NX (with a whisper.cpp STT server
and a voice agent co-resident): warm short reply ~3.3 s, streaming first audio
~0.62 s, 15 s paragraph at RTF ~1.1, cold start ~17 s.

---

## 1. Architecture

```
your app ──GET /v1/audio/stream?input=hello──▶ tts_adapter.py (:8399)
                                                 │  POST /v1/audio/speech {stream:true, stream_format:"audio", response_format:"pcm"}
                                                 ▼
                                          audiocpp_server (:8402)  ←─ VoxCPM2 Q4_K GGUF, CUDA backend
                                                 │
                                                 ▼  chunked 48 kHz s16le mono PCM
                              adapter resamples 48k→24k, wraps in WAV header, streams to your app
```

The engine is pure C++/ggml — no Python, no PyTorch at runtime. The GGUF is
self-contained (tokenizer/config sidecars embedded): LM at Q4_K (~1.6 GB file),
AudioVAE kept in f16.

## 2. Prerequisites (Jetson)

- Jetson Xavier NX, JetPack 5.x (L4T 35.x, Ubuntu 20.04, CUDA 11.4). The 8 GB model
  is assumed; everything below also works with swap on the USB SSD (recommended:
  ≥8 GB swapfile).
- The host has no recent cmake and (depending on image) no nvcc — build inside the
  `dustynv/l4t-pytorch:2.2-r35.4.1` container (ships CUDA 11.4 toolkit; add
  `cmake>=3.26`, `ninja`, `gcc-13`/`g++-13` from `ppa:ubuntu-toolchain-r/test`).
  Example builder Dockerfile:

  ```dockerfile
  FROM dustynv/l4t-pytorch:2.2-r35.4.1
  RUN apt-get update && apt-get install -y software-properties-common git \
   && (add-apt-repository -y ppa:ubuntu-toolchain-r/test || true) && apt-get update \
   && apt-get install -y gcc-13 g++-13
  RUN python3 -m pip install -q --upgrade pip && python3 -m pip install -q "cmake>=3.26" ninja
  ```
- Python ≥3.9 for the adapter (≤3.12 unless you `pip install audioop-lts` — 3.13
  removed `audioop`).
- ~10 GB free disk (weights + GGUF + build tree).

## 3. Get and prepare the weights

```bash
git clone https://huggingface.co/openbmb/VoxCPM2   # or huggingface-cli download
```

Required files: `model.safetensors`, `audiovae.pth`, `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `tokenization_voxcpm2.py`, `special_tokens_map.json`.

**Gotcha:** `audiovae.pth` stores `decoder.sr_bin_boundaries` as int32, which chokes
the safetensors/ggml path. Convert to safetensors first, casting ints to f32:

```python
import torch
from safetensors.torch import save_file

sd = torch.load("VoxCPM2/audiovae.pth", map_location="cpu")
if "state_dict" in sd:
    sd = sd["state_dict"]
sd = {k: (v.float() if v.dtype in (torch.int32, torch.int64) else v)
      for k, v in sd.items()}
save_file(sd, "VoxCPM2/audiovae.safetensors")
```

## 4. Build audio.cpp with the patches

```bash
curl -L https://codeload.github.com/0xShug0/audio.cpp/tar.gz/refs/heads/main | tar xz
mv audio.cpp-main src
patch -d src -p1 < patches/0001-cuda114-toolkit-floor.patch   # CUDA 11.4 floor
patch -d src -p1 < patches/0002-carmel-isa-flags.patch        # Carmel ISA flags
patch -d src -p1 < patches/0003-voxcpm2-jetson-xavier-performance.patch

cmake -S src -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_CUDA=ON \
  -DGGML_CUDA_NO_VMM=ON \
  -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm2
ninja -C build audiocpp_cli audiocpp_server audiocpp_gguf
```

Why the flags:

- **`-DGGML_CUDA_NO_VMM=ON`** — ggml-cuda allocates via `cuMemCreate`/`cuMemMap` (CUDA
  VMM API), which is broken on Tegra CUDA 11.4 (fails at ~357 MB with gigabytes free).
  Plain `cudaMalloc` works. (On desktop GPUs with working VMM, leave the default.)
- **`-DCMAKE_CUDA_ARCHITECTURES=72`** — Volta. `native` also works on the board.
- **`-DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm2`** — lean voxcpm2-only
  composite; faster to compile, smaller binaries.
- **Clock sanity:** if ninja loops on "manifest still dirty after 100 tries", the
  board's clock has reset (no RTC): `sudo date -s ...` / fix NTP first.

## 5. Convert to GGUF

```bash
./build/bin/audiocpp_gguf \
  --input weights=VoxCPM2/model.safetensors \
  --input audiovae_weights=VoxCPM2/audiovae.safetensors \
  --output voxcpm2-q4_k.gguf \
  --type q4_k \
  --keep-type "audiovae*=f16" \
  --family voxcpm2 \
  --root VoxCPM2
```

Quantizer notes, measured on the Xavier: **Q4_K wins** — Q4_0 was *slower* than Q4_K
on Volta (K-quants dequantize better there). Q8_0 on CPU is RTF ~37 — this is a GPU
project. The VAE stays f16 (`--keep-type "audiovae*=f16"`): small and
quality-sensitive. Speaker-identity cosine similarity vs the reference clip measured
0.888 (steps=4) / 0.900 (steps=10) — Q4_K is already excellent.

## 6. Reference voice

8–10 s, mono, 16-bit PCM, 22050 Hz WAV, one clean speaker, no music/noise. Longer is
not meaningfully better; clean is everything. **Do not publish your reference clip**
if it is a real person's voice — keep it local.

## 7. Server configuration

Use `config/voxcpm2_server.json` from this repo (adjust paths). Field-by-field:

- `"mode": "streaming"` — required for chunked output.
- `"retry_badcase": false` — **REQUIRED for streaming**; the engine's bad-case retry
  is incompatible with streaming mode and hangs/errors otherwise.
- `"num_inference_steps": 6` — the production sweet spot, measured on the Xavier:
  steps=10 is ~2× slower for +0.012 identity; steps=4 is fastest but a whisper
  round-trip battery flagged 1/10 short clips as garbled; **steps ≤3 silently drops
  content on paragraphs (rejected)**; steps=6 passed 10/10 clean at ~+35–50 % wall
  over steps=4.
- `guidance_scale` — keep the default 2.0. **Measured: guidance 1.0 breaks EOS
  discipline — the model rambles and never stops.**
- `"voxcpm2.mem_saver": false` — ~2× faster warm (holds all compute graphs resident).
  Flip to `true` only if you co-locate other big models and accept ~4 s/request.
- `"voxcpm2.audiovae_encoder_sample_capacity": 192000` — VAE encoder headroom
  (8 s × 24 kHz).
- `"threads": 1` — CPU-side ops; the model is GPU-bound.
- `"lazy_load": true` — model loads on first request; keeps container boot instant
  and avoids boot-time memory races with co-resident services.
- `"default_voice_preset"` — pins the reference voice so clients don't send one per
  request.

### Streaming API contract

`audiocpp_server` already streams chunked audio on the stock endpoint:

```http
POST /v1/audio/speech
{"model": "voxcpm2", "input": "Hello there",
 "stream": true, "stream_format": "audio", "response_format": "pcm"}
```

→ chunked raw 48 kHz s16le mono PCM. Without `"stream": true` the response is
buffered and TTFB equals total generation time.

## 8. The adapter

`scripts/tts_adapter.py` (stdlib-only) serves `GET /v1/audio/stream?input=...` on
:8399 and proxies the engine's streaming endpoint, resampling 48 kHz→24 kHz on the
fly with *stateful* `audioop.ratecv`. Non-obvious details it handles: fake max-size
WAV header for unknown length, odd-byte guard so a network chunk never splits a
16-bit sample, `Connection: close` as body terminator, 600 s timeout.

## 8a. The assistant brain (LLM bridge) — bring your own provider

For a full voice-assistant pipeline you need an LLM behind the agent.
`scripts/llm_bridge.py` provides `POST /ask {"text": ...}` → `{"reply": ...}` against
OpenAI, Anthropic, Moonshot, Mistral, LM Studio, or any OpenAI-compatible endpoint,
configured ONLY via environment / Docker variables:

```bash
export LLM_PROVIDER=anthropic            # or openai / moonshot / mistral / lmstudio / openai_compatible
export LLM_API_KEY=...                   # set in the environment or .env — never in code or git
python3 scripts/llm_bridge.py            # :8401 — POST /ask, GET /health
```

Per-provider default endpoints and models are built in (see the README table);
override with `LLM_BASE_URL` / `LLM_MODEL`. LM Studio and other local endpoints need
no key. Tunables: `LLM_SYSTEM_PROMPT` (defaults to short speakable answers),
`LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, `LLM_HISTORY` (in-memory turn count),
`LISTEN_PORT`. The bridge exits with a clear error if a cloud provider is selected
without `LLM_API_KEY`. `bench/test_llm_bridge.py` verifies every provider's request
shape against a local mock — run it after any change: `python3 bench/test_llm_bridge.py`.

`docker-compose.example.yml` wires the whole on-device stack (TTS engine + adapter +
LLM bridge) and injects secrets from an untracked `.env` (template: `.env.example`).

## 8b. Watchdog (optional but recommended on-device)

Docker's `--restart unless-stopped` only covers crashes. Two real failure modes slip
through: the voice agent wedging while alive (process pinned at high CPU, log silent —
the wake word stops answering), and a service container showing "Up" while its HTTP
endpoint is dead. `scripts/voice_watchdog.py` covers both. It runs in a small
container with the docker socket mounted, checks every minute, and restarts the
offender — rate-limited to one restart per service per 10 minutes.

```bash
docker run -d --name voice-watchdog --network host --restart unless-stopped \
  --entrypoint python3 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /opt/voxcpm2/scripts:/voice:ro \
  python:3.12-slim /voice/voice_watchdog.py
```

Defaults calibrated on the Xavier: wedge = >22% sustained container CPU plus zero log
output for 10 minutes, twice in a row (healthy wake-word listening sits at ~14-15%);
health checks restart after 3 consecutive dead minutes. Override with
`WATCHDOG_WEDGE_CPU_PCT`, `WATCHDOG_LOG_SILENCE_MIN`, `WATCHDOG_WEDGE_CHECKS`,
`WATCHDOG_HEALTH_FAILS`, `WATCHDOG_RESTART_COOLDOWN_S`. Edit the HEALTH map at the top
of the script to match your service names and ports. Follow its decisions with
`docker logs voice-watchdog`.

## 9. The Tegra memory shim (you will need it on a lived-in box)

Tegra's NvMap fails large `cudaMalloc`s on *contiguity*, not total memory — a
157 MB graph buffer can fail with gigabytes free. `scripts/cuda_compact_shim.c`
interposes `cudaMalloc`, and on failure runs `drop_caches` + `compact_memory` and
retries twice; critically, it clears the latched `cudaErrorOutOfMemory` via
`cudaGetLastError()` so a later ggml op doesn't abort on the stale error.

```bash
gcc -shared -fPIC -O2 -I/usr/local/cuda/include cuda_compact_shim.c -ldl -lcudart -o cuda_compact_shim.so
# run with privilege to write /proc/sys/vm:
LD_PRELOAD=./cuda_compact_shim.so audiocpp_server --config voxcpm2_server.json --no-ui
```

Docker reference (note `--privileged`, `--runtime nvidia`, host network):

```bash
docker run -d --name tts-voxcpm --runtime nvidia --privileged --network host \
  --restart unless-stopped -e LD_PRELOAD=/bench/cuda_compact_shim.so \
  -v /opt/voxcpm2:/bench -v /opt/voxcpm2/build/bin:/build/bin \
  <your-builder-image> /build/bin/audiocpp_server --config /bench/voxcpm2_server.json --no-ui
```

## 10. Verify (don't skip)

```bash
curl -s http://127.0.0.1:8402/health

# buffered baseline
time curl -s http://127.0.0.1:8402/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"voxcpm2","input":"Hoi. Dit is een test."}' -o /tmp/t1.wav

# streaming (watch first-byte time)
time curl -sN http://127.0.0.1:8402/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"voxcpm2","input":"Hoi. Dit is een test.","stream":true,"stream_format":"audio","response_format":"pcm"}' \
  -o /tmp/t2.pcm

# adapter path (24 kHz WAV stream)
time curl -sN "http://127.0.0.1:8399/v1/audio/stream?input=Hoi.%20Dit%20is%20een%20test." -o /tmp/t3.wav
```

Bench harness for repeatable numbers: `bench/bench_vx.py <label>` (edit the engine
address at the top if needed; it reports warm short-reply wall, streaming TTFA, and
paragraph RTF). What to record: cold start, warm short-reply wall, streaming
first-audio, paragraph RTF, and — by listening, not by file size — a long paragraph
with **no dropped content** (the steps≤3 failure mode). Optional objective check:
speaker-encoder cosine similarity vs the reference clip (acceptance floor 0.875;
Q4_K steps=4 scores ~0.888).

## 11. Fix log (what was broken, what fixed it)

1. **CUDA VMM broken on Tegra** → `-DGGML_CUDA_NO_VMM=ON`.
2. **Fragmentation OOMs** (NvMap contiguity, not total RAM) → `cuda_compact_shim.so`
   LD_PRELOAD. Symptom: `cudaMalloc failed: out of memory` for ~150–360 MB buffers
   with GBs free.
3. **CUDA 12 compat drivers don't work on Jetson r35** (`cuda-compat-12-2`/`-12-4`
   fail `cuInit(100)`) → stuck at 11.4, hence patch 0001.
4. **gcc-13 dropped `-mcpu=carmel`** → broken armv8-a baseline, fp16/NEON mismatch in
   ggml-cpu → patch 0002.
5. **Clock reset to 2000-01-01** (no RTC) → ninja regen loop; check `date` first on
   any haunted build.
6. **`retry_badcase=false` is mandatory for streaming.**
7. **Steps ladder:** 10 best identity/2× slower; 4 = production; ≤3 = content loss.
8. **guidance < 2.0 breaks EOS.** Never lower it.
9. **mem_saver=false ≈ 2× faster warm.**
10. **Q4_K > Q4_0 on Volta.**
11. **Decode-path hotspots fixed by patch 0003** (all measured on-box, output
    bit-identical): full-`max_length` f32 KV attention per token (the biggest one —
    the cache was contiguous-copied and f32→f16-converted in full every token),
    1.2 GB of per-request KV zero-fills, per-token single embedding graph launches,
    over-broad projection outputs, per-step DiT conditioning re-uploads, and
    `ggml_repeat` materialization in the AudioVAE snake activations.

### Dead ends (measured, don't repeat)

- ggml CUDA graphs on sm_72: runtime-disabled by arch check; force-enabling via
  `GGML_CUDA_GRAPHS_PRE_AMPERE=1` crashes at `cudaGraphInstantiate` (Tegra allocator).
- CPU inference (Q8_0): RTF ~37.
- Vulkan: no glslc for aarch64 on Ubuntu 20.04 (LunarG SDK is x86-only).
- fp16 KV caches: after patch 0003 the remaining attention traffic is tiny; the
  accuracy risk isn't worth it. Same for fused mega-graphs — without CUDA-graph
  replay the sync savings are small.
