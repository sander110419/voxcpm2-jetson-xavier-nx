# VoxCPM2 on Jetson Xavier NX — the fastest zero-shot voice-cloning TTS pipeline on this board

Zero-shot, multilingual (incl. Dutch) voice-cloning TTS running **entirely on a 15 W
Jetson Xavier NX** (Volta sm_72, CUDA 11.4, JetPack 5) — no cloud, no desktop GPU.
Built on [audio.cpp](https://github.com/0xShug0/audio.cpp) (pure C++/ggml, no Python
runtime) and [OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

**Measured on a lived-in Xavier NX (full voice stack resident), before → after this
patch series, output bit-identical:**

| metric | stock audio.cpp | + these patches | Δ |
|---|---|---|---|
| warm short reply (2.7 s audio) | 5.29 s | **3.28 s** | −38 % |
| streaming time-to-first-audio | 0.96 s | **0.62 s** | −35 % |
| paragraph, 15.2 s audio | 25.4 s (RTF 1.67) | **16.6 s (RTF 1.09)** | −35 % |
| cold start (model load + first request) | ~29 s | **17.5 s** | −40 % |
| per-request KV cache allocation | ~605 MB | **19–38 MB** | −94 % |

The ~37.6×-RTF CPU lane and the "no LLM-class TTS runs on Xavier" folklore are out of
date: with the CUDA backend patched to build on JP5 and the decode path fixed, a
2.4 B-param zero-shot cloning model answers a short line in ~3 s and speaks its first
audio in ~0.6 s.

## What's here

- `patches/0001-cuda114-toolkit-floor.patch` — build audio.cpp against CUDA 11.4
  (upstream floor is 12.0).
- `patches/0002-carmel-isa-flags.patch` — hardcode `armv8.2-a+fp16+dotprod`
  (gcc-13 dropped `-mcpu=carmel`; the native probe yields a broken baseline).
- `patches/0003-voxcpm2-jetson-xavier-performance.patch` — the performance series
  (details below). Numerically exact: before/after PCM output is bit-identical
  (same fixed seed), verified with per-sample diff on multiple texts.
- `config/voxcpm2_server.json` — production server config (streaming, Q4_K GGUF).
- `scripts/tts_adapter.py` — stdlib-only streaming HTTP adapter:
  `GET /v1/audio/stream?input=...` → 24 kHz WAV stream, ready to front a voice agent.
- `scripts/llm_bridge.py` — the assistant brain: `POST /ask {"text": ...}` →
  `{"reply": ...}` against **your** LLM provider (OpenAI, Anthropic, Moonshot,
  Mistral, LM Studio, or any OpenAI-compatible endpoint), configured entirely by
  environment variables — no API key ever lives in code (see below).
- `scripts/cuda_compact_shim.c` — `LD_PRELOAD` interposer that retries failed
  `cudaMalloc` after cache-drop + memory compaction (Tegra NvMap contiguity fix).
- `bench/bench_vx.py` — the benchmark harness used for the numbers above.
- `bench/test_llm_bridge.py` — provider-shape tests for the LLM bridge (runs against
  a local mock, no real keys needed).
- `docker-compose.example.yml` + `.env.example` — three-service pipeline (TTS engine,
  adapter, LLM bridge) with all secrets injected from `.env`.
- `docs/GUIDE.md` — full build/convert/run/verify guide with every pitfall included.

## Plug in your own LLM (OpenAI / Anthropic / Moonshot / Mistral / LM Studio)

`scripts/llm_bridge.py` is a stdlib-only service that maps the voice pipeline's
`POST /ask {"text": "..."}` onto your provider of choice:

```bash
# pick one provider and export its key — NEVER put the key in code or git:
export LLM_PROVIDER=openai            # openai | anthropic | moonshot | mistral | lmstudio | openai_compatible
export LLM_API_KEY=sk-...             # not needed for lmstudio / local endpoints
# optional: export LLM_MODEL=..., LLM_BASE_URL=..., LLM_SYSTEM_PROMPT=...
python3 scripts/llm_bridge.py         # serves :8401  (POST /ask, GET /health)
```

Provider defaults (override with `LLM_BASE_URL` / `LLM_MODEL`):

| provider | default endpoint | default model | key needed |
|---|---|---|---|
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` | yes |
| `anthropic` | `https://api.anthropic.com/v1` | `claude-3-5-haiku-latest` | yes |
| `moonshot` | `https://api.moonshot.ai/v1` | `moonshot-v1-8k` | yes |
| `mistral` | `https://api.mistral.ai/v1` | `mistral-small-latest` | yes |
| `lmstudio` | `http://localhost:1234/v1` | loaded model | no |
| `openai_compatible` | `http://localhost:1234/v1` | `local-model` | no |

With Docker, put the key in an untracked `.env` (see `.env.example`) and use
`docker-compose.example.yml` — the compose file injects `LLM_API_KEY` as a container
environment variable only. The bridge refuses to start for cloud providers when the
key is missing, and the request shapes for every provider are covered by
`bench/test_llm_bridge.py` (mock-tested; no real credentials required).

The bridge keeps a short per-session history (`LLM_HISTORY`, default 10 turns) and
defaults to a speakable-answers system prompt — tune with `LLM_SYSTEM_PROMPT`.

## The performance patch, in order of impact

1. **KV step caches sized to the request, not to `max_length`.** The stock decode loop
   allocates both transformer KV caches at the model's 8192-step maximum (f32) and runs
   attention over the *entire* cache every token — with a contiguous-copy and an
   f32→f16 conversion of all 8192 rows per layer per token. The patch sizes the step
   graphs to `prefill_rows + max_tokens` (256-step buckets): a short reply attends over
   ~256 rows instead of 8192. On the 59 GB/s unified-memory bus this was the dominant
   per-token cost (~40 ms/token → <2 ms).
2. **Per-request KV reset/import without the 1.2 GB host round-trip.** Stock code
   zero-fills and uploads both full caches on `reset()` and again on prefill import,
   twice per request. Now: zero once at graph build, counter-only reset, prefix-only
   import. (Also removes the OOM-killer magnet: that transient is what was getting the
   server killed on a co-resident box.)
3. **Batched text embedding.** One `get_rows` graph per request instead of one synced
   single-token graph compute per prompt token.
4. **Split projection graphs.** The per-token "advance" and per-patch "emit" calls now
   compute and download only the outputs they actually consume (2 resp. 3 tensors
   instead of all 8).
5. **DiT/CFM loop hoisting.** Conditioning tensors upload once per patch (not per
   diffusion step); the sinusoidal time-step embeddings are computed once per request
   instead of per patch per step.
6. **AudioVAE decoder:** snake activation and sample-rate conditioning use broadcast
   mul/div instead of materializing a full-size `ggml_repeat` copy per activation.

Notes on what does **not** work on this board (measured, so you don't have to):
ggml CUDA graphs are disabled at runtime on sm_72, and force-enabling them crashes at
`cudaGraphInstantiate` (Tegra allocator contiguity). fp16 KV storage and fused
mega-graphs are the remaining ideas; see `docs/GUIDE.md` for the analysis.

## Quickstart

```bash
# 1. Build audio.cpp for the Xavier (inside a dustynv/l4t-pytorch container):
patch -d src -p1 < patches/0001-cuda114-toolkit-floor.patch
patch -d src -p1 < patches/0002-carmel-isa-flags.patch
patch -d src -p1 < patches/0003-voxcpm2-jetson-xavier-performance.patch
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm2
ninja -C build audiocpp_cli audiocpp_server audiocpp_gguf

# 2. Convert the weights (or download the ready GGUF from the Hugging Face mirror):
./build/bin/audiocpp_gguf --input weights=VoxCPM2/model.safetensors \
  --input audiovae_weights=VoxCPM2/audiovae.safetensors \
  --output voxcpm2-q4_k.gguf --type q4_k --keep-type "audiovae*=f16" \
  --family voxcpm2 --root VoxCPM2

# 3. Run (config in config/): engine on :8402, adapter on :8399
audiocpp_server --config voxcpm2_server.json --no-ui
python3 scripts/tts_adapter.py

# 4. Optional — the assistant brain, with YOUR provider (keys via env only):
export LLM_PROVIDER=openai   # anthropic / moonshot / mistral / lmstudio / ...
export LLM_API_KEY=sk-...    # never in code or git
python3 scripts/llm_bridge.py   # POST :8401/ask
```

Full instructions, the reference-voice recipe, the streaming API contract, and the
complete fix log are in **[docs/GUIDE.md](docs/GUIDE.md)**.

## License & credits

- Patches and scripts: Apache-2.0, same as upstream
  [audio.cpp](https://github.com/0xShug0/audio.cpp) (© ShugoAI LLC).
- [VoxCPM2](https://github.com/OpenBMB/VoxCPM2) model and weights: Apache-2.0, OpenBMB.
  This project redistributes a derived GGUF quantization on Hugging Face with
  attribution; see the model card there for details.
- Bring your own reference voice: an 8–10 s clean mono WAV. No voice data is included
  in this repository.
