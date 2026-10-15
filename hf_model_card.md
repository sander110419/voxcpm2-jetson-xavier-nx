---
license: apache-2.0
base_model: openbmb/VoxCPM2
tags:
  - gguf
  - tts
  - voice-cloning
  - zero-shot
  - jetson
  - audio.cpp
  - voxcpm2
library_name: ggml
---

# VoxCPM2 Q4_K GGUF — tuned for Jetson Xavier NX (and any CUDA GPU)

A self-contained GGUF of [OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
(2.4 B-param multilingual zero-shot voice-cloning TTS, ~30 languages incl. Dutch),
quantized for the
[audio.cpp](https://github.com/0xShug0/audio.cpp) pure-C++/ggml inference engine:

- **LM (base + residual MiniCPM stack) at Q4_K**
- **AudioVAE decoder kept at f16** (quality-sensitive, small)
- tokenizer/config sidecars embedded — one file, no external spec needed

Converted with `audiocpp_gguf` from the original safetensors; all
int32/int64 tensors (e.g. `decoder.sr_bin_boundaries`) cast to f32 so the ggml path
loads cleanly.

## Why this quant

Measured on a Jetson Xavier NX (Volta sm_72): **Q4_K beats Q4_0 on speed**, not just
size (K-quants dequantize better on Volta). Speaker-identity cosine similarity vs the
reference clip (higher = more like the reference): 0.888 at 4 diffusion steps,
0.900 at 10 — already at the quality floor needed for production voice use.

## Measured performance (Jetson Xavier NX, JP5/CUDA 11.4)

With the companion patch series
([github.com/sander110419/voxcpm2-jetson-xavier-nx](https://github.com/sander110419/voxcpm2-jetson-xavier-nx))
on a lived-in Xavier NX with a full voice stack resident:

| metric | value |
|---|---|
| warm short reply (~2.7 s audio) | **3.3 s** total |
| streaming time-to-first-audio | **0.62 s** |
| 15.2 s paragraph | **16.6 s** (RTF ≈ 1.09) |
| cold start (load + first request) | ~17 s |
| per-request KV cache allocation | 19–38 MB |

On a desktop GPU (RTX-class) expect well under RTF 0.5 — the same GGUF and engine
run unmodified there.

## Usage

```bash
# engine (needs audio.cpp; on Jetson apply the patches from the GitHub repo above)
audiocpp_server --config voxcpm2_server.json --no-ui

curl -sN http://127.0.0.1:8402/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"voxcpm2","input":"Hoi. Dit is een test.","stream":true,"stream_format":"audio","response_format":"pcm"}' \
  -o out.pcm   # 48 kHz s16le mono
```

Recommended request/session options: `retry_badcase=false` (mandatory for
streaming), `num_inference_steps=4`, `guidance_scale=2.0` (do not lower — breaks EOS),
`voxcpm2.mem_saver=false` (~2× faster warm). Voice cloning needs an 8–10 s clean mono
22 050 Hz reference WAV (`default_voice_preset.voice_ref` in the server config).

Full build/convert/run guide with all Jetson pitfalls:
[docs/GUIDE.md](https://github.com/sander110419/voxcpm2-jetson-xavier-nx/blob/master/docs/GUIDE.md).
The same repo also carries a small stdlib-only LLM bridge (`scripts/llm_bridge.py`)
that turns the TTS into a full assistant pipeline with your own provider —
OpenAI, Anthropic, Moonshot, Mistral, LM Studio or any OpenAI-compatible endpoint —
configured purely through environment/Docker variables (no keys in code).

## License & attribution

- Quantized from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2),
  Apache-2.0 © OpenBMB. This redistribution is Apache-2.0 with attribution.
- Runtime: [audio.cpp](https://github.com/0xShug0/audio.cpp), Apache-2.0 © ShugoAI LLC.
- No reference voice audio is included; supply your own clip.
