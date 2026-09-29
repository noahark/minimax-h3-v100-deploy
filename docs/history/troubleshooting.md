# Troubleshooting history — how this recipe was found

This records the dead-ends and the diagnosis, so others don't repeat them.
The **twist**: the V100 was **never** the problem — the *stack* was.

> **⚠️ Kept as history (2026-09-29):** two conclusions in this doc were later overturned
> by measurement. They are annotated inline below and fixed in the
> [README Corrections section](../../README.md#corrections--what-we-got-wrong-at-first-2026-09-29).

## Abandoned approach 1 — ComfyUI (0.33.2) + comfy-kitchen + T8 + v100-patch

Three symptoms, each chased down:

- **`cudaErrorNotSupported` (sticky)** — after the first CUDA op, every `tensor.to(cuda)`
  (even a legal copy) reported the same error. Diagnosis: the `cudaMallocAsync` allocator on
  Windows + V100 + cu126 poisons the CUDA context; the async error surfaces at the next sync
  point. Workaround (verified standalone): `--disable-cuda-malloc --disable-pinned-memory`.
  **This was not a hardware limit.**
- **dtype mismatch** — `Float vs Half` in the diffusion model's `condition_proj` / `token_refiner`;
  the text encoder emitted fp32 latents into an fp16 pipeline.
- **OOM** — the fp16 main model alone is ~37 GB > 32 GB VRAM.

Conclusion: ComfyUI 0.33.2 + comfy-kitchen + T8 + v100-patch is **incompatible with V100**.
The two public V100 success reports don't use T8, and we could not reproduce them on Windows.

> **Correction (2026-09-18):** this conclusion was overturned **for a different route** —
> ComfyUI with an fp8_scaled DiT + nvfp4 encoders + the `ComfyUI-MiniMaxH3-FP16Safe`
> plugin + an eager-execution patch **does work on V100** (T2V/I2V verified, same
> 209-frame ceiling at 768p, 3–6× slower than sd.cpp). The *default* route documented
> above still fails. "Impossible on V100" was wrong; "not the production stack" stands.

## Abandoned approach 2 — fp32 (disable the v100-patch)

fp16 weights double to ~74 GB + 47 GB encoder → exceeds 128 GB RAM → hard crash. Dead end.

## Abandoned approach 3 — int8 (pruned `int8_convrot`, 19.53 GB)

Fits in 32 GB, **but V100 (sm_70) has no int8 tensor core** (int8 needs sm_75+ / Turing) →
`cudaErrorNotSupported`. Dead end.

## Abandoned approach 4 — GGUF inside ComfyUI

`ComfyUI-GGUF` does not know the `minimax` architecture (`Unknown model architecture!`), and the
GGUF CLIP was incompatible. Dead end.

## The approach that worked — stable-diffusion.cpp + GGUF

- sd.cpp **natively supports H3** ([docs/minimax_h3.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/minimax_h3.md)).
- Quantized **GGUF + `--offload-to-cpu`** fits the whole model set on 32 GB (all ~35 GB of
  weights in system RAM, 0 MB VRAM resident; the generation peak is activations only).
- V100 (sm_70) compile support via ggml PR [#1062](https://github.com/leejet/stable-diffusion.cpp/pull/1062).
- **Verified end-to-end**: 256×256×5 = 135 s; 864×480×56 = 687 s (official demo spec).
- *(As of 2026-09-29 the same rig runs 1344×768×209 = 8.71 s per shot — see the README
  "What worked" table and `docs/resolution_duration_guide.md`.)*

## Key gotchas (all discovered by running, not by docs)

| Gotcha | Reality |
|---|---|
| `--offload-to-cpu` | **CORRECTED 2026-09-10 — the original claim here was wrong.** Offload is the *recommended* mode; the official sd.cpp H3 example pairs it with `--diffusion-fa`. Measured cost ~1.7%, weights → system RAM, 0 MB VRAM resident. The "extremely slow" behavior came from passing `--backend`/`--params-backend`, which overrides offload and disables `--auto-fit`. |
| default `--auto-fit` | correct for small runs — but never combine with explicit `--backend` flags (they disable it) |
| `te=cpu` (large runs) | **obsolete** — with `--diffusion-fa --offload-to-cpu` all weights live in RAM and large runs need no manual backend placement |
| `--diffusion-fa` | *(added 2026-09-29)* the single biggest lever: streamed attention instead of materialized attention matrices. Moved the 1024×576 ceiling from 39 to 243 frames. |
| `--cfg-scale 1.0` | mandatory — default 7.0 makes H3 abort. |
| `-DCMAKE_CUDA_ARCHITECTURES=70` | must build from source; prebuilt PTX breaks on V100. |
| CUDA bin on PATH | else the exe fails with `0xC0000135` (DLL not found). |
| `git clone --depth 1` | does not fetch submodules; run `submodule update --init --recursive`. |
