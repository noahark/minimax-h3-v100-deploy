# Troubleshooting history — how this recipe was found

This records the dead-ends and the diagnosis, so others don't repeat them.
The **twist**: the V100 was **never** the problem — the *stack* was.

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
- Quantized **GGUF + `auto-fit`** fits the whole model on 32 GB (auto-fit put encoder 18.8 GB +
  diffusion 10.9 GB on VRAM, VAE on RAM).
- V100 (sm_70) compile support via ggml PR [#1062](https://github.com/leejet/stable-diffusion.cpp/pull/1062).
- **Verified end-to-end**: 256×256×5 = 135 s; 864×480×56 = 687 s (official demo spec).

## Key gotchas (all discovered by running, not by docs)

| Gotcha | Reality |
|---|---|
| `--offload-to-cpu` | moves **every** backend (incl. diffusion) to CPU → extremely slow. Avoid. |
| default `--auto-fit` | the correct mode for small runs; let it place weights. |
| `te=cpu` (large runs) | move the one-time encoder (18.8 GB) to RAM so diffusion activations fit. |
| `--cfg-scale 1.0` | mandatory — default 7.0 makes H3 abort. |
| `-DCMAKE_CUDA_ARCHITECTURES=70` | must build from source; prebuilt PTX breaks on V100. |
| CUDA bin on PATH | else the exe fails with `0xC0000135` (DLL not found). |
| `git clone --depth 1` | does not fetch submodules; run `submodule update --init --recursive`. |
