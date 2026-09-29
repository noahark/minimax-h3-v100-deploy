# MiniMax-H3 on a Single Tesla V100 32GB — Resolution × Duration Field Guide

> Practical, measured numbers for running MiniMax-H3 video+audio generation locally at different resolutions and durations, collected over ~3 weeks of daily production runs (Sept 2026). Every number below comes from an actual run on our rig unless explicitly marked *extrapolated*. Sharing so others can calibrate expectations before burning GPU hours.
>
> Companion data guide to this repository — the rig is described in [§0](#0-the-rig-these-numbers-come-from); the deployment recipe itself is in the [README](../README.md).

---

## 0. The rig these numbers come from

| Component | Value |
|---|---|
| GPU | Tesla V100-PCIE-32GB (Volta, sm_70), passively cooled |
| System RAM | 128 GB |
| Driver / CUDA toolkit | 581.80 / CUDA 12.4 (CUDA 13 dropped Volta offline compilation — don't upgrade the toolkit past 12.x on Volta) |
| Inference stack | stable-diffusion.cpp @ commit `d8fb10c`, self-compiled, `GGML_CUDA_FA=ON` |
| DiT | MiniMax-H3 FL2VA (pruned), **Q4_K_M** GGUF, 11 GB |
| Text encoder | Qwen3-VL-32B, Q4_K_M GGUF, 17 GB |
| VAEs | video fp16 (4.9 GB) + audio fp32 (578 MB) |
| Distillation LoRA | community Turbo LoRA v4 (step-600 EMA), converted for sd.cpp key naming, **multiplier 1.0** |
| Sampling | **8 steps**, CFG 1.0, video shift 12 (default), fps 24 |
| Server flags | `--diffusion-fa --offload-to-cpu --mmap` |
| Output | joint video + audio in one pass (H3 is a joint AV model — **there is no video-only mode**; skipping audio saves nothing) |

---

## 1. Hard constraint #1: frame counts must be on the `17k+5` grid

Duration = frames ÷ 24. Off-grid values are not accepted.

| Frames | 5 | 22 | 39 | 56 | 73 | 90 | 107 | 124 | 141 | 158 | 175 | 192 | 209 | 226 | 243 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Duration (s) | 0.21 | 0.92 | 1.63 | 2.33 | 3.04 | 3.75 | 4.46 | **5.17** | 5.88 | **6.58** | 7.29 | **8.00** | **8.71** | 9.42 | 10.13 |

Width/height must be multiples of 32 — `1280×720` requested actually renders as **1280×736**.

---

## 2. Headline: resolution ↔ duration ceiling map

| Canvas | Pixels | Role in our pipeline | Single-shot duration ceiling | Evidence |
|---|---|---|---|---|
| 512×288 | 0.15 MP | probe / seed screening | ≥ 6.58 s (158 f) | measured; longer untested |
| 864×480 | 0.41 MP | draft | — | pre-flash-attention this was our daily maximum |
| 960×544 | 0.52 MP | draft | ~14.4 s | *extrapolated, untested* |
| 1024×576 | 0.59 MP | long-form / 16:9 delivery source | **10.12 s (243 f) measured**; ~13 s *extrapolated* | 243 f ran in 31.5 min |
| 1280×720 | 0.92 MP | — (poor value for us) | ≥ 5.17 s (124 f) | measured, 23 min per shot |
| **1344×768** | **1.03 MP** | **final / production canvas** | **8.71 s (209 f) measured**; 192 f recommended default; 226 f *extrapolates past 32 GB* | boundary battle, §4 |
| 1536×864 | 1.33 MP | experiments | ≥ 1.63 s (39 f) | measured; longer untested |
| 1920×1088 | 2.09 MP | experiments (delivery is upscaled instead) | ≥ 1.63 s (39 f) measured; ~3.5 s *extrapolated* | 39 f costs 14 min |

**The lesson that cost us days:** before enabling flash attention, 1024×576 capped at 39 frames (1.63 s) and we concluded it was a "physical 32GB VRAM wall". One launch flag — `--diffusion-fa` (streamed attention instead of materialized attention matrices) — moved that ceiling to 243 frames, and made 720p/1536×864/1080p runnable at all. **Probe implementation switches before declaring hardware limits.**

Resolution and duration trade roughly 1:1 against each other inside a fixed VRAM budget. **Longer than the ceiling → chain shots, don't stretch a single shot.**

---

## 3. Measured benchmarks — every verified combo

### 3a. Production config (Q4 + Turbo LoRA @1.0, 8 steps)

| Canvas | Frames | Duration | Sampling | End-to-end | Notes |
|---|---|---|---:|---:|---|
| 512×288 | 39 | 1.63 s | ~60–65 s | ~1 min | seed-screening probe (T2VA) |
| 960×544 | 56 | 2.33 s | 235.8 s | 278.3 s | draft tier |
| 1024×576 | 124 | 5.17 s | 688–715 s | 793–817 s | 3 runs (I2VA), spread < 4% |
| 1024×576 | 243 | 10.12 s | 1,665.9 s | 1,889.9 s | longest verified 576p shot |
| 1344×768 | 39 | 1.63 s | 405–408 s | 466–469 s | probe tier at full canvas |
| 1344×768 | 158 | 6.58 s | 2,011–2,146 s | 2,238–2,387 s | 4 runs; ~36.6 MB webm per shot |
| 1344×768 | 175 | 7.29 s | 2,530.8 s | 47.8 min wall | CLI, boundary probe |
| 1344×768 | 192 | 7.96 s | 2,914.1 s | 54.7 min wall | CLI |
| 1344×768 | 209 | 8.71 s | 3,224.0 s | 60.0 min wall | CLI, 0.55 GB VRAM headroom left |

Realistic planning anchor for 1344×768×158 including submission/loading overhead: **42.4 min per clip** (three identical runs: 2541 / 2542 / 2557 s, spread ≤ 16 s).

### 3b. Baseline without LoRA (8 steps, seed 3001) — flash-attention breakthrough runs

| Canvas | Frames | Sampling | End-to-end |
|---|---|---:|---:|
| 1024×576 | 39 | 258.3 s | ~294 s |
| 1024×576 | 56 | 248.8–252.9 s | 306–310 s |
| 1024×576 | 124 | 643.3 s | 752.8 s |
| 1024×576 | 158 | 890.4 s | 1,032 s (17 min) |
| 1280×720 | 39 | 279.2 s | 330.6 s |
| 1280×720 | 124 | 1,221.5 s | 1,390 s (23 min) |
| 1536×864 | 39 | 424.6 s | 496.0 s |
| 1920×1088 | 39 | 761.7 s | 854.6 s (14 min) |

Note: 56 frames at 1024×576 samples *faster* than 39 frames did under the old non-FA kernel (248.8 s vs 258 s) — the old wall was pure attention-matrix overhead, not compute.

The Turbo LoRA itself costs only **~+6%** per run at the same step count (688–715 s vs 643 s at 1024×576×124); its value is cutting 20 steps down to 8, not speeding up each step.

---

## 4. The 768p VRAM boundary, measured frame-by-frame (1344×768)

| Frames | Duration | Peak VRAM | Headroom | Sampling | Decode | Wall clock |
|---|---|---:|---:|---:|---:|---:|
| 158 | 6.58 s | 28,284 MiB | 3.9 GB | 2,011–2,146 s | — | 38–42.5 min |
| 175 | 7.29 s | 29,486 MiB | 3.2 GB | 2,530.8 s | 266.1 s | 47.8 min |
| 192 | 7.96 s | 30,925 MiB | 1.8 GB | 2,914.1 s | ~290 s | 54.7 min |
| 209 | 8.71 s | **32,206 MiB** | **0.55 GB** | 3,224.0 s | 301.4 s | 60.0 min |
| 226 | 9.42 s | ~33.9 GB | negative | — | — | *will OOM, not attempted* |

- VRAM slope at this canvas: **≈ 84.6 MiB per frame**.
- The HTTP server keeps ~3.3 GB idle → **through-server ceiling is 158 frames**; 175/192/209 all required stopping the server and running the CLI with the full card.
- **Production default is 192** (1.8 GB headroom absorbs input-type variation); 209 is reserved for non-critical T2VA/I2VA runs.
- Decode is tiled (~8.5 GB in 28+ chunks) — **the peak is always in sampling, decode is never the bottleneck**.
- Reference-video inputs (Ref2VA) raise VRAM further (more tokens) — probe before combining them with high frame counts.

---

## 5. Predict before you run: the cost model

Everything below was validated to a few percent, so you can compute OOM risk and runtime *before* launching:

1. **Sampling time ∝ pixels × frames^1.2** (fit at 576p). Validation: 1024×576×158 predicted 906 s vs measured 890.4 s — **2% error**.
2. **Peak VRAM ∝ pixels × frames** at a fixed canvas. Validation: 1344×768×158 predicted 29,040 MiB vs measured 28,284 MiB — **2.6% error**.
3. **Per-step time is independent of step count**: 8 steps → 32.3 s/step, 20 steps → 32.4 s/step (1024×576×39). Time scales linearly with steps; quality differences between step counts are real, speed differences per step are not.
4. With `--offload-to-cpu`, all ~35.4 GB of weights sit in system RAM and stream over PCIe each step: ≈ 0.9 s of a ~32 s step (**≤ 4%**, overlapped by prefetch). Measured cost of offload vs weights-on-GPU: **+1.7%**.
5. Sampling pins the GPU's **250 W power cap** → the workload is compute-bound, not transfer-bound. RAM/caching tricks have single-digit-% upside.
6. End-to-end split: **sampling 87% / VAE decode 11% / text encoding 0.4%**.

---

## 6. Throughput cheat sheet (what a shot actually costs)

| Task | GPU cost |
|---|---|
| Prompt/seed probe, 512×288×39 | ~1 min each; a 5-seed screen ≈ 6 min |
| Draft shot, 960×544×56 (2.33 s) | ~5 min |
| 576p shot, 1024×576×124 (5.17 s) | ~13–14 min |
| 576p max shot, 1024×576×243 (10.12 s) | ~31.5 min |
| 768p shot, 1344×768×158 (6.58 s) | ~38 min best case; **42.4 min realistic anchor** |
| 768p max shot, 1344×768×209 (8.71 s) | ~60 min |
| 15.4 s film @ 576p (3 × 124 f chained) | ~40 min |
| 13.2 s film @ 768p (2 × 158 f chained) | ~77 min |
| Delivery upscale of 15.4 s (ESRGAN ×4 → color → encode) | ~18 min (upscale 1.95 s/frame) |
| Delivery upscale extrapolated to a 10-min film | ~11.5 h + 263 GB of intermediate frames — batch by scene |

---

## 7. Speed levers: measured wins and measured failures

| Lever | Measured effect | Verdict |
|---|---|---|
| **Turbo LoRA, 8 steps** (vs plain 20 steps) | 649.0 → 261.1 s sampling @ 1024×576×39 = **2.5×** | ✅ production default. **8 steps is the quality floor** (6-step output was rejected by eye); LoRA author's own range is 4–8, so 8 sits at their ceiling |
| `--mmap` model loading | cold-start end-to-end 1,236 → 286 s = **4.3×**; output bit-identical (PSNR = ∞, audio hash identical) | ✅ always on |
| EasyCache @ 20 steps | 649.0 → 323.4 s = **2.01×**, small drift | ✅ fine *if* you run 20 steps |
| EasyCache @ 8 steps | 1.33× faster, **but temporal jitter 2.1×** (0.548 vs 0.259), visible structural popping between frames | ❌ rejected — the cache family is structurally dead in the 8-step distilled regime |
| DBCache @ ≤ 8 steps | frames visibly destroyed (PSNR 15.6 dB) | ❌ never |
| Latent-space second pass ("hires") | hard error — LTX-only in sd.cpp | ❌ not available |
| Adding a 2nd GPU (GTX 1060 6 GB) | slowest card sets the pace; 6 GB can't hold one transformer block's activations; no tensor cores | ❌ negative value |
| Same seed, same params, re-run | **bit-identical** output (verified cold vs warm) | ✅ makes A/B testing clean |

**Audio side-finding:** with the Turbo LoRA, 4/4 runs produced audible audio; without it, 2/2 runs were near-silent (−69 / −42 dB). If you want sound, keep the LoRA on (or explicitly describe the sound in the prompt).

---

## 8. How we actually produce (draft → final → deliver)

1. **Draft tier** — 864×480 or 960×544, 39–56 frames, text-only, ~1–5 min: decide whether the scene works at all.
2. **Seed screening before any expensive shot** — the same final prompt at 512×288×39 across **5 seeds** (~6 min total). ≥ 4/5 usable → pick the best seed and burn the final; ≤ 2/5 → rewrite the prompt first. Seed variance is real: same config, same prompt, different seeds can differ by a full quality tier.
3. **Final tier** — 1344×768 native (7:4 canvas), default 124 frames, ceiling 192 (209 CLI-only). First-frame conditioning (I2VA) when a still exists.
4. **Longer films = chaining, not longer shots** — next shot's init frame = previous shot's last frame. Seams measured near-invisible (same face/pose/lighting/composition); VRAM cost is just the sum of independent shots. Verified: 13.2 s two-shot @ 768p, 15.5 s three-shot @ 576p. Multi-angle coverage works when planned as **one continuous camera arc, then split** — hard-cutting separate angles reads as inconsistent.
5. **Delivery tier** — don't generate 1920 natively as the norm: 1344×768 native → ESRGAN ×4 → **global single-curve** color restore (ESRGAN drops chroma 31% and looks gray without it; per-frame color matching flattens scene progression — don't) → scale to 1920 → remux the original audio. Upscale is CPU-bound and runs ~1.95 s/frame.

---

## 9. Gotchas that cost us real time

| Gotcha | Symptom | Fix |
|---|---|---|
| PCM audio inside WebM | players silently ignore the audio track | remux before delivery: webm → `-c:v copy -c:a libopus`; mp4 → `-c:v libx264 -crf 16 -c:a aac` |
| 1344×768 is **7:4, not 16:9** | scaling straight to 1920×1080 distorts | `scale=1920:1096:flags=lanczos,crop=1920:1080:0:8` (1024×576 is exactly 16:9 — zero-stretch path to 1080p) |
| Auto-burned Chinese hard subtitles | dialogue tags render as correct-glyph hardsubs; generic "no text" negation does NOT suppress them | append "That line is heard only as audio…" immediately after each dialogue tag; spot-check the lower band of frames |
| Frame count off the 17k+5 grid | rejected | submit only grid values |
| Dimensions not multiples of 32 | silently rounded (1280×720 → 1280×736) | pick multiples of 32 up front |
| Server holds 3.3 GB idle | 768p > 158 frames OOMs through the server | stop the server, run the CLI with the full card |
| Thermal throttling (passive V100) | mysterious 2.1× slowdown; SM clock 135–420 MHz, 81 °C | fix airflow first; healthy state: 72–73 °C, 1230–1380 MHz, only power-capped |
| Believing "OOM = physical wall" | we lost days on a wrong 39-frame ceiling | probe implementation switches (flash attention, tiling, offload) first; an OOM conclusion without a failure log is not a conclusion |
| Literal prompt execution | "a dragon's head hangs in the blackness" → literally a floating, detached head | spell out anatomy and physical relationships explicitly |
| `medium close-up` doesn't work | renders as a medium side view, face barely visible | "a tight close-up … fills the frame, shot from the front at eye level" |

---

## 10. Scope and caveats

- All numbers are from **one rig**: Volta sm_70 (no bf16/fp8/native-int8 paths in this stack), GGUF Q4_K_M quantization, sd.cpp `d8fb10c`. Absolute times will differ on other GPUs, quantizations, or builds — the **scaling laws, VRAM-relative ceilings, and the practices** are the transferable part.
- Where we say "verified", the clip exists and passed decode/stability checks; where we extrapolate, it is marked.
- Frozen production recipe (for reproducibility): 8 steps / Turbo LoRA @1.0 / CFG 1.0 / shift 12 / `--diffusion-fa --offload-to-cpu --mmap` / 1344×768 final canvas.
