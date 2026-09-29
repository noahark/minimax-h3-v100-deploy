# MiniMax-H3 on V100 — Run command & prompt

## Reproduce the exact README demo (one command)

The README demo video (surf cat, 864×480×56) is produced by
[`reproduce_demo.ps1`](reproduce_demo.ps1). Edit the **3 paths at the top** and run:

```powershell
powershell -ExecutionPolicy Bypass -File examples\reproduce_demo.ps1
```

> The embedded demo video was rendered 2026-09-07 with the then-current flags in 687 s.
> The script now uses the current recommended flags (`--diffusion-fa --offload-to-cpu
> --mmap` + Turbo LoRA at 8 steps) — same prompt, same spec, several times faster.
> Expect roughly 4–6 min on a healthy V100 (measured: 278 s end-to-end at the slightly
> larger 960×544×56 with the same step count).

**Exact prompt used (verbatim):**

```text
A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten's energetic surfing.
```

**Exact settings:** `-M vid_gen --cfg-scale 1.0 -W 864 -H 480 --video-frames 56 --fps 24
--steps 8 --rng cpu --diffusion-fa --offload-to-cpu --mmap` (plus the Turbo LoRA tag —
see §3; set `--steps 20` and drop the tag if you skipped the LoRA).

## 1. Verify the CUDA backend sees the V100

```powershell
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"
sd-cli.exe --list-devices
# expect:  CUDA0  Tesla V100-PCIE-32GB
```

> ⚠️ CUDA bin **must** be on PATH, else the exe dies with `0xC0000135` (DLL not found).

## 2. Generate (text-to-audio-video, T2AV)

Small / quick test first:

```powershell
sd-cli.exe -M vid_gen ^
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va_pruned-Q4_K_M.gguf ^
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors ^
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors ^
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf ^
  -p "a cat" --cfg-scale 1.0 -W 256 -H 256 --video-frames 5 ^
  --diffusion-fa --offload-to-cpu --mmap ^
  -o out.webm
```

### CRITICAL flags

| Flag | Why |
|---|---|
| `-M vid_gen` | video generation mode |
| `--cfg-scale 1.0` | **mandatory** — default 7.0 aborts H3 |
| `--diffusion-fa` | flash attention. Without it the attention matrix materializes in VRAM and the frame count caps early (1024×576: 39 f instead of 243 f). This one flag was our biggest single gain. |
| `--offload-to-cpu` | parks all ~35 GB of weights in system RAM (0 MB VRAM resident; generation peak is activations only). Measured cost: ~1.7%. |
| `--mmap` | 4.3× faster cold start, **bit-identical** output |
| (do **not** pass `--backend` / `--params-backend`) | explicit module assignments **override** `--offload-to-cpu` and **disable** `--auto-fit` — the root cause of our original "offload is slow" misdiagnosis |

### Larger / longer generations

The old `te=cpu` backend trick from the first release is **obsolete** — with
`--diffusion-fa --offload-to-cpu`, all weights live in RAM and you are bounded only by
activation VRAM, which is predictable:

- sampling time ∝ `pixels × frames^1.2` (~2% error), peak VRAM ∝ `pixels × frames`;
- measured ceilings on 32 GB: **1344×768×209 (8.71 s)**, **1024×576×243 (10.12 s)**,
  1920×1088×39 verified.

Just raise `-W/-H` (multiples of 32) and `--video-frames` (on the `17k+5` grid). Full
map with timings and VRAM: [`../docs/resolution_duration_guide.md`](../docs/resolution_duration_guide.md).

## 3. Turbo LoRA (optional but recommended — 2.5× sampling speedup)

1. **Convert once** (the original HF file loads as a *silent no-op* in sd.cpp — no
   error, zero effect):

   ```powershell
   python tools\convert_h3_lora.py minimax_h3_turbo_v4_step600_ema.safetensors minimax_h3_turbo_v4_step600_ema_sdcpp.safetensors mm
   ```

2. Put the converted file in a folder and pass `--lora-model-dir <that folder>`.
3. Set `--steps 8` and **prefix the prompt** with the LoRA tag:

   ```text
   <lora:minimax_h3_turbo_v4_step600_ema_sdcpp.safetensors:1.0> A cute American Shorthair…
   ```

Measured: 20 steps 649 s → 8 steps + LoRA 261 s sampling at 1024×576×39 (**2.5×**).
**Do not go below 8 steps** — 4/6-step output was rejected on quality; the LoRA author's
own usable range is 4–8, and 8 is our floor. Bonus finding: runs without the LoRA came
out near-silent; with it, audio is reliably present.

## 4. Longer films — chaining

Each generation is an independent shot; for films longer than the single-shot ceiling,
feed the previous shot's **last frame** as the next shot's init image:

```powershell
ffmpeg -y -i shot01.webm -vf "select=eq(n\,55)" -frames:v 1 -q:v 1 shot02_first.png
sd-cli.exe -M vid_gen … -i shot02_first.png -W 1344 -H 768 --video-frames 124 …
```

Keep resolution / steps / LoRA identical across shots; allocate dialogue within a single
shot; reuse the subject description verbatim. Seams measured near-invisible.

## 5. Duration / frame count

- FPS fixed at **24**. Frame count must be on the **`17k+5`** grid (min 5):
  `5, 22, 39, 56, 73, 90, 107, 124…`. Duration = `frames / 24`.
- Width/height must be multiples of 32 (1280×720 renders as 1280×736).

## 6. Prompt template (video + music)

```
A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave,
riding a white surfboard with the clear text 'sd.cpp' on it.
Cinematic tracking shot, realistic water, bright sunlight, smooth motion,
and consistent character appearance.
Add upbeat tropical surf-rock background music with cheerful drums and guitar,
synchronized with the kitten's energetic surfing.
Technical spec: 864x480 resolution, 56 frames at 24 fps, 2.33 seconds, 9:5 aspect.
```

Prompt writing tips:
1. English, concrete (species, colors, action, setting).
2. Always include a camera word (close-up / tracking shot / slow pan).
3. Always include "consistent character appearance" to stop identity drift.
4. Add music keywords + "synchronized with" for the audio branch (H3 does joint AV).
5. H3 executes wording **literally** — spell out anatomy and physical relations
   ("a dragon's head hangs in the blackness" produced a literal floating, detached head).
