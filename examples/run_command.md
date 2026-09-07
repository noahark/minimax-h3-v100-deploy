# MiniMax-H3 on V100 — Run command & prompt

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
  -p "a cat" --cfg-scale 1.0 -W 256 -H 256 --video-frames 5 -o out.webm
```

### CRITICAL flags
| Flag | Why |
|---|---|
| `-M vid_gen` | video generation mode |
| `--cfg-scale 1.0` | **mandatory** — default 7.0 aborts H3 |
| (do **not** pass `--offload-to-cpu`) | it moves everything (incl. diffusion) to CPU → extremely slow |

### Large generation (864x480x56) — MUST move encoder to RAM
```powershell
sd-cli.exe -M vid_gen ^
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va_pruned-Q4_K_M.gguf ^
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors ^
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors ^
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf ^
  -p "<prompt>" --cfg-scale 1.0 -W 864 -H 480 --video-frames 56 --fps 24 ^
  --backend "diffusion=cuda0,vae=cuda0,te=cpu" --params-backend "diffusion=cuda0,vae=cuda0,te=cpu" ^
  -o out.webm
```
> Without `te=cpu`, the 18.8 GB encoder stays on VRAM and the 864x480x56 diffusion
> activations OOM. `te=cpu` keeps diffusion+VAE on GPU and the one-time encoder on RAM.

## 3. Duration / frame count
- FPS fixed at **24**. Frame count must be on the **`17k+5`** grid (min 5):
  `5, 22, 39, 56, 73, 90, 107, 124…`. Duration = `frames / 24`.

## 4. Prompt template (video + music)
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
