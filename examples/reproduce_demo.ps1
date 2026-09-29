# ============================================================================
# Reproduce the README demo on a Tesla V100 (sm_70) -- MiniMax-H3, 864x480x56
#   Run:  powershell -ExecutionPolicy Bypass -File reproduce_demo.ps1
#   EDIT the three paths below to match your machine, then run.
#
# NOTE (2026-09-29): the demo video embedded in the README was rendered
#   2026-09-07 with the flags of that time (687 s, te=cpu backend trick).
#   This script uses the CURRENT recommended flags:
#     --diffusion-fa --offload-to-cpu --mmap   (see README "Corrections")
#   Same prompt, same spec, several times faster.
#
#   $UseLora = $true  -> 8 steps + Turbo LoRA (recommended; requires the
#                        converted LoRA, see tools\convert_h3_lora.py)
#   $UseLora = $false -> 20 steps, no LoRA (base models only)
# ============================================================================
$SD_CLI   = "D:\minimax-h3\sd_build\stable-diffusion.cpp\build\bin\sd-cli.exe"
$MODELS   = "D:\minimax-h3\sdcpp\models"
$CUDA_BIN = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
$LORA_DIR = "$MODELS\loras"
$OUT      = "$MODELS\surfcat_864x480x56.webm"
$UseLora  = $true
$Seed     = 3001

# Required: CUDA runtime DLLs (cudart/cublas) must be on PATH, else 0xC0000135.
$env:PATH = "$CUDA_BIN;$env:PATH"

# ==== Exact prompt used for the README demo (video + music) ====
$Prompt = "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten's energetic surfing."

if ($UseLora) {
  $Steps  = 8
  $Prompt = "<lora:minimax_h3_turbo_v4_step600_ema_sdcpp.safetensors:1.0> $Prompt"
} else {
  $Steps = 20
}

& $SD_CLI -M vid_gen `
  --diffusion-model "$MODELS\diffusion_models\minimax_h3_fl2va_pruned-Q4_K_M.gguf" `
  --vae "$MODELS\vae\minimax_h3_video_vae_fp16.safetensors" `
  --audio-vae "$MODELS\vae\minimax_h3_audio_vae_fp32.safetensors" `
  --llm "$MODELS\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf" `
  --lora-model-dir "$LORA_DIR" `
  -p $Prompt `
  --cfg-scale 1.0 -W 864 -H 480 --video-frames 56 --fps 24 `
  --steps $Steps --seed $Seed --rng cpu `
  --diffusion-fa --offload-to-cpu --mmap `
  -o $OUT

Write-Host "`nDone. Output -> $OUT" -ForegroundColor Green
