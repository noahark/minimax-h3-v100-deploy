# ============================================================================
# Reproduce the README demo on a Tesla V100 (sm_70) — MiniMax-H3, 864x480x56
#   Run:  powershell -ExecutionPolicy Bypass -File reproduce_demo.ps1
#   EDIT the three paths below to match your machine, then run.
# ============================================================================
$SD_CLI   = "D:\minimax-h3\sd_build\stable-diffusion.cpp\build\bin\sd-cli.exe"
$MODELS   = "D:\minimax-h3\sdcpp\models"
$CUDA_BIN = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
$OUT      = "$MODELS\surfcat_864x480x56.webm"

# Required: CUDA runtime DLLs (cudart/cublas) must be on PATH, else 0xC0000135.
$env:PATH = "$CUDA_BIN;$env:PATH"

# ==== Exact prompt used for the README demo (video + music) ====
$Prompt = "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten's energetic surfing."

& $SD_CLI -M vid_gen `
  --diffusion-model "$MODELS\diffusion_models\minimax_h3_fl2va_pruned-Q4_K_M.gguf" `
  --vae "$MODELS\vae\minimax_h3_video_vae_fp16.safetensors" `
  --audio-vae "$MODELS\vae\minimax_h3_audio_vae_fp32.safetensors" `
  --llm "$MODELS\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf" `
  -p $Prompt `
  --cfg-scale 1.0 -W 864 -H 480 --video-frames 56 --fps 24 `
  --backend "diffusion=cuda0,vae=cuda0,te=cpu" `
  --params-backend "diffusion=cuda0,vae=cuda0,te=cpu" `
  -o $OUT

Write-Host "`nDone. Output -> $OUT" -ForegroundColor Green
