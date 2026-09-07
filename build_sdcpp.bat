@echo off
REM ============================================================================
REM  MiniMax-H3 on Tesla V100 — build stable-diffusion.cpp (sd.cpp) with CUDA sm_70
REM  Run from a standard cmd (no admin needed). Requires the toolchain below.
REM ============================================================================

REM ---- EDIT THESE to match your machine --------------------------------------
set "SD_SRC=D:\minimax-h3\sd_build\stable-diffusion.cpp"   REM sd.cpp source dir
set "MSVC_BAT=C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"             REM VS Build Tools
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin" REM CUDA toolkit bin
set "NINJA=D:\minimax-h3\python\Scripts"                    REM dir containing ninja.exe
set "CMAKE=D:\minimax-h3\sd_build\tools\cmake\bin\cmake.exe"              REM cmake.exe path
REM ----------------------------------------------------------------------------

call "%MSVC_BAT%" x64
if errorlevel 1 (echo VCVARS_FAILED & exit /b 1)
set "CUDA_PATH=%CUDA_BIN%\.."
set "PATH=%CUDA_BIN%;%NINJA%;%PATH%"

cd /d "%SD_SRC%"
echo ==== cmake configure (CUDA + sm_70) ====
"%CMAKE%" -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 ^
  -DCMAKE_CUDA_COMPILER="%CUDA_BIN%\nvcc.exe"
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 1)
echo ==== build ====
"%CMAKE%" --build build -j 12
echo BUILD_EXIT=%errorlevel%
if exist build\bin\sd-cli.exe (echo OK: build\bin\sd-cli.exe)
