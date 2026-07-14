# Run the Slang->PyTorch (CUDA) test suite with a hand-built MSVC 14.44 + CUDA 12.6.3 environment.
# Why manual env: this box has VS18 (MSVC 14.51, too new for CUDA 12.x nvcc) AND VS2022 (14.44, the
# supported host compiler), but VsDevCmd/vcvars fail to put cl on PATH here, and torch's cpp_extension
# re-activates the LATEST VS unless VSCMD_ARG_TGT_ARCH is set. So: pin all paths explicitly.
#
# Usage: powershell -File scripts/run_slang_tests.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

$msvc = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
$sdk = "C:\Program Files (x86)\Windows Kits\10"
$sdkv = "10.0.26100.0"
$cuda = "$env:USERPROFILE\scoop\apps\cuda-12.6.3\current"

$cache = Join-Path $repo "trellis_slat_fsq\.slangtorch_cache"
if (Test-Path $cache) { Remove-Item -Recurse -Force $cache -Confirm:$false }

# pixi-managed env (pixi install -e slang): torch 2.7.1+cu126 + slangtorch + ninja.
$py = "$repo\.pixi\envs\slang"

$env:PATH = "$py;$py\Scripts;$cuda\bin;$msvc\bin\Hostx64\x64;" + $env:PATH
$env:INCLUDE = "$msvc\include;$sdk\Include\$sdkv\ucrt;$sdk\Include\$sdkv\um;$sdk\Include\$sdkv\shared"
$env:LIB = "$msvc\lib\x64;$sdk\Lib\$sdkv\ucrt\x64;$sdk\Lib\$sdkv\um\x64"
$env:CUDA_HOME = $cuda
$env:CUDA_PATH = $cuda
$env:TORCH_CUDA_ARCH_LIST = "8.9"     # RTX 4090 (sm_89) only; skip other arch codegen
$env:VSCMD_ARG_TGT_ARCH = "x64"        # stop torch cpp_extension re-activating the newest VS

Set-Location $repo
& "$py\python.exe" -m pytest tests/ -v --tb=short
exit $LASTEXITCODE
