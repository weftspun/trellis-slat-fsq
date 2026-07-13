# Build the Slang FSQ NIF on Windows: slangc (CPU C++ target) + MSVC cl -> priv/fsq_nif.dll.
#
# Requirements: slangc (shader-slang release; set $env:SLANGC or have it on PATH), Visual Studio
# (cl.exe located via vswhere), Erlang (erl on PATH, for erl_nif.h).
#
# Usage: powershell -File native/build_windows.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$slangc = if ($env:SLANGC) { $env:SLANGC } else { "slangc" }
$slangInclude = Split-Path -Parent (Split-Path -Parent (Get-Command $slangc).Source)  # <release>/

# 1. Slang -> C++ (kernel + entry-point dispatcher; includes the slang-cpp prelude by absolute path).
& $slangc "$root/priv/slang/fsq_nif.slang" -target cpp -entry fsq_encode -stage compute `
    -o "$root/native/fsq_nif.gen.cpp"
if ($LASTEXITCODE -ne 0) { throw "slangc failed" }

# 2. Locate MSVC + Erlang headers (path math, no erl -eval: OTP >= 28 parses ~s as a sigil there).
$pf86 = [Environment]::GetFolderPath("ProgramFilesX86")
$vswhere = Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe"
$vsroot = & $vswhere -latest -property installationPath
# scoop shims hide the real install dir; prefer `scoop prefix`, fall back to path math.
$erlRoot = if (Get-Command scoop -ErrorAction SilentlyContinue) { scoop prefix erlang } else { $null }
if (-not $erlRoot) { $erlRoot = Split-Path -Parent (Split-Path -Parent (Get-Command erl).Source) }
$ertsInclude = Join-Path $erlRoot "usr\include"
if (-not (Test-Path (Join-Path $ertsInclude "erl_nif.h"))) { throw "erl_nif.h not found under $ertsInclude" }

# 3. Compile the NIF DLL inside a VS dev environment.
New-Item -ItemType Directory -Force "$root/priv" | Out-Null
$vsdevcmd = "$vsroot\Common7\Tools\VsDevCmd.bat"
cmd /c "`"$vsdevcmd`" -arch=x64 -no_logo && cl /nologo /LD /O2 /EHsc /std:c++17 /I `"$ertsInclude`" `"$root/native/fsq_nif.cpp`" /Fe:`"$root/priv/fsq_nif.dll`" /Fo:`"$root/native/`""
if ($LASTEXITCODE -ne 0) { throw "cl failed" }

Write-Host "OK: $root/priv/fsq_nif.dll"
