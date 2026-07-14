"""Pin the slangtorch JIT toolchain into os.environ (Windows).

Python twin of scripts/run_slang_tests.ps1 — import BEFORE anything that triggers a slangtorch
compile. The only combo that builds on this box: CUDA 12.6.3 toolkit + MSVC 14.44 (VS2022) host
compiler + VSCMD_ARG_TGT_ARCH set (else torch cpp_extension re-activates the newest VS).
"""

from __future__ import annotations

import os

MSVC = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
SDK = r"C:\Program Files (x86)\Windows Kits\10"
SDK_VERSION = "10.0.26100.0"
CUDA = os.path.expanduser(r"~\scoop\apps\cuda-12.6.3\current")


def pin() -> None:
    if os.name != "nt":
        return
    os.environ["PATH"] = os.pathsep.join(
        [rf"{CUDA}\bin", rf"{MSVC}\bin\Hostx64\x64", os.environ.get("PATH", "")]
    )
    os.environ["INCLUDE"] = os.pathsep.join(
        [rf"{MSVC}\include", rf"{SDK}\Include\{SDK_VERSION}\ucrt",
         rf"{SDK}\Include\{SDK_VERSION}\um", rf"{SDK}\Include\{SDK_VERSION}\shared"]
    )
    os.environ["LIB"] = os.pathsep.join(
        [rf"{MSVC}\lib\x64", rf"{SDK}\Lib\{SDK_VERSION}\ucrt\x64", rf"{SDK}\Lib\{SDK_VERSION}\um\x64"]
    )
    os.environ["CUDA_HOME"] = CUDA
    os.environ["CUDA_PATH"] = CUDA
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ["VSCMD_ARG_TGT_ARCH"] = "x64"
