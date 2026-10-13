# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ROCm's AMD GPU device libraries rewritten for SYCL device code compiled on Windows.

The libraries are built on Linux, where wchar_t is 4 bytes, and their wchar_size module flag says so; SYCL device
code compiled on Windows has a 2-byte wchar_t, and the link refuses modules whose flags differ.  The libraries do not
use wchar_t: only the flag changes.  llvm-dis and llvm-as come from the intel/llvm build that will read the result,
so the libraries must be bitcode it can read (an older ROCm's, not newer than its LLVM).

usage: python tools/windows/device_libs_wchar.py LLVM_BIN_DIR SRC_DIR DST_DIR
"""
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    bin_dir, src, dst = (Path(a) for a in sys.argv[1:])
    dst.mkdir(parents=True, exist_ok=True)
    flag = re.compile(r'(!\{i32 \d+, !"wchar_size", i32 )4(\})')
    libs = sorted(src.glob("*.bc"))
    changed = 0
    for bc in libs:
        text = subprocess.run([str(bin_dir / "llvm-dis"), str(bc), "-o", "-"], capture_output=True, text=True,
                              check=True).stdout
        text, n = flag.subn(r"\g<1>2\g<2>", text)
        changed += n
        subprocess.run([str(bin_dir / "llvm-as"), "-", "-o", str(dst / bc.name)], input=text, text=True, check=True)
    print(f"{len(libs)} libraries written to {dst}, wchar_size rewritten in {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
