# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare chunked logits across independent starts and resets."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    if not os.environ.get("STRATA_DSV41_TEST_PACK"):
        print("SKIP: set STRATA_DSV41_TEST_PACK to a synthetic model pack")
        return 77
    with tempfile.TemporaryDirectory(prefix="ds4-chunk-repeat-") as directory:
        root = Path(directory)
        if shutil.disk_usage(root).free < 256 * 1024 * 1024:
            print("chunk repeat: insufficient space for temporary logits", file=sys.stderr)
            return 1
        results = []
        for repeat in range(2):
            path = root / f"logits-{repeat}.f32"
            print(f"chunk repeat: independent start {repeat + 1}/2", flush=True)
            result = subprocess.run(
                [sys.argv[1], "--dump", str(path)], check=False, timeout=120,
            )
            if result.returncode:
                return result.returncode
            results.append((path.stat().st_size, hashlib.sha256(path.read_bytes()).digest()))
        if not results[0][0] or results[0] != results[1]:
            print("chunk repeat: logits differ between independent starts", file=sys.stderr)
            return 1
    print("chunk repeat: independent starts are bit equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
