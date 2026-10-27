#!/bin/bash
# Build the DeepSeek-V4.1-Flash engine component for Apple Silicon (strata_ds4, backend metal).
#
#   bash packaging/mac/build_dsv41_engine.sh [--version 0.2.7] [--build DIR] [--out DIR]
#
# Run it in the source tree on the Mac, after `ninja -C <build> strata_ds4` (the script builds it too when the
# binary is missing). It stages the binary and the kernel sources the engine compiles at start-up, packages them with
# packaging/package_engine.py and verifies every hash. It does NOT sign: the feed key stays offline, and a package
# made here is unsigned and loads (see packaging/ENGINE-COMPONENT.md, Signature).
#
# Layout (strata_ds4 finds these beside its own executable, metal/src/resource_path.h):
#   <exe>/metal/kernels/*.metal         the shared Qwen kernels the DeepSeek host also compiles
#   <exe>/metal/src/engine/msl/*.metal  ds4_init, ds4_iq, ds4_mm, ds4_quant, mtp, mtp_chain, ple
#   <exe>/metal/ggml-common.h
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VERSION=0.2.7
BUILD="$ROOT/build-metal"
OUT="$ROOT/pkg/engine"
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --build) BUILD="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "usage: $0 [--version V] [--build DIR] [--out DIR]"; exit 2;;
  esac
done
NINJA=${NINJA:-$HOME/strata-mac/tools/bin/ninja}
EXE="$BUILD/strata_ds4"
[ -x "$EXE" ] || "$NINJA" -C "$BUILD" strata_ds4 || exit 1

STAGE="$ROOT/pkg/dsv41-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/metal/kernels" "$STAGE/metal/src/engine/msl"
cp "$EXE" "$STAGE/"; chmod +x "$STAGE/strata_ds4"
cp "$ROOT"/metal/kernels/*.metal "$STAGE/metal/kernels/"
for f in ds4_init ds4_iq ds4_mm ds4_quant mtp mtp_chain ple; do
  cp "$ROOT/metal/src/engine/msl/$f.metal" "$STAGE/metal/src/engine/msl/" || exit 1
done
cp "$ROOT/third_party/ggml/ggml-common.h" "$STAGE/metal/"

python3 "$ROOT/packaging/package_engine.py" --src "$STAGE" --backend metal --version "$VERSION" \
  --platform macos-arm64 --arch apple-gpu --entrypoint strata_ds4 --include 'metal/**' --out "$OUT" || exit 1
DIR="$OUT/$VERSION-macos-arm64-metal"
python3 "$ROOT/packaging/package_engine.py" --verify "$DIR" || exit 1

# self-containment: the package must run with the source tree's kernels hidden
PACK=${PACK:-$HOME/strata-mac/pack-2.25-real}
if [ ! -d "$PACK" ]; then echo "== no pack at $PACK: package built, smoke test skipped"; exit 0; fi
echo "== self-containment test (source kernels hidden, run from /tmp)"
mv "$ROOT/metal/kernels" "$ROOT/metal/kernels.hidden"
mv "$ROOT/metal/src/engine/msl" "$ROOT/metal/src/engine/msl.hidden"
( cd /tmp && "$DIR/strata_ds4" --pack "$PACK" --ids 0 --n 1 \
    --expert-profile none --vision off --dspark off --cpu-threads 4 --threads off --tp 1 2>&1 | grep -a -E 'prompt:|decode:|cannot read|failed' | head -3 )
rc=$?
mv "$ROOT/metal/kernels.hidden" "$ROOT/metal/kernels"
mv "$ROOT/metal/src/engine/msl.hidden" "$ROOT/metal/src/engine/msl"
echo "== done: $DIR (rc=$rc)"
