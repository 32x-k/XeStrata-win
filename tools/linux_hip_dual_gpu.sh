#!/usr/bin/env bash
# Configure and build: tools/linux_hip_dual_gpu.sh build
# Run a built executable: tools/linux_hip_dual_gpu.sh run strata-device --list-devices
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
llvm=${INTEL_LLVM_DIR:-/home/x/projects/xestrata/.tools/intel-llvm/install}
case ${1:-build} in
  build)
    cmake -S "$root" -B "$root/build/hip" \
      -DCMAKE_C_COMPILER="$llvm/bin/clang" -DCMAKE_CXX_COMPILER="$llvm/bin/clang++" \
      '-DSTRATA_HIP_ARCHS=gfx1201;gfx1151' -DSTRATA_AOTRITON_ROOT=/nonexistent
    cmake --build "$root/build/hip" --target strata strata-device -j6
    ;;
  run)
    shift
    if (($# == 0)); then echo 'Usage: linux_hip_dual_gpu.sh run <executable> [args...]' >&2; exit 2; fi
    exe=$1; shift
    export LD_LIBRARY_PATH="$llvm/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
    exec "$root/build/hip/$exe" "$@"
    ;;
  *) echo 'Usage: linux_hip_dual_gpu.sh {build|run <executable> [args...]}' >&2; exit 2 ;;
esac
