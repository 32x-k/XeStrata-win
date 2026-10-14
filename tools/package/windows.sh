#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/windows.sh - the inside of `tools/package/build.sh windows cuda12.4`: run as root in a fresh Ubuntu 26.04
# container, with the commit's files in /src, /out for the 7z and /cache for the downloads kept between builds (xwin's
# and NuGet's).  Everything is built here, on Linux, for Windows (docs/DEVTOOLS.md#windows-cross-build): intel/llvm's
# contrib toolchain (whose tools the next step runs), intel/llvm's SYCL runtime for Windows, the engine (the
# contrib-llvm mode: Intel's GPUs through Level Zero, NVIDIA's through CUDA; oneMath is not on Windows yet), the
# launchers (windows/launcher, .NET), and Python's embeddable distribution with the packages setup.py uses;
# tools/package/windows_bundle.py makes the folder, its notices and the 7z.
set -eu
src=/src
variant=$XESTRATA_VARIANT
cuda_ver=${variant#cuda}
# as the cuda12.4 deb (tools/package/container.sh): from Pascal on
archs="sm_60;sm_61;sm_70;sm_75;sm_80;sm_86;sm_89;sm_90"
xwin_ver=0.10.0
python_ver=3.14.8

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
# llvm-21 and lld-21: llvm-lib, llvm-rc, llvm-mt, llvm-ml and lld-link, which intel/llvm does not install;
# nvidia-cuda-toolkit: CUDA 12.4, for intel/llvm's CUDA target and the engine's NVIDIA code (not free software);
# dotnet-sdk-10.0 for the launchers
apt-get install -y -q --no-install-recommends git cmake ninja-build g++ libhwloc-dev libzstd-dev python3 python3-pip \
  curl ca-certificates unzip ccache lld lld-21 llvm-21 dotnet-sdk-10.0 7zip nvidia-cuda-toolkit
cuda_path=/usr/lib/cuda

# build.sh's compiler cache, shared with the Ubuntu packages (container.sh)
if [ -n "${CCACHE_DIR:-}" ]; then
  export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CCACHE_COMPILERCHECK=content
  trap 'chown -R "$XESTRATA_UID:$XESTRATA_GID" "$CCACHE_DIR" /cache' EXIT
  ccache -z
fi
jobs=${XESTRATA_JOBS:-$(nproc)}

# llama.cpp at the commit setup.py pins: ggml for the engine, gguf-py for the tools
commit=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)"$/\1/p' "$src/setup.py")
curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/$commit.zip"
unzip -q /tmp/llama.zip -d /tmp/llama
mv "/tmp/llama/llama.cpp-$commit" "$src/third_party/main/llama.cpp"
llama=$src/third_party/main/llama.cpp

# intel/llvm: the contrib toolchain for this machine, then the SYCL runtime for Windows with its tools
python3 "$src/tools/intel_llvm_build.py" --contrib --yes ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}

# the MSVC CRT and the Windows SDK (not free software: Microsoft's license, accepted here; never in the 7z)
mkdir -p /cache/xwin /build
xwin_tar=xwin-$xwin_ver-x86_64-unknown-linux-musl.tar.gz
curl -fsSL -o "/build/$xwin_tar" "https://github.com/Jake-Shadle/xwin/releases/download/$xwin_ver/$xwin_tar"
curl -fsSL -o "/build/$xwin_tar.sha256" "https://github.com/Jake-Shadle/xwin/releases/download/$xwin_ver/$xwin_tar.sha256"
# the release's .sha256 holds the hash alone
echo "$(cat "/build/$xwin_tar.sha256")  /build/$xwin_tar" | sha256sum -c
tar -xzf "/build/$xwin_tar" -C /build
"/build/xwin-$xwin_ver-x86_64-unknown-linux-musl/xwin" --accept-license --cache-dir /cache/xwin --arch x86_64 \
  splat --copy --output /build/xwin   # copied: /cache is another file system
python3 "$src/tools/intel_llvm_build.py" --windows --xwin /build/xwin ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}
sycl=$src/.tools/intel-llvm-win/install

# the engine
cmake -G Ninja -S "$src" -B /build/engine -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$src/cmake/toolchain-windows-msvc.cmake" \
  -DCMAKE_CXX_COMPILER="$src/.tools/intel-llvm-contrib/install/bin/clang++" -DSTRATA_LICENSE=contrib \
  -DSTRATA_XWIN=/build/xwin -DSTRATA_SYCL_WINDOWS="$sycl" -DSTRATA_CUDA_ARCHS="$archs" -DSTRATA_CUDA_PATH="$cuda_path" \
  -DSTRATA_CUDA_PTX= -DSTRATA_ONEMKL=OFF -DSTRATA_HIP_ARCHS= -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR="$llama"
cmake --build /build/engine --target strata strata-device -j "$jobs"

# the launchers (framework-dependent: Windows' .NET 10 runtime runs them); Avalonia's build telemetry off
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 AVALONIA_TELEMETRY_OPTOUT=1 NUGET_PACKAGES=/cache/nuget
for p in Gui Cli; do
  dotnet publish "$src/windows/launcher/$p/$p.fsproj" -c Release -r win-x64 --self-contained false -o /build/launcher
done

# Python's embeddable distribution and the Windows wheels of requirements.txt (without cmake and ninja, which build
# the engine from source on Linux) and colorama, which tqdm takes on Windows
curl -fsSL -o "/build/python-$python_ver-embed-amd64.zip" \
  "https://www.python.org/ftp/python/$python_ver/python-$python_ver-embed-amd64.zip"
grep -vE '^(cmake|ninja)==' "$src/requirements.txt" > /build/requirements.txt
echo colorama >> /build/requirements.txt
pyver=$(echo "$python_ver" | cut -d. -f1-2)
python3 -m pip download -q --only-binary=:all: --platform win_amd64 --python-version "$pyver" --implementation cp \
  -r /build/requirements.txt -d /build/wheels

version=$(sed -n 's/^set(STRATA_UPSTREAM_VERSION "\(.*\)")$/\1/p' "$src/CMakeLists.txt")-$(sed -n \
  's/^set(XESTRATA_REVISION \([0-9]*\))$/\1/p' "$src/CMakeLists.txt")
python3 "$src/tools/package/windows_bundle.py" --src "$src" --engine /build/engine --sycl "$sycl" \
  --launcher /build/launcher --nuget /cache/nuget --python-zip "/build/python-$python_ver-embed-amd64.zip" \
  --wheels /build/wheels --llama "$llama" --version "$version" --cuda-archs "$archs" --out /build/out
cp /build/out/*.7z /out/

{
  echo "commit: $XESTRATA_COMMIT"
  echo "image: $XESTRATA_IMAGE"
  echo "variant: windows $variant (contrib, CUDA $cuda_ver)"
  echo "intel/llvm: $(sed -n 's/.*"tag": "\(.*\)".*/\1/p' "$sycl/XESTRATA.json") (Windows, cross-built)"
  echo "xwin: $xwin_ver"
  echo "python: $python_ver"
  echo "dotnet: $(dotnet --version)"
  echo "llama.cpp: $commit"
  echo "jobs: ${XESTRATA_JOBS:-default}"
  [ -z "${CCACHE_DIR:-}" ] || ccache --print-stats | awk -F'\t' '/^(direct|preprocessed)_cache_hit\t/ { h += $2 }
    /^cache_miss\t/ { m = $2 } END { print "ccache: " h + 0 " hits, " m + 0 " misses" }'
  echo "wheels:"
  for w in /build/wheels/*.whl; do echo "  $(basename "$w")"; done
  echo "packages:"
  dpkg-query -W -f '  ${Package} ${Version}\n'
} > /out/BUILDINFO
chown -R "$XESTRATA_UID:$XESTRATA_GID" /out
