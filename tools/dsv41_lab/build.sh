#!/bin/bash
# build the trellis kernel lab on a CUDA box: tools/dsv41_lab/build.sh [arch, default sm_86]
set -e
cd "$(dirname "$0")"
ARCH=${1:-sm_86}
NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
mkdir -p out
g++ -O2 -std=c++17 -I../../third_party/ggml -c ../trellis/tqenc.cpp -o out/tqenc.o
$NVCC -O3 -std=c++17 -arch=$ARCH -lineinfo -o out/tqk_lab tqk_lab.cu out/tqenc.o -lpthread
