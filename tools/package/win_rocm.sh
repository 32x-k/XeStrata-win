#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/win_rocm.sh DEST - rocBLAS and hipBLASLt for a Windows cross build (STRATA_ROCM_WINDOWS_LIBS, oneMath's
# rocBLAS backend): their headers and those of hipblas-common (DEST/include), and Windows' import libraries
# (DEST/lib/rocblas.lib, hipblaslt.lib), made with llvm-dlltool from the C functions this machine's Linux libraries
# export, which the Windows DLLs of AMD's Python packages (rocm-sdk-libraries-<family>: rocblas.dll, libhipblaslt.dll)
# export by the same names; and their license texts (DEST/LICENSE-<package>.txt: the headers' inline code goes into the
# backend's DLL).  Needs Ubuntu's librocblas-dev, libhipblaslt-dev and libhipblas-common-dev, and llvm-21.
set -eu
dest=$1
lib=/usr/lib/x86_64-linux-gnu
rm -rf "$dest"
mkdir -p "$dest/include" "$dest/lib"
for d in rocblas hipblaslt hipblas-common; do
  cp -r "/usr/include/$d" "$dest/include/"
done
for p in librocblas-dev libhipblaslt-dev libhipblas-common-dev; do
  cp "/usr/share/doc/$p/copyright" "$dest/LICENSE-$p.txt"
done
implib() {   # Linux library, Windows DLL, import library's name, the C++ functions used by their MSVC names
  so=$1 dll=$2 name=$3
  shift 3
  { echo "LIBRARY $dll"
    echo EXPORTS
    # the C functions, without their symbol versions; the Itanium-mangled C++ ones have other names on Windows
    nm -D --defined-only --format=posix "$so" |
      awk '($2 == "T" || $2 == "W") && $1 !~ /^_Z/ { sub(/@.*/, "", $1); print "  " $1 }' | sort -u
    for f in "$@"; do echo "  $f"; done
  } > "$dest/lib/$name.def"
  llvm-dlltool-21 -m i386:x86-64 -d "$dest/lib/$name.def" -l "$dest/lib/$name.lib"
}
implib "$lib/librocblas.so" rocblas.dll rocblas
# hipblaslt_ext::getIndexFromAlgo (the backend's tuning, rocblas_hipblaslt.cpp), as libhipblaslt.dll exports it
implib "$lib/libhipblaslt.so" libhipblaslt.dll hipblaslt '?getIndexFromAlgo@hipblaslt_ext@@YAHAEAU_hipblasLtMatmulAlgo_t@@@Z'
