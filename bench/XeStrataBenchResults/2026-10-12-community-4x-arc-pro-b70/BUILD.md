# How this record's toolchain and engine were built

Short notes, because two of them cost time and are worth passing on.

## The compiler: intel/llvm v7.1.1 from source

Ubuntu 26.04's own DPC++ (6.2, `dpclang++`) is refused by CMake: its SYCL runtime reports no XMX for
the Arc Pro B70, and intel/llvm added that in v7.0.0. So the toolchain came from this project's own
script, unmodified:

```
python3 tools/intel_llvm_build.py --yes --jobs 24
```

- It clones the v7.1.1 tag (`504366f4b82ff00bbac7b0c956635eed8bc599d8`), builds, and installs into
  `.tools/intel-llvm/install` — **3.4 GB** total, libsycl 9, and it keeps `XESTRATA.json` with the
  tag, the commit and each GPU's matrix engines as its probe saw them.
- The script's own need, beyond the packages `docs/DEVTOOLS.md` lists: **`lld`**. Without it the link
  step has no linker to use.
- **`libstdc++-16-dev` is also needed on Ubuntu 26.04**, and its absence fails late and confusingly.
  The distro ships g++ 15 with its C++ headers in `/usr/include/c++/15`, and also installs `gcc-16`
  plus `libgcc-16-dev` *without* `libstdc++-16-dev`. clang-22 picks the newest GCC install directory
  (16), finds no C++ headers there, and the device-library bitcode targets cannot compile:

```
src/libdevice/imf/../imf_impl_utils.hpp:12:10: fatal error: 'cstddef' file not found
lib/dpcpp-7/sycl/libsycl-bfloat16.bc: .../sycl/stl_wrappers/cassert:14:15: fatal error: 'cassert' file not found
```

  The output even says so in a warning: *"'/usr/lib/gcc/x86_64-linux-gnu/15' would be chosen over
  '/usr/lib/gcc/x86_64-linux-gnu/16'"*. Installing `libstdc++-16-dev` fixes it, and an interrupted
  build resumes: ~2,900 objects were reused and only the failed targets rebuilt (~1 minute on 24
  threads).

## The engine: both modes, with tests

Configured with ninja and ccache, `CMAKE_CXX_COMPILER` pointing at the toolchain's `clang++`:

```
cmake -G Ninja -S . -B build/free          -DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++ \
      -DSTRATA_LICENSE=free  -DSTRATA_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake -G Ninja -S . -B build/contrib-llvm  -DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++
      -DSTRATA_LICENSE=contrib -DSTRATA_BUILD_TESTS=ON
cmake --build build/free -j 24 && (cd build/free && ctest)
cmake --build build/contrib-llvm -j 24 && (cd build/contrib-llvm && ctest)
```

- `STRATA_BUILD_TESTS=ON` is needed explicitly: the published source ships no `tests/CMakeLists.txt`,
  so the test suite defaults to OFF.
- **92/92 CTest cases pass in each mode.** `contrib-llvm` needs oneMKL (`intel-oneapi-mkl-devel`,
  installed here as 2026.1); it stops without it on a machine with an Intel GPU.
- The binary this record measured is the `contrib-llvm` one, installed by `setup.sh` as `engine/strata`
  with `BUILD.json` recording the compiler, mode and oneMKL. The `free` binary was built and passed its
  tests but was not measured.
