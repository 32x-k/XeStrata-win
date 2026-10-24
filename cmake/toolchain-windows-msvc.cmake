# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# cmake/toolchain-windows-msvc.cmake - build the Windows engine on Linux (docs/DEVTOOLS.md#windows-cross-build).
#
# intel/llvm's SYCL runtime on Windows takes the MSVC ABI only, so the target is x86_64-pc-windows-msvc: intel/llvm's
# clang++ (-DCMAKE_CXX_COMPILER) compiles, lld-link links, the MSVC CRT and the Windows SDK come from xwin's splat
# (STRATA_XWIN) and the SYCL headers and import libraries from an intel/llvm Windows build (STRATA_SYCL_WINDOWS).
# The result is not run here: the device code is the same SPIR-V, the host code needs a Windows PC.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

set(STRATA_XWIN "$ENV{STRATA_XWIN}" CACHE PATH "xwin splat's output: crt/ and sdk/")
set(STRATA_SYCL_WINDOWS "$ENV{STRATA_SYCL_WINDOWS}" CACHE PATH
    "an intel/llvm Windows build (sycl_windows.tar.gz): include/ and lib/")
foreach(dir STRATA_XWIN STRATA_SYCL_WINDOWS)
  if(NOT IS_DIRECTORY "${${dir}}")
    message(FATAL_ERROR "${dir}: set it (-D${dir}=... or the environment) to an existing folder")
  endif()
endforeach()
# try_compile projects see these and the compiler, not the cache
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES STRATA_XWIN STRATA_SYCL_WINDOWS)

set(CMAKE_C_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
if(CMAKE_CXX_COMPILER AND NOT CMAKE_C_COMPILER)
  string(REGEX REPLACE "clang\\+\\+$" "clang" CMAKE_C_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

find_program(STRATA_LLD_LINK NAMES lld-link PATHS /usr/lib/llvm-22/bin /usr/lib/llvm-21/bin /usr/lib/llvm-20/bin
             REQUIRED)
# its LLVM's own folder (/usr/bin/lld-link is a link into /usr/lib/llvm-<version>/bin), where llvm-ar and the others are
get_filename_component(_strata_llvm_bin "${STRATA_LLD_LINK}" REALPATH)
get_filename_component(_strata_llvm_bin "${_strata_llvm_bin}" DIRECTORY)
find_program(CMAKE_AR NAMES llvm-ar HINTS "${_strata_llvm_bin}" REQUIRED)
find_program(CMAKE_RC_COMPILER NAMES llvm-rc HINTS "${_strata_llvm_bin}")
find_program(CMAKE_MT NAMES llvm-mt HINTS "${_strata_llvm_bin}")

# the SYCL headers before the compiler's own SYCL headers (Linux's build of the same release); the CRT and the SDK
# after clang's built-in headers, as on Windows (MSVC's immintrin.h declares the intrinsics as functions).
# -nostdlibinc: the SYCL device compile would otherwise search the build machine's /usr/include.
set(_strata_win_flags "-nostdlibinc -isystem \"${STRATA_SYCL_WINDOWS}/include\"")
foreach(dir "${STRATA_XWIN}/crt/include" "${STRATA_XWIN}/sdk/include/ucrt" "${STRATA_XWIN}/sdk/include/um"
            "${STRATA_XWIN}/sdk/include/shared")
  string(APPEND _strata_win_flags " -idirafter \"${dir}\"")
endforeach()
set(CMAKE_C_FLAGS_INIT "${_strata_win_flags}")
# llvm-rc's (a library's version resource, xz's among them)
set(CMAKE_RC_FLAGS_INIT "-I\"${STRATA_XWIN}/sdk/include/um\" -I\"${STRATA_XWIN}/sdk/include/shared\" \
-I\"${STRATA_XWIN}/crt/include\" -I\"${STRATA_XWIN}/sdk/include/ucrt\"")
set(CMAKE_CXX_FLAGS_INIT "${_strata_win_flags}")
# CMake adds -fuse-ld=lld-link itself and clang finds it through -B: a folder with lld-link alone, as the LLVM folder's
# other tools (clang-offload-bundler) would take the place of intel/llvm's
set(_strata_lld_dir "${CMAKE_BINARY_DIR}/strata-lld-link")
file(MAKE_DIRECTORY "${_strata_lld_dir}")
file(CREATE_LINK "${STRATA_LLD_LINK}" "${_strata_lld_dir}/lld-link" SYMBOLIC)
# the AMD device code's link (ELF) takes -fuse-ld=lld-link as the name ld.lld-link: the compiler's own ld.lld under it
if(CMAKE_CXX_COMPILER)
  get_filename_component(_strata_cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  if(EXISTS "${_strata_cxx_bin}/ld.lld")
    file(CREATE_LINK "${_strata_cxx_bin}/ld.lld" "${_strata_lld_dir}/ld.lld-link" SYMBOLIC)
  endif()
  # and its lld (-flavor gnu -m elf64_amdgpu), found here first, ignores the libraries the device objects name: the MSVC
  # and Windows SDK headers' #pragma comment(lib, ...) and CMake's --dependent-lib=msvcrt reach the device compilation
  # too, and intel/llvm v7.2.0's lld (LLVM 23) refuses a library it cannot find there (msvcrt, msvcprt, uuid.lib)
  if(EXISTS "${_strata_cxx_bin}/lld")
    file(WRITE "${_strata_lld_dir}/lld"
         "#!/bin/sh
case \" $* \" in *\" elf64_amdgpu \"*) exec \"${_strata_cxx_bin}/lld\" \"$@\" --no-dependent-libraries ;; esac
"
         "exec \"${_strata_cxx_bin}/lld\" \"$@\"
")
    execute_process(COMMAND chmod 755 "${_strata_lld_dir}/lld")
  endif()
endif()
set(_strata_win_link "-B\"${_strata_lld_dir}\" -L\"${STRATA_XWIN}/crt/lib/x86_64\" \
-L\"${STRATA_XWIN}/sdk/lib/um/x86_64\" -L\"${STRATA_XWIN}/sdk/lib/ucrt/x86_64\" -L\"${STRATA_SYCL_WINDOWS}/lib\"")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_strata_win_link}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_strata_win_link}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_strata_win_link}")

# the checks' test programs as Release builds: a Debug one links the SYCL runtime's debug library (sycl9d.lib), which
# a SYCL tree built for Release alone does not have
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)

set(CMAKE_FIND_ROOT_PATH "${STRATA_XWIN}" "${STRATA_SYCL_WINDOWS}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
