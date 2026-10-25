# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# cmake/StrataHip.cmake - the AMD architectures (intel/llvm's HIP target) and ROCm's device libraries.
#
# ROCm is free software, so the AMD code is built in the free and contrib modes alike.  The engine's kernels are
# written for wave32 GPUs with integer dot instructions: RDNA2 (gfx103x) and later.

include(${CMAKE_CURRENT_LIST_DIR}/StrataCuda.cmake)   # strata_sycl_compiles

# ROCm's CMake packages (hip's, rocBLAS's and hipBLASLt's, which oneMath's rocBLAS backend asks for) are in the
# distribution's prefix where it packages them (/usr/lib64/cmake, /usr/lib/<triplet>/cmake), and under AMD's own tree,
# /opt/rocm/lib/cmake, where ROCm itself was installed (Arch's packages too) - a prefix CMake does not search by
# itself.  This file is included before oneMath is added (CMakeLists.txt), so the backends see it.
if(EXISTS /opt/rocm/lib/cmake)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
endif()

# This PC's AMD GPUs, as the ISA names the compiler takes (gfx1200), from the kernel's KFD topology:
# gfx_target_version is major * 10000 + minor * 100 + stepping, the last two in hex in the name.
function(strata_hip_detect out)
  set(archs "")
  file(GLOB nodes /sys/class/kfd/kfd/topology/nodes/*/properties)
  foreach(props IN LISTS nodes)
    file(STRINGS "${props}" version REGEX "^gfx_target_version ")
    string(REGEX REPLACE "^gfx_target_version " "" version "${version}")
    if(version AND NOT version EQUAL 0)
      math(EXPR major "${version} / 10000")
      math(EXPR minor "(${version} / 100) % 100" OUTPUT_FORMAT HEXADECIMAL)
      math(EXPR step "${version} % 100" OUTPUT_FORMAT HEXADECIMAL)
      string(REGEX REPLACE "^0x" "" minor "${minor}")
      string(REGEX REPLACE "^0x" "" step "${step}")
      list(APPEND archs "gfx${major}${minor}${step}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES archs)
  set(${out} "${archs}" PARENT_SCOPE)
endfunction()

# Whether the engine's kernels run on `arch`: RDNA2 and later (RDNA1 lacks the dot instructions, GCN and CDNA are
# wave64).
function(strata_hip_supported arch out)
  if(arch MATCHES "^gfx(103[0-9a-f]|11[0-9a-f][0-9a-f]|12[0-9a-f][0-9a-f])$")
    set(${out} ON PARENT_SCOPE)
  else()
    set(${out} OFF PARENT_SCOPE)
  endif()
endfunction()

# The architectures the distribution's rocBLAS has code for, of those the engine runs on and the SYCL compiler builds
# for (STRATA_HIP_ARCHS=rocblas, for a package: the engine's dense products run in rocBLAS, so a GPU it lacks cannot
# run the engine), from its kernels' file names (Fedora: /usr/lib64/rocblas/library; Ubuntu:
# /usr/lib/<triplet>/rocblas/<version>/library).
function(strata_hip_rocblas out)
  file(GLOB files /usr/lib64/rocblas/library/* /usr/lib/*/rocblas/*/library/* /opt/rocm/lib/rocblas/library/*)
  set(archs "")
  foreach(file_path IN LISTS files)
    get_filename_component(name "${file_path}" NAME)
    if(name MATCHES "gfx[0-9a-f]+")
      set(arch ${CMAKE_MATCH_0})
      strata_hip_supported(${arch} ok)
      if(ok)
        list(APPEND archs ${arch})
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES archs)
  list(SORT archs)
  set(built "")
  foreach(arch IN LISTS archs)
    strata_hip_compiler_knows(${arch} ok)
    if(ok)
      list(APPEND built ${arch})
    else()
      message(STATUS "AMD GPU ${arch}: rocBLAS has code for it, the SYCL compiler has no target for it")
    endif()
  endforeach()
  set(${out} "${built}" PARENT_SCOPE)
endfunction()

# Whether the SYCL compiler has a target for `arch` (amd_gpu_gfx1200): intel/llvm 7.1.1 has none for gfx1152 and
# gfx1153, which ROCm 7.1's rocBLAS has code for.  Asked of the driver alone (-###), without ROCm's device libraries.
function(strata_hip_compiler_knows arch out)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -fsycl -fsycl-targets=amd_gpu_${arch} -nogpulib "-###" -x c++ -c
                          /dev/null -o /dev/null
                  RESULT_VARIABLE r OUTPUT_QUIET ERROR_QUIET)
  if(r EQUAL 0)
    set(${out} ON PARENT_SCOPE)
  else()
    set(${out} OFF PARENT_SCOPE)
  endif()
endfunction()

# Whether the SYCL compiler builds a kernel for `arch` with the device libraries in `dir` (a kernel with ocml's math,
# which links them), and if not, why: bitcode of a newer LLVM than the compiler's (it reads its own and older), files
# the compiler asks for that are missing (ROCm 7.14 dropped the oclc_correctly_rounded_sqrt and oclc_daz_opt ones,
# which LLVM 22 still links), or the compiler's own first error.  Cached per folder and architecture.
function(strata_hip_device_libs_check dir arch ok_out why_out)
  string(MD5 key "${dir} ${arch} ${CMAKE_CXX_COMPILER}")
  set(var STRATA_HIP_DEVICE_LIBS_${key})
  if(NOT DEFINED ${var})
    set(src "${CMAKE_BINARY_DIR}/strata_hip_device_libs_probe.cpp")
    file(WRITE "${src}" "#include <sycl/sycl.hpp>\nint main() {\n  sycl::queue q;\n"
                        "  float* p = sycl::malloc_shared<float>(1, q);\n"
                        "  q.single_task([=] { p[0] = sycl::sqrt(p[0]) * sycl::exp(p[0]); }).wait();\n"
                        "  sycl::free(p, q);\n}\n")
    set(flags -fsycl -fsycl-targets=amd_gpu_${arch} --rocm-device-lib-path=${dir})
    message(CHECK_START "Building an AMD kernel (${arch}) with the device libraries in ${dir}")
    strata_try_compile_defs("${flags}" defs)
    try_compile(result "${CMAKE_BINARY_DIR}/strata_hip_device_libs_probe" SOURCES "${src}"
                COMPILE_DEFINITIONS ${defs} LINK_OPTIONS ${flags} OUTPUT_VARIABLE out)
    set(why "")
    if(result)
      message(CHECK_PASS "yes")
    else()
      string(REGEX MATCH "Producer: '[^']*' *Reader: '[^']*'" versions "${out}")
      string(REGEX REPLACE "^gfx" "oclc_isa_version_" isa "${arch}")
      set(missing "")
      foreach(name ocml ockl oclc_correctly_rounded_sqrt_on oclc_daz_opt_on oclc_finite_only_off
                   oclc_unsafe_math_off oclc_wavefrontsize64_off ${isa})
        if(NOT EXISTS "${dir}/${name}.bc")
          list(APPEND missing ${name}.bc)
        endif()
      endforeach()
      if(versions OR out MATCHES "Unknown attribute kind")
        set(why "its bitcode is of a newer LLVM than the SYCL compiler's, which cannot read it (${versions})")
      elseif(missing)
        list(JOIN missing ", " missing)
        set(why "the SYCL compiler needs files it lacks: ${missing}")
      else()
        string(REGEX MATCH "[^\n]*error:[^\n]*" first "${out}")
        set(why "the SYCL compiler did not build with it: ${first}")
      endif()
      message(CHECK_FAIL "no: ${why}")              # said also when a later folder is taken
    endif()
    set(${var} "${result}" CACHE INTERNAL "")
    set(${var}_WHY "${why}" CACHE INTERNAL "")
  endif()
  set(${ok_out} "${${var}}" PARENT_SCOPE)
  set(${why_out} "${${var}_WHY}" PARENT_SCOPE)
endfunction()

# ROCm's device libraries for `arch`'s kernels, given (STRATA_ROCM_DEVICE_LIBS) or the first folder found that a test
# kernel builds with (strata_hip_device_libs_check).  The search before this one kept the folder it found in the cache
# under the same name, which then read as given: such an entry (its help text) is searched for again.
function(strata_hip_device_libs arch out)
  get_property(help CACHE STRATA_ROCM_DEVICE_LIBS PROPERTY HELPSTRING)
  if(help STREQUAL "ROCm's device libraries (ockl.bc), for the AMD code")
    unset(STRATA_ROCM_DEVICE_LIBS CACHE)
  endif()
  if(STRATA_ROCM_DEVICE_LIBS)                         # given: used as it is, or the reason it cannot be
    strata_hip_device_libs_check("${STRATA_ROCM_DEVICE_LIBS}" ${arch} ok why)
    if(NOT ok)
      message(FATAL_ERROR "STRATA_ROCM_DEVICE_LIBS=${STRATA_ROCM_DEVICE_LIBS}: ${why}")
    endif()
    set(libs "${STRATA_ROCM_DEVICE_LIBS}")
  else()
    # ROCm's device libraries (ockl.bc and the others): AMD's trees and the distribution's, which keeps them with
    # its ROCm clang (Fedora: /usr/lib64/rocm/llvm/lib/clang/<N>/lib/amdgcn/bitcode; Ubuntu:
    # /usr/lib/llvm-<N>/lib/clang/<N>/amdgcn/bitcode), and last the SYCL compiler's own (tools/intel_llvm_build.py
    # builds them with it).  The first one the compiler builds a kernel with: a clang reads bitcode of its own LLVM
    # or older, and ROCm 7.14's (LLVM 23, without files LLVM 22 asks for) stopped intel/llvm 7.1.1's build on an
    # RX 7900 XTX (a field report).  The ROCm's own first, so that a ROCm whose libraries the compiler reads (Fedora
    # 44's and Ubuntu 26.04's 7.1.1) gets the code it was linked with before, matched with its runtime; the
    # compiler's own only where none of them builds
    get_filename_component(own "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(own "${own}/../amdgcn/bitcode" ABSOLUTE)
    file(GLOB dirs /opt/rocm/amdgcn/bitcode /opt/rocm*/amdgcn/bitcode /opt/rocm*/core-*/amdgcn/bitcode
                   /opt/rocm*/lib/llvm/amdgcn/bitcode /opt/rocm*/core-*/lib/llvm/amdgcn/bitcode
                   /opt/rocm*/lib/llvm/lib/clang/*/lib/amdgcn/bitcode
                   /usr/lib64/rocm/llvm/lib/clang/*/lib/amdgcn/bitcode
                   /usr/lib/llvm-*/lib/clang/*/amdgcn/bitcode /usr/lib/llvm-*/lib/clang/*/lib/amdgcn/bitcode)
    list(SORT dirs COMPARE NATURAL ORDER DESCENDING)
    if(CMAKE_CROSSCOMPILING AND WIN32)
      set(dirs "")                                  # this machine's are a Linux host's
    endif()
    set(libs "")
    set(tried "")
    foreach(dir IN ITEMS ${dirs} "${own}")
      get_filename_component(real "${dir}" REALPATH)
      if(libs OR NOT EXISTS "${real}/ockl.bc" OR real IN_LIST tried)
        continue()
      endif()
      list(APPEND tried "${real}")
      strata_hip_device_libs_check("${real}" ${arch} ok why)
      if(ok)
        set(libs "${real}")
      else()
        string(APPEND reasons "\n  ${real}: ${why}")
      endif()
    endforeach()
    if(NOT libs AND NOT tried)
      message(FATAL_ERROR "STRATA_HIP_ARCHS (${arch}): ROCm's device libraries (ockl.bc) were not found: build "
                          "intel/llvm with ROCm's HIP installed (tools/intel_llvm_build.py builds them with it), "
                          "install them (Fedora: rocm-device-libs; Ubuntu: rocm-device-libs-21), or give "
                          "-DSTRATA_ROCM_DEVICE_LIBS=")
    elseif(NOT libs)
      message(FATAL_ERROR "STRATA_HIP_ARCHS (${arch}): the SYCL compiler builds with none of ROCm's device libraries "
                          "found:${reasons}\nBuild intel/llvm again with tools/intel_llvm_build.py --rebuild (it "
                          "builds device libraries of its own LLVM), or give -DSTRATA_ROCM_DEVICE_LIBS= a folder of "
                          "an older ROCm's")
    endif()
    message(STATUS "ROCm's device libraries: ${libs}")
  endif()
  set(${out} "${libs}" PARENT_SCOPE)
endfunction()

# The SYCL targets (amd_gpu_gfx1200) and options (ROCm's device libraries) for `archs`, after checking that each is
# supported and that the SYCL compiler builds for them.
function(strata_hip_choose archs targets_out opts_out)
  set(targets "")
  foreach(arch IN LISTS archs)
    strata_hip_supported(${arch} ok)
    if(NOT ok)
      message(FATAL_ERROR "STRATA_HIP_ARCHS: ${arch} is not an RDNA2 or later AMD GPU (gfx103x, gfx11xx, gfx12xx); "
                          "set the others with -DSTRATA_HIP_ARCHS=, or none with -DSTRATA_HIP_ARCHS=\"\"")
    endif()
    list(APPEND targets amd_gpu_${arch})
  endforeach()
  strata_win_spirv_check(amdgcn-amd-amdhsa)
  list(GET archs 0 first)
  strata_hip_device_libs(${first} libs)
  set(opts "--rocm-device-lib-path=${libs}")
  list(JOIN targets "," str)
  string(MD5 key "${str} ${opts}")
  strata_sycl_compiles("-fsycl-targets=spir64,${str};${opts}" STRATA_SYCL_HIP_${key} ok)
  if(NOT ok)
    message(FATAL_ERROR "STRATA_HIP_ARCHS=${archs}: the SYCL compiler does not build for these AMD GPUs: build "
                        "intel/llvm with ROCm's HIP installed (tools/intel_llvm_build.py, docs/DEVTOOLS.md), or "
                        "configure without them (-DSTRATA_HIP_ARCHS=\"\")")
  endif()
  set(${targets_out} "${targets}" PARENT_SCOPE)
  set(${opts_out} "${opts}" PARENT_SCOPE)
endfunction()

# AMD's ROCm tree with CMake packages (hip, rocblas, hipblaslt), for CMAKE_PREFIX_PATH: /opt/rocm or the versioned
# /opt/rocm-<version>, or the parts ROCm 7.14 splits its tree into (/opt/rocm/core-7.14, where hip-config.cmake is;
# /opt/rocm itself has no lib/cmake there, and find_package(hip) failed: a field report).  One tree, so that two
# installed versions are not mixed, in setup.py's order (rocm_roots): /opt/rocm, its alternatives link core, the newest
# core-<version>, the newest /opt/rocm-<version>; NATURAL sorts core-10.1 after core-7.14.
function(strata_hip_prefixes out)
  file(GLOB cores LIST_DIRECTORIES true /opt/rocm/core-*)
  file(GLOB versioned LIST_DIRECTORIES true /opt/rocm-*)
  list(SORT cores COMPARE NATURAL ORDER DESCENDING)
  list(SORT versioned COMPARE NATURAL ORDER DESCENDING)
  set(${out} "" PARENT_SCOPE)
  foreach(root IN ITEMS /opt/rocm /opt/rocm/core ${cores} ${versioned})
    if(IS_DIRECTORY "${root}/lib/cmake")
      set(${out} "${root}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

# ROCm's clang (AMD's tree's, or the distribution's ROCm one: Fedora's /usr/lib64/rocm/llvm, Ubuntu's
# /usr/lib/llvm-<N>), for HIP's CMake package (HIP_CXX_COMPILER): it links clang's runtime builtins where that compiler
# has them, and intel/llvm's install has none, so asked with intel/llvm's clang it named a file that is not there.  The
# newest one whose builtins exist; empty when there is none.
function(strata_hip_cxx out)
  file(GLOB compilers /opt/rocm/llvm/bin/clang++ /opt/rocm/lib/llvm/bin/clang++ /opt/rocm*/core-*/llvm/bin/clang++
                      /opt/rocm*/core-*/lib/llvm/bin/clang++ /usr/lib64/rocm/llvm/bin/clang++
                      /usr/lib/llvm-*/bin/clang++)
  list(SORT compilers COMPARE NATURAL ORDER DESCENDING)
  foreach(cxx IN LISTS compilers)
    execute_process(COMMAND "${cxx}" -print-libgcc-file-name --rtlib=compiler-rt
                    OUTPUT_VARIABLE builtins OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE r)
    if(r EQUAL 0 AND EXISTS "${builtins}")
      set(${out} "${cxx}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out} "" PARENT_SCOPE)
endfunction()
