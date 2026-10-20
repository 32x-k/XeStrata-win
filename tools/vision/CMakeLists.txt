# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/main/llama.cpp
#   cmake --build build-vision-cpu --target strata-vision
#
# On the GPU through Vulkan (ggml-vulkan; free software: Mesa's drivers, libvulkan-dev, glslc):
#   cmake -S tools/vision -B build-vision-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_DIR=third_party/main/llama.cpp -DSTRATA_VISION_VULKAN=ON
#
# On Windows (the zip's strata-vision-cpu.exe and strata-vision-vulkan.exe): the same commands from a Visual Studio
# x64 prompt; the Vulkan one needs the LunarG Vulkan SDK (VULKAN_SDK: its headers, vulkan-1.lib and glslc), and runs
# on the GPU driver's vulkan-1.dll.
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT WIN32)
  message(FATAL_ERROR "Strata vision supports Linux and Windows only")
endif()

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/main/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_VULKAN "run the vision encoder on the GPU through Vulkan (builds ggml-vulkan)" OFF)
option(STRATA_PORTABLE "ggml for any AVX2 CPU; OFF: -march=native" ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${STRATA_VISION_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(WIN32 AND STRATA_VISION_VULKAN)
  # --gpu-pci on a driver without VK_EXT_pci_bus_info: the card by its adapter LUID (D3DKMT in gdi32, and Vulkan's
  # deviceLUID), see vk_index_by_pci in strata_vision.cpp
  find_package(Vulkan REQUIRED)
  target_compile_definitions(strata-vision PRIVATE STRATA_VISION_VULKAN)
  target_link_libraries(strata-vision PRIVATE Vulkan::Vulkan gdi32)
endif()

# opens a GGUF that has tensor types llama.cpp does not know (a Gyro quantization's TQK6 / TQK7) for its vocabulary only
add_executable(strata-vision-gguf-test EXCLUDE_FROM_ALL gguf_meta_test.cpp)
target_link_libraries(strata-vision-gguf-test PRIVATE llama)
target_compile_features(strata-vision-gguf-test PRIVATE cxx_std_17)
set_target_properties(strata-vision-gguf-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
