# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Included after strata_kernels_cpu and its optional ggml dependencies have been defined.

add_library(strata_dsv41_host STATIC
    src/dsv41/ds4_gguf_cpu.cpp
    src/dsv41/ds4_iq_cpu.cpp
    src/dsv41/ds4_tq_cpu.cpp
    src/dsv41/ds4_gg_cpu.cpp
    src/dsv41/ds4_packinfo.cpp
    src/dsv41/vendor/exl3_cpu.cpp)
target_compile_features(strata_dsv41_host PUBLIC cxx_std_17)
target_include_directories(strata_dsv41_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
target_link_libraries(strata_dsv41_host PUBLIC strata_platform strata_warnings Threads::Threads)
if(STRATA_NATIVE_EXPERTS)
    target_link_libraries(strata_dsv41_host PUBLIC strata_kernels_cpu)
    target_compile_definitions(strata_dsv41_host PRIVATE STRATA_DSV41_TQ_CPU=1)
endif()
if(WIN32)
    target_link_libraries(strata_dsv41_host PUBLIC winmm)
endif()

if(STRATA_BUILD_TESTS)
    enable_testing()
    foreach(_ds4_host_test slot experts_rule pick_rule store pool mx_tier stream_selection)
        add_executable(ds4_${_ds4_host_test}_test tests/dsv41/${_ds4_host_test}_test.cpp)
        target_link_libraries(ds4_${_ds4_host_test}_test PRIVATE strata_dsv41_host)
        if(_ds4_host_test STREQUAL "store")
            add_test(NAME ds4_store_test COMMAND ds4_store_test --selftest)
        else()
            add_test(NAME ds4_${_ds4_host_test}_test COMMAND ds4_${_ds4_host_test}_test)
        endif()
        set_tests_properties(ds4_${_ds4_host_test}_test PROPERTIES LABELS "dsv41;host" TIMEOUT 60)
    endforeach()
    if(STRATA_NATIVE_EXPERTS)
        add_executable(ds4_gg_bench tests/dsv41/gg_dot_bench.cpp)
        target_link_libraries(ds4_gg_bench PRIVATE strata_dsv41_host)
        foreach(_ds4_type 16 17 18 21 29)
            foreach(_ds4_nt 1 3 8)
                set(_ds4_test ds4_gg_check_${_ds4_type}_${_ds4_nt})
                add_test(NAME ${_ds4_test}
                    COMMAND ds4_gg_bench --check-only --type ${_ds4_type} --nt ${_ds4_nt})
                set_tests_properties(${_ds4_test} PROPERTIES LABELS "dsv41;host" TIMEOUT 60)
            endforeach()
        endforeach()
        add_test(NAME ds4_gg_check_avx2 COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1
            $<TARGET_FILE:ds4_gg_bench> --check-only --type 18 --nt 3)
        set_tests_properties(ds4_gg_check_avx2 PROPERTIES LABELS "dsv41;host" TIMEOUT 60)
    endif()
endif()

if(STRATA_BUILD_TESTS)
  foreach(ds4_mx_tier scalar avx2 avxvnni avx512 avx512vnni)
    add_test(NAME ds4_mx_tier_override_${ds4_mx_tier}
      COMMAND ${CMAKE_COMMAND} -E env DS4_MX_TIER=${ds4_mx_tier}
        $<TARGET_FILE:ds4_mx_tier_test> --selection-only --expect ${ds4_mx_tier})
    set_tests_properties(ds4_mx_tier_override_${ds4_mx_tier} PROPERTIES LABELS "dsv41;host" TIMEOUT 60)
  endforeach()
  add_test(NAME ds4_mx_tier_avx2 COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1 DS4_MX_TIER=
    $<TARGET_FILE:ds4_mx_tier_test> --selection-only)
  add_test(NAME ds4_mx_tier_invalid COMMAND ${CMAKE_COMMAND} -E env DS4_MX_TIER=invalid
    $<TARGET_FILE:ds4_mx_tier_test> --selection-only)
  set_tests_properties(ds4_mx_tier_avx2 ds4_mx_tier_invalid PROPERTIES LABELS "dsv41;host" TIMEOUT 60)
endif()
