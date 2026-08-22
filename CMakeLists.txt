cmake_minimum_required(VERSION 3.25)
project(mip-heuristics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional instrumentation for the LocalMIP warm-start branch counters.
# These atomic counters fire on every dispatch in the production binary
# but are only consumed by two tests.  When OFF the counter increments
# compile out via `if constexpr` and the public API returns zeros; when
# ON, the counters fire and the test suite uses them as regression
# sentinels.  Defaults ON because this project always builds tests
# (BUILD_TESTING is force-set ON below); production builds wanting zero
# counter overhead pass `-DMIP_HEURISTICS_INSTRUMENT=OFF` explicitly.
# R3-1 round-4 review; R5 R1-1/R2-2/R3-2 dropped the dead else-branch
# whose default was unreachable due to the BUILD_TESTING force later.
option(MIP_HEURISTICS_INSTRUMENT
       "Enable LocalMIP warm-start branch instrumentation counters" ON)

option(MIP_HEURISTICS_INSTALL_GIT_HOOKS
       "Point this checkout's core.hooksPath at the tracked .githooks/" ON)

# The git hooks live in .githooks/ under version control, so a fresh clone gets
# them for free -- but git only runs hooks from core.hooksPath, which is
# per-checkout config and cannot be tracked.  Configure sets it, since
# `cmake -B build` is the first thing anyone runs here.  Idempotent, and it
# never overwrites a value someone else set deliberately: only an unset
# hooksPath, or one already pointing at .githooks, is touched.
# EXISTS, not IS_DIRECTORY: in a linked worktree .git is a file.
if(MIP_HEURISTICS_INSTALL_GIT_HOOKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" config --get core.hooksPath
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE MIP_HEURISTICS_HOOKS_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT MIP_HEURISTICS_HOOKS_PATH)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" config core.hooksPath .githooks
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
            message(STATUS "git hooks: core.hooksPath -> .githooks")
        elseif(NOT MIP_HEURISTICS_HOOKS_PATH STREQUAL ".githooks")
            message(WARNING
                "git hooks: core.hooksPath is '${MIP_HEURISTICS_HOOKS_PATH}', not '.githooks'. "
                "The tracked hooks under .githooks/ are NOT running. "
                "Fix with: git config core.hooksPath .githooks")
        endif()
    endif()
endif()

include(cmake/FetchHiGHS.cmake)

add_library(mip_heuristics OBJECT
    src/contested_pdlp.cpp
    src/effort_ledger.cpp
    src/fj.cpp
    src/fj_worker.cpp
    src/fpr.cpp
    src/fpr_core.cpp
    src/fpr_lp.cpp
    src/fpr_lp_refs.cpp
    src/fpr_val_select.cpp
    src/fpr_var_order.cpp
    src/incumbent_sink.cpp
    src/local_mip.cpp
    src/local_mip_construction.cpp
    src/local_mip_core.cpp
    src/local_mip_search.cpp
    src/local_mip_worker.cpp
    src/prop_engine.cpp
    src/repair_search.cpp
    src/mode_dispatch.cpp
    src/scylla_worker.cpp
    src/solution_pool.cpp
    src/scylla.cpp
    src/walksat.cpp
)
# `IncumbentSink::offer` is `[[nodiscard]]` because dropping its verdict is
# the bug issue #111 fixed: every presolve worker computed the pool's
# answer, discarded it, and substituted a worker-local notion of
# "improved" that reset on every rebuild.  `[[nodiscard]]` alone does not
# prevent that returning — nothing in this project passes `-Werror`, the
# `.clang-tidy` config opens with `-*` (so `clang-diagnostic-*` is off)
# and sets `WarningsAsErrors: ''`, so a re-drop compiles to exit 0 with a
# warning that scrolls past mid-build.  That is precisely the silent
# failure mode CLAUDE.md warns about for the lint gates.
#
# Promoting just this one diagnostic to an error makes the guard real.
# PRIVATE and per-target on purpose: `mip_heuristics` is an OBJECT library
# whose *already compiled* objects are injected into `highs`, so the flag
# cannot reach HiGHS's own translation units, which are third-party and
# not ours to police.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mip_heuristics PRIVATE -Werror=unused-result)
endif()

target_include_directories(mip_heuristics PRIVATE ${highs_SOURCE_DIR}/highs ${highs_BINARY_DIR} src)
if(MIP_HEURISTICS_INSTRUMENT)
    target_compile_definitions(mip_heuristics PRIVATE MIP_HEURISTICS_INSTRUMENT_ENABLED=1)
endif()

target_sources(highs PRIVATE $<TARGET_OBJECTS:mip_heuristics>)
target_include_directories(highs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Tests ---
set(BUILD_TESTING ON CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_executable(mip_heuristics_tests
    tests/test_smoke.cpp
    tests/test_suite_option.cpp
    tests/test_fpr.cpp
    tests/test_fj.cpp
    tests/test_local_mip.cpp
    tests/test_scylla.cpp
    tests/test_fpr_lp.cpp
    tests/test_primitives.cpp
    tests/test_execution_modes.cpp
    tests/test_attribution.cpp
    tests/test_prop_engine.cpp
    tests/test_continuous_loop.cpp
    tests/test_contested_pdlp.cpp
    tests/test_stall_gate.cpp
)
# Same guard as `mip_heuristics` above — a test that drops the verdict
# would model the bug rather than catch it.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mip_heuristics_tests PRIVATE -Werror=unused-result)
endif()

target_include_directories(mip_heuristics_tests PRIVATE ${highs_SOURCE_DIR}/highs ${highs_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mip_heuristics_tests PRIVATE highs Catch2::Catch2WithMain)
target_compile_definitions(mip_heuristics_tests PRIVATE
    "INSTANCES_DIR=\"${highs_SOURCE_DIR}/check/instances\"")
# Mirror the instrumentation flag onto the tests target so
# `local_mip::kInstrumented` has a consistent value across TUs (ODR).
# When MIP_HEURISTICS_INSTRUMENT=OFF the tests that consume the
# counters will simply read zeros — see the per-test guards below.
if(MIP_HEURISTICS_INSTRUMENT)
    target_compile_definitions(mip_heuristics_tests PRIVATE MIP_HEURISTICS_INSTRUMENT_ENABLED=1)
endif()
catch_discover_tests(mip_heuristics_tests)

# Python bench tests — unit tests for the log parser, the benchmark
# harness and the results analysis.  The measurements themselves run off a
# real MIPLIB results tree and stay a manual/CI step; these tests only cover
# the scripts' own logic (regexes, aggregation, exit codes, error paths) so
# they are kept honest as the harness evolves.
#
# Prefer `.venv/bin/python` (where `pip install pytest` typically lands in
# this project) over the system interpreter.  Skip silently when neither
# has pytest so `ctest` on a fresh clone stays green.
set(_bench_py "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    set(_bench_py "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
else()
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(_bench_py "${Python3_EXECUTABLE}")
    endif()
endif()
if(_bench_py)
    execute_process(
        COMMAND ${_bench_py} -c "import pytest"
        RESULT_VARIABLE _have_pytest
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_have_pytest EQUAL 0)
        add_test(
            NAME bench_python_tests
            COMMAND ${_bench_py} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/bench
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    # docs/PARAMETERS.md documents ~50 tuning constants by name; this fails the
    # suite when one is renamed or deleted without the entry following it (a
    # `kWeightFj` entry outlived its constant that way, default value and all).
    # Registered outside the pytest gate above because it needs only the
    # standard library — a checkout without pytest still gets the check.
    add_test(
        NAME docs_parameter_references
        COMMAND ${_bench_py} ${CMAKE_CURRENT_SOURCE_DIR}/bench/check_docs_refs.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# --- Lint gates ---
# clang-format (check-only) and clang-tidy over src/ and tests/, registered as
# ctest tests labelled `lint` and as the `lint` build target.  Must come after
# include(CTest) so add_test() is available.  See cmake/Lint.cmake for the
# pinned tool versions and for why CI has to pass
# -DMIP_HEURISTICS_REQUIRE_LINT=ON.
include(cmake/Lint.cmake)
