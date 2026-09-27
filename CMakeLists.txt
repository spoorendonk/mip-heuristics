cmake_minimum_required(VERSION 3.25)
project(mip-heuristics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The Catch2 suite, the bench and docs checks and the lint gates serve this
# repo's own development only.  A parent that pulls this project in with
# FetchContent or add_subdirectory wants the patched `highs` target, not
# these in its ctest (and not a `lint` target colliding with its own), so
# they are on only as the top-level project; a parent can force them on by
# setting this ON.  The early return() at "--- Tests ---" below does the
# skipping.
option(MIP_HEURISTICS_BUILD_TESTS
       "Build and register the tests, bench checks and lint gates" ${PROJECT_IS_TOP_LEVEL})

# Optional instrumentation for the LocalMIP counters: the warm-start
# branches and the deadline polls.  These atomic counters fire on every
# dispatch in the production binary but are only read by five tests.
# When OFF the counter increments compile out via `if constexpr` and the
# public API returns zeros; when ON, the counters fire and the test suite
# uses them as regression sentinels.  So the default follows
# MIP_HEURISTICS_BUILD_TESTS: ON wherever the tests are built to read them,
# OFF for a parent consuming only the library, which would pay the
# overhead for nothing.  A top-level production build wanting zero counter
# overhead passes `-DMIP_HEURISTICS_INSTRUMENT=OFF` explicitly.  Like any
# option default, this only seeds a fresh cache: a tree first configured
# with the tests off keeps INSTRUMENT=OFF until it is set ON explicitly.
option(MIP_HEURISTICS_INSTRUMENT
       "Enable LocalMIP warm-start and deadline-poll instrumentation counters" ${MIP_HEURISTICS_BUILD_TESTS})

option(MIP_HEURISTICS_INSTALL_GIT_HOOKS
       "Point this checkout's core.hooksPath at the tracked .githooks/" ${PROJECT_IS_TOP_LEVEL})

# The git hooks live in .githooks/ under version control, so a fresh clone gets
# them for free -- but git only runs hooks from core.hooksPath, which is
# per-checkout config and cannot be tracked.  Configure sets it, since
# `cmake -B build` is the first thing anyone runs here.  Idempotent, and it
# never overwrites a value someone else set deliberately: only an unset
# hooksPath, or one already pointing at .githooks, is touched.  Defaults OFF
# as a subproject: a copy FetchContent cloned into a parent's build tree is
# not a checkout anyone commits from.
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

# Compiler cache.  The pre-push gate is a *clean* rebuild (`rm -rf build`),
# which is deliberate — HiGHS is patched in place and `apply_patch.cmake`
# refuses a tree whose PATCH_VERSION marker does not match rather than
# upgrading it, so a wiped tree is the one state guaranteed consistent with
# `src/`.  That guarantee costs a full recompile of the fetched HiGHS tree
# every time, and ccache gives it back without weakening it: the cache is
# keyed on source content, so a differently-patched tree simply misses.
#
# Set before `include(cmake/FetchHiGHS.cmake)` on purpose.  The launcher is a
# directory property that targets read at creation, so HiGHS's own targets --
# which are the recompile worth caching -- only inherit it if it is in scope
# before FetchContent adds their subdirectory.
#
# Both languages: HiGHS enables C for cuPDLP after we set these, and the
# variables still seed the property when that happens.
option(MIP_HEURISTICS_CCACHE
       "Route compilation through ccache when it is installed" ON)
if(MIP_HEURISTICS_CCACHE)
    find_program(CCACHE_EXE ccache)
    if(CCACHE_EXE)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXE}")
        message(STATUS "ccache: enabled (${CCACHE_EXE})")
    else()
        message(STATUS "ccache: not found, compiling without a cache")
    endif()
endif()

include(cmake/FetchHiGHS.cmake)

# The heuristics in two layers (#170).
#
# `mip_heuristics_core` is the algorithms, the solution pool and sink, the
# runner scaffold and the model view, and needs no `HighsMipSolver`: it
# includes no `mip/` header but the header-only `feasibilityjump.hh`, and a
# caller can run FJ and LocalMIP from it on a model it owns.  The ctest
# `core_boundary` enforces both halves of that.  Headers are listed so the
# check can read them off the target.
#
# `mip_heuristics` is the HiGHS adapter: the presolve chain and `fpr_lp`
# entry points, the solver-facing sink, the effort ledger, FPR's clique
# rankings and Scylla.  Its objects are injected into `highs`, as the
# whole library's were before the split.
add_library(mip_heuristics_core STATIC
    src/continuous_loop.h
    src/deadline.h
    src/fj.cpp
    src/fj.h
    src/fj_worker.cpp
    src/fj_worker.h
    src/fpr_core.cpp
    src/fpr_core.h
    src/fpr_strategies.h
    src/fpr_val_select.cpp
    src/fpr_val_select.h
    src/heuristic_common.h
    src/heuristic_context.cpp
    src/heuristic_context.h
    src/local_mip.cpp
    src/local_mip.h
    src/local_mip_caches.h
    src/local_mip_construction.cpp
    src/local_mip_construction.h
    src/local_mip_core.cpp
    src/local_mip_core.h
    src/local_mip_search.cpp
    src/local_mip_worker.cpp
    src/local_mip_worker.h
    src/opportunistic_runner.h
    src/prop_engine.cpp
    src/prop_engine.h
    src/repair_search.cpp
    src/repair_search.h
    src/repair_walk.cpp
    src/repair_walk.h
    src/rng.h
    src/solution_pool.cpp
    src/solution_pool.h
    src/solution_sink.cpp
    src/solution_sink.h
    src/walksat.cpp
    src/walksat.h
    src/worker_base.h
)
add_library(mip_heuristics OBJECT
    src/clique_cover.cpp
    src/contested_pdlp.cpp
    src/effort_ledger.cpp
    src/fpr.cpp
    src/fpr_lp.cpp
    src/fpr_lp_refs.cpp
    src/fpr_var_order.cpp
    src/incumbent_sink.cpp
    src/mode_dispatch.cpp
    src/scylla_worker.cpp
    src/scylla.cpp
)
# `SolutionSink::offer` is `[[nodiscard]]` because dropping its verdict is
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
# PRIVATE and per-target on purpose: the flag must not reach HiGHS's own
# translation units, which are third-party and not ours to police.
#
# `-Werror=switch` joins it for the same reason (issue #128's cold
# review): a `switch` over one of our own scoped enums with no `default`
# and a case an edit forgot to add compiles clean today, silently falling
# through to whatever the switch's target was last assigned (or, worse,
# reading it uninitialized) — exactly how `fpr_lp::select_ref`'s
# class-to-pointer mapping could drift without a single diagnostic. Every
# switch presently in `src/` and `tests/` either covers its enum
# exhaustively or carries an explicit `default:`, so this costs nothing
# today; it is meant to catch the next one that doesn't.
foreach(_target mip_heuristics_core mip_heuristics)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${_target} PRIVATE -Werror=unused-result -Werror=switch)
    endif()
    if(MIP_HEURISTICS_INSTRUMENT)
        target_compile_definitions(${_target} PRIVATE MIP_HEURISTICS_INSTRUMENT_ENABLED=1)
    endif()
endforeach()

# The include directories are spelled out rather than taken from `highs`'s
# usage requirements, which would also bring `-fPIC` and HiGHS's own
# warning flags: the heuristics compile exactly as they did before the
# split.  `$<LINK_ONLY:highs>` supplies the HiGHS utility symbols the core
# needs (timer, sparse transpose, logging) without those requirements.
# Installed, the core's headers sit beside HiGHS's (see the install rules
# below), which they include by the same relative paths.
target_include_directories(mip_heuristics_core PUBLIC
    $<BUILD_INTERFACE:${highs_SOURCE_DIR}/highs>
    $<BUILD_INTERFACE:${highs_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/highs>
    $<INSTALL_INTERFACE:include/mip_heuristics>)
target_link_libraries(mip_heuristics_core PUBLIC $<LINK_ONLY:highs>)
# The core's headers are C++23 (`std::expected`, designated initialisers,
# `[[nodiscard]]` structs), so a caller compiling against them needs it too;
# `CMAKE_CXX_STANDARD` above only reaches this project's own targets.
target_compile_features(mip_heuristics_core PUBLIC cxx_std_23)
target_include_directories(mip_heuristics PRIVATE ${highs_SOURCE_DIR}/highs ${highs_BINARY_DIR} src)

# `libhighs` holds both layers: the adapter's objects, as it always has, and
# the core's.  Self-contained, so a plain `-lhighs` link (pkg-config, or any
# build system but CMake) resolves on its own, as on a build without the
# split.  Linking the core to `highs` instead would make the two archives
# each need the other, a cycle CMake can order but a `.pc` file cannot:
# pkg-config collapses the repeated `-lhighs` it would take.  The core
# library below is the same objects again, for a caller that wants the core
# without the adapter; a link naming both archives pulls each object once.
target_sources(highs PRIVATE $<TARGET_OBJECTS:mip_heuristics> $<TARGET_OBJECTS:mip_heuristics_core>)
target_include_directories(highs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# The core, installed into the export set HiGHS puts `highs` in
# (`install(EXPORT highs-targets)`, cmake/cpp-highs.cmake upstream, plus the
# build-tree `highs-targets.cmake`), so a `find_package(highs)` caller gets
# `highs::mip_heuristics_core` beside `highs::highs`, and its headers under
# `include/mip_heuristics`.
include(GNUInstallDirs)
install(TARGETS mip_heuristics_core
    EXPORT highs-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libs)
export(TARGETS mip_heuristics_core
    NAMESPACE highs::
    APPEND FILE "${highs_BINARY_DIR}/highs-targets.cmake")
get_target_property(_core_sources mip_heuristics_core SOURCES)
list(FILTER _core_sources INCLUDE REGEX "\\.h$")
install(FILES ${_core_sources}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mip_heuristics
    COMPONENT libs)

# --- Tests ---
# Everything from here to the end of the file is development-only and skipped
# unless MIP_HEURISTICS_BUILD_TESTS is on (see its declaration at the top).
# Consumer-facing targets therefore go above this line.
if(NOT MIP_HEURISTICS_BUILD_TESTS)
    return()
endif()

# A normal variable, overriding the OFF that cmake/FetchHiGHS.cmake set for
# HiGHS's own suite, so include(CTest) below enables testing here.
set(BUILD_TESTING ON)
include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_executable(mip_heuristics_tests
    tests/test_smoke.cpp
    tests/test_fpr.cpp
    tests/test_fj.cpp
    tests/test_local_mip.cpp
    tests/test_scylla.cpp
    tests/test_fpr_lp.cpp
    tests/test_primitives.cpp
    tests/test_execution_modes.cpp
    tests/test_attribution.cpp
    tests/test_prop_engine.cpp
    tests/test_repair_search.cpp
    tests/test_repair_walk.cpp
    tests/test_continuous_loop.cpp
    tests/test_contested_pdlp.cpp
    tests/test_patience_gate.cpp
    tests/test_presolve_only.cpp
    tests/test_effort_zero.cpp
    tests/test_heursol_trace.cpp
    tests/test_deadline.cpp
    tests/test_setup_bail.cpp
    tests/test_clique_cover.cpp
    tests/test_fj_jump_value.cpp
    tests/test_fj_objective_sign.cpp
    tests/test_problem_view.cpp
)
# Same guard as `mip_heuristics` above — a test that drops the verdict
# would model the bug rather than catch it. `-Werror=switch` joins it for
# the same reason as the `mip_heuristics` target: see that target's comment.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mip_heuristics_tests PRIVATE -Werror=unused-result -Werror=switch)
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
# Two registrations partitioning the suite on the `[serial]` tag (#146).
#
# `[serial]` marks the cases that measure elapsed time against a threshold:
# the presolve-deadline cases in tests/test_deadline.cpp (a 0.1 s solver
# limit that a descheduled thread spends doing nothing, after which
# `run_sequential` skips the heuristic and no `[Heur]` line is emitted),
# effort-zero's setup-free window, and the two `ContestedPdlp` cases that
# race a sleeping thread against a deadline.  All were calibrated on an idle
# machine and all can fail under `ctest -j$(nproc)` on a saturated host with
# nothing wrong with the code.
#
# The tag is the *last* resort, not the first.  Prefer removing the clock
# from the fixture: the same deadline cases also failed at `readModel`,
# because the MPS parser takes `time_limit` as its own parse budget, and
# that one was fixed by writing the option after the read rather than by
# tagging around it (see the note at that write).  A wall-clock threshold
# tuned until it stops failing is the same test with a longer fuse, and a
# tag is only honest where nothing in the code can bound the wait.
#
# `RUN_SERIAL` rather than `RESOURCE_LOCK`: these cases need an *unloaded*
# machine, not merely exclusion from each other.  A lock only serialises its
# own holders, so the other 160-odd tests would still be running alongside
# them — which is the load that makes them fail.  Neither, though, excludes
# load from *outside* ctest: a shared CI runner is beyond what either can
# do, which is the other half of why the parser fix was worth making.
#
# The two specs partition the suite exactly: Catch2's `~[serial]` is the
# complement of `[serial]` over the non-hidden cases, so no test is dropped
# and none is registered twice (`ctest -N` counts and a `uniq -d` over the
# names both confirm it).  The distinct `TEST_LIST` names are hygiene, not
# a fix for a live bug: `Catch.cmake` keys its generated files on a SHA1 of
# `TEST_SPEC`, so the two calls already write separate include scripts and
# nothing in this project reads either variable.  Keep them distinct anyway
# — the default is `<target>_TESTS` for both, and a future reader of the
# list should not have to work out which call won.
catch_discover_tests(mip_heuristics_tests
    TEST_SPEC "[serial]"
    TEST_LIST mip_heuristics_tests_SERIAL
    PROPERTIES RUN_SERIAL TRUE)
catch_discover_tests(mip_heuristics_tests
    TEST_SPEC "~[serial]"
    TEST_LIST mip_heuristics_tests_PARALLEL)

# The heuristic core on its own (#170): links `mip_heuristics_core` and
# nothing else of ours (HiGHS itself comes with it, for `readModel`), and
# runs FJ and LocalMIP on an original model with no `HighsMipSolver`.  A
# separate executable so that the link line, not only the source, is the
# one a caller of the core would have.  No `[serial]` case: nothing here
# reads a clock.
add_executable(mip_heuristics_core_tests tests/test_core.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mip_heuristics_core_tests PRIVATE -Werror=unused-result -Werror=switch)
endif()
target_link_libraries(mip_heuristics_core_tests PRIVATE mip_heuristics_core Catch2::Catch2WithMain)
target_compile_definitions(mip_heuristics_core_tests PRIVATE
    "INSTANCES_DIR=\"${highs_SOURCE_DIR}/check/instances\"")
if(MIP_HEURISTICS_INSTRUMENT)
    target_compile_definitions(mip_heuristics_core_tests PRIVATE MIP_HEURISTICS_INSTRUMENT_ENABLED=1)
endif()
catch_discover_tests(mip_heuristics_core_tests TEST_LIST mip_heuristics_core_tests_ALL)

# The core linked against a libhighs with the MIP solver and the adapter
# taken out: every member built from `highs/mip/*.cpp`, and every object of
# `mip_heuristics` (and of the core itself).  The core archive goes in whole
# (`WHOLE_ARCHIVE`): an ordinary archive link extracts only the members these
# cases reach, and a dependency in any other member — the reviewer's
# `Highs::run` planted in `repair_search.cpp` — would link unseen.  Whole,
# every core member has to resolve here, so this executable does not link if
# any of them needs the MIP solver or the adapter, directly or through
# another libhighs member.  The archive is named as a file, not linked as a
# target, because the target's `$<LINK_ONLY:highs>` would bring the whole of
# libhighs back.  Its cases build their model by hand: the `Highs` class
# itself reaches the MIP solver.
file(GLOB _mip_sources ${highs_SOURCE_DIR}/highs/mip/*.cpp)
get_target_property(_adapter_sources mip_heuristics SOURCES)
get_target_property(_core_cpp mip_heuristics_core SOURCES)
list(FILTER _core_cpp INCLUDE REGEX "\\.cpp$")
set(_drop "")
# The core's own copies go too (libhighs carries them; see `target_sources`
# on `highs`), so the whole-archive core below is the only one linked.
foreach(_source IN LISTS _mip_sources _adapter_sources _core_cpp)
    get_filename_component(_name ${_source} NAME)
    list(APPEND _drop "${_name}.o")
endforeach()
list(JOIN _drop "|" _drop)
set(_highs_without_mip ${CMAKE_CURRENT_BINARY_DIR}/libhighs_without_mip.a)
add_custom_command(
    OUTPUT ${_highs_without_mip}
    COMMAND ${CMAKE_COMMAND} -DAR=${CMAKE_AR} -DIN=$<TARGET_FILE:highs>
            -DOUT=${_highs_without_mip} -DDROP=${_drop}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_archive.cmake
    DEPENDS highs ${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_archive.cmake
    COMMENT "libhighs without the MIP solver and the adapter"
    VERBATIM)
add_custom_target(highs_without_mip DEPENDS ${_highs_without_mip})

add_executable(mip_heuristics_core_standalone_tests tests/test_core_standalone.cpp)
add_dependencies(mip_heuristics_core_standalone_tests mip_heuristics_core highs_without_mip)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mip_heuristics_core_standalone_tests PRIVATE -Werror=unused-result
                                                                        -Werror=switch)
endif()
target_include_directories(mip_heuristics_core_standalone_tests PRIVATE
    $<TARGET_PROPERTY:mip_heuristics_core,INTERFACE_INCLUDE_DIRECTORIES>)
find_package(Threads REQUIRED)
target_link_libraries(mip_heuristics_core_standalone_tests PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,$<TARGET_FILE:mip_heuristics_core>> ${_highs_without_mip}
    Threads::Threads ${CMAKE_DL_LIBS}
    Catch2::Catch2WithMain)
if(MIP_HEURISTICS_INSTRUMENT)
    target_compile_definitions(mip_heuristics_core_standalone_tests PRIVATE
        MIP_HEURISTICS_INSTRUMENT_ENABLED=1)
endif()
catch_discover_tests(mip_heuristics_core_standalone_tests
    TEST_LIST mip_heuristics_core_standalone_tests_ALL)

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

    # The core/adapter boundary (#170): no core unit reaches an adapter or
    # `mip/` header, transitively, and no symbol the core leaves undefined
    # lives in the MIP solver or the adapter.  Standard library only, like
    # the docs check above.  The shared C++ runtime is passed in so what it
    # exports is not charged to whichever member carries a copy.
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE _cxx_runtime
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    add_test(
        NAME core_boundary
        COMMAND ${_bench_py} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_core_boundary.py
                --nm ${CMAKE_NM}
                --core-lib $<TARGET_FILE:mip_heuristics_core>
                --highs-lib $<TARGET_FILE:highs>
                --cxx-runtime ${_cxx_runtime}
                --highs-mip-dir ${highs_SOURCE_DIR}/highs/mip
                --src-dir ${CMAKE_CURRENT_SOURCE_DIR}/src
                --compile-commands ${CMAKE_BINARY_DIR}/compile_commands.json
                --core-sources $<TARGET_PROPERTY:mip_heuristics_core,SOURCES>
                --adapter-sources $<TARGET_PROPERTY:mip_heuristics,SOURCES>
        COMMAND_EXPAND_LISTS
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
