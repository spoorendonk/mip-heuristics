# clang-format and clang-tidy gates over the first-party sources (issue #101).
#
# Both are wired as ctest tests labelled `lint`, so `ctest` — the command the
# devkit pre-push hook and CI both run — enforces them, and `ctest -LE lint`
# skips them while iterating.  They are also reachable as build targets
# (`cmake --build build --target lint`) for a CI job that wants them separately.
#
# Scope is `src/` and `tests/` only.  The build fetches and patches a large
# upstream HiGHS tree plus Catch2 under `${CMAKE_BINARY_DIR}/_deps`; neither
# tool is ever pointed at those.  The `.clang-tidy` HeaderFilterRegex enforces
# the same boundary for headers reached through an include.
#
# TOOL VERSIONS ARE PART OF THE CONTRACT.  A format gate is only reproducible
# if everyone runs the same major version — clang-format's output changes
# between releases.  The gate was written against clang-format 22.1.8 and
# clang-tidy 22.1.8; a different major version is a configure-time warning
# (a hard error when MIP_HEURISTICS_REQUIRE_LINT is ON).
#
# Install with:  python3 -m venv .venv && .venv/bin/pip install
#                clang-format==22.1.8 clang-tidy==22.1.8
# `.venv/bin` is searched first, so that install is picked up automatically.

set(MIP_HEURISTICS_CLANG_TOOLS_VERSION "22"
    CACHE STRING "Expected clang-format / clang-tidy major version for the lint gates")

option(MIP_HEURISTICS_REQUIRE_LINT
       "Fail configuration when the clang tools are missing or the wrong major version. \
CI must set this ON so a missing tool can never silently drop the gate." OFF)

set(_lint_hint "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin")

find_program(CLANG_FORMAT_EXE
             NAMES clang-format-${MIP_HEURISTICS_CLANG_TOOLS_VERSION} clang-format
             HINTS "${_lint_hint}")
find_program(CLANG_TIDY_EXE
             NAMES clang-tidy-${MIP_HEURISTICS_CLANG_TOOLS_VERSION} clang-tidy
             HINTS "${_lint_hint}")

# `<tool> --version` prints e.g. "clang-format version 22.1.8" or, for
# clang-tidy, a multi-line banner containing "LLVM version 22.1.8".
function(_mip_lint_check_version exe label out_ok)
    execute_process(COMMAND "${exe}" --version
                    OUTPUT_VARIABLE _v OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        _mip_lint_complain("${label} at ${exe} could not be run")
        set(${out_ok} FALSE PARENT_SCOPE)
        return()
    endif()
    string(REGEX MATCH "version ([0-9]+)\\.[0-9]+" _m "${_v}")
    if(NOT CMAKE_MATCH_1 STREQUAL MIP_HEURISTICS_CLANG_TOOLS_VERSION)
        _mip_lint_complain(
            "${label} is major version ${CMAKE_MATCH_1}, the gate expects "
            "${MIP_HEURISTICS_CLANG_TOOLS_VERSION}. Both tools change their output between "
            "major versions, so this one will report findings that CI does not see, and "
            "miss findings that it does")
        set(${out_ok} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${out_ok} TRUE PARENT_SCOPE)
endfunction()

function(_mip_lint_complain)
    string(JOIN "" _msg ${ARGN})
    if(MIP_HEURISTICS_REQUIRE_LINT)
        message(FATAL_ERROR "lint gate: ${_msg} (MIP_HEURISTICS_REQUIRE_LINT is ON)")
    else()
        message(WARNING "lint gate DISABLED: ${_msg}. Configure with "
                        "-DMIP_HEURISTICS_REQUIRE_LINT=ON to make this fatal.")
    endif()
endfunction()

set(_lint_format_ok FALSE)
set(_lint_tidy_ok FALSE)

if(CLANG_FORMAT_EXE)
    _mip_lint_check_version("${CLANG_FORMAT_EXE}" "clang-format" _lint_format_ok)
else()
    _mip_lint_complain("clang-format not found on PATH or in .venv/bin")
endif()

if(CLANG_TIDY_EXE)
    _mip_lint_check_version("${CLANG_TIDY_EXE}" "clang-tidy" _lint_tidy_ok)
else()
    _mip_lint_complain("clang-tidy not found on PATH or in .venv/bin")
endif()

add_custom_target(lint)

if(_lint_format_ok)
    set(_format_cmd
        "${CMAKE_COMMAND}"
        "-DCLANG_FORMAT_EXE=${CLANG_FORMAT_EXE}"
        "-DPROJECT_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_clang_format.cmake")
    add_test(NAME clang_format COMMAND ${_format_cmd})
    set_tests_properties(clang_format PROPERTIES LABELS lint)
    add_custom_target(format-check COMMAND ${_format_cmd}
                      COMMENT "clang-format --dry-run -Werror over src/ and tests/")
    add_dependencies(lint format-check)
endif()

# The tidy gate needs `compile_commands.json` and the generated HiGHS headers,
# so it only makes sense against a configured, built tree — which is where
# ctest runs it.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(_lint_tidy_ok AND NOT Python3_Interpreter_FOUND)
    _mip_lint_complain("clang-tidy was found but no Python 3 interpreter was, and the "
                       "tidy gate is a Python wrapper (see cmake/clang_tidy_gate.py for "
                       "why clang-tidy's own exit status cannot be used here)")
    set(_lint_tidy_ok FALSE)
endif()

if(_lint_tidy_ok)
    include(ProcessorCount)
    ProcessorCount(_lint_jobs)
    if(_lint_jobs EQUAL 0)
        set(_lint_jobs 1)
    endif()
    set(_tidy_cmd
        "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/clang_tidy_gate.py"
        --clang-tidy "${CLANG_TIDY_EXE}"
        --build-dir "${CMAKE_BINARY_DIR}"
        --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
        --jobs "${_lint_jobs}")
    add_test(NAME clang_tidy COMMAND ${_tidy_cmd})
    # Roughly 30 s on 12 cores; give a small CI runner plenty of room.
    set_tests_properties(clang_tidy PROPERTIES LABELS lint TIMEOUT 1800)
    add_custom_target(tidy COMMAND ${_tidy_cmd}
                      COMMENT "clang-tidy over src/ and tests/ translation units")
    add_dependencies(lint tidy)
endif()
