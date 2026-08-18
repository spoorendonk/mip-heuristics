# Check-only clang-format gate.  Run via `cmake -P`; never rewrites a file.
#
# Required cache-style arguments (pass with -D):
#   CLANG_FORMAT_EXE  path to the clang-format binary
#   PROJECT_ROOT      repository root (holds .clang-format)
#
# The file list is globbed here rather than at configure time so a source
# file added after the last `cmake` run is still covered.

if(NOT CLANG_FORMAT_EXE OR NOT PROJECT_ROOT)
    message(FATAL_ERROR "run_clang_format.cmake needs -DCLANG_FORMAT_EXE and -DPROJECT_ROOT")
endif()

file(GLOB LINT_FILES
     "${PROJECT_ROOT}/src/*.cpp" "${PROJECT_ROOT}/src/*.h"
     "${PROJECT_ROOT}/tests/*.cpp" "${PROJECT_ROOT}/tests/*.h")
list(LENGTH LINT_FILES LINT_FILE_COUNT)
if(LINT_FILE_COUNT EQUAL 0)
    message(FATAL_ERROR "clang-format gate found no first-party sources under ${PROJECT_ROOT}")
endif()

execute_process(
    COMMAND "${CLANG_FORMAT_EXE}" --version
    OUTPUT_VARIABLE CF_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

# --dry-run reports what it would change and -Werror turns that into a
# non-zero exit.  Neither ever writes to the tree.
execute_process(
    COMMAND "${CLANG_FORMAT_EXE}" --style=file --dry-run -Werror ${LINT_FILES}
    WORKING_DIRECTORY "${PROJECT_ROOT}"
    RESULT_VARIABLE CF_RESULT)

if(NOT CF_RESULT EQUAL 0)
    message(FATAL_ERROR
            "clang-format found unformatted code in ${LINT_FILE_COUNT} first-party files "
            "(${CF_VERSION}).\n"
            "Fix with:  ${CLANG_FORMAT_EXE} --style=file -i src/*.cpp src/*.h "
            "tests/*.cpp tests/*.h")
endif()

message(STATUS "clang-format clean: ${LINT_FILE_COUNT} files (${CF_VERSION})")
