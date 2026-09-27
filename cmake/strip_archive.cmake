# Copy a static archive without some of its members (#170).
#
#   cmake -DAR=<ar> -DIN=<lib.a> -DOUT=<lib.a> -DDROP=<member|member|...> -P strip_archive.cmake
#
# Used to build a libhighs that contains neither the MIP solver (every member
# compiled from highs/mip/*.cpp) nor the HiGHS adapter, for the test that
# links the heuristic core against it.  Every name in DROP must be a member
# of IN, and none may survive: a member that silently failed to drop would
# let that test pass for the wrong reason.

cmake_minimum_required(VERSION 3.25)

# `|`-separated on the command line, where a `;` would split the argument.
string(REPLACE "|" ";" DROP "${DROP}")
execute_process(COMMAND "${AR}" t "${IN}" OUTPUT_VARIABLE members RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "strip_archive: cannot list ${IN}")
endif()
string(REPLACE "\n" ";" members "${members}")
foreach(member IN LISTS DROP)
    if(NOT member IN_LIST members)
        message(FATAL_ERROR "strip_archive: ${member} is not a member of ${IN}")
    endif()
endforeach()

file(COPY_FILE "${IN}" "${OUT}")
execute_process(COMMAND "${AR}" ds "${OUT}" ${DROP} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "strip_archive: ar d failed on ${OUT}")
endif()

execute_process(COMMAND "${AR}" t "${OUT}" OUTPUT_VARIABLE left)
string(REPLACE "\n" ";" left "${left}")
foreach(member IN LISTS DROP)
    if(member IN_LIST left)
        message(FATAL_ERROR "strip_archive: ${member} survived in ${OUT}")
    endif()
endforeach()
