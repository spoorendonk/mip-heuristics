# Patch script for HiGHS: insert heuristic call sites
# Called by FetchContent PATCH_COMMAND
# Idempotent: safe to run multiple times.

# Detect source layout: v1.13+ uses highs/ subdirectory
if(EXISTS "${SOURCE_DIR}/highs/mip")
    set(MIP_DIR "${SOURCE_DIR}/highs/mip")
else()
    set(MIP_DIR "${SOURCE_DIR}/src/mip")
endif()

# ── Patch HighsMipSolver.cpp: insert heuristic call sites ──
file(READ "${MIP_DIR}/HighsMipSolver.cpp" CONTENT)

string(FIND "${CONTENT}" "fpr::run" _found)
if(_found EQUAL -1)
    # Add includes at top (after existing includes)
    string(REPLACE
      "#include \"mip/HighsMipSolver.h\""
      "#include \"mip/HighsMipSolver.h\"\n#include \"fpr.h\"\n#include \"local_mip.h\"\n#include \"scylla_fpr.h\""
      CONTENT "${CONTENT}")

    # Patch A: after feasibilityJump block (pre-root-node, LP-free heuristics)
    string(REPLACE
      "    }\n    // End of pre-root-node heuristics"
      "    }\n    fpr::run(*this);\n    local_mip::run(*this);\n\n    // End of pre-root-node heuristics"
      CONTENT "${CONTENT}")

    # Patch B: after RINS/RENS block closing brace (B&B dive)
    string(REPLACE
      "          }\n\n          mipdata_->heuristics.flushStatistics();"
      "          }\n          scylla_fpr::run(*this);\n\n          mipdata_->heuristics.flushStatistics();"
      CONTENT "${CONTENT}")

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied heuristic call site patches to HighsMipSolver.cpp")
else()
    message(STATUS "Heuristic patches already applied to HighsMipSolver.cpp, skipping")
endif()
