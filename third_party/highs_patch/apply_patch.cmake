# Patch script for HiGHS: insert heuristic call sites and options
# Called by FetchContent PATCH_COMMAND
# Idempotent: safe to run multiple times.

# Detect source layout: v1.13+ uses highs/ subdirectory
if(EXISTS "${SOURCE_DIR}/highs/mip")
    set(MIP_DIR "${SOURCE_DIR}/highs/mip")
    set(LP_DATA_DIR "${SOURCE_DIR}/highs/lp_data")
else()
    set(MIP_DIR "${SOURCE_DIR}/src/mip")
    set(LP_DATA_DIR "${SOURCE_DIR}/src/lp_data")
endif()

# ── Patch HighsOptions.h: register custom heuristic options ──
file(READ "${LP_DATA_DIR}/HighsOptions.h" OPTIONS_CONTENT)

string(FIND "${OPTIONS_CONTENT}" "mip_heuristic_run_fpr" _opts_found)
if(_opts_found EQUAL -1)
    # Member variables: insert after mip_heuristic_run_shifting
    string(REPLACE
      "bool mip_heuristic_run_shifting;\n"
      "bool mip_heuristic_run_shifting;\n  bool mip_heuristic_run_fpr;\n  bool mip_heuristic_run_local_mip;\n  bool mip_heuristic_run_scylla_fpr;\n  bool mip_heuristic_portfolio;\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Constructor initializer list: insert after mip_heuristic_run_shifting(false),
    string(REPLACE
      "mip_heuristic_run_shifting(false),\n"
      "mip_heuristic_run_shifting(false),\n        mip_heuristic_run_fpr(false),\n        mip_heuristic_run_local_mip(false),\n        mip_heuristic_run_scylla_fpr(false),\n        mip_heuristic_portfolio(false),\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Record registration: insert after the mip_heuristic_run_shifting record block
    string(REPLACE
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);"
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_run_fpr\",\n                                       \"Use the FPR heuristic\", advanced,\n                                       &mip_heuristic_run_fpr, true);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_run_local_mip\",\n                                       \"Use the LocalMIP heuristic\", advanced,\n                                       &mip_heuristic_run_local_mip, true);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_run_scylla_fpr\",\n                                       \"Use the ScyllaFPR heuristic\", advanced,\n                                       &mip_heuristic_run_scylla_fpr, true);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_portfolio\",\n                                       \"Use adaptive portfolio mode for custom heuristics\", advanced,\n                                       &mip_heuristic_portfolio, false);\n    records.push_back(record_bool);"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    file(WRITE "${LP_DATA_DIR}/HighsOptions.h" "${OPTIONS_CONTENT}")
    message(STATUS "Applied option patches to HighsOptions.h")
else()
    message(STATUS "Option patches already applied to HighsOptions.h, skipping")
endif()

# ── Patch HighsMipSolver.cpp: insert heuristic call sites ──
file(READ "${MIP_DIR}/HighsMipSolver.cpp" CONTENT)

string(FIND "${CONTENT}" "fpr::run" _found)
if(_found EQUAL -1)
    # Add includes at top (after existing includes)
    string(REPLACE
      "#include \"mip/HighsMipSolver.h\""
      "#include \"mip/HighsMipSolver.h\"\n#include \"fpr.h\"\n#include \"local_mip.h\"\n#include \"scylla_fpr.h\"\n#include \"adaptive/portfolio.h\""
      CONTENT "${CONTENT}")

    # Patch A: after feasibilityJump block (pre-root-node, LP-free heuristics)
    # Portfolio mode: run portfolio. Sequential mode: run individually.
    string(REPLACE
      "    }\n    // End of pre-root-node heuristics"
      "    }\n    if (options_mip_->mip_heuristic_portfolio) {\n      portfolio::run_presolve(*this);\n    } else {\n      if (options_mip_->mip_heuristic_run_fpr) fpr::run(*this);\n      if (options_mip_->mip_heuristic_run_local_mip) local_mip::run(*this);\n    }\n\n    // End of pre-root-node heuristics"
      CONTENT "${CONTENT}")

    # Patch B: after RINS/RENS block closing brace (B&B dive)
    string(REPLACE
      "          }\n\n          mipdata_->heuristics.flushStatistics();"
      "          }\n          if (options_mip_->mip_heuristic_portfolio) {\n            portfolio::run_lp_based(*this);\n          } else {\n            if (options_mip_->mip_heuristic_run_scylla_fpr) scylla_fpr::run(*this);\n          }\n\n          mipdata_->heuristics.flushStatistics();"
      CONTENT "${CONTENT}")

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied heuristic call site patches to HighsMipSolver.cpp")
else()
    message(STATUS "Heuristic patches already applied to HighsMipSolver.cpp, skipping")
endif()
