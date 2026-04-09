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
      "bool mip_heuristic_run_shifting;\n  bool mip_heuristic_run_fpr;\n  bool mip_heuristic_run_local_mip;\n  bool mip_heuristic_local_mip_parallel;\n  bool mip_heuristic_run_scylla;\n  bool mip_heuristic_scylla_parallel;\n  bool mip_heuristic_portfolio;\n  bool mip_heuristic_portfolio_opportunistic;\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Constructor initializer list: insert after mip_heuristic_run_shifting(false),
    string(REPLACE
      "mip_heuristic_run_shifting(false),\n"
      "mip_heuristic_run_shifting(false),\n        mip_heuristic_run_fpr(false),\n        mip_heuristic_run_local_mip(false),\n        mip_heuristic_local_mip_parallel(false),\n        mip_heuristic_run_scylla(false),\n        mip_heuristic_scylla_parallel(false),\n        mip_heuristic_portfolio(false),\n        mip_heuristic_portfolio_opportunistic(false),\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Record registration: insert after the mip_heuristic_run_shifting record block
    string(REPLACE
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);"
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_run_fpr\",\n                                       \"Use the FPR heuristic\", advanced,\n                                       &mip_heuristic_run_fpr, true);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_run_local_mip\",\n                                       \"Use the LocalMIP heuristic\", advanced,\n                                       &mip_heuristic_run_local_mip, true);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_local_mip_parallel\",\n                                       \"Run LocalMIP workers in parallel\", advanced,\n                                       &mip_heuristic_local_mip_parallel, false);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_run_scylla\",\n                                       \"Use the Scylla heuristic\", advanced,\n                                       &mip_heuristic_run_scylla, true);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_scylla_parallel\",\n                                       \"Run Scylla pump chains in parallel\", advanced,\n                                       &mip_heuristic_scylla_parallel, false);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_portfolio\",\n                                       \"Use adaptive portfolio mode for custom heuristics\", advanced,\n                                       &mip_heuristic_portfolio, false);\n    records.push_back(record_bool);\n\n    record_bool = new OptionRecordBool(\"mip_heuristic_portfolio_opportunistic\",\n                                       \"Use opportunistic (non-deterministic) mode for presolve portfolio\", advanced,\n                                       &mip_heuristic_portfolio_opportunistic, false);\n    records.push_back(record_bool);"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    file(WRITE "${LP_DATA_DIR}/HighsOptions.h" "${OPTIONS_CONTENT}")
    message(STATUS "Applied option patches to HighsOptions.h")
else()
    message(STATUS "Option patches already applied to HighsOptions.h, skipping")
endif()

# ── Patch HighsMipSolverData.h: add capture overload + custom solution source enums ──
file(READ "${MIP_DIR}/HighsMipSolverData.h" MIPDATA_H)

# Add heuristic_effort_used field
string(FIND "${MIPDATA_H}" "heuristic_effort_used" _effort_field_found)
if(_effort_field_found EQUAL -1)
    string(REPLACE
      "double heuristic_effort;"
      "double heuristic_effort;\n  size_t heuristic_effort_used = 0;"
      MIPDATA_H "${MIPDATA_H}")
    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied heuristic_effort_used field to HighsMipSolverData.h")
else()
    message(STATUS "heuristic_effort_used field already applied, skipping")
endif()

string(FIND "${MIPDATA_H}" "feasibilityJumpCapture" _fj_h_found)
if(_fj_h_found EQUAL -1)
    string(REPLACE
      "HighsModelStatus feasibilityJump();"
      "HighsModelStatus feasibilityJump();\n  HighsModelStatus feasibilityJumpCapture(std::vector<double>& captured_solution, double& captured_obj, size_t& captured_effort, size_t max_effort = 0, const std::vector<double>* hint_incumbent = nullptr, int seed_override = -1);"
      MIPDATA_H "${MIPDATA_H}")

    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied feasibilityJumpCapture declaration to HighsMipSolverData.h")
else()
    message(STATUS "feasibilityJumpCapture patch already applied, skipping")
endif()

# Add per-heuristic solution source enum entries
string(FIND "${MIPDATA_H}" "kSolutionSourceFPR" _src_enum_found)
if(_src_enum_found EQUAL -1)
    string(REPLACE
      "  kSolutionSourceTrivialZ,            // z\n  kSolutionSourceCleanup,"
      "  kSolutionSourceTrivialZ,            // z\n  kSolutionSourceFPR,                 // A (fix-propagate-repair)\n  kSolutionSourceLocalMIP,            // M (local MIP search)\n  kSolutionSourceScylla,              // G (Scylla)\n  kSolutionSourceFJ,                  // J (feasibility jump)\n  kSolutionSourceCleanup,"
      MIPDATA_H "${MIPDATA_H}")

    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied custom solution source enums to HighsMipSolverData.h")
else()
    message(STATUS "Custom solution source enums already applied, skipping")
endif()

# ── Patch HighsMipSolverData.cpp: add source strings + fix key display ──
file(READ "${MIP_DIR}/HighsMipSolverData.cpp" MIPDATA_CPP)

string(FIND "${MIPDATA_CPP}" "kSolutionSourceFPR" _src_cpp_found)
if(_src_cpp_found EQUAL -1)
    # Add source-to-string entries before kSolutionSourceCleanup
    string(REPLACE
      "} else if (solution_source == kSolutionSourceCleanup) {\n    if (code) return \" \";\n    return \"\";"
      "} else if (solution_source == kSolutionSourceFPR) {\n    if (code) return \"A\";\n    return \"FPR\";\n  } else if (solution_source == kSolutionSourceLocalMIP) {\n    if (code) return \"M\";\n    return \"Local MIP\";\n  } else if (solution_source == kSolutionSourceScylla) {\n    if (code) return \"G\";\n    return \"Scylla\";\n  } else if (solution_source == kSolutionSourceFJ) {\n    if (code) return \"J\";\n    return \"FJ\";\n  } else if (solution_source == kSolutionSourceCleanup) {\n    if (code) return \" \";\n    return \"\";"
      MIPDATA_CPP "${MIPDATA_CPP}")

    # Update printSolutionSourceKey limits for the 3 new entries
    string(REPLACE
      "std::vector<int> limits = {4, 9, 14, last_enum};"
      "std::vector<int> limits = {4, 9, 14, 18, last_enum};"
      MIPDATA_CPP "${MIPDATA_CPP}")

    file(WRITE "${MIP_DIR}/HighsMipSolverData.cpp" "${MIPDATA_CPP}")
    message(STATUS "Applied solution source strings to HighsMipSolverData.cpp")
else()
    message(STATUS "Solution source strings already applied, skipping")
endif()

# ── Patch HighsFeasibilityJump.cpp: add capture implementation ──
file(READ "${MIP_DIR}/HighsFeasibilityJump.cpp" FJ_CONTENT)

string(FIND "${FJ_CONTENT}" "feasibilityJumpCapture" _fj_found)
if(_fj_found EQUAL -1)
    # Append the capture variant after the original function
    string(APPEND FJ_CONTENT "\n\
HighsModelStatus HighsMipSolverData::feasibilityJumpCapture(\n\
    std::vector<double>& captured_solution, double& captured_obj,\n\
    size_t& captured_effort, size_t max_effort,\n\
    const std::vector<double>* hint_incumbent, int seed_override) {\n\
  const HighsLp* model = this->mipsolver.model_;\n\
  const HighsLogOptions& log_options = mipsolver.options_mip_->log_options;\n\
  double sense_multiplier = static_cast<double>(model->sense_);\n\
\n\
#ifdef HIGHSINT64\n\
  highsLogUser(log_options, HighsLogType::kInfo,\n\
               \"Feasibility Jump code isn't currently compatible \"\n\
               \"with a 64-bit HighsInt: skipping Feasibility Jump\\n\");\n\
  return HighsModelStatus::kNotset;\n\
#else\n\
\n\
  bool found_integer_feasible_solution = false;\n\
  std::vector<double> col_value(model->num_col_, 0.0);\n\
  double objective_function_value = 0.0;\n\
\n\
  const auto& inc = hint_incumbent ? *hint_incumbent : incumbent;\n\
  const bool use_incumbent = !inc.empty();\n\
\n\
  const int fj_seed = (seed_override >= 0) ? seed_override : mipsolver.options_mip_->random_seed;\n\
  auto solver = external_feasibilityjump::FeasibilityJumpSolver(\n\
      log_options,\n\
      fj_seed,\n\
      epsilon,\n\
      feastol);\n\
\n\
  for (HighsInt col = 0; col < model->num_col_; ++col) {\n\
    double lower = model->col_lower_[col];\n\
    double upper = model->col_upper_[col];\n\
\n\
    external_feasibilityjump::VarType fjVarType;\n\
    if (model->integrality_[col] == HighsVarType::kContinuous) {\n\
      fjVarType = external_feasibilityjump::VarType::Continuous;\n\
    } else {\n\
      fjVarType = external_feasibilityjump::VarType::Integer;\n\
      lower = std::ceil(lower - feastol);\n\
      upper = std::floor(upper + feastol);\n\
    }\n\
\n\
    const bool legal_bounds = lower <= upper && lower < kHighsInf &&\n\
                              upper > -kHighsInf && !std::isnan(lower) &&\n\
                              !std::isnan(upper);\n\
    if (!legal_bounds) {\n\
      return HighsModelStatus::kInfeasible;\n\
    }\n\
    solver.addVar(fjVarType, lower, upper,\n\
                  sense_multiplier * model->col_cost_[col]);\n\
\n\
    double initial_assignment = 0.0;\n\
    if (use_incumbent && std::isfinite(inc[col])) {\n\
      initial_assignment = std::max(lower, std::min(upper, inc[col]));\n\
    } else {\n\
      if (std::isfinite(lower)) {\n\
        initial_assignment = lower;\n\
      } else if (std::isfinite(upper)) {\n\
        initial_assignment = upper;\n\
      }\n\
    }\n\
    col_value[col] = initial_assignment;\n\
  }\n\
\n\
  HighsSparseMatrix a_matrix;\n\
  a_matrix.createRowwise(model->a_matrix_);\n\
\n\
  for (HighsInt row = 0; row < model->num_row_; ++row) {\n\
    bool hasFiniteLower = std::isfinite(model->row_lower_[row]);\n\
    bool hasFiniteUpper = std::isfinite(model->row_upper_[row]);\n\
    if (hasFiniteLower || hasFiniteUpper) {\n\
      HighsInt row_num_nz = a_matrix.start_[row + 1] - a_matrix.start_[row];\n\
      auto row_index = a_matrix.index_.data() + a_matrix.start_[row];\n\
      auto row_value = a_matrix.value_.data() + a_matrix.start_[row];\n\
      if (hasFiniteLower) {\n\
        solver.addConstraint(external_feasibilityjump::RowType::Gte,\n\
                             model->row_lower_[row], row_num_nz, row_index,\n\
                             row_value, 0);\n\
      }\n\
      if (hasFiniteUpper) {\n\
        solver.addConstraint(external_feasibilityjump::RowType::Lte,\n\
                             model->row_upper_[row], row_num_nz, row_index,\n\
                             row_value, 0);\n\
      }\n\
    }\n\
  }\n\
\n\
  const HighsInt nnz = a_matrix.numNz();\n\
  const size_t kMaxTotalEffort = (max_effort > 0) ? max_effort : ((size_t)nnz << 10);\n\
  const size_t kMaxEffortSinceLastImprovement = std::min((size_t)nnz << 8, (max_effort > 0) ? max_effort : ((size_t)nnz << 8));\n\
\n\
  size_t last_total_effort = 0;\n\
  auto fjControlCallback =\n\
      [=, &col_value, &found_integer_feasible_solution,\n\
       &objective_function_value, &last_total_effort](external_feasibilityjump::FJStatus status)\n\
      -> external_feasibilityjump::CallbackControlFlow {\n\
    last_total_effort = status.totalEffort;\n\
    if (status.solution != nullptr) {\n\
      found_integer_feasible_solution = true;\n\
      col_value = std::vector<double>(status.solution,\n\
                                      status.solution + status.numVars);\n\
      objective_function_value =\n\
          model->offset_ + sense_multiplier * status.solutionObjectiveValue;\n\
    }\n\
    if (status.effortSinceLastImprovement > kMaxEffortSinceLastImprovement ||\n\
        status.totalEffort > kMaxTotalEffort) {\n\
      return external_feasibilityjump::CallbackControlFlow::Terminate;\n\
    } else {\n\
      return external_feasibilityjump::CallbackControlFlow::Continue;\n\
    }\n\
  };\n\
\n\
  solver.solve(col_value.data(), fjControlCallback);\n\
  captured_effort = last_total_effort;\n\
\n\
  if (found_integer_feasible_solution) {\n\
    captured_solution = std::move(col_value);\n\
    captured_obj = objective_function_value;\n\
  }\n\
  return HighsModelStatus::kNotset;\n\
#endif\n\
}\n")

    file(WRITE "${MIP_DIR}/HighsFeasibilityJump.cpp" "${FJ_CONTENT}")
    message(STATUS "Applied feasibilityJumpCapture to HighsFeasibilityJump.cpp")
else()
    message(STATUS "feasibilityJumpCapture already applied, skipping")
endif()

# ── Patch feasibilityjump.hh: add resume parameter to solve() ──
file(READ "${MIP_DIR}/feasibilityjump.hh" FJ_HH)

string(FIND "${FJ_HH}" "bool resume = false" _fj_resume_found)
if(_fj_resume_found EQUAL -1)
    string(REPLACE
      "  int solve(double* initialValues,\n            std::function<CallbackControlFlow(FJStatus)> callback) {\n    assert(callback);\n    highsLogDev(logOptions, HighsLogType::kInfo,\n                FJ_LOG_PREFIX\n                \"starting solve. weightUpdateDecay=%g, relaxContinuous=%d  \\n\",\n                weightUpdateDecay, problem.usedRelaxContinuous);\n\n    init(initialValues);\n\n    effortAtLastLogging = -kMinEffortToLogging;  // Enabling step=0 logging\n    int num_logging_lines_since_header = 0;"
      "  int solve(double* initialValues,\n            std::function<CallbackControlFlow(FJStatus)> callback,\n            bool resume = false) {\n    assert(callback);\n    if (!resume) {\n      highsLogDev(logOptions, HighsLogType::kInfo,\n                  FJ_LOG_PREFIX\n                  \"starting solve. weightUpdateDecay=%g, relaxContinuous=%d  \\n\",\n                  weightUpdateDecay, problem.usedRelaxContinuous);\n      init(initialValues);\n      effortAtLastLogging = -kMinEffortToLogging;  // Enabling step=0 logging\n    }\n    int num_logging_lines_since_header = 0;"
      FJ_HH "${FJ_HH}")

    file(WRITE "${MIP_DIR}/feasibilityjump.hh" "${FJ_HH}")
    message(STATUS "Applied resume parameter patch to feasibilityjump.hh")
else()
    message(STATUS "Resume parameter patch already applied to feasibilityjump.hh, skipping")
endif()

# ── Patch standalone feasibilityJump() to store effort ──
file(READ "${MIP_DIR}/HighsFeasibilityJump.cpp" FJ_CONTENT2)
string(FIND "${FJ_CONTENT2}" "heuristic_effort_used" _fj_effort_found)
if(_fj_effort_found EQUAL -1)
    # Add effort capture variable to original FJ callback
    string(REPLACE
      "  auto fjControlCallback =\n      [=, &col_value, &found_integer_feasible_solution,\n       &objective_function_value](external_feasibilityjump::FJStatus status)\n      -> external_feasibilityjump::CallbackControlFlow {"
      "  size_t fj_last_effort = 0;\n  auto fjControlCallback =\n      [=, &col_value, &found_integer_feasible_solution,\n       &objective_function_value, &fj_last_effort](external_feasibilityjump::FJStatus status)\n      -> external_feasibilityjump::CallbackControlFlow {\n    fj_last_effort = status.totalEffort;"
      FJ_CONTENT2 "${FJ_CONTENT2}")

    # Store effort after solve
    string(REPLACE
      "  solver.solve(col_value.data(), fjControlCallback);\n\n  if (found_integer_feasible_solution) {\n    // Initial assignments"
      "  solver.solve(col_value.data(), fjControlCallback);\n  heuristic_effort_used += fj_last_effort;\n\n  if (found_integer_feasible_solution) {\n    // Initial assignments"
      FJ_CONTENT2 "${FJ_CONTENT2}")

    file(WRITE "${MIP_DIR}/HighsFeasibilityJump.cpp" "${FJ_CONTENT2}")
    message(STATUS "Applied effort tracking to standalone feasibilityJump()")
else()
    message(STATUS "Standalone FJ effort tracking already applied, skipping")
endif()

# ── Patch HighsMipSolver.cpp: insert heuristic call sites ──
file(READ "${MIP_DIR}/HighsMipSolver.cpp" CONTENT)

string(FIND "${CONTENT}" "heuristics::run_presolve" _found)
if(_found EQUAL -1)
    # Add includes at top (after existing includes)
    string(REPLACE
      "#include \"mip/HighsMipSolver.h\""
      "#include \"mip/HighsMipSolver.h\"\n#include \"fpr_lp.h\"\n#include \"heuristic_common.h\"\n#include \"mode_dispatch.h\""
      CONTENT "${CONTENT}")

    # Patch A: disable standalone FJ (we handle it in our presolve block)
    string(REPLACE
      "if (options_mip_->mip_heuristic_run_feasibility_jump) {"
      "if (false) { // FJ runs via custom presolve heuristics block"
      CONTENT "${CONTENT}")

    # Patch A2: insert custom heuristics block via mode_dispatch
    string(REPLACE
      "    }\n    // End of pre-root-node heuristics"
      "    }\n    {\n      const size_t nnz = mipdata_->ARindex_.size();\n      const size_t budget = heuristic_effort_budget(nnz, options_mip_->mip_heuristic_effort);\n      if (heuristics::run_presolve(*this, budget)) {\n        modelstatus_ = HighsModelStatus::kInfeasible;\n        cleanupSolve();\n        return;\n      }\n    }\n\n    // End of pre-root-node heuristics"
      CONTENT "${CONTENT}")

    # Patch B: guard standalone RINS/RENS when portfolio is on (B&B dive)
    # In portfolio mode, RINS/RENS run as bandit arms inside run_lp_based.
    string(REPLACE
      "if (options_mip_->mip_heuristic_run_rens) {"
      "if (options_mip_->mip_heuristic_run_rens && !options_mip_->mip_heuristic_portfolio) {"
      CONTENT "${CONTENT}")
    string(REPLACE
      "if (options_mip_->mip_heuristic_run_rins) {"
      "if (options_mip_->mip_heuristic_run_rins && !options_mip_->mip_heuristic_portfolio) {"
      CONTENT "${CONTENT}")

    # Patch C: after RINS/RENS block, insert LP-dependent FPR (Scylla moved to presolve)
    string(REPLACE
      "          }\n\n          mipdata_->heuristics.flushStatistics();"
      "          }\n          if (options_mip_->mip_heuristic_run_fpr) {\n            const size_t fpr_lp_nnz = mipdata_->ARindex_.size();\n            fpr_lp::run(*this, heuristic_effort_budget(fpr_lp_nnz, options_mip_->mip_heuristic_effort));\n          }\n\n          mipdata_->heuristics.flushStatistics();"
      CONTENT "${CONTENT}")

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied heuristic call site patches to HighsMipSolver.cpp")
else()
    message(STATUS "Heuristic patches already applied to HighsMipSolver.cpp, skipping")
endif()
