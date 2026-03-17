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

# ── Patch HighsMipSolverData.h: add capture overload for feasibilityJump ──
file(READ "${MIP_DIR}/HighsMipSolverData.h" MIPDATA_H)

string(FIND "${MIPDATA_H}" "feasibilityJumpCapture" _fj_h_found)
if(_fj_h_found EQUAL -1)
    string(REPLACE
      "HighsModelStatus feasibilityJump();"
      "HighsModelStatus feasibilityJump();\n  HighsModelStatus feasibilityJumpCapture(std::vector<double>& captured_solution, double& captured_obj);"
      MIPDATA_H "${MIPDATA_H}")

    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied feasibilityJumpCapture declaration to HighsMipSolverData.h")
else()
    message(STATUS "feasibilityJumpCapture patch already applied, skipping")
endif()

# ── Patch HighsFeasibilityJump.cpp: add capture implementation ──
file(READ "${MIP_DIR}/HighsFeasibilityJump.cpp" FJ_CONTENT)

string(FIND "${FJ_CONTENT}" "feasibilityJumpCapture" _fj_found)
if(_fj_found EQUAL -1)
    # Append the capture variant after the original function
    string(APPEND FJ_CONTENT "\n\
HighsModelStatus HighsMipSolverData::feasibilityJumpCapture(\n\
    std::vector<double>& captured_solution, double& captured_obj) {\n\
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
  const bool use_incumbent = !incumbent.empty();\n\
\n\
  auto solver = external_feasibilityjump::FeasibilityJumpSolver(\n\
      log_options,\n\
      mipsolver.options_mip_->random_seed,\n\
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
    if (use_incumbent && std::isfinite(incumbent[col])) {\n\
      initial_assignment = std::max(lower, std::min(upper, incumbent[col]));\n\
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
  const size_t kMaxTotalEffort = (size_t)nnz << 10;\n\
  const size_t kMaxEffortSinceLastImprovement = (size_t)nnz << 8;\n\
\n\
  auto fjControlCallback =\n\
      [=, &col_value, &found_integer_feasible_solution,\n\
       &objective_function_value](external_feasibilityjump::FJStatus status)\n\
      -> external_feasibilityjump::CallbackControlFlow {\n\
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
    # In portfolio mode: skip standalone FJ (it runs as a portfolio arm).
    # In sequential mode: FJ already ran above, then run FPR + LocalMIP.
    string(REPLACE
      "    if (options_mip_->mip_heuristic_run_feasibility_jump) {\n      analysis_.mipTimerStart(kMipClockFeasibilityJump);\n      HighsModelStatus returned_model_status = mipdata_->feasibilityJump();"
      "    if (options_mip_->mip_heuristic_run_feasibility_jump && !options_mip_->mip_heuristic_portfolio) {\n      analysis_.mipTimerStart(kMipClockFeasibilityJump);\n      HighsModelStatus returned_model_status = mipdata_->feasibilityJump();"
      CONTENT "${CONTENT}")

    string(REPLACE
      "    }\n    // End of pre-root-node heuristics"
      "    }\n    if (options_mip_->mip_heuristic_portfolio) {\n      portfolio::run_presolve(*this);\n    } else {\n      if (options_mip_->mip_heuristic_run_fpr) fpr::run(*this);\n      if (options_mip_->mip_heuristic_run_local_mip) local_mip::run(*this);\n    }\n\n    // End of pre-root-node heuristics"
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

    # Patch C: after RINS/RENS block closing brace, insert portfolio/scylla dispatch
    string(REPLACE
      "          }\n\n          mipdata_->heuristics.flushStatistics();"
      "          }\n          if (options_mip_->mip_heuristic_portfolio) {\n            portfolio::run_lp_based(*this);\n          } else {\n            if (options_mip_->mip_heuristic_run_scylla_fpr) scylla_fpr::run(*this);\n          }\n\n          mipdata_->heuristics.flushStatistics();"
      CONTENT "${CONTENT}")

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied heuristic call site patches to HighsMipSolver.cpp")
else()
    message(STATUS "Heuristic patches already applied to HighsMipSolver.cpp, skipping")
endif()
