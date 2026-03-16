#pragma once

#include <vector>

#include "util/HighsInt.h"
#include "lp_data/HConst.h"

struct CscMatrix {
  std::vector<HighsInt> col_start;
  std::vector<HighsInt> col_row;
  std::vector<double> col_val;
};

inline CscMatrix build_csc(HighsInt ncol, HighsInt nrow,
                            const std::vector<HighsInt>& ARstart,
                            const std::vector<HighsInt>& ARindex,
                            const std::vector<double>& ARvalue) {
  const HighsInt nnz = static_cast<HighsInt>(ARindex.size());
  CscMatrix csc;
  csc.col_start.assign(ncol + 1, 0);
  for (HighsInt k = 0; k < nnz; ++k) csc.col_start[ARindex[k] + 1]++;
  for (HighsInt j = 0; j < ncol; ++j)
    csc.col_start[j + 1] += csc.col_start[j];
  csc.col_row.resize(nnz);
  csc.col_val.resize(nnz);
  {
    std::vector<HighsInt> pos(csc.col_start);
    for (HighsInt i = 0; i < nrow; ++i)
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        HighsInt j = ARindex[k];
        csc.col_row[pos[j]] = i;
        csc.col_val[pos[j]] = ARvalue[k];
        pos[j]++;
      }
  }
  return csc;
}

inline bool is_integer(const std::vector<HighsVarType>& integrality,
                       HighsInt j) {
  return integrality[j] != HighsVarType::kContinuous;
}
