#include "pdm.hpp"
#include <iostream>

PDM::PDM(double c, double o): c(c), o(o) {}

double PDM::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  const size_t cand_len = candidate.size(), query_len = query.size();
  if(cand_len == 0 && query_len == 0) {
    return 1.;
  } else if(cand_len == 0 || query_len == 0) {
    return -9999.;
  }

  std::vector<double> M_THIS_ROW;
  M_THIS_ROW.resize(cand_len);
  std::vector<double> M_PAST_ROW;
  M_PAST_ROW.resize(cand_len);

  for(size_t i = 0; i < query_len; i++) {
    M_THIS_ROW.swap(M_PAST_ROW);

    for(size_t j = 0; j < cand_len; j++) {
      const double previous_row = (i == 0) ? 0. : M_PAST_ROW[j];
      const double previous_col = (j == 0) ? 0. : M_THIS_ROW[j - 1];
      double previous_row_col   = (j == 0 || i == 0) ? 0. : M_PAST_ROW[j - 1];

      if(query[i] == candidate[j]) {
        // Add cost
        const double first = std::exp(-c * std::min(i, j)), second = std::exp(-o * std::fabs(static_cast<double>(i) - j)),
                     third = std::fabs(i - j);
        previous_row_col +=  first * second;
      }

      M_THIS_ROW[j] = std::max({previous_row, previous_col, previous_row_col});
    }
  }
  const double sim = M_THIS_ROW.back();
  double dividend = 0.;
  for(size_t j = 0; j < std::min(query_len, cand_len); j++) {
    dividend += std::exp(-c * j);
  }

  return sim / dividend;
}
