
#include "brodie_05.hpp"
#include <cmath>

//////////////////////////////////
// Brodie 05 Implementation
// DOI : https://doi.org/10.1109/icac.2005.49
//////////////////////////////////

Brodie05::Brodie05(const std::vector<double> &df, double gap_penalty): gap_penalty(gap_penalty), df(df) {}

double Brodie05::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  const size_t offset = candidate.size() + 1;
  // We need to keep only two rows of the matrix M
  std::vector<double> M_PAST_ROW;
  std::vector<double> M_THIS_ROW;
  M_THIS_ROW.resize(offset);
  M_PAST_ROW.resize(offset);
  const double fpQuerySize = static_cast<double>(query.size()); // Floating Point Query Size

  // Create first row
  for(size_t i = 0; i < offset; i++) {
    M_THIS_ROW[i] = -gap_penalty * i;
  }

  for(size_t i = 0; i < query.size(); i++) {
    // Copy THIS_ROW to ONE_AGO
    M_THIS_ROW.swap(M_PAST_ROW);

    // Set first column of the row
    M_THIS_ROW[0] = -gap_penalty * (i + 1);

    for(size_t j = 0; j < candidate.size(); j++) {
      // Gap
      const double previous_row = M_PAST_ROW[(j + 1)] - gap_penalty;
      const double previous_col = M_THIS_ROW[j] - gap_penalty;
      double previous_row_col   = M_PAST_ROW[j];

      if(query[i] == candidate[j]) {
        // IDF* Function call position* Shift between calls
        previous_row_col += (1. - df[query[i]]) * (1. - (static_cast<double>(i) / fpQuerySize)) *
                            std::exp(-std::fabs(static_cast<double>(i) - static_cast<double>(j)) / 2.);
      }

      M_THIS_ROW[(j + 1)] = std::max({previous_row, previous_col, previous_row_col});
    }
  }
  return M_THIS_ROW.back();
}
