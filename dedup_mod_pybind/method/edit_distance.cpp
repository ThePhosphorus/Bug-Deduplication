#include "edit_distance.hpp"

//////////////////////////////////
// NeedlemanWunsch Implementation
// DOI : https://doi.org/10.1016%2F0022-2836%2870%2990057-4
//////////////////////////////////

NeedlemanWunsch::NeedlemanWunsch(double indel_penalty, double mismatch_penalty, double match_cost)
    : indel_penalty(indel_penalty), mismatch_penalty(mismatch_penalty), match_cost(match_cost) {}

double NeedlemanWunsch::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  // We need to keep only two rows of the matrix M
  const size_t offset = candidate.size() + 1;
  std::vector<double> M_THIS_ROW;
  std::vector<double> M_PAST_ROW;
  M_THIS_ROW.resize(offset);
  M_PAST_ROW.resize(offset);
  std::vector<double> WS_THIS_ROW;
  std::vector<double> WS_PAST_ROW;
  WS_THIS_ROW.resize(offset);
  WS_PAST_ROW.resize(offset);

  // Create first row
  for(size_t i = 0; i < offset; i++) {
    M_THIS_ROW[i]  = -indel_penalty * i;
    WS_THIS_ROW[i] = -indel_penalty * i;
  }

  for(size_t i = 0; i < query.size(); i++) {
    // Copy THIS_ROW to ONE_AGO
    M_THIS_ROW.swap(M_PAST_ROW);
    WS_THIS_ROW.swap(WS_PAST_ROW);

    // Set first column of the row
    M_THIS_ROW[0]  = -indel_penalty * i;
    WS_THIS_ROW[0] = -indel_penalty * i;

    for(size_t j = 0; j < candidate.size(); j++) {
      // Gap
      double previous_row     = M_PAST_ROW[j + 1] - indel_penalty;
      double previous_col     = M_THIS_ROW[j] - indel_penalty;
      double previous_row_col = M_PAST_ROW[j];

      previous_row_col += (query[i] == candidate[j]) ? match_cost : -mismatch_penalty;

      M_THIS_ROW[j + 1]  = std::max({previous_row, previous_col, previous_row_col});
      WS_THIS_ROW[j + 1] = std::max(
          {WS_PAST_ROW[j + 1] - indel_penalty, WS_THIS_ROW[j] - indel_penalty, WS_PAST_ROW[j] - mismatch_penalty});
    }
  }

  const double sim = M_THIS_ROW.back();

  const double min_value = WS_THIS_ROW.back();
  const double max_value = std::min(query.size(), candidate.size()) * match_cost;

  const double normalized_sim = (sim - min_value) / (max_value - min_value);

  return normalized_sim;
}

//////////////////////////////////
// DamerayLevenshtein Implementation
// DOI : https://doi.org/10.1145/363958.363994
//////////////////////////////////

DamerayLevenshtein::DamerayLevenshtein(double insert_penalty, double delete_penalty, double subs_penalty,
                                       double trans_penalty, bool enable_trans)
    : insert_penalty(insert_penalty), delete_penalty(delete_penalty), subs_penalty(subs_penalty),
      trans_penalty(trans_penalty), enable_trans(enable_trans) {}

double DamerayLevenshtein::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  const size_t cand_len = candidate.size(), query_len = query.size();
  const size_t offset = cand_len + 1;

  std::vector<double> M_THIS_ROW;
  std::vector<double> M_PAST_ROW;
  std::vector<double> M_TWO_AGO;
  M_THIS_ROW.resize(offset);
  M_PAST_ROW.resize(offset);
  M_TWO_AGO.resize(offset);

  std::vector<double> WS_THIS_ROW;
  std::vector<double> WS_PAST_ROW;
  WS_THIS_ROW.resize(offset);
  WS_PAST_ROW.resize(offset);

  // initialize THIS_ROW
  for(size_t i = 0; i < cand_len; i++) {
    // Insert from query to candidate
    M_THIS_ROW[i]  = (i + 1) * insert_penalty;
    WS_THIS_ROW[i] = (i + 1) * insert_penalty;
  }

  for(size_t i = 0; i < query.size(); i++) {

    // swap/initialize vectors
    M_PAST_ROW.swap(M_TWO_AGO);
    M_THIS_ROW.swap(M_PAST_ROW);
    WS_THIS_ROW.swap(WS_PAST_ROW);

    // Delete from query to candidate
    M_THIS_ROW[0]  = (i + 1) * delete_penalty;
    WS_THIS_ROW[0] = (i + 1) * delete_penalty;

    // now compute costs
    for(size_t j = 1; j < offset; j++) {
      const double delete_cost       = M_PAST_ROW[j] + delete_penalty;
      const double add_cost          = M_THIS_ROW[j - 1] + insert_penalty;
      const double substitution_cost = M_PAST_ROW[j - 1] + static_cast<double>(query[i] != candidate[j - 1]) * subs_penalty;
      M_THIS_ROW[j]                  = std::min({delete_cost, add_cost, substitution_cost});
      WS_THIS_ROW[j]                 = std::min(
          {WS_PAST_ROW[j] + delete_penalty, WS_THIS_ROW[j - 1] + insert_penalty, WS_PAST_ROW[j - 1] + subs_penalty});

      // deal with transpositions
      if(enable_trans && i > 0 && j > 1 && query[i] == candidate[j - 2] && query[i - 1] == candidate[j - 1] &&
         query[i] != candidate[j - 1]) {
        M_THIS_ROW[j] = std::min(M_THIS_ROW[j], M_TWO_AGO[j - 2] + trans_penalty);
      }
    }
  }

  return 1. - (M_THIS_ROW.back() / WS_THIS_ROW.back());
}
