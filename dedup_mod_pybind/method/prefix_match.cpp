#include "prefix_match.hpp"

double PrefixMatch::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  double lcp     = 0.;
  const size_t min_len = std::min(query.size(), candidate.size());

  for(size_t i = 0; i < min_len; i++) {
    if(query[i] != candidate[i]) {
      break;
    }

    lcp += 1.;
  }

  return lcp / std::max(query.size(), candidate.size());
}
