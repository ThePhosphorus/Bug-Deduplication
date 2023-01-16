#include "crash_graph.hpp"

//////////////////////////////////
// Crash Graph Implementation
// DOI : https://doi.org/10.1109/dsn.2011.5958261
//////////////////////////////////

double CrashGraph::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  double n_edge = 0;
  
  for(size_t i = 0, j = 0; i < query.size() && j < candidate.size();) {
    const int diff = static_cast<int>(query[i]) - static_cast<int>(candidate[j]);
    n_edge += static_cast<size_t>(diff == 0);
    i += static_cast<size_t>(diff <= 0);
    j += static_cast<size_t>(diff >= 0);
  }
  const size_t minSize = std::min(query.size(), candidate.size());
  return (minSize == 0)? 0. : (n_edge / minSize);
}
