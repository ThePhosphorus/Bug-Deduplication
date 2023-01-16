#pragma once

#include "../imports.hpp"

enum class SimilarityMethod
{
  TRACE_SIM = 0,
  BRODIE_05,
  DARMERAU_LEVENSHTEIN,
  OPT_ALIGN,
  PDM_METHOD,
  PREFIX_MATCH,
  CRASH_GRAPH,
  FAST
};

class Method {
public:
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const = 0;
};
