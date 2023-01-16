#pragma once

#include "../imports.hpp"
#include "method.hpp"

//////////////////////////////////
// Brodie 05 Implementation
// DOI : https://doi.org/10.1109/icac.2005.49
//////////////////////////////////

class Brodie05: public Method {
public:
  const std::vector<double> &df;
  double gap_penalty;

  Brodie05(const std::vector<double> &df, double gap_penalty);
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;
};
