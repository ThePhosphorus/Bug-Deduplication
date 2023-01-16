#pragma once

#include "../imports.hpp"
#include "method.hpp"

class PDM: public Method {
public:
  double c;
  double o;

  PDM(double c, double o);
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;
};