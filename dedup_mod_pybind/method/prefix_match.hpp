#pragma once

#include "../imports.hpp"
#include "method.hpp"

class PrefixMatch: public Method {
public:
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;
};