#pragma once

#include "../imports.hpp"
#include "method.hpp"

//////////////////////////////////
// Crash Graph Implementation
// DOI : https://doi.org/10.1109/dsn.2011.5958261
//////////////////////////////////

class CrashGraph: public Method {
public:
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;
};
