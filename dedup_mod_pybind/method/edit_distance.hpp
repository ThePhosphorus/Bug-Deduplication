#pragma once

#include "../imports.hpp"
#include "method.hpp"

//////////////////////////////////
// NeedlemanWunsch Implementation
// DOI : https://doi.org/10.1016%2F0022-2836%2870%2990057-4
//////////////////////////////////

class NeedlemanWunsch: public Method {
public:
  double indel_penalty;
  double mismatch_penalty;
  double match_cost;

  NeedlemanWunsch(double indel_penalty, double mismatch_penalty, double match_cost);
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;
};

//////////////////////////////////
// DamerayLevenshtein Implementation
// DOI : https://doi.org/10.1145/363958.363994
//////////////////////////////////

class DamerayLevenshtein: public Method {
public:
  double insert_penalty;
  double delete_penalty;
  double subs_penalty;
  double trans_penalty;
  bool enable_trans;

  DamerayLevenshtein(double insert_penalty, double delete_penalty, double subs_penalty, double trans_penalty,
                     bool enable_trans);
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;
};