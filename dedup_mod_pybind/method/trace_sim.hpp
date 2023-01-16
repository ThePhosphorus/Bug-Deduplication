#pragma once

#include "../imports.hpp"
#include "method.hpp"

//////////////////////////////////
// TraceSim Implementation
// DOI : https://doi.org/10.1145/3416505.3423561
//////////////////////////////////

struct WeightedFrame {
  size_t id;
  double weight;
};

class TraceSim: public Method {
public:
  const std::vector<double> &df;
  double df_coef;
  double pos_coef;
  double diff_coef;
  double match_cost;
  double gap_penalty;
  double mismatch_penalty;

  bool sigmoid;
  double gamma;
  bool useSum;
  bool useIdf;
  bool const_match;
  bool reciprocal_func;
  bool no_norm;

  TraceSim(const std::vector<double> &df, double df_coef, double pos_coef, double diff_coef, double match_cost,
           double gap_penalty, double mismatch_penalty, bool sigmoid, double gamma, bool useSum, bool useIdf,
           bool const_match, bool reciprocal_func, bool no_norm);
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;
  double calculate_weight(const std::vector<size_t> &trace, size_t pos) const;

private:
  inline double getIDF(size_t i) const {
    const auto elem = df[i];
    return useIdf ? std::log(1. / ((elem == 0.) ? 1. : elem)) : elem;
  };
};
