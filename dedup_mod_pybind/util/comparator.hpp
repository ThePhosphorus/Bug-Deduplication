#pragma once
#include "../imports.hpp"

#include "structures.hpp"
#include <vector>

enum class AggStrategy {
  MAX,
  AVG_QUERY,
  AVG_CAND,
  AVG_SHORT,
  AVG_LONG,
  AVG_QUERY_CAND
};

struct ScoreWeight {
  double *score;
  double *weights;
  bool source;
  size_t len;
};

enum class MeanType { QUERY, CANDIDATE, SHORTEST, LONGEST, SYM, QUERY_CAND };

enum class WeightType { OFF, AVG, MAXI };

struct StackTraceInfo {
  size_t idx;
  double weight;
  const Stacktrace * stack;
};

struct DfInfo {
  size_t idx;
  double max_value;
  double mean_value;
};

class Comparator {
public:
  virtual std::vector<StackTraceInfo>
  prepare(const std::vector<Stacktrace> &stacks,
          const std::vector<Stacktrace> &other_stack) const ;
  virtual double aggregate(std::vector<double> &matrix_score,
                           const std::vector<StackTraceInfo> &query_stacks,
                           const std::vector<StackTraceInfo> &candidate_stacks) const;
};

class Max : public Comparator {
public:
  virtual double
  aggregate(std::vector<double> &matrix_score,
            const std::vector<StackTraceInfo> &query_stacks,
            const std::vector<StackTraceInfo> &candidate_stacks) const override;
};

class Mean : public Comparator {
public:
  const MeanType mean_type;
  const WeightType weight_type;
  const std::vector<double> &df;

  Mean(MeanType mean_type, WeightType weightType, const std::vector<double> &df);
  virtual std::vector<StackTraceInfo>
  prepare(const std::vector<Stacktrace> &stacks,
          const std::vector<Stacktrace> &other_stack) const override;
  virtual double
  aggregate(std::vector<double> &matrix_score,
            const std::vector<StackTraceInfo> &query_stacks,
            const std::vector<StackTraceInfo> &candidate_stacks) const override;
};

inline bool compare_align_elem(const DfInfo& lhs, const DfInfo& rhs);
double compute_weight(const Stacktrace *stack, const std::vector<double> &df);
double compute_weight_max(const Stacktrace *stack, const std::vector<double> &df);

enum class FilterStrategy { NONE = 0, SELECT_ONE, TOP_K_FUNC };

class Filter {
public:
  virtual std::vector<Stacktrace> filter(const std::vector<Stacktrace> &stacks) const {
    return stacks;
  };
};

class SelectOne : public Filter {
public:
  const std::vector<double> &df;

  SelectOne(const std::vector<double> &df);
  virtual std::vector<Stacktrace>
  filter(const std::vector<Stacktrace> &stacks) const override;
};

class KTopFunction : public Filter {
public:
  double threshold;
  const std::vector<double> &df;

  KTopFunction(double filter_k, const std::vector<double> &df);
  virtual std::vector<Stacktrace>
  filter(const std::vector<Stacktrace> &stacks) const override;
};
