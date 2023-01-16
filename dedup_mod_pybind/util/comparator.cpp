#include "comparator.hpp"

std::vector<StackTraceInfo> Comparator::prepare(const std::vector<Stacktrace> &stacks,
                                                const std::vector<Stacktrace> &other_stack) const {
  std::vector<StackTraceInfo> v;
  for(size_t i = 0; i < stacks.size(); i++) {
    StackTraceInfo info;
    info.idx    = i;
    info.weight = 1.;
    info.stack  = &stacks[i];

    v.push_back(info);
  }

  return v;
}

double Comparator::aggregate(std::vector<double> &matrix_score, const std::vector<StackTraceInfo> &query_stacks,
                             const std::vector<StackTraceInfo> &candidate_stacks) const {
  throw std::logic_error("Not implemented");
}

double Max::aggregate(std::vector<double> &matrix_score, const std::vector<StackTraceInfo> &query_stacks,
                      const std::vector<StackTraceInfo> &candidate_stacks) const {
  double max_result = -9999999.99;

  for(size_t i = 0; i < query_stacks.size(); i++) {
    for(size_t j = 0; j < candidate_stacks.size(); j++) {
      const double result = matrix_score[(i * candidate_stacks.size() + j)];
      max_result          = std::max(max_result, result);
    }
  }

  return max_result;
}

Mean::Mean(MeanType mean_type, WeightType weight_type, const std::vector<double> &df)
    : mean_type(mean_type), weight_type(weight_type), df(df) {}

std::vector<StackTraceInfo> Mean::prepare(const std::vector<Stacktrace> &stacks,
                                          const std::vector<Stacktrace> &other_stack) const {
  std::vector<StackTraceInfo> v;
  v.reserve(stacks.size());

  for(size_t i = 0; i < stacks.size(); i++) {
    StackTraceInfo info;

    info.idx   = i;
    info.stack = &stacks[i];

    switch(weight_type) {
      default:
      case WeightType::OFF: info.weight = 1.; break;
      case WeightType::AVG: info.weight = compute_weight(info.stack, df); break;
      case WeightType::MAXI: info.weight = compute_weight_max(info.stack, df); break;
    }
    v.push_back(info);
  }

  return v;
}

double Mean::aggregate(std::vector<double> &matrix_score, const std::vector<StackTraceInfo> &query_stacks,
                       const std::vector<StackTraceInfo> &candidate_stacks) const {
  int avg_row;
  size_t length;

  switch(mean_type) {
    default:
    case MeanType::QUERY:
      avg_row = 1;
      length  = query_stacks.size();
      break;
    case MeanType::CANDIDATE:
      avg_row = 0;
      length  = candidate_stacks.size();
      break;
    case MeanType::SHORTEST:
      avg_row = static_cast<int>(query_stacks.size() <= candidate_stacks.size());
      length  = std::min(query_stacks.size(), candidate_stacks.size());
      break;
    case MeanType::LONGEST:
      avg_row = static_cast<int>(query_stacks.size() >= candidate_stacks.size());
      length  = std::max(query_stacks.size(), candidate_stacks.size());
      break;
    case MeanType::QUERY_CAND:
      avg_row = -1;
      length  = query_stacks.size() + candidate_stacks.size();
      break;
  }

  std::vector<double> max_values;
  max_values.resize(length, -9999999.99);

  for(size_t query_idx = 0; query_idx < query_stacks.size(); query_idx++) {
    for(size_t cand_idx = 0; cand_idx < candidate_stacks.size(); cand_idx++) {
      const double score = matrix_score[(query_idx * candidate_stacks.size()) + cand_idx];

      switch(avg_row) {
        case -1:
          max_values[query_idx]                      = std::max(max_values[query_idx], score);
          max_values[cand_idx + query_stacks.size()] = std::max(max_values[cand_idx + query_stacks.size()], score);
        case 1: max_values[query_idx] = std::max(max_values[query_idx], score); break;
        default:
        case 0: max_values[cand_idx] = std::max(max_values[cand_idx], score); break;
      }
    }
  }

  double sum  = 0.;
  double sum2 = 0.;
  double den  = 0.;
  double den2 = 0.;

  for(size_t i = 0; i < length; i++) {
    if(avg_row == -1) {
      if(i < query_stacks.size()) {
        const double weight = query_stacks[i].weight;
        sum += weight * max_values[i];
        den += 2. * weight;
      } else {
        const double weight = candidate_stacks[i - query_stacks.size()].weight;
        sum2 += weight * max_values[i];
        den2 += 2. * weight;
      }
    } else {
      const double weight = ((avg_row == 1) ? query_stacks : candidate_stacks)[i].weight;
      sum += weight * max_values[i];
      den += weight;

      den2 = 1.;
    }
  }

  return sum / den + sum2 / den2;
}

inline bool compare_align_elem(const DfInfo &lhs, const DfInfo &rhs) {
  return (lhs.max_value < rhs.max_value || (lhs.max_value == rhs.max_value && lhs.mean_value < rhs.mean_value));
}

double compute_weight(const Stacktrace *stack, const std::vector<double> &df) {
  double sum = 0.;

  const auto stack_length = stack->stack.size();
  for(size_t i = 0; i < stack_length; i++) {
    sum += 1. - df[stack->stack[i]];
  }
  return sum / stack_length;
}

double compute_weight_max(const Stacktrace *stack, const std::vector<double> &df) {
  double max_val = -1.;

  const auto stack_length = stack->stack.size();
  for(size_t i = 0; i < stack_length; i++) {
    const double s = 1. - df[stack->stack[i]];
    max_val        = std::max(max_val, s);
  }

  return max_val;
}

SelectOne::SelectOne(const std::vector<double> &df): df(df) {}

std::vector<Stacktrace> SelectOne::filter(const std::vector<Stacktrace> &stacks) const {
  if(stacks.size() == 1) {
    return stacks;
  }

  DfInfo best{.idx = 1, .max_value = -1., .mean_value = -1.};

  for(size_t i = 0; i < stacks.size(); i++) {
    DfInfo current {
        .idx = i, .max_value = compute_weight_max(&stacks[i], df), .mean_value = compute_weight(&stacks[i], df)};

    if(compare_align_elem(best, current)) {
      best = current;
    }
  }

  std::vector<Stacktrace> new_stacks;
  new_stacks.push_back(stacks[best.idx]);

  return new_stacks;
}

KTopFunction::KTopFunction(double filter_k, const std::vector<double> &df): df(df) {
  const size_t vocab_size = df.size();
  std::vector<double> idfs(vocab_size);

  std::transform(df.begin(), df.end(), idfs.begin(), [](double df) { return 1. - df; });
  std::sort(idfs.begin(), idfs.end());

  if(filter_k <= 0. || filter_k > 1.)
    throw std::runtime_error(("filter_k " + std::to_string(filter_k) + " is invalid").c_str());

  const size_t start = static_cast<size_t>((1. - filter_k) * vocab_size);

  threshold = idfs[start];
}

std::vector<Stacktrace> KTopFunction::filter(const std::vector<Stacktrace> &stacks) const {
  std::vector<Stacktrace> new_stacks;

  for(const auto &st: stacks) {
    if(std::any_of(st.stack.begin(), st.stack.end(),
                   [&](size_t frameId) { return (1. - df[frameId]) >= threshold; })) {
      new_stacks.push_back(st);
    }
  }

  return new_stacks;
}

#pragma region PYBIND_DECLARATIONS

PYBIND11_MODULE(comparator, m) {
  m.attr("__name__") = "dedup_mod.util.comparator"; // Set Name as submodule
  py::enum_<AggStrategy>(m, "AggStrategy", py::arithmetic())
      .value("MAX", AggStrategy::MAX)
      .value("AVG_QUERY", AggStrategy::AVG_QUERY)
      .value("AVG_CAND", AggStrategy::AVG_CAND)
      .value("AVG_SHORT", AggStrategy::AVG_SHORT)
      .value("AVG_LONG", AggStrategy::AVG_LONG)
      .value("AVG_QUERY_CAND", AggStrategy::AVG_QUERY_CAND)
      .export_values();
  py::enum_<MeanType>(m, "MeanType", py::arithmetic())
      .value("QUERY", MeanType::QUERY)
      .value("CANDIDATE", MeanType::CANDIDATE)
      .value("SHORTEST", MeanType::SHORTEST)
      .value("LONGEST", MeanType::LONGEST)
      .value("SYM", MeanType::SYM)
      .value("QUERY_CAND", MeanType::QUERY_CAND)
      .export_values();
  py::enum_<WeightType>(m, "WeightType", py::arithmetic())
      .value("OFF", WeightType::OFF)
      .value("AVG", WeightType::AVG)
      .value("MAXI", WeightType::MAXI)
      .export_values();
  py::enum_<FilterStrategy>(m, "FilterStrategy", py::arithmetic())
      .value("NONE", FilterStrategy::NONE)
      .value("SELECT_ONE", FilterStrategy::SELECT_ONE)
      .value("TOP_K_FUNC", FilterStrategy::TOP_K_FUNC)
      .export_values();
}
#pragma endregion // PYBIND_DECLARATIONS