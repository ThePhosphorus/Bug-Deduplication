#include "imports.hpp"

#include <iostream>
#include <tbb/tbb.h>
#include <vector>

#include "./cmp_func.hpp"

[[nodiscard("Returns allocated unique_ptr")]] //
static std::unique_ptr<Method>
    getMethod(SimilarityMethod method_type, const std::vector<double> &df, const py::list &args) {
  std::unique_ptr<Method> method;

  switch(method_type) {
    case SimilarityMethod::TRACE_SIM:
      method = std::make_unique<TraceSim>(df, args[0].cast<double>(), args[1].cast<double>(), args[2].cast<double>(),
                                          args[3].cast<double>(), args[4].cast<double>(), args[5].cast<double>(),
                                          args[6].cast<bool>(), args[7].cast<double>(), args[8].cast<bool>(),
                                          args[9].cast<bool>(), args[10].cast<bool>(), args[11].cast<bool>(),
                                          args[12].cast<bool>());
      break;
    case SimilarityMethod::BRODIE_05: {
      method = std::make_unique<Brodie05>(df, args[0].cast<double>());
    } break;
    case SimilarityMethod::DARMERAU_LEVENSHTEIN:
      method =
          std::make_unique<DamerayLevenshtein>(args[0].cast<double>(), args[1].cast<double>(), args[2].cast<double>(),
                                               args[3].cast<double>(), args[4].cast<bool>());
      break;
    case SimilarityMethod::OPT_ALIGN:
      method =
          std::make_unique<NeedlemanWunsch>(args[0].cast<double>(), args[1].cast<double>(), args[2].cast<double>());
      break;
    case SimilarityMethod::PDM_METHOD:
      method = std::make_unique<PDM>(args[0].cast<double>(), args[1].cast<double>());
      break;
    case SimilarityMethod::FAST:
      method = std::make_unique<FaST>(df, args[0].cast<double>(), args[1].cast<double>(), args[2].cast<double>());
      break;
    case SimilarityMethod::PREFIX_MATCH: method = std::make_unique<PrefixMatch>(); break;
    case SimilarityMethod::CRASH_GRAPH: method = std::make_unique<CrashGraph>(); break;
  }

  return std::move(method);
}

[[nodiscard("Returns allocated unique_ptr")]] //
static std::unique_ptr<Filter>
    getFilter(FilterStrategy filter_strategy, const std::vector<double> &df, double filter_k) {
  std::unique_ptr<Filter> filter;
  switch(filter_strategy) {
    case FilterStrategy::NONE: filter = std::make_unique<Filter>(); break;
    case FilterStrategy::SELECT_ONE: filter = std::make_unique<SelectOne>(df); break;
    case FilterStrategy::TOP_K_FUNC: filter = std::make_unique<KTopFunction>(filter_k, df); break;
    default: break;
  }

  return std::move(filter);
}

[[nodiscard("Returns allocated unique_ptr")]] //
static std::unique_ptr<Comparator>
    getAggregator(AggStrategy agg_strategy, const std::vector<double> &df) {
  std::unique_ptr<Comparator> agg;
  switch(agg_strategy) {
    case AggStrategy::MAX: agg = std::make_unique<Max>(); break;
    default: {
      MeanType mean_type;
      switch(agg_strategy) {
        case AggStrategy::AVG_QUERY: mean_type = MeanType::QUERY; break;
        case AggStrategy::AVG_CAND: mean_type = MeanType::CANDIDATE; break;
        case AggStrategy::AVG_SHORT: mean_type = MeanType::SHORTEST; break;
        case AggStrategy::AVG_LONG: mean_type = MeanType::LONGEST; break;
        case AggStrategy::AVG_QUERY_CAND: mean_type = MeanType::QUERY_CAND; break;
      }
      agg = std::make_unique<Mean>(mean_type, WeightType::OFF, df);
    } break;
  }

  return std::move(agg);
}

std::vector<DTYPE_t> compare(SimilarityMethod method_type, AggStrategy agg_strategy, const py::list & args,
                             const std::vector<std::vector<size_t>>& query,
                             const std::vector<std::vector<std::vector<size_t>>> &candidates,
                             const std::vector<double> &df,
                             FilterStrategy filter_strategy, double filter_k, const std::vector<bool> &is_stop_word,
                             bool beg_trail_trim) {

  std::vector<Stacktrace> query_stacks = generateStackTraces(0, query, is_stop_word, beg_trail_trim);
  std::vector<std::vector<Stacktrace>> stack_by_report =
      generateStackTracesByReport(candidates, is_stop_word, beg_trail_trim);

  size_t n_candidates = candidates.size();
  std::vector<DTYPE_t> d;
  std::unique_ptr<Method> method = getMethod(method_type, df, args);
  std::unique_ptr<Filter> filter = getFilter(filter_strategy, df, filter_k);
  std::unique_ptr<Comparator> agg = getAggregator(agg_strategy, df);

  std::vector<DTYPE_t> results(n_candidates, -9999999.99);

  std::vector<Stacktrace> f_query_stacks = filter->filter(query_stacks);

  if(f_query_stacks.size() > 0) {
    tbb::parallel_for(size_t(0), n_candidates, [&](size_t report_idx) {
      if(stack_by_report[report_idx].size() == 0)
        return;
      const std::vector<Stacktrace> f_cand_stacks = filter->filter(stack_by_report[report_idx]);

      if(f_cand_stacks.size() == 0)
        return;

      const std::vector<StackTraceInfo> query_stacks_info = agg->prepare(f_query_stacks, stack_by_report[report_idx]);
      const std::vector<StackTraceInfo> cand_stacks_info  = agg->prepare(f_cand_stacks, query_stacks);

      std::vector<double> matrix_score(query_stacks_info.size() * cand_stacks_info.size());

      for(size_t i = 0; i < query_stacks_info.size(); i++) {
        const Stacktrace *query_st = query_stacks_info[i].stack;
        for(size_t j = 0; j < cand_stacks_info.size(); j++) {
          const Stacktrace *cand_st = cand_stacks_info[j].stack;

          matrix_score[(i * cand_stacks_info.size()) + j] = method->similarity(query_st->stack, cand_st->stack);
        }
      }
      results[report_idx] = agg->aggregate(matrix_score, query_stacks_info, cand_stacks_info);
    });
  }

  return results;
}

#pragma region PYBIND_DECLARATIONS

PYBIND11_MODULE(cmp_func, m) {
  py::class_<std::vector<double>>(m, "doubleVector")
      .def(py::init<>())
      .def("clear", &std::vector<double>::clear)
      .def("pop_back", &std::vector<double>::pop_back)
      .def("push_back", (void (std::vector<double>::*)(const double &)) & std::vector<double>::push_back)
      .def("reserve", (void (std::vector<double>::*)(size_t)) & std::vector<double>::reserve)
      .def("__len__", [](const std::vector<double> &v) { return v.size(); })
      .def(
          "__iter__", [](std::vector<double> &v) { return py::make_iterator(v.begin(), v.end()); },
          py::keep_alive<0, 1>())
      .def("__getitem__", [](const std::vector<double> &v, size_t key) { return v[key]; })
      .def("__setitem__", [](std::vector<double> &v, size_t key, double value) { return v[key] = value; });
  py::class_<std::vector<bool>>(m, "boolVector")
      .def(py::init<>())
      .def("clear", &std::vector<bool>::clear)
      .def("pop_back", &std::vector<bool>::pop_back)
      .def("push_back", (void (std::vector<bool>::*)(const bool &)) & std::vector<bool>::push_back)
      .def("reserve", (void (std::vector<bool>::*)(size_t)) & std::vector<bool>::reserve)
      .def("__len__", [](const std::vector<bool> &v) { return v.size(); })
      .def(
          "__iter__", [](std::vector<bool> &v) { return py::make_iterator(v.begin(), v.end()); },
          py::keep_alive<0, 1>())
      .def("__getitem__", [](const std::vector<bool> &v, size_t key) { return v[key]; })
      .def("__setitem__", [](std::vector<bool> &v, size_t key, bool value) { v[key] = value; });
  m.attr("__name__") = "dedup_mod.cmp_func"; // Set Name as submodule
  m.def("compare", &compare);
}
#pragma endregion // PYBIND_DECLARATIONS
