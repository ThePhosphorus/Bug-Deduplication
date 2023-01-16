#include "method.hpp"

#pragma region PYBIND_DECLARATIONS

PYBIND11_MODULE(method, m) {
  m.attr("__name__") = "dedup_mod.method.method"; // Set Name as submodule
  py::enum_<SimilarityMethod>(m, "SimilarityMethod", py::arithmetic())
      .value("TRACE_SIM", SimilarityMethod::TRACE_SIM)
      .value("BRODIE_05", SimilarityMethod::BRODIE_05)
      .value("DARMERAU_LEVENSHTEIN", SimilarityMethod::DARMERAU_LEVENSHTEIN)
      .value("OPT_ALIGN", SimilarityMethod::OPT_ALIGN)
      .value("PDM_METHOD", SimilarityMethod::PDM_METHOD)
      .value("PREFIX_MATCH", SimilarityMethod::PREFIX_MATCH)
      .value("CRASH_GRAPH", SimilarityMethod::CRASH_GRAPH)
      .value("FAST", SimilarityMethod::FAST)
      .export_values();
  ;
}
#pragma endregion // PYBIND_DECLARATIONS