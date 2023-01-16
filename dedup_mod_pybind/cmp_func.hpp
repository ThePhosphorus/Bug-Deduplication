#include "method/brodie_05.hpp"
#include "method/crash_graph.hpp"
#include "method/edit_distance.hpp"
#include "method/method.hpp"
#include "method/pdm.hpp"
#include "method/prefix_match.hpp"
#include "method/trace_sim.hpp"
#include "method/fast.hpp"
#include "util/comparator.hpp"
#include "util/utils.hpp"
#include "util/structures.hpp"

std::vector<DTYPE_t> compare(SimilarityMethod method_type, AggStrategy agg_strategy, const py::list& args,
                             const std::vector<std::vector<size_t>> &query,
                             const std::vector<std::vector<std::vector<size_t>>> &candidates,
                             const std::vector<double> &df, FilterStrategy filter_strategy, double filter_k,
                             const std::vector<bool> &is_stop_word, bool beg_trail_trim);
