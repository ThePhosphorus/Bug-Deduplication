#define BOOST_TEST_MODULE FILTER
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>

#include <pybind11/embed.h>

#include <vector>

#include "../../../dedup_mod_pybind/util/comparator.hpp"
#include "../../../dedup_mod_pybind/util/structures.hpp"

boost::random::mt19937 seed;

BOOST_AUTO_TEST_CASE(TestTrue) { BOOST_CHECK(true); }

BOOST_AUTO_TEST_SUITE(BaseFilter)
BOOST_AUTO_TEST_CASE(filter) {
  Filter filter;
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  auto filteredStacktraces = filter.filter(stacktraces);

  // Should return the same elements (no filtering is done)
  BOOST_CHECK_EQUAL(stacktraces.size(), filteredStacktraces.size());

  for(size_t i = 0; i < filteredStacktraces.size(); i++) {
    const Stacktrace &expected = stacktraces[i];
    const Stacktrace &got      = filteredStacktraces[i];
    BOOST_CHECK_EQUAL(expected.report_idx, got.report_idx);
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.stack.begin(), expected.stack.end(), got.stack.begin(), got.stack.end());
  }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(SelectOneFilter)
BOOST_AUTO_TEST_CASE(filter) {
  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }
  SelectOne filter(document_frequencies);
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  std::vector<Stacktrace> oneStack = {stacktraces[0]};

  auto filteredStacktracesOne = filter.filter(oneStack);

  // Should return the same elements (no filtering is done) if there is only one stacktrace
  BOOST_CHECK_EQUAL(filteredStacktracesOne.size(), 1);

  BOOST_CHECK_EQUAL(oneStack[0].report_idx, filteredStacktracesOne[0].report_idx);
  BOOST_CHECK_EQUAL_COLLECTIONS(oneStack[0].stack.begin(), oneStack[0].stack.end(),
                                filteredStacktracesOne[0].stack.begin(), filteredStacktracesOne[0].stack.end());

  auto filteredStacktraces = filter.filter(stacktraces);

  // Should only return one result
  BOOST_CHECK_EQUAL(filteredStacktraces.size(), 1);

  // The report should have the best score
  std::vector<DfInfo> dfInfos = {};
  for(const auto &stacktrace: stacktraces) {
    dfInfos.push_back({.idx        = stacktrace.report_idx,
                       .max_value  = compute_weight_max(&stacktrace, document_frequencies),
                       .mean_value = compute_weight(&stacktrace, document_frequencies)});
  }

  // sort to have the best
  std::sort(dfInfos.begin(), dfInfos.end(), compare_align_elem);

  BOOST_CHECK_EQUAL(filteredStacktraces[0].report_idx, dfInfos.back().idx);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(KTopFunctionFilter)

BOOST_AUTO_TEST_CASE(Constructor) {
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 6; i++) {
    document_frequencies.push_back(dist(seed));
  }

  // Should throw if filter_k is outside ]0,1]
  BOOST_CHECK_THROW({ const auto value = KTopFunction(0, document_frequencies); }, std::runtime_error);
  BOOST_CHECK_THROW({ KTopFunction(-1, document_frequencies); }, std::runtime_error);
  BOOST_CHECK_THROW({ KTopFunction(-INFINITY, document_frequencies); }, std::runtime_error);
  BOOST_CHECK_THROW({ KTopFunction(2, document_frequencies); }, std::runtime_error);
  BOOST_CHECK_THROW({ KTopFunction(INFINITY, document_frequencies); }, std::runtime_error);
  // Should not throw inside ]0,1]
  BOOST_CHECK_NO_THROW({ KTopFunction(1, document_frequencies); });
  BOOST_CHECK_NO_THROW({ KTopFunction(0.001, document_frequencies); });

  std::vector<DTYPE_t> dfManual{0.63, 0.42, 0.88, 0.40, 0.65, 0.33, 1., 0.02, 0.60, 0.10};
  // threashold should be second best with 20% with 10 elems

  BOOST_CHECK_EQUAL(KTopFunction(0.2, dfManual).threshold, 1. - 0.10);
}

BOOST_AUTO_TEST_CASE(filter) {
  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 100; i++) {
    document_frequencies.push_back(dist(seed));
  }
  KTopFunction filter(0.5, document_frequencies);
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  const double threashold = filter.threshold;

  auto filteredStacktraces = filter.filter(stacktraces);

  // Safety check for filtered size
  BOOST_CHECK_LE(filteredStacktraces.size(), stacktraces.size());

  // Should return the same elements (no filtering is done) if there is only one stacktrace
  for(const auto &stacktrace: filteredStacktraces) {
    bool oneFrameAboveThreashold = false;
    for(const auto &frame: stacktrace.stack) {
      if(1. - document_frequencies[frame] >= threashold) {
        oneFrameAboveThreashold = true;
        break;
      }
    }
    BOOST_CHECK(oneFrameAboveThreashold);
  }
}
BOOST_AUTO_TEST_SUITE_END()
