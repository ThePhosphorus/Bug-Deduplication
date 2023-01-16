#define BOOST_TEST_MODULE TRACE_SIM
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>

#include <pybind11/embed.h>


#include <vector>

#include "../../../dedup_mod_pybind/method/trace_sim.hpp"
#include "../../../dedup_mod_pybind/util/structures.hpp"

boost::random::mt19937 seed;

BOOST_AUTO_TEST_CASE(TestTrue) { BOOST_CHECK(true); }

BOOST_AUTO_TEST_CASE(Similarity) {
  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  TraceSim model(document_frequencies, 1., 1., 1., 1., 1., 1., true, true, true, true, true, true, true);

  const size_t nb_st = stacktraces.size();
  // Create 2D score array
  std::vector<std::vector<double>> results;
  results.assign(nb_st, std::vector<double>(nb_st));

  for(size_t i = 0; i < nb_st; i++) {
    for(size_t j = 0; j < nb_st; j++) {
      results[i][j] = model.similarity(stacktraces[i].stack, stacktraces[j].stack);
    }
  }

  // For each stack itself should have the best score
  for(size_t i = 0; i < nb_st; i++) {
    double maxValue = -999999999.;
    size_t maxId    = -1;
    for(size_t j = 0; j < nb_st; j++) {
      if(results[i][j] > maxValue) {
        maxValue = results[i][j];
        maxId    = j;
      }
    }
    BOOST_CHECK_EQUAL(maxId, i);
  }

  // Check 2 similar stack traces, should have a better score than dissimilar stack traces
  std::vector<Stacktrace> pair1{{.report_idx = 6, .stack = {10, 9, 8, 5, 4, 3, 2, 1, 0}},
                                {.report_idx = 7, .stack = {10, 9, 8, 7, 6, 5, 4, 1, 0}}};

  std::vector<Stacktrace> pair2{{.report_idx = 8, .stack = {15, 22, 16, 26, 3, 2, 1, 0}},
                                {.report_idx = 9, .stack = {15, 22, 16, 26, 4, 3, 2, 1, 0}}};

  BOOST_CHECK_GT(model.similarity(pair1[0].stack, pair1[1].stack), model.similarity(pair1[0].stack, pair2[0].stack));
  BOOST_CHECK_GT(model.similarity(pair2[0].stack, pair2[1].stack), model.similarity(pair2[0].stack, pair1[0].stack));

  // Score similarity should be symetric
  BOOST_CHECK_EQUAL(model.similarity(pair1[0].stack, pair1[1].stack), model.similarity(pair1[1].stack, pair1[0].stack));
}
