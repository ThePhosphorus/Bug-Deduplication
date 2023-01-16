#define BOOST_TEST_MODULE utilsS
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/test/unit_test.hpp>

#include <pybind11/embed.h>

#include <vector>

#include "../../../dedup_mod_pybind/util/utils.hpp"

boost::random::mt19937 seed;

BOOST_AUTO_TEST_CASE(TestTrue) { BOOST_CHECK(true); }
using TripleSt = std::pair<std::vector<size_t>, std::pair<std::vector<size_t>, std::vector<size_t>>>;

BOOST_AUTO_TEST_CASE(Strip) {
  // Create test stack traces
  const std::vector<bool> stopWords{1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1}; //{11, 5, 1, 0, 9, 2, 10};
  const std::vector<TripleSt> stacktraces = {
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {{3, 4, 6, 7, 8}, {3, 4, 5, 6, 7, 8}}},
      {{3, 4, 5, 6, 7, 8, 9, 10}, {{3, 4, 6, 7, 8}, {3, 4, 5, 6, 7, 8}}},
      {{0, 1, 2, 3, 4, 5, 6, 7, 8}, {{3, 4, 6, 7, 8}, {3, 4, 5, 6, 7, 8}}},
      {{11, 5, 1, 0, 9, 2, 10}, {{}, {}}},
      {{}, {{}, {}}},
  };
  for(const TripleSt &st: stacktraces) {
    std::vector<size_t> strippedStTrail                                 = st.first;
    std::vector<size_t> strippedStNoTrail                               = st.first;
    const std::pair<std::vector<size_t>, std::vector<size_t>> &expected = st.second;

    strip(strippedStTrail, stopWords, true);
    strip(strippedStNoTrail, stopWords, false);
    BOOST_CHECK_EQUAL_COLLECTIONS(strippedStNoTrail.begin(), strippedStNoTrail.end(), expected.first.begin(),
                                  expected.first.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(strippedStTrail.begin(), strippedStTrail.end(), expected.second.begin(),
                                  expected.second.end());
  }
}

BOOST_AUTO_TEST_CASE(toC) {
  const std::vector<bool> stopWords{1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1}; //{11, 5, 1, 0, 9, 2, 10};

  const std::pair<std::vector<std::vector<size_t>>,
                  std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<size_t>>>>
      stacktraces = {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                      {3, 4, 5, 6, 7, 8, 9, 10},
                      {0, 1, 2, 3, 4, 5, 6, 7, 8},
                      {11, 5, 1, 0, 9, 2, 10},
                      {}},
                     {{{3, 4, 6, 7, 8}, {3, 4, 6, 7, 8}, {3, 4, 6, 7, 8}, {}, {}},
                      {{3, 4, 5, 6, 7, 8}, {3, 4, 5, 6, 7, 8}, {3, 4, 5, 6, 7, 8}, {}, {}}}};

  // Generate random report_idx
  boost::random::uniform_int_distribution dist(0, 100000);

  // Test with and without strip

  // Check  that all frames are here if no stop word exists, and beg_trail_trim should change nothing
  {
    const size_t report_idx = 0;
    std::vector<Stacktrace> stacksNoStrip = generateStackTraces(report_idx, stacktraces.first, {}, true);
    // One of the stack traces is empty and should not be in the results
    BOOST_CHECK_EQUAL(stacksNoStrip.size(), stacktraces.first.size() - 1);

    for(size_t i = 0; i < stacksNoStrip.size(); i++) {
      const Stacktrace &st                   = stacksNoStrip[i];
      const std::vector<size_t> &expected_st = stacktraces.first[i];
      BOOST_CHECK_EQUAL(st.report_idx, report_idx);
      BOOST_CHECK_EQUAL_COLLECTIONS(st.stack.begin(), st.stack.end(), expected_st.begin(), expected_st.end());
    }
  }
  {
    const size_t report_idx = dist(seed);
    std::vector<Stacktrace> stacksNoStrip = generateStackTraces(report_idx, stacktraces.first, {}, true);
    // One of the stack traces is empty and should not be in the results

    BOOST_CHECK_EQUAL(stacksNoStrip.size(), stacktraces.first.size() - 1);

    for(size_t i = 0; i < stacksNoStrip.size(); i++) {
      const Stacktrace &st                   = stacksNoStrip[i];
      const std::vector<size_t> &expected_st = stacktraces.first[i];
      BOOST_CHECK_EQUAL(st.report_idx, report_idx);
      BOOST_CHECK_EQUAL_COLLECTIONS(st.stack.begin(), st.stack.end(), expected_st.begin(), expected_st.end());
    }
  }
  {
    const size_t report_idx = dist(seed);
    std::vector<Stacktrace> stacksNoTrail = generateStackTraces(report_idx, stacktraces.first, stopWords, false);
    // Two resulting stack traces should be empty and should not be added into the results
    BOOST_CHECK_EQUAL(stacksNoTrail.size(), stacktraces.second.first.size() - 2);

    for(size_t i = 0; i < stacksNoTrail.size(); i++) {
      const Stacktrace &st                   = stacksNoTrail[i];
      const std::vector<size_t> &expected_st = stacktraces.second.first[i];
      BOOST_CHECK_EQUAL(st.report_idx, report_idx);
      BOOST_CHECK_EQUAL_COLLECTIONS(st.stack.begin(), st.stack.end(), expected_st.begin(), expected_st.end());
    }
  }
  {
    const size_t report_idx             = dist(seed);
    std::vector<Stacktrace> stacksTrail = generateStackTraces(report_idx, stacktraces.first, stopWords, true);
    // Two resulting stack traces should be empty and should not be added into the results
    BOOST_CHECK_EQUAL(stacksTrail.size(), stacktraces.second.second.size() - 2);

    for(size_t i = 0; i < stacksTrail.size(); i++) {
      const Stacktrace &st                   = stacksTrail[i];
      const std::vector<size_t> &expected_st = stacktraces.second.second[i];
      BOOST_CHECK_EQUAL(st.report_idx, report_idx);
      BOOST_CHECK_EQUAL_COLLECTIONS(st.stack.begin(), st.stack.end(), expected_st.begin(), expected_st.end());
    }
  }
}

BOOST_AUTO_TEST_CASE(toCByReport) {
  const std::vector<bool> stopWords{1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1}; //{11, 5, 1, 0, 9, 2, 10};

  std::vector<std::vector<std::vector<size_t>>> stackstracesByReports;
  // Generate a random nb of reports
  boost::random::uniform_int_distribution nbReportsDist(10, 20);
  boost::random::uniform_int_distribution framesDist(0, 11);

  const auto nbReports = nbReportsDist(seed);
  stackstracesByReports.reserve(nbReports);
  for(size_t i = 0; i < nbReports; i++) {
    const size_t nbStackTraces = nbReportsDist(seed);
    std::vector<std::vector<size_t>> stacktraces;
    stacktraces.reserve(nbStackTraces);
    for(size_t j = 0; j < nbStackTraces; j++) {
      const size_t nbFrames = nbReportsDist(seed);
      std::vector<size_t> stacktrace;
      stacktrace.reserve(nbFrames);
      for(size_t k = 0; k < nbFrames; k++) {
        stacktrace.push_back(framesDist(seed));
      }
      stacktraces.push_back(stacktrace);
    }
    stackstracesByReports.emplace_back(stacktraces);
  }
  // use no stop words
  {
    std::vector<std::vector<Stacktrace>> results = generateStackTracesByReport(stackstracesByReports, {}, true);
    BOOST_CHECK_EQUAL(results.size(), stackstracesByReports.size());
    // For each report check it's the same result as running to_C on it
    for(size_t i = 0; i < results.size(); i++) {
      std::vector<Stacktrace> expected = generateStackTraces(i, stackstracesByReports[i], {}, true);
      auto result = results[i];
      BOOST_CHECK_EQUAL(result.size(), expected.size());
      for(size_t j = 0; j < result.size(); j++) {
        auto expectedSt = expected[j];
        auto resultSt   = result[j];
        BOOST_CHECK_EQUAL(expectedSt.report_idx, resultSt.report_idx);
        BOOST_CHECK_EQUAL_COLLECTIONS(expectedSt.stack.begin(), expectedSt.stack.end(), resultSt.stack.begin(),
                                      resultSt.stack.end());
      }
    }
  }
  {
    std::vector<std::vector<Stacktrace>> results = generateStackTracesByReport(stackstracesByReports, {}, false);
    BOOST_CHECK_EQUAL(results.size(), stackstracesByReports.size());
    // For each report check it's the same result as running to_C on it
    for(size_t i = 0; i < results.size(); i++) {
      std::vector<Stacktrace> expected = generateStackTraces(i, stackstracesByReports[i], {}, false);
      auto result = results[i];
      BOOST_CHECK_EQUAL(result.size(), expected.size());
      for(size_t j = 0; j < result.size(); j++) {
        auto expectedSt = expected[j];
        auto resultSt   = result[j];
        BOOST_CHECK_EQUAL(expectedSt.report_idx, resultSt.report_idx);
        BOOST_CHECK_EQUAL_COLLECTIONS(expectedSt.stack.begin(), expectedSt.stack.end(), resultSt.stack.begin(),
                                      resultSt.stack.end());
      }
    }
  }
  // Use stop word
  {
    std::vector<std::vector<Stacktrace>> results = generateStackTracesByReport(stackstracesByReports, stopWords, true);
    BOOST_CHECK_EQUAL(results.size(), stackstracesByReports.size());
    // For each report check it's the same result as running to_C on it
    for(size_t i = 0; i < results.size(); i++) {
      std::vector<Stacktrace> expected = generateStackTraces(i, stackstracesByReports[i], stopWords, true);
      auto result                      = results[i];
      BOOST_CHECK_EQUAL(result.size(), expected.size());
      for(size_t j = 0; j < result.size(); j++) {
        auto expectedSt = expected[j];
        auto resultSt   = result[j];
        BOOST_CHECK_EQUAL(expectedSt.report_idx, resultSt.report_idx);
        BOOST_CHECK_EQUAL_COLLECTIONS(expectedSt.stack.begin(), expectedSt.stack.end(), resultSt.stack.begin(),
                                      resultSt.stack.end());
      }
    }
  }
  {
    std::vector<std::vector<Stacktrace>> results =
        generateStackTracesByReport(stackstracesByReports, stopWords, false);
    BOOST_CHECK_EQUAL(results.size(), stackstracesByReports.size());
    // For each report check it's the same result as running to_C on it
    for(size_t i = 0; i < results.size(); i++) {
      std::vector<Stacktrace> expected = generateStackTraces(i, stackstracesByReports[i], stopWords, false);
      auto result                      = results[i];
      BOOST_CHECK_EQUAL(result.size(), expected.size());
      for(size_t j = 0; j < result.size(); j++) {
        auto expectedSt = expected[j];
        auto resultSt   = result[j];
        BOOST_CHECK_EQUAL(expectedSt.report_idx, resultSt.report_idx);
        BOOST_CHECK_EQUAL_COLLECTIONS(expectedSt.stack.begin(), expectedSt.stack.end(), resultSt.stack.begin(),
                                      resultSt.stack.end());
      }
    }
  }
}
