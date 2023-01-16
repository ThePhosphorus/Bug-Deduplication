#define BOOST_TEST_MODULE COMPARATOR
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>

#include <pybind11/embed.h>

#include <vector>

#include "../../../dedup_mod_pybind/util/comparator.hpp"

boost::random::mt19937 seed;

BOOST_AUTO_TEST_CASE(TestTrue) { BOOST_CHECK(true); }

BOOST_AUTO_TEST_CASE(CompareAlignElem) {
  std::vector<DfInfo> documentFreq     = {{.idx = 0, .max_value = 100., .mean_value = 28.},
                                      {.idx = 1, .max_value = 100., .mean_value = 30.},
                                      {.idx = 2, .max_value = 30., .mean_value = 30.},
                                      {.idx = 3, .max_value = 0., .mean_value = 1.},
                                      {.idx = 4, .max_value = 30., .mean_value = 1.}};
  std::vector<size_t> expected_results = {3, 4, 2, 0, 1};

  std::sort(documentFreq.begin(), documentFreq.end(), compare_align_elem);

  for(size_t i = 0; i < documentFreq.size(); i++) {
    BOOST_CHECK_EQUAL(documentFreq[i].idx, expected_results[i]);
  }
}

BOOST_AUTO_TEST_CASE(ComputeMeanWeight) {
  Stacktrace st{.report_idx = 0, .stack = {0, 1, 2, 3, 4, 5}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 6; i++) {
    document_frequencies.push_back(dist(seed));
  }

  double df_mean = 0.;
  for(size_t i = 0; i < document_frequencies.size(); i++) {
    df_mean += 1. - document_frequencies[i];
  }
  df_mean /= document_frequencies.size();

  BOOST_CHECK_EQUAL(compute_weight(&st, document_frequencies), df_mean);
}

BOOST_AUTO_TEST_CASE(ComputeMaxWeight) {
  Stacktrace st{.report_idx = 0, .stack = {0, 1, 2, 3, 4, 5}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 6; i++) {
    document_frequencies.push_back(dist(seed));
  }

  double df_max = -INFINITY;
  for(size_t i = 0; i < document_frequencies.size(); i++) {
    df_max = std::max(df_max, 1. - document_frequencies[i]);
  }

  BOOST_CHECK_EQUAL(compute_weight_max(&st, document_frequencies), df_max);
}

BOOST_AUTO_TEST_SUITE(BaseComparator)
BOOST_AUTO_TEST_CASE(Prepare) {
  // Create obj instance
  Comparator comparator;

  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  std::vector<StackTraceInfo> stInfos = comparator.prepare(stacktraces, {});

  // We should get as many results as stack traces given
  BOOST_CHECK_EQUAL(stacktraces.size(), stInfos.size());

  for(size_t i = 0; i < stInfos.size(); i++) {
    const StackTraceInfo &stinfo = stInfos[i];
    // Index should be incremetal from 0
    BOOST_CHECK_EQUAL(stinfo.idx, i);
    // All weights should be 1
    BOOST_CHECK_EQUAL(stinfo.weight, 1.);
    // Should have a valid pointer to the stacktrace
    BOOST_CHECK_EQUAL(stinfo.stack, &stacktraces[i]);
  }
}

BOOST_AUTO_TEST_CASE(aggregate) {
  // Create obj instance
  Comparator comparator;

  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  std::vector<StackTraceInfo> stInfos = comparator.prepare(stacktraces, {});
  std::vector<double> matrix_score;

  // Function not implemented and therefor should throw
  BOOST_CHECK_THROW(comparator.aggregate(matrix_score, stInfos, stInfos), std::logic_error);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MaxComparator)
BOOST_AUTO_TEST_CASE(Prepare) {
  // Create obj instance
  Max comparator;

  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  std::vector<StackTraceInfo> stInfos = comparator.prepare(stacktraces, {});

  // We should get as many results as stack traces given
  BOOST_CHECK_EQUAL(stacktraces.size(), stInfos.size());

  for(size_t i = 0; i < stInfos.size(); i++) {
    const StackTraceInfo &stinfo = stInfos[i];
    // Index should be incremetal from 0
    BOOST_CHECK_EQUAL(stinfo.idx, i);
    // All weights should be 1
    BOOST_CHECK_EQUAL(stinfo.weight, 1.);
    // Should have a valid pointer to the stacktrace
    BOOST_CHECK_EQUAL(stinfo.stack, &stacktraces[i]);
  }
}

BOOST_AUTO_TEST_CASE(aggregate) {
  // Create obj instance
  Max comparator;

  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  std::vector<StackTraceInfo> stInfos = comparator.prepare(stacktraces, {});
  std::vector<double> matrix_score    = {1.0, 5.0, 7.0, -999.0, 0., 60., 999.9, 2000, 2};

  // Should return top matrix score
  BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos, stInfos), 2000.);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MeanComparator)
BOOST_AUTO_TEST_CASE(Prepare) {
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {1}}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }
  // Create obj instance for each MeanType and WeightType
  std::vector<Mean> comparators;
  for(size_t meanType = 0; meanType <= static_cast<size_t>(MeanType::QUERY_CAND); meanType++) {
    for(size_t weightType = 0; weightType <= static_cast<size_t>(WeightType::MAXI); weightType++) {
      comparators.push_back(
          Mean(static_cast<MeanType>(meanType), static_cast<WeightType>(weightType), document_frequencies));
    }
  }

  for(const Mean &comparator: comparators) {
    std::vector<StackTraceInfo> stInfos = comparator.prepare(stacktraces, {});

    // We should get as many results as stack traces given
    BOOST_CHECK_EQUAL(stacktraces.size(), stInfos.size());

    for(size_t i = 0; i < stInfos.size(); i++) {
      const StackTraceInfo &stinfo = stInfos[i];
      // Index should be incremetal from 0
      BOOST_CHECK_EQUAL(stinfo.idx, i);
      // Should have a valid pointer to the stacktrace
      BOOST_CHECK_EQUAL(stinfo.stack, &stacktraces[i]);

      if(comparator.weight_type == WeightType::OFF) {
        BOOST_CHECK_EQUAL(stinfo.weight, 1.);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(aggregate_QUERY) {
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {1}}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }

  // Create obj instance for each WeightType
  std::vector<Mean> comparators;
  for(size_t weightType = 0; weightType <= static_cast<size_t>(WeightType::MAXI); weightType++) {
    comparators.push_back(
        Mean(static_cast<MeanType>(MeanType::QUERY), static_cast<WeightType>(weightType), document_frequencies));
  }

  for(const Mean &comparator: comparators) {
    std::vector<StackTraceInfo> stInfos = comparator.prepare(stacktraces, {});
    std::vector<double> matrix_score    = {1., 5., 7., -999., 0., 60., 999.9, 2000., 2.};

    // Should be the weighted average of max per query
    const double expectedResult = ((7. * stInfos[0].weight) + (60. * stInfos[1].weight) + (2000. * stInfos[2].weight)) /
                                  (stInfos[0].weight + stInfos[1].weight + stInfos[2].weight);
    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos, stInfos), expectedResult);
  }
}

BOOST_AUTO_TEST_CASE(aggregate_CANDIDATE) {
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {1}}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }

  // Create obj instance for each WeightType
  std::vector<Mean> comparators;
  for(size_t weightType = 0; weightType <= static_cast<size_t>(WeightType::MAXI); weightType++) {
    comparators.push_back(
        Mean(static_cast<MeanType>(MeanType::CANDIDATE), static_cast<WeightType>(weightType), document_frequencies));
  }

  for(const Mean &comparator: comparators) {
    std::vector<StackTraceInfo> stInfos = comparator.prepare(stacktraces, {});
    std::vector<double> matrix_score    = {1., 5., 7., -999., 0., 60., 999.9, 2000., 2.};

    // Should be the weighted average of max per candidate
    const double expectedResult =
        ((999.9 * stInfos[0].weight) + (2000. * stInfos[1].weight) + (60. * stInfos[2].weight)) /
        (stInfos[0].weight + stInfos[1].weight + stInfos[2].weight);
    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos, stInfos), expectedResult);
  }
}

BOOST_AUTO_TEST_CASE(aggregate_SHORTEST) {
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces1{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                       {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                       {.report_idx = 0, .stack = {1}}};

  std::vector<Stacktrace> stacktraces2{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                       {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                       {.report_idx = 0, .stack = {1}},
                                       {.report_idx = 2, .stack = {25, 3, 2, 1, 0}}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }

  // Create obj instance for each WeightType
  std::vector<Mean> comparators;
  for(size_t weightType = 0; weightType <= static_cast<size_t>(WeightType::MAXI); weightType++) {
    comparators.push_back(
        Mean(static_cast<MeanType>(MeanType::SHORTEST), static_cast<WeightType>(weightType), document_frequencies));
  }

  for(const Mean &comparator: comparators) {
    std::vector<StackTraceInfo> stInfos1 = comparator.prepare(stacktraces1, {});
    std::vector<StackTraceInfo> stInfos2 = comparator.prepare(stacktraces2, {});
    std::vector<double> matrix_score     = {1., 5., 7., -999., 0., 60., 999.9, 2000., 2., 23., 44., -20.};

    // Should be the weighted average of max for shortest (1)
    const double expectedResult1 =
        ((7. * stInfos1[0].weight) + (2000. * stInfos1[1].weight) + (44. * stInfos1[2].weight)) /
        (stInfos1[0].weight + stInfos1[1].weight + stInfos1[2].weight);
    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos1, stInfos2), expectedResult1);

    const double expectedResult2 =
        ((999.9 * stInfos1[0].weight) + (2000. * stInfos1[1].weight) + (60. * stInfos1[2].weight)) /
        (stInfos1[0].weight + stInfos1[1].weight + stInfos1[2].weight);
    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos2, stInfos1), expectedResult2);
  }
}

BOOST_AUTO_TEST_CASE(aggregate_LONGEST) {
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces1{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                       {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                       {.report_idx = 0, .stack = {1}}};

  std::vector<Stacktrace> stacktraces2{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                       {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                       {.report_idx = 0, .stack = {1}},
                                       {.report_idx = 2, .stack = {25, 3, 2, 1, 0}}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }

  // Create obj instance for each WeightType
  std::vector<Mean> comparators;
  for(size_t weightType = 0; weightType <= static_cast<size_t>(WeightType::MAXI); weightType++) {
    comparators.push_back(
        Mean(static_cast<MeanType>(MeanType::LONGEST), static_cast<WeightType>(weightType), document_frequencies));
  }

  for(const Mean &comparator: comparators) {
    std::vector<StackTraceInfo> stInfos1 = comparator.prepare(stacktraces1, {});
    std::vector<StackTraceInfo> stInfos2 = comparator.prepare(stacktraces2, {});
    std::vector<double> matrix_score     = {1., 5., 7., -999., 0., 60., 999.9, 2000., 2., 23., 44., -20.};

    // Should be the weighted average of max for shortest (1)
    const double expectedResult1 = ((2. * stInfos2[0].weight) + (60. * stInfos2[1].weight) +
                                    (999.9 * stInfos2[2].weight) + (2000. * stInfos2[3].weight)) /
                                   (stInfos2[0].weight + stInfos2[1].weight + stInfos2[2].weight + stInfos2[3].weight);
    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos1, stInfos2), expectedResult1);

    const double expectedResult2 = ((7. * stInfos2[0].weight) + (60. * stInfos2[1].weight) +
                                    (2000. * stInfos2[2].weight) + (44. * stInfos2[3].weight)) /
                                   (stInfos2[0].weight + stInfos2[1].weight + stInfos2[2].weight + stInfos2[3].weight);
    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos2, stInfos1), expectedResult2);
  }
}

BOOST_AUTO_TEST_CASE(aggregate_QUERY_CAND) {
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces1{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                       {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                       {.report_idx = 0, .stack = {1}}};

  std::vector<Stacktrace> stacktraces2{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                       {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                       {.report_idx = 0, .stack = {1}},
                                       {.report_idx = 2, .stack = {25, 3, 2, 1, 0}}};

  // Generate random document frequencies
  boost::random::uniform_real_distribution dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }

  // Create obj instance for each WeightType
  std::vector<Mean> comparators;
  for(size_t weightType = 0; weightType <= static_cast<size_t>(WeightType::MAXI); weightType++) {
    comparators.push_back(
        Mean(static_cast<MeanType>(MeanType::QUERY_CAND), static_cast<WeightType>(weightType), document_frequencies));
  }

  for(const Mean &comparator: comparators) {
    std::vector<StackTraceInfo> stInfos1 = comparator.prepare(stacktraces1, {});
    std::vector<StackTraceInfo> stInfos2 = comparator.prepare(stacktraces2, {});
    std::vector<double> matrix_score     = {1., 5., 7., -999., 0., 60., 999.9, 2000., 2., 23., 44., -20.};

    // Should be the weighted average of max for shortest (1)
    const double expectedResult11 = ((2. * stInfos2[0].weight) + (60. * stInfos2[1].weight) +
                                     (999.9 * stInfos2[2].weight) + (2000. * stInfos2[3].weight)) /
                                    (stInfos2[0].weight + stInfos2[1].weight + stInfos2[2].weight + stInfos2[3].weight);
    const double expectedResult12 =
        ((7. * stInfos1[0].weight) + (2000. * stInfos1[1].weight) + (44. * stInfos1[2].weight)) /
        (stInfos1[0].weight + stInfos1[1].weight + stInfos1[2].weight);

    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos1, stInfos2),
                      (expectedResult11 + expectedResult12) / 2.);

    const double expectedResult21 = ((7. * stInfos2[0].weight) + (60. * stInfos2[1].weight) +
                                     (2000. * stInfos2[2].weight) + (44. * stInfos2[3].weight)) /
                                    (stInfos2[0].weight + stInfos2[1].weight + stInfos2[2].weight + stInfos2[3].weight);
    const double expectedResult22 =
        ((999.9 * stInfos1[0].weight) + (2000. * stInfos1[1].weight) + (60. * stInfos1[2].weight)) /
        (stInfos1[0].weight + stInfos1[1].weight + stInfos1[2].weight);

    BOOST_CHECK_EQUAL(comparator.aggregate(matrix_score, stInfos2, stInfos1),
                      (expectedResult21 + expectedResult22) / 2.);
  }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(BaseFilter)
BOOST_AUTO_TEST_CASE(filter) {
  // Create obj instance
  Filter filter;

  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  const auto filtered = filter.filter(stacktraces);

  // We should get as many results as stack traces given
  BOOST_CHECK_EQUAL(stacktraces.size(), filtered.size());

  // Should not change any stack trace
  for(size_t i = 0; i < filtered.size(); i++) {
    BOOST_CHECK_EQUAL(filtered[i].report_idx, stacktraces[i].report_idx);
    BOOST_CHECK_EQUAL_COLLECTIONS(filtered[i].stack.begin(), filtered[i].stack.end(), stacktraces[i].stack.begin(),
                                  stacktraces[i].stack.end());
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

  // Create obj instance
  SelectOne filter(document_frequencies);

  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {}}};

  const auto filtered = filter.filter(stacktraces);

  // We should get only one result (Hopefully it's the best)
  BOOST_CHECK_EQUAL(1, filtered.size());
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(KTopFilter)
BOOST_AUTO_TEST_CASE(filterNoTresh) {
  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }

  // Create obj instance
  KTopFunction filter(1., document_frequencies);

  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {1}}};

  const auto filtered = filter.filter(stacktraces);

  // We should get as many results as stack traces given
  BOOST_CHECK_EQUAL(stacktraces.size(), filtered.size());
}

BOOST_AUTO_TEST_CASE(filterRandTresh) {
  // Generate random document frequencies
  boost::random::uniform_real_distribution<> dist(0., 1.);
  std::vector<DTYPE_t> document_frequencies;

  for(size_t i = 0; i < 33; i++) {
    document_frequencies.push_back(dist(seed));
  }

  std::vector<DTYPE_t> sortedDf(document_frequencies.size());
  std::transform(document_frequencies.begin(), document_frequencies.end(), sortedDf.begin(),
                 [](DTYPE_t df) { return 1. - df; });
  std::sort(sortedDf.begin(), sortedDf.end());
  // Create stack trace objs
  std::vector<Stacktrace> stacktraces{{.report_idx = 420, .stack = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
                                      {.report_idx = 69, .stack = {32, 2, 1, 0}},
                                      {.report_idx = 0, .stack = {1}}};

  // Because of randomness run this test A LOT of times
  for(size_t i = 0; i < 200; i++) {
    // Create obj instance
    double filter_k        = dist(seed);
    const double threshold = sortedDf[static_cast<size_t>((1. - filter_k) * document_frequencies.size())];
    KTopFunction filter(filter_k, document_frequencies);

    const auto filtered = filter.filter(stacktraces);
    for(const Stacktrace &st: filtered) {
      // Each stack trace should have at least one stack above threshold
      BOOST_CHECK(std::any_of(st.stack.begin(), st.stack.end(),
                              [&](size_t frameId) { return (1. - document_frequencies[frameId]) >= threshold; }));
    }
  }
}
BOOST_AUTO_TEST_SUITE_END()
