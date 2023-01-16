#include "trace_sim.hpp"

//////////////////////////////////
// TraceSim Implementation
// DOI : https://doi.org/10.1145/3416505.3423561
//////////////////////////////////

static bool compareFrame(const WeightedFrame &lhs, const WeightedFrame &rhs) {
  if(lhs.id == rhs.id) {
    return lhs.weight < rhs.weight;
  } else {
    return lhs.id < rhs.id;
  }
}

static void sumEqualFunctions(std::vector<WeightedFrame>& frames) {
  const size_t seq_len = frames.size();
  if(seq_len == 0) {
    return;
  }

  size_t firstFrameId = 0;
  for(size_t i = 1; i < seq_len; i++) {
    const WeightedFrame &curFrame = frames[i];

    if(frames[i - 1].id == curFrame.id) {
      frames[firstFrameId].weight += curFrame.weight;
    } else {
      firstFrameId = i;
    }
  }
}

static std::vector<WeightedFrame> createFrames(const std::vector<size_t> &ids, const std::vector<double> &weights) {
  if(ids.size() != weights.size()) {
    throw std::runtime_error("Size Mismatch when creating frames");
  }

  std::vector<WeightedFrame> results;
  const size_t size = ids.size();

  results.reserve(size);
  for(size_t i = 0; i < size; i++) {
    WeightedFrame frame = {.id = ids[i], .weight = weights[i]};
    results.push_back(frame);
  }
  return results;
}

static double calculateDen(const std::vector<size_t> &query, const std::vector<double> &queryPosValues,
                           const std::vector<size_t> &candidate, const std::vector<double> &candPosValues) {
  //const size_t querySize = query.size(), candidSize = candidate.size();
  //std::vector<WeightedFrame> queryFrames = createFrames(query, queryPosValues);
  //std::vector<WeightedFrame> candFrames  = createFrames(candidate, candPosValues);

  //std::sort(queryFrames.begin(), queryFrames.end(), compareFrame);
  //std::sort(candFrames.begin(), candFrames.end(), compareFrame);

  //sumEqualFunctions(queryFrames);
  //sumEqualFunctions(candFrames);

  //std::vector<WeightedFrame> merged(queryFrames.size() + candFrames.size());
  //// Merge both sorted arrays
  //auto endIt = std::set_union(queryFrames.begin(), queryFrames.end(), candFrames.begin(), candFrames.end(), merged.begin(),
  //           compareFrame);
  //// Sort from biggest to smallest
  //std::reverse(merged.begin(), endIt);
  //
  //// Remove other weights (keep max)
  //auto itToRemove = std::unique(merged.begin(), endIt, [](const WeightedFrame &lhs, const WeightedFrame &rhs) { return lhs.id == rhs.id; });

  //// Sum of weights
  //return std::accumulate(merged.begin(), itToRemove, 0.,
  //                       [](double sum, WeightedFrame elem) { return sum + elem.weight; });
  return std::accumulate(queryPosValues.begin(), queryPosValues.end(), 0.,
                         [](double sum, double elem) { return sum + elem; }) + 
      std::accumulate(candPosValues.begin(), candPosValues.end(), 0.,
                         [](double sum, double elem) { return sum + elem; });
}

TraceSim::TraceSim(const std::vector<double> &df, double df_coef, double pos_coef, double diff_coef, double match_cost,
                   double gap_penalty, double mismatch_penalty, bool sigmoid, double gamma, bool useSum, bool useIdf,
                   bool const_match, bool reciprocal_func, bool no_norm)
    : df(df), df_coef(df_coef), pos_coef(pos_coef), diff_coef(diff_coef), match_cost(match_cost),
      gap_penalty(gap_penalty), mismatch_penalty(mismatch_penalty), sigmoid(sigmoid), gamma(gamma), useSum(useSum), useIdf(useIdf),
      const_match(const_match), reciprocal_func(reciprocal_func), no_norm(no_norm) {
  if(!sigmoid && useIdf) {
    throw std::runtime_error("Exponential function cannot receive IDF");
  }
}

double TraceSim::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  const size_t cand_len = candidate.size(), query_len = query.size();
  const size_t offset = cand_len + 1;
  std::vector<double> M_THIS_ROW;
  M_THIS_ROW.resize(offset);
  M_THIS_ROW[0] = 0.;
  std::vector<double> M_PAST_ROW;
  M_PAST_ROW.resize(offset);

  std::vector<double> queryPosValues;
  queryPosValues.resize(query_len);
  std::vector<double> candPosValues;
  candPosValues.resize(cand_len);


  for(size_t queryPos = 0; queryPos < query_len; queryPos++) {
    queryPosValues[queryPos] = calculate_weight(query, queryPos);
  }

  for(size_t candPos = 0; candPos < cand_len; candPos++) {
    candPosValues[candPos] = calculate_weight(candidate, candPos);
  }

  // Craete first row
  for(size_t i = 0; i < cand_len; i++) {
    M_THIS_ROW [(i + 1)] = M_THIS_ROW[i] - gap_penalty * candPosValues[i];
  }

  for(size_t i = 0; i < query_len; i++) {
    M_THIS_ROW.swap(M_PAST_ROW);

    M_THIS_ROW [0] = M_PAST_ROW [0] - gap_penalty * queryPosValues[i];

    for(size_t j = 0; j < cand_len; j++) {
      // Gap
      const double delete_cost  = -gap_penalty * queryPosValues[i];
      const double previous_row = M_PAST_ROW [j + 1] + delete_cost;

      const double insert_cost  = -gap_penalty * candPosValues[j];
      const double previous_col = M_THIS_ROW [j] + insert_cost;

      double previous_row_col    = M_PAST_ROW [j];
      const double mismatch_cost = -mismatch_penalty * (useSum ? (queryPosValues[i] + candPosValues[j])
                                                            : std::max(queryPosValues[i], candPosValues[j]));

      if(query[i] == candidate[j]) {
        const double cost =
            const_match ? 1.
                        : (std::max(queryPosValues[i], candPosValues[j]) *
                           std::exp(-diff_coef * std::fabs(static_cast<double>(i) - static_cast<double>(j))));
        previous_row_col += match_cost * cost;
      } else {

        previous_row_col += mismatch_cost;
      }

      M_THIS_ROW[j + 1] = std::max({previous_row, previous_col, previous_row_col});
    }
  }

  const auto sim = M_THIS_ROW.back();

  double normalized_sim;
  if(no_norm) {
    normalized_sim = sim;    
  } else {
    const double den = calculateDen(query, queryPosValues, candidate, candPosValues);

    normalized_sim = (den == 0.) ? 0. : sim / den;
  }

  return normalized_sim;
}

double TraceSim::calculate_weight(const std::vector<size_t> &trace, size_t pos) const {
  double gw = 0;
  
  const double lw = (reciprocal_func) ? (1. / std::pow((pos + 1), pos_coef)) : std::exp(-pos_coef * pos);

  if(sigmoid) {
    double sigmoid_val = 1. / (1 + std::exp(-df_coef * (getIDF(trace[pos]) - gamma)));

    if(!useIdf) { // TODO : Check validity
      sigmoid_val = 1. - sigmoid_val;
    }
    gw = sigmoid_val;
  } else {
    gw = std::exp(-df_coef * getIDF(trace[pos]));
  }

  return lw * gw;
}
