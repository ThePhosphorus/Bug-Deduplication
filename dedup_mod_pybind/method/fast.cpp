
#include "fast.hpp"
#include <cmath>
#include <vector>

//////////////////////////////////
// FaST Implementation
// DOI : https://doi.org/10.1145/3524842.3527951
//////////////////////////////////

static bool compareFramePos(const FramePosition &lhs, const FramePosition &rhs) {
  if(lhs.id == rhs.id) {
    return lhs.position < rhs.position;
  } else {
    return lhs.id < rhs.id;
  }
}

static std::vector<FramePosition> generate_posframe(const std::vector<size_t> &stack) {
  std::vector<FramePosition> result;
  result.reserve(stack.size());
  for(size_t i = 0; i < stack.size(); i++) {
    result.push_back({.id = stack[i], .position = i});
  }

  std::sort(result.begin(), result.end(), compareFramePos);

  return result;
}

FaST::FaST(const std::vector<double> &df, double alpha, double beta, double gamma)
    : df(df), alpha(alpha), beta(beta), gamma(gamma) {}

double FaST::similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const {
  const auto sortedQuery = generate_posframe(query), sortedCan = generate_posframe(candidate);
  const size_t q_size = query.size(), c_size = candidate.size();

  double sim = 0.;
  size_t i = 0, j = 0;

  while(i < q_size && j < c_size) {
    const FramePosition q_frame = sortedQuery[i];
    const FramePosition c_frame = sortedCan[j];

    if(q_frame.id == c_frame.id) {
      sim += match(q_frame, c_frame);
      i++;
      j++;
    } else if(q_frame.id < c_frame.id) {
      sim -= gap(q_frame);
      i++;
    } else {
      sim -= gap(c_frame);
      j++;
    }
  }

  while(i < q_size) {
    sim -= gap(sortedQuery[i]);
    i++;
  }
  while(j < c_size) {
    sim -= gap(sortedCan[j]);
    j++;
  }

  double den = 0.;

  for(const auto &framePos: sortedQuery) {
    den += weight(framePos);
  }
  for(const auto &framePos: sortedCan) {
    den += weight(framePos);
  }

  return sim / den;
}

inline double FaST::match(const FramePosition &q_frame, const FramePosition &c_frame) const {
  return (weight(q_frame) + weight(c_frame)) * diff(q_frame, c_frame);
}

inline double FaST::gap(const FramePosition &frame) const { return weight(frame); }

inline double FaST::diff(const FramePosition &q_frame, const FramePosition &c_frame) const {
  return std::exp(-gamma * std::abs(static_cast<int>(q_frame.position) - static_cast<int>(c_frame.position)));
}

inline double FaST::weight(const FramePosition &q_frame) const {
  return std::pow((q_frame.position + 1), -alpha) * std::exp(-beta * df[q_frame.id]);
}
