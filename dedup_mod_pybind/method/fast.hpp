#pragma once

#include "../imports.hpp"
#include "method.hpp"

//////////////////////////////////
// FaST Implementation
// DOI : https://doi.org/10.1145/3524842.3527951
//////////////////////////////////

struct FramePosition {
  size_t id;
  size_t position;
};

class FaST: public Method {
public:
  const std::vector<double> &df;
  double alpha;
  double beta;
  double gamma;
  FaST(const std::vector<double> &df, double alpha, double beta, double gamma);
  virtual double similarity(const std::vector<size_t> &query, const std::vector<size_t> &candidate) const override;

private:
  inline double match(const FramePosition &q_frame, const FramePosition &c_frame) const;
  inline double gap(const FramePosition &frame) const;
  inline double diff(const FramePosition &q_frame, const FramePosition &c_frame) const;
  inline double weight(const FramePosition &q_frame) const;
};
