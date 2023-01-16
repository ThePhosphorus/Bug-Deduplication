#pragma once

struct Stacktrace {
  size_t report_idx;
  std::vector<size_t> stack;
};

struct TermFreq {
  int term;
  double freq;
};