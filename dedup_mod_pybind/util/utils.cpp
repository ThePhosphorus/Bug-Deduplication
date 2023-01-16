#include "utils.hpp"

#include <algorithm>

std::vector<std::vector<Stacktrace>> generateStackTracesByReport(
    const std::vector<std::vector<std::vector<size_t>>> &report_stacks, const std::vector<bool> &is_stop_word,
    bool beg_trail_trim) {
  std::vector<std::vector<Stacktrace>> stacks_by_reportid;
  for(size_t report_idx = 0; report_idx < report_stacks.size(); report_idx++) {
    stacks_by_reportid.push_back(
        generateStackTraces(report_idx, report_stacks.at(report_idx), is_stop_word, beg_trail_trim));
  }
  return stacks_by_reportid;
}

void strip(std::vector<size_t> &st, const std::vector<bool> &is_stop_word, bool beg_trail_trim) {
  auto it = st.begin();

  // Trim forward
  while(it != st.end()) {
    const auto function_id = *it;

    if(is_stop_word[function_id]) {
      it = st.erase(it);
    } else {
      // We found a non stop word
      if(beg_trail_trim)
        break;
      else
        it++;
    }
  }

  if(it != st.end()) {
    // Trim backward
    auto rit = st.end();
    rit      = std::prev(rit);

    while(rit != it) {
      const auto function_id = *rit;
      if(is_stop_word[function_id]) {
        rit = std::prev(st.erase(rit));

      } else
        break;
    }
  }
}

std::vector<Stacktrace> generateStackTraces(size_t report_idx, const std::vector<std::vector<size_t>> &report_stacks,
                                            const std::vector<bool> &is_stop_word, bool beg_trail_trim) {
  std::vector<Stacktrace> stacks;
  stacks.reserve(report_stacks.size());
  for(size_t stack_list_idx = 0; stack_list_idx < report_stacks.size(); stack_list_idx++) {
    Stacktrace s;
    std::vector<size_t> st = report_stacks.at(stack_list_idx);

    if(is_stop_word.size() != 0) {
      // Filter functions
      strip(st, is_stop_word, beg_trail_trim);
    }
    if(st.size() == 0) {
      // Filtered all functions
      continue;
    }
    s.stack      = st;
    s.report_idx = report_idx;

    stacks.push_back(s);
  }

  return stacks;
}
