#pragma once

#include "../imports.hpp"
#include "structures.hpp"

std::vector<Stacktrace> generateStackTraces(size_t report_idx,
          const std::vector<std::vector<size_t>> &report_stacks,
                                            const std::vector<bool> &is_stop_word, bool beg_trail_trim);
std::vector<std::vector<Stacktrace>> generateStackTracesByReport(
    const std::vector<std::vector<std::vector<size_t>>> &report_stacks, const std::vector<bool> &is_stop_word,
    bool beg_trail_trim);

void strip(std::vector<size_t> &st, const std::vector<bool> &is_stop_word, bool beg_trail_trim);
