#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters are classes that change the token, like transform to lower case the letters 
"""
import hashlib
import json
import string
from math import log

from typing import Callable, Dict, List, Optional, Set, Tuple, DefaultDict, Union

from data.bug_report_database import BugReportDatabase, Frame, StackTrace

#preprocess_stacktrace, std_function_preprocess

PreprocessFunc = Callable[[Dict[str, int],
                           List[Frame], str, int, int], List[int]]


def std_function_preprocess(vocab: Dict[str, int], frames: List[Frame], ukn_tkn_main: str, ukn_id: int, max_depth: int) -> List[int]:
    function_ids: List[int] = []

    for depth, frame in enumerate(frames):
        func_name = str(frame)

        if depth >= max_depth:
            break

        if func_name is None or func_name == ukn_tkn_main:
            token_id = ukn_id
        else:
            token_id = vocab.setdefault(func_name, len(vocab))

        function_ids.append(token_id)

    return function_ids


def rm_duplicate_stacks(frames: List[List[int]], st_set: Set[str] = None) -> List[List[int]]:
    st_set = st_set if st_set else set()

    result: List[List[int]] = []

    for st in frames:
        st_hash = hashlib.sha512(json.dumps(st).encode("utf-8")).hexdigest()

        if not st_hash in st_set:
            st_set.add(st_hash)
            result.append(st)

    return result


def preprocess_stacktrace(all_report_ids: List[int], report_db: BugReportDatabase, max_depth: int, ukn_tkn_main: str = "??", vocab: Dict[str, int] = None,
                          stacktraces_by_id: Dict[int, List[List[int]]] = None, unique_ukn_report: bool = True,
                          preprocess_func: PreprocessFunc = std_function_preprocess,
                          ukn_set: Set[int] = None, rm_dup_stacks: bool = False) -> Tuple[Dict[int, List[List[int]]], Dict[str, int]]:
    vocab = vocab if vocab else dict()
    stacktraces_by_id = stacktraces_by_id if stacktraces_by_id else dict()

    for report_id in all_report_ids:
        stacktrace: StackTrace = report_db.get_report(report_id).stacktrace

        ukn_tkn = ukn_tkn_main
        if unique_ukn_report:
            # Create token for unknown function name. Give a unique id to the unknown values in stacktrace
            # Following ABRT, we only compare the function names and consider two unknown function(??) as different
            ukn_tkn = '{}{}'.format(ukn_tkn_main, report_id)

            if ukn_tkn in vocab:
                raise Exception(
                    "Token for unknown function name in the report {} already exists.".format(report_id))

        ukn_id = vocab.setdefault(ukn_tkn, len(vocab))

        if ukn_set is not None:
            ukn_set.add(ukn_id)

        frames: List[List[Frame]] = stacktrace.frames_lists
        frames_ids: List[List[int]] = list()

        for frame in frames:
            frames_ids.append(preprocess_func(
                vocab, frame, ukn_tkn, ukn_id, max_depth))

        if len(frames_ids) == 0:
            raise Exception("{} contains 0 stacktraces".format(report_id))
        else:
            sum_lengths: int = 0

            for fr in frames_ids:
                sum_lengths += len(fr)

            if sum_lengths == 0:
                raise Exception(
                    "{} contains 0 frames".format(report_id))

        if rm_dup_stacks:
            st_set: Optional[Set[str]] = set()

            frames_ids = rm_duplicate_stacks(frames_ids, st_set)

        stacktraces_by_id[report_id] = frames_ids

    return stacktraces_by_id, vocab
