"""
This class represents a bug report database where we can find all bug reports that are available.
"""
from __future__ import annotations

import codecs
from collections import defaultdict
import logging
import json
import math
from typing import Any, Iterator, List, Mapping, Union, Set, Dict, Optional, Tuple, DefaultDict
import numpy as np

from tqdm import tqdm

import torch

import os
import re

UNK_TOKEN: str = "UNK"

class Frame(object):
    def __init__(self,trim_len:int, function:  Optional[str] = None, depth:  Optional[str] = None, args:  Optional[str] = None, extra:  Optional[List[str]] = None, address:  Optional[str] = None, dylib:  Optional[str] = None, file:  Optional[str] = None, fileline:  Optional[str] = None, is_crash: Optional[str] = None, lib_tag: Optional[str] = None, code:Optional[str] = None) -> None:
        self.trim_len = trim_len
        self.function: Optional[str] = function
        self.depth: int = int(depth) if depth is not None else -1
        self.args: Optional[str] = args
        self.extra: Optional[List[str]] = extra
        self.address: Optional[str] = address
        self.dylib: Optional[str] = Frame.preprocess_dylib(dylib) if dylib is not None else None
        self.file: Optional[str] = file
        self.fileline: Optional[str] = fileline
        self.is_crash: Optional[str] = is_crash
        self.lib_tag: Optional[str] = lib_tag
        self.code:Optional[str] = code

    def trim(self) -> None :
        self.function: Optional[str] = Frame._trim_function(Frame.preprocess_function(self.function), self.trim_len) if self.function is not None else None

    def __str__(self) -> str:
        return self.function or self.dylib or UNK_TOKEN

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o) if type(Frame) == type(__o) else str(self) == __o
    
    @staticmethod
    def _trim_function(func : str, trim: int = 0 ) -> str :
        if trim > 0 :
            return '.'.join(func.split('.')[:-trim])
        else :
            return func
    
    @staticmethod
    def preprocess_function(func_name: str) -> str:

        func_name = re.sub(r'\(.*\)', '', func_name)
        # Strip __GI__ in the begining of function call
        # todo:  It doesn't clean '*__GI_raise') '*__GI_abort')
        func_name = re.sub(r'^_*GI_+', '', func_name)
        # Strip _ in the begining of function call
        func_name = re.sub(r'^_+', '', func_name)
        # todo: remove ^IA__

        return func_name

    @staticmethod
    def preprocess_dylib(dylib: str) -> str:
        # keep only basename
        dylib = os.path.basename(dylib)
        # remove last version
        dylib = re.sub(r'\.\d+$', '', dylib)

        return dylib

class StackTrace(object):
    def __init__(self, remove_recursion: bool, trim_len:int, frames: List[Mapping[str,Any]],  nested: Optional[List[Mapping[str,Any]]] = None, message:  Optional[str] = None, exception:  Optional[str] = None) -> None:
        self.message: Optional[str] = message
        self.exception: Optional[str] = exception
        self.frames_lists: List[List[Frame]] = StackTrace.__get_nested_list_of_frames(StackTrace.__clean_frames([Frame(trim_len, **frame) for frame in frames], remove_recursion),
                                                                                      [StackTrace(remove_recursion, trim_len, **stack) for stack in nested] if nested else [])
        # self.frames_lists = sorted(StackTrace.__split_threads(self.frames_lists), key=lambda x: len(x), reverse=True)
        self.frames_lists = sorted(self.frames_lists, key=lambda x: len(x), reverse=True)
        # Reverse

        for frame_list in self.frames_lists :
            for frame in frame_list :
                frame.trim()

        self.candidates : Optional[List[Tuple[int, int]]] = None
        self.prediction: Optional[int] = None

    @staticmethod
    def __split_threads(frame_lists: List[List[Frame]]) -> List[List[Frame]]:
        new_frame_lists: List[List[Frame]] = []

        for frames in frame_lists:
            if len(frames) == 0 :
                continue
            sub_frames: List[List[Frame]] = []

            past_depth = -1
            begin = 0
            for idx, frame in enumerate(frames):
                if frame.depth < past_depth :
                    sub_frames.append(frames[begin:idx])
                    begin = idx
                past_depth = frame.depth
            sub_frames.append(frames[begin:])

            new_frame_lists.extend(sub_frames)

        return new_frame_lists

    @staticmethod
    def __get_nested_list_of_frames(main_frames: List[Frame], nested: List[StackTrace]) -> List[List[Frame]]:
        stacktraces: List[List[Frame]] = [main_frames]
        for sub_stacktrace in nested:
            stacktraces.extend(sub_stacktrace.frames_lists)

        return stacktraces

    @staticmethod
    def __clean_frames(frames: List[Frame], remove_recursion: bool) -> List[Frame]:
        new_list : List[Frame] = []
        # Remove all frames with no functions
        for frame in frames:
            if (frame.function is not None) or (frame.dylib is not None):
                new_list.append(frame)
        return StackTrace.remove_recursion(new_list) if remove_recursion else new_list

    @staticmethod
    def remove_recursion(input_frames: List[Frame]) -> List[Frame]:
        frames: List[Frame] = input_frames
        continue_loop = True
        while continue_loop:
            continue_loop = False

            seen_frames_set: Set[str] = set()
            i: int = 0
            while i < len(frames):
                frame = frames[i]
                if str(frame) not in seen_frames_set:
                    seen_frames_set.add(str(frame))
                    i += 1
                else:
                    # Get all possible repitions
                    repetitions_by_length: Dict[int, List[List[Frame]]] = {}
                    max_repetition_length: int = 0
                    repetition: List[Frame] = []
                    for j in range(i-1, -1, -1):
                        past_frame = frames[j]
                        repetition.append(past_frame)
                        if str(past_frame) == str(frame):
                            repetition_length = len(repetition)
                            current_repetions = repetitions_by_length.get(
                                repetition_length, [])
                            repetition_copy = repetition.copy()
                            repetition_copy.reverse()
                            current_repetions.append(repetition_copy)
                            repetitions_by_length[repetition_length] = current_repetions
                            max_repetition_length = max(
                                max_repetition_length, len(repetition))

                    # remove possible repetition stemming from current frame
                    is_repetition_found = False
                    found_repetition = []
                    for j in range(i, min(i + max_repetition_length, len(frames))):
                        length = j-i + 1
                        possible_repetitions = repetitions_by_length.get(
                            length, [])
                        # For each possible repition check for a match
                        for possible_repetition in possible_repetitions:
                            if frames[i:i+length] == possible_repetition:
                                is_repetition_found = True
                                found_repetition = possible_repetition
                                break

                        if is_repetition_found:
                            end: int = i+length
                            for idx in range(i+length, len(frames), length):
                                if frames[idx:idx+length] == found_repetition:
                                    end = idx + length
                                else:
                                    break

                            frames: List[Frame]  = frames[:i] + (frames[end:]
                                                   if end < len(frames) else [])
                            continue_loop = True
                            break
                    if not is_repetition_found:
                        i += 1
                    else:
                        break
        return frames

class BugReport(object):
    def __init__(self, remove_recursion: bool, trim_len:int , stacktrace: List[Mapping[str,Any]], bug_id: int,creation_ts: int,dup_id:  Optional[int] = None, **kargs) -> None:
        if not isinstance(stacktrace, list):
            stacktrace = [stacktrace]

        self.stacktraces: List[StackTrace] = [StackTrace(remove_recursion, trim_len, **single_stacktrace) for single_stacktrace in stacktrace]
        # remove empty stack traces
        non_empty_stacks = [ stack for stack in self.stacktraces if len(stack.frames_lists) > 0 and len(stack.frames_lists[0]) > 0]

        hashs : Set[str] = set()
        stacks: List[StackTrace] = []
        for stack in non_empty_stacks:
            stack_hash = str(hash(tuple([str(frame) for frame in stack.frames_lists[0]])))
            if stack_hash not in hashs :
                stacks.append(stack)
                hashs.add(stack_hash)

        self.stacktrace: StackTrace = self.stacktraces[0] if len(stacks) == 0 else stacks[0] # TO not break previous code # TODO: Fix
        self.stacktraces = stacks
        self.bug_id: int = bug_id
        self.dup_id: Optional[int] = dup_id
        self.creation_ts: int = int(creation_ts)
        self.max_length: int = max([len(stack.frames_lists[0]) for stack in self.stacktraces]) if len(self.stacktraces) > 0 else 0

    def __getitem__(self, stack: int) -> List[Frame]:
        return self.stacktrace.frames_lists[stack]

    def __iter__(self) -> Iterator[List[Frame]]:
        return iter(self.stacktrace.frames_lists)

    def __len__(self) -> int:
        return len(self.stacktrace.frames_lists)

    def is_possible_bad_candidate(self, rhs : BugReport) -> bool:
        return rhs.creation_ts < self.creation_ts and (rhs.dup_id != self.dup_id and rhs.bug_id != self.dup_id and  rhs.dup_id != self.bug_id and rhs.bug_id != self.bug_id)

class Vocab(object) :
    def __init__(self, bugList: List[BugReport]) -> None:
        self.vocab: Dict[str, int] = dict()
        self.frames: List[str] = []
        self.df_values: List[float] = []
        self.nb_documents: int = 0

        self.frames.append(UNK_TOKEN)
        self.df_values.append(0)
        self.vocab.setdefault(UNK_TOKEN, 0)

        self.generate_lists(bugList=bugList)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, frame: str) -> int:
        return self.vocab.get(frame, 0)

    def get_frame_str(self, frame_id: int) -> str:
        return self.frames[frame_id]

    def __contains__(self, frame: str) -> bool :
        return frame in self.vocab

    def get_idf_value(self, frame_id: int) -> float:
        return math.log(self.nb_documents / self.df_values[frame_id]) if self.df_values[frame_id] > 0 else 0.

    def turn_to_id_tensor(self, frame_list: List[Frame], length : Optional[int] = None) -> torch.Tensor :
        length = len(frame_list) if not length else length
        result: torch.Tensor = torch.zeros((length), dtype=torch.long)

        for idx, frame in enumerate(frame_list) :
             result[idx] = self[str(frame)]

        return result

    def get_idf_tensor(self, id_tensor: torch.Tensor) -> torch.Tensor :
        flattened_tensor = id_tensor.flatten()
        idf_tensor: torch.Tensor = torch.empty_like(flattened_tensor, dtype=torch.float)
    
        for i in range(flattened_tensor.shape[0]) :
            idf_tensor[i] = self.get_idf_value(flattened_tensor[i].detach().item())
        return idf_tensor.reshape(id_tensor.shape)

    def get_lerch_score(self, query_set: Set[str], candidate: List[Frame]) -> float:
        return sum([1 for frame in candidate if str(frame) in query_set])
        # return sum([self.get_idf_value(self[str(frame)]) ** 2 for frame in candidate if str(frame) in query_set])

    def generate_lists(self, bugList: List[BugReport]) -> None:
        for bug in bugList:
            for stack in bug.stacktraces:
                for frame_list in stack.frames_lists:
                    frame_set: Set[int] = set()
                    self.nb_documents += 1
                    for frame in frame_list:
                        assert str(frame) in self
                        frame_set.add(self[str(frame)])
                    for frame in frame_set : 
                        self.df_values[frame] += 1

    def register_ids(self, bugList: List[BugReport]) -> None:
        for bug in bugList:
            for stack in bug.stacktraces:
                for frame_list in stack.frames_lists:
                    for frame in frame_list:
                        if str(frame) not in self:
                            id = len(self.frames)
                            self.frames.append(str(frame))
                            self.df_values.append(0)
                            self.vocab.setdefault(str(frame), id)

    def turn_report_to_ids(self, bug_report: BugReport) -> List[List[int]] :
        return [ [ self[str(frame)] for frame in stack.frames_lists[0] ] for stack in bug_report.stacktraces ]

class BugReportDatabase(object):
    '''
    Load bug report data (categorical information, summary and description) from json file.
    '''

    def __init__(self, iterator: Iterator[Dict[str,Any]], remove_recursion: bool, trim_len:int = 0):
        self.report_by_id: Dict[int, BugReport] = dict()
        self.report_list: List[BugReport] = []
        self.logger: logging.Logger = logging.getLogger()
        self.master_set: Dict[int, List[int]] = dict()
        self.master_set_newest: Dict[int, int] = dict()
        self.max_window: int = 99999999999
        self.window_mean: float = 0.
        self.window_std: float = 0.
        self.window_median: float = 0.
        self.window_within_2y: float = 0.
        self.frame_stack_set : DefaultDict[str, List[Tuple[int,int]]] = defaultdict(list)

        self.current_timestamp: Tuple[int, int] = (0,0) # timestamp, self.report_list pos

        for report_dict in iterator:
            if report_dict is None:
                continue
            report = BugReport(remove_recursion,trim_len, **report_dict)

            if len(report.stacktrace.frames_lists) != 0 and len(report.stacktrace.frames_lists[0]) != 0:
                report_id: int = report.bug_id

                self.report_by_id[report_id] = report
                self.report_list.append(report)

        # sort report_list
        self.report_list = sorted(self.report_list, key= lambda x : x.creation_ts, reverse= False)

        self.vocab = Vocab([])
        self._gen_master_sets()
        self._gen_frame_stack_map()

    @staticmethod
    def from_json(fileToLoad: str, remove_recursion: bool = True, trim_len:int = 0):
        f = codecs.open(fileToLoad, 'r')
        return BugReportDatabase(json.load(f), remove_recursion, trim_len)

    def load_until(self, timestamp: int) -> Tuple[int, int]: # Returns dup_count and end idx
        assert self.current_timestamp[0] < timestamp

        past_pos  = self.current_timestamp[1]
        dup_count = 0
        for pos in range(past_pos, len(self.report_list) + 1) :
            if pos == len(self.report_list) :
                self.current_timestamp = (self.report_list[-1].creation_ts, len(self.report_list))
                break
            if self.report_list[pos].creation_ts > timestamp :
                self.current_timestamp = (self.report_list[pos].creation_ts, pos)
                break

            if self.report_list[pos].dup_id is not None :
                dup_count += 1

        #self.vocab.generate_lists(self.report_list[past_pos: self.current_timestamp[1]])
        return dup_count, self.current_timestamp[1]

    def get_report(self, report_id: int) -> BugReport:
        return self.report_by_id[report_id]

    def __getitem__(self, report_id: int) -> BugReport:
        return self.report_by_id[report_id]

    def __iter__(self) -> Iterator[BugReport]:
        return iter(self.report_list)

    def __len__(self) -> int:
        return len(self.report_list)

    def ref(self, bug: Union[int, BugReport]) -> int:
        return bug.bug_id if isinstance(bug, BugReport) else bug

    def deref(self, bug: Union[int, BugReport]) -> BugReport:
        return bug if isinstance(bug, BugReport) else self[bug]

    def ref_list(self, bugs: Union[List[int], List[BugReport]]) -> List[int]:
        return [self.ref(bug) for bug in bugs]

    def deref_list(self, bugs: Union[List[int], List[BugReport]]) -> List[BugReport]:
        return [self.deref(bug) for bug in bugs]

    def __contains__(self, bug: Union[int, BugReport]) -> bool:
        return self.ref(bug) in self.report_by_id

    def get_master_by_report(self, bugs: Union[List[int], Optional[List[BugReport]]] = None) -> Dict[int, int]:
        masterIdByBugId: Dict[int, int] = dict()
        bugs = self.report_list if bugs is None else self.deref_list(bugs)

        for bug in bugs:
            bugId: int = bug.bug_id
            dupId: Optional[int] = bug.dup_id

            masterIdByBugId[bugId] = bugId if dupId is None else dupId

        return masterIdByBugId

    def get_master_set_by_id(self, bugs: Union[List[int], Optional[List[BugReport]]] = None) -> Dict[int, Set[int]]:
        masterSetById: Dict[int, Set[int]] = dict()
        bugs = self.report_list if bugs is None else self.deref_list(bugs)

        for bug in bugs:
            bugId: int = bug.bug_id
            dupId: Optional[int] = bug.dup_id

            if dupId is not None:
                masterSet = masterSetById.setdefault(dupId, set())
                masterSet.add(bugId)

        # Insert id of the master bugs in your master sets
        for masterId, masterSet in masterSetById.items():
            if masterId in self:
                masterSet.add(masterId)

        return masterSetById

    def _gen_master_sets(self) -> None:
        self.master_set = dict()
        alias_table: Dict[int, int] = dict()
        for report in self.report_list :
            root = report
            root_stack: List[BugReport] = []
            while root.dup_id is not None:
                if root.dup_id == root.bug_id :
                    root.dup_id = None
                    break
                if root in root_stack :
                    # Loop found assign root to oldest report
                    root_stack = sorted(root_stack, key= lambda x : x.creation_ts, reverse=True)
                    root = root_stack[-1]
                    root.dup_id = None
                    break

                root_stack.append(root)
                if root.dup_id not in self :
                    if root.dup_id in alias_table:
                        root.dup_id = alias_table[root.dup_id]
                        root = self[root.dup_id]
                    else :
                        alias_table.setdefault(root.dup_id, root.bug_id)
                        root.dup_id = None
                        break
                else :
                    root = self[root.dup_id]
            # Set root to all path for quicker next steps
            for subroot in root_stack :
                subroot.dup_id = root.bug_id if subroot.bug_id != root.bug_id else None

            if root.bug_id not in self.master_set:
                self.master_set.setdefault(root.bug_id, [root.bug_id])
            
            if root.bug_id is not report.bug_id :
                self.master_set[root.bug_id].append(report.bug_id)
        self._fix_master_sets()
        self._compute_master_set_metrics()

    def _fix_master_sets(self) -> None :
        # Got through each master set and fix it
        old_master_set_ids : List[int] = list(self.master_set.keys())
        for master_id in old_master_set_ids:
            reports = self.master_set[master_id]
            # Find oldest report
            oldest_report_id: Optional[int] = None
            for report_id in reports :
                if oldest_report_id is None or self[report_id].creation_ts < self[oldest_report_id].creation_ts :
                    oldest_report_id = report_id

            if oldest_report_id is None:
                self.master_set.pop(master_id)
                continue

            if oldest_report_id is not master_id:                    
                # Change all dup_id to newone
                for report_id in reports :
                    self[report_id].dup_id = oldest_report_id if report_id is not oldest_report_id else None

                # Remove master_set and add new one
                self.master_set.pop(master_id)
                self.master_set[oldest_report_id] = reports

    def _compute_master_set_metrics(self) -> None:
        windows: List[int] = []
        for master_id in self.master_set:
            reports = self.master_set[master_id]
            # Find newest Report
            timestamps: List[int] = [self[report_id].creation_ts for report_id in reports]
            timestamps = sorted(timestamps, reverse= True)
            self.master_set_newest[master_id] = timestamps[0]
            if len(timestamps) > 1 :
                diffs = [ timestamps[i] - timestamps[i + 1] for i in range(len(timestamps) - 1) if (timestamps[i] - timestamps[i + 1]) != 0 ]
                if len(diffs) != 0 :
                    windows.append(max(diffs))

        self.max_window = max(windows)
        self.window_mean = np.mean(windows)
        self.window_median = np.median(windows)
        self.window_std = np.std(windows)
        self.window_within_2y = sum([ int(window <= (2 * 365.2425 * 24 * 3600)) for window in windows ])/ len(windows)

        print(f"Window is : Max {self.max_window}, Mean {self.window_mean}, median {self.window_median} and std {self.window_std}")

    def get_duplicates(self) -> List[int] :
        duplicates: List[int] = []

        for report in self:
            if report.dup_id != None and report.dup_id in self:
                duplicates.append(report.bug_id)
        return duplicates

    def get_non_duplicates(self) -> List[int] :
        non_duplicates: List[int] = []

        for report in self:
            if report.dup_id == None:
                non_duplicates.append(report.bug_id)
        return non_duplicates

    def _gen_frame_stack_map(self) -> None:
        print("Generating frame map.")        
        for report in tqdm(self.report_list):
            for stack_id, stack in enumerate(report.stacktraces):
                for frame in stack.frames_lists[0] :
                    self.frame_stack_set[str(frame)].append((report.bug_id ,stack_id))

    def gen_candidates(self, query_reports: List[int]) -> None:
        print(f"Generating candiates for {len(query_reports)} queries")

        for query_id in tqdm(query_reports) :
            query_report = self[query_id]

            for stack in query_report.stacktraces:
                # Compute lerch score
                frames = stack.frames_lists[0]
                frame_set : Set[str] = set([str(frame) for frame in frames])

                possible_candidates : List[Tuple[int, int]] = []
                for frame in frame_set :
                    possible_candidates.extend(self.frame_stack_set[frame])


                results : List[Tuple[Tuple[int, int], float]] = []

                for can_report_id, can_stack in set(possible_candidates) :
                    can_report = self[can_report_id]

                    if not query_report.is_possible_bad_candidate(can_report):
                        continue
                    
                    can_frames = can_report.stacktraces[can_stack].frames_lists[0]
                    results.append( ( (can_report_id, can_stack), self.vocab.get_lerch_score(frame_set, can_frames)))

                results = sorted(results, key=lambda x : x[1], reverse= True)[:50]

                stack.candidates = [ can for can, _ in results ]
        print("Done generating candidates")

    def get_all_candidates(self, query_id: int) -> List[BugReport] :
        candiadates: List[BugReport] = []

        query: BugReport = self[query_id]

        for report in self.report_list :
            if report.creation_ts > query.creation_ts:
                break
            if report.bug_id != query.bug_id :
                candiadates.append(report)

        return candiadates
