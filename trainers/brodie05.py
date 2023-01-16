
from typing import Dict, Any, List, Tuple

from hyperopt import hp
import numpy as np

from trainers.trainer import ClassicTrainer, ClassicBatch, aggregate_map
from data.bug_report_database import BugReport, BugReportDatabase


from dedup_mod.cmp_func import compare, boolVector
from dedup_mod.util.comparator import FilterStrategy
from dedup_mod.method.method import SimilarityMethod


class Brodie05Trainer(ClassicTrainer):
    def __init__(self, report_db: BugReportDatabase, is_incremental: bool) -> None:
        super().__init__(report_db, is_incremental)

        self.fixed_values: Dict[str, Any] = {"J": 2}

        self.space: Dict[str, Any] = {
            "gap_penalty": hp.uniform("gap_penalty", 0, 10),
            "aggregate": hp.choice("aggregate", ('max', 'avg_query', 'avg_cand', 'avg_short', 'avg_long', 'avg_query_cand')),
        }
        self.current_value: Dict[str, Any] = {
            "gap_penalty": 2.5,
            "aggregate": 0,
        }

    def run(self, space: Dict[str, Any], batch: ClassicBatch) -> Tuple[List[int], Tuple[List[float], List[int]]]:
        rank_results: List[int] = []
        top_scores: List[float] = []
        top_scores_label: List[int] = []

        aggregate = self.current_value["aggregate"] if isinstance(
            self.current_value["aggregate"], str) else list(aggregate_map.keys())[self.current_value["aggregate"]]

        for query_stacks, candidates, same_bucket_can in zip(batch.queries, batch.candidates_per_query, batch.same_bucket_can_per_query):
            result = np.array(list(compare(SimilarityMethod.BRODIE_05, aggregate_map[aggregate], [
                              space["gap_penalty"]], query_stacks, candidates, batch.df_values, FilterStrategy.NONE, 0., boolVector(), False)))
            if len(same_bucket_can) == 0 :
                rank_results.append(99999)
                continue

            np_same_bucket_can = np.array(same_bucket_can)

            sorted_cans = np.argsort(result)[::-1]
            rank = min(np.argsort(sorted_cans)[np_same_bucket_can])

            rank_results.append(rank)

            top_scores.append(result[sorted_cans[0]])
            top_scores_label.append(1 if rank == 0 else 0)

        # run metrics
        return rank_results, (top_scores, top_scores_label)

    def filter_candidates(self, query: BugReport, candidates: List[BugReport]) -> List[BugReport]:
        J: int = self.fixed_values["J"]
        keys: List[str] = [".".join([str(frame) for frame in stack.frames_lists[0][:J]])
                           for stack in query.stacktraces if len(stack.frames_lists[0]) > J]
        return list(filter(lambda x: any([(any([".".join([str(frame) for frame in stack.frames_lists[0][:J]]) == key for key in keys]))
               for stack in x.stacktraces if len(stack.frames_lists[0]) > J]), candidates))
