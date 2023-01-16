
from typing import Dict, Any, List, Tuple

from hyperopt import hp
import numpy as np

from trainers.trainer import ClassicTrainer, ClassicBatch, aggregate_map
from data.bug_report_database import BugReportDatabase


from dedup_mod.cmp_func import compare, boolVector
from dedup_mod.util.comparator import FilterStrategy
from dedup_mod.method.method import SimilarityMethod

class TracesimTrainer( ClassicTrainer ) :
    def __init__(self, report_db: BugReportDatabase, is_incremental: bool) -> None:
        super().__init__(report_db, is_incremental)
        
        self.fixed_values: Dict[str, Any] = {
            "match_cost": 1.0,
            "gap_penalty": 1.0,
            "mismatch_penalty": 1.0,
            "diff_coef": .5,
            "sigmoid": True,
            "gamma": 1.0,
            "idf": True,
            "sum": True,
            "const_match": False,
            "reciprocal_func": True,
            "no_norm": False,
            # "filter_func": "threshold_trim"
        }


        self.space: Dict[str, Any] = {
            "df_coef": hp.uniform("df_coef", 0.0, 3.0),
            "pos_coef": hp.uniform("pos_coef", 0.0, 50.0),
            "diff_coef": hp.uniform("diff_coef", 0.0, 2.0),

            # "keep_ukn": hp.choice("keep_ukn", (False, True)),
            # "static_df_ukn": hp.choice("static_df_ukn", (False, True)),
            "aggregate": hp.choice("aggregate", aggregate_map.keys()),
            # "rm_dup_stacks": hp.choice("rm_dup_stacks", (False, True)),
            # "freq_by_stacks": hp.choice("freq_by_stacks", (False, True)),
            # "filter_func_k": hp.uniform("filter_func_k", 0.0, 130.0),
        }

        self.current_value: Dict[str, Any] = {
            "df_coef" : 0.5,
            "pos_coef" : 1.0,
            "diff_coef" : 1.0,
            "aggregate" : "max",
        }

    def run(self, space: Dict[str, Any], batch: ClassicBatch) -> Tuple[List[int], Tuple[List[float], List[int]]]:
        rank_results: List[int] = []
        top_scores: List[float] = []
        top_scores_label: List[int] = []
        aggregate = self.current_value["aggregate"] if isinstance(self.current_value["aggregate"], str) else list(aggregate_map.keys())[self.current_value["aggregate"]]

        for query_stacks, candidates, same_bucket_can in zip(batch.queries, batch.candidates_per_query, batch.same_bucket_can_per_query) :            
            result = np.array(list(compare(SimilarityMethod.TRACE_SIM, aggregate_map[aggregate],
            [   space["df_coef"], space["pos_coef"], space["diff_coef"],
                self.fixed_values["match_cost"], self.fixed_values["gap_penalty"], self.fixed_values["mismatch_penalty"],
                self.fixed_values["sigmoid"], self.fixed_values["gamma"], self.fixed_values["sum"], self.fixed_values["idf"],
                self.fixed_values["const_match"], self.fixed_values["reciprocal_func"], self.fixed_values["no_norm"]
            ],
            query_stacks, candidates, batch.df_values, FilterStrategy.NONE, 0., boolVector(), False)))
            
            np_same_bucket_can = np.array(same_bucket_can)

            sorted_cans = np.argsort(result)[::-1]
            rank = min(np.argsort(sorted_cans)[np_same_bucket_can])

            rank_results.append(rank)

            top_scores.append(result[sorted_cans[0]])
            top_scores_label.append(1 if rank == 0 else 0)
        # run metrics
        return rank_results, (top_scores, top_scores_label)