
from typing import Dict, Any, List, Tuple

from hyperopt import hp
import numpy as np

from sklearn import metrics

from trainers.trainer import ClassicTrainer, ClassicBatch, aggregate_map
from data.bug_report_database import BugReportDatabase


from dedup_mod.cmp_func import compare, boolVector
from dedup_mod.util.comparator import FilterStrategy
from dedup_mod.method.method import SimilarityMethod

class FaSTTrainer( ClassicTrainer ) :
    def __init__(self, report_db: BugReportDatabase, is_incremental: bool) -> None:
        super().__init__(report_db, is_incremental)
        
        self.fixed_values: Dict[str, Any] = {
        }


        self.space: Dict[str, Any] = {
            "alpha": hp.uniform("alpha", 0.0, 50.0),
            "beta": hp.uniform("beta", 0.0, 50.0),
            "gamma": hp.uniform("gamma", 0.0, 50.0),

            "aggregate": hp.choice("aggregate", aggregate_map.keys()),
        }

        self.current_value: Dict[str, Any] = {
            "alpha" : 1.0,
            "beta" : 1.0,
            "gamma" : 1.0,
            "aggregate" : "max",
        }

    def run(self, space: Dict[str, Any], batch: ClassicBatch) -> Tuple[List[int], Tuple[List[float], List[int]]]:
        rank_results: List[int] = []
        top_scores: List[float] = []
        top_scores_label: List[int] = []
        aggregate = self.current_value["aggregate"] if isinstance(self.current_value["aggregate"], str) else list(aggregate_map.keys())[self.current_value["aggregate"]]

        for query_stacks, candidates, same_bucket_can in zip(batch.queries, batch.candidates_per_query, batch.same_bucket_can_per_query) :            
            result = np.array(list(compare(SimilarityMethod.FAST, aggregate_map[aggregate],
            [ space["alpha"], space["beta"], space["gamma"]],
            query_stacks, candidates, batch.df_values, FilterStrategy.NONE, 0., boolVector(), False)))
            
            np_same_bucket_can = np.array(same_bucket_can)

            sorted_cans = np.argsort(result)[::-1]
            rank = min(np.argsort(sorted_cans)[np_same_bucket_can])

            rank_results.append(rank)

            top_scores.append(result[sorted_cans[0]])
            top_scores_label.append(1 if rank == 0 else 0)

        # run metrics
        return rank_results, (top_scores, top_scores_label)
        