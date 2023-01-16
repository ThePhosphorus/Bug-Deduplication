from __future__ import annotations

from typing import Any, Optional, Dict, List, Tuple, Generator
import time
import random
from itertools import chain

from hyperopt import fmin, tpe
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics


import wandb

from dedup_mod.cmp_func import doubleVector
from data.bug_report_database import BugReport, Vocab, BugReportDatabase, Frame, StackTrace
from data.report_dataset import ReportDataset
from data.embed_map import get_packed_seq_from_frames, PackedSequence

from dedup_mod.util.comparator import AggStrategy

aggregate_map: Dict[str, AggStrategy] = {
    'max': AggStrategy.MAX,
    'avg_query': AggStrategy.AVG_QUERY,
    'avg_cand': AggStrategy.AVG_CAND,
    'avg_short': AggStrategy.AVG_SHORT,
    'avg_long': AggStrategy.AVG_LONG,
    'avg_query_cand': AggStrategy.AVG_QUERY_CAND
}


class Trainer:
    def __init__(self, report_db: BugReportDatabase) -> None:
        self.report_db: BugReportDatabase = report_db

    def fit(self, dataset: ReportDataset) -> None:
        raise NotImplementedError()

    def compute(self, dataset: ReportDataset) -> None:
        raise NotImplementedError()


class ClassicBatch():
    def __init__(self, report_db: BugReportDatabase, vocab: Vocab, dup_ids: List[int], trainer: ClassicTrainer) -> None:
        self.df_values: doubleVector = trainer.get_df_vector(vocab=vocab)
        self.queries: List[List[List[int]]] = []
        self.candidates_per_query: List[List[List[List[int]]]] = []
        self.same_bucket_can_per_query: List[List[int]] = []

        for dup_id in dup_ids:
            query = report_db[dup_id]
            self.queries.append(vocab.turn_report_to_ids(query))
            candidates = trainer.filter_candidates(
                query, report_db.get_all_candidates(dup_id))
            self.candidates_per_query.append(
                [vocab.turn_report_to_ids(report) for report in candidates])
            self.same_bucket_can_per_query.append([i for i, report in enumerate(
                candidates) if report.dup_id == query.dup_id or report.bug_id == query.dup_id])


class ClassicTrainer(Trainer):
    def __init__(self, report_db: BugReportDatabase, is_incremental: bool) -> None:
        super().__init__(report_db)
        self.space: Dict[str, Any] = {}
        self.fixed_values: Dict[str, Any] = {}
        self.is_incremental: bool = is_incremental

        self.current_value: Dict[str, Any] = {}  # set as default
        self.already_fitted: bool = False

    def objective_func(self, space: Dict[str, Any], batch: ClassicBatch) -> float:
        ranks, (top_scores, top_scores_label) = self.run(space, batch)

        np_ranks = np.array(ranks)
        # Compute MAP, RR@1, AUC

        mrr = np.sum(1/(np_ranks + 1)) / len(ranks)
        bin_count = np.bincount(np_ranks, minlength=10)
        rr1 = bin_count[0] / len(ranks)
        auc_roc = metrics.roc_auc_score(top_scores_label, top_scores) if any(top_scores_label) else 0.5

        return 1. - (0.4 * mrr + 0.3 * rr1 + 0.3 * auc_roc)

    def generate_batch(self, dataset: ReportDataset, get_validation: bool = False) -> ClassicBatch:
        return ClassicBatch(self.report_db, self.report_db.vocab, dataset.duplicateIds if not get_validation else dataset.duplicateIds[(len(dataset.duplicateIds)//2):], self)

    def fit(self, dataset: ReportDataset) -> None:
        # copy vocab
        bugs = [self.report_db[bug_id] for bug_id in dataset.bugIds]
        self.report_db.vocab.register_ids(bugs)
        self.report_db.vocab.generate_lists(bugs)

        batch = self.generate_batch(dataset)

        self.current_value = fmin(fn=lambda x: self.objective_func(x, batch), space=self.space, algo=tpe.suggest,
                                  max_evals=50, points_to_evaluate=[self.current_value] if self.already_fitted else None)
        print(self.current_value)
        if self.is_incremental:
            self.already_fitted = True

    def compute(self, dataset: ReportDataset) -> None:
        bugs = [self.report_db[bug_id] for bug_id in dataset.bugIds]

        self.report_db.vocab.register_ids(bugs)

        batch = self.generate_batch(dataset, get_validation=True)

        start_time = time.time()
        ranks, (top_scores, top_scores_label) = self.run(self.current_value, batch)
        end_time = time.time()
        np_ranks = np.array(ranks, dtype=np.longlong)

        mrr = np.sum(1/(np_ranks + 1)) / len(ranks)
        bin_count = np.bincount(np_ranks, minlength=10)
        rr1: float = sum(bin_count[:1]) / len(ranks)
        rr5: float = sum(bin_count[:5]) / len(ranks)
        rr10: float = sum(bin_count[:10]) / len(ranks)

        auc_roc = metrics.roc_auc_score(top_scores_label, top_scores) if any(top_scores_label) else 0.5

        print({"rr@1": rr1, "rr@5": rr5, "rr@10": rr10, "mrr": mrr, "auc_roc": auc_roc,
              "Throughput": len(batch.queries)/(end_time - start_time)})
        wandb.log({"rr@1": rr1, "rr@5": rr5, "rr@10": rr10, "mrr": mrr,
                  "auc_roc": auc_roc, "Throughput": len(batch.queries)/(end_time - start_time)})

    def run(self, space: Dict[str, Any], batch: ClassicBatch) -> Tuple[List[int], Tuple[List[float], List[int]]]:
        raise NotImplementedError()

    def get_df_vector(self, vocab: Vocab) -> doubleVector:
        df_values: doubleVector = doubleVector()
        df_values.reserve(len(vocab.df_values))
        for df in vocab.df_values:
            df_values.push_back(df/(vocab.nb_documents or 1))
        return df_values

    def filter_candidates(self, query: BugReport, candidates: List[BugReport]) -> List[BugReport]:
        return candidates


class MLTrainer(Trainer):
    def __init__(self, report_db: BugReportDatabase, model: nn.Module, batch_size: int, embed_dim: int, nb_reports: int, nb_epochs: int) -> None:
        super().__init__(report_db)
        self.batch_size: int = batch_size
        self.embed_dim: int = embed_dim
        self.nb_reports: int = nb_reports
        self.model: nn.Module = model
        self.nb_epochs : int = nb_epochs

        self._ids_to_register: Optional[List[BugReport]] = None

        self.embed_layer: nn.Embedding = nn.Embedding(
            len(self.report_db.vocab), self.embed_dim)

        self.available_device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.available_device)
        self.embed_layer = self.embed_layer.to(self.available_device)

    def _generate_new_embed(self) -> None:
        current_weights = self.embed_layer.weight
        # create new embedding layer
        self.embed_layer: nn.Embedding = nn.Embedding(
            len(self.report_db.vocab), self.embed_dim)

        
        with torch.no_grad():
            self.embed_layer.weight[:current_weights.shape[0], :current_weights.shape[1]] = current_weights.detach()

    def _generate_train_batch(self, dataset: ReportDataset) -> Generator[Tuple[PackedSequence, Tuple[List[PackedSequence], List[PackedSequence]]], None, None]:
        # get all dups
        duplicates: List[Tuple[int, BugReport]] = []
        for idx, report in enumerate(self.report_db.report_list):
            if report.creation_ts > dataset.end_ts:
                break
            if report.dup_id is not None:
                duplicates.append((idx, report))

        random.shuffle(duplicates)
        master_sets = self.report_db.master_set

        for i in range(0, len(duplicates), self.batch_size):
            dup_batch = duplicates[i:i+self.batch_size]

            queries: List[List[Frame]] = []
            good_reports_frame_lists: List[List[List[Frame]]] = []
            bad_reports_frame_lists: List[List[List[Frame]]] = []
            for pos, duplicate in dup_batch:
                # Get master
                assert duplicate.dup_id is not None
                master_set = master_sets[duplicate.dup_id]

                query: StackTrace = random.choice(duplicate.stacktraces)

                queries.append(query.frames_lists[0])

                possible_good_reports: List[List[Frame]] = [stack.frames_lists[0] for candidate in master_set if candidate !=
                                                            duplicate.bug_id and self.report_db[candidate].creation_ts <= duplicate.creation_ts for stack in self.report_db[candidate].stacktraces]

                good_reports: List[List[Frame]] = random.choices(
                    possible_good_reports, k=self.nb_reports)

                bad_reports: List[List[Frame]] = []
                while len(bad_reports) != self.nb_reports:
                    random_stack = random.choice(
                        self.report_db.report_list[:pos])

                    if (random_stack.dup_id == duplicate.dup_id or random_stack.bug_id == duplicate.dup_id or random_stack.dup_id == duplicate.bug_id or random_stack.bug_id == duplicate.bug_id):
                        continue
                    bad_reports.append(random.choice(
                        random_stack.stacktraces).frames_lists[0])
                good_reports_frame_lists.append(good_reports)
                bad_reports_frame_lists.append(bad_reports)

            yield (get_packed_seq_from_frames(queries, self.report_db),  # Issues
                   ([get_packed_seq_from_frames([frame_lists[item] for frame_lists in good_reports_frame_lists], self.report_db) for item in range(self.nb_reports)],  # good reports
                    [get_packed_seq_from_frames([frame_lists[item] for frame_lists in bad_reports_frame_lists], self.report_db) for item in range(self.nb_reports)]))  # Bad reports

    def fit(self, dataset: ReportDataset) -> None:
        # if self._ids_to_register is not None:
        #     self.report_db.vocab.register_ids(self._ids_to_register)
        self.report_db.vocab.register_ids([self.report_db[bug_id]
                                 for bug_id in dataset.bugIds])


        # Extand embedding to fit new vocab
        self._generate_new_embed()

        # init optimizer
        optimizer: torch.optim.Optimizer = torch.optim.Adam(params= chain(self.model.parameters(), self.embed_layer.parameters() ),lr=1e-4)

        self.model = self.model.to(self.available_device)
        self.embed_layer = self.embed_layer.to(self.available_device)

        # train a number of epochs
        for e in range(self.nb_epochs) :
            # Generate training batch (using all queries up to and including the dataset)
            print({"epoch" : e})
            for batch in enumerate(self._generate_train_batch(dataset)):
                # init
                self._train_step(batch, optimizer)
        
        # self._ids_to_register = [self.report_db[bug_id]
        #                          for bug_id in dataset.bugIds]

    def _embed_packed_seq(self, input: PackedSequence) -> PackedSequence :
        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first=True)
        x = self.embed_layer(x)
        emb_f = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        return emb_f

    def _train_step(self, batch : Tuple[int, Tuple[PackedSequence, Tuple[List[PackedSequence], List[PackedSequence]]]], optimizer: torch.optim.Optimizer) -> None :
        raise NotImplementedError()

    def _prepare_candidates(self, dataset: ReportDataset) -> None:
        raise NotImplementedError()
    
    def _run_metrics(self, dataset:ReportDataset) -> Tuple[List[int], Tuple[List[float], List[int]]]:
        raise NotImplementedError()

    def compute(self, dataset: ReportDataset) -> None:
        # Prepare embed database
        self._prepare_candidates(dataset)

        self.model.cpu()
        self.embed_layer.cpu()

        # run metrics
        start_time = time.time()
        ranks, (top_scores, top_scores_label) = self._run_metrics(dataset)
        end_time = time.time()
        np_ranks = np.array(ranks, dtype=np.longlong)

        mrr = np.sum(1/(np_ranks + 1)) / len(ranks)
        bin_count = np.bincount(np_ranks, minlength=10)
        rr1: float = sum(bin_count[:1]) / len(ranks)
        rr5: float = sum(bin_count[:5]) / len(ranks)
        rr10: float = sum(bin_count[:10]) / len(ranks)
        auc_roc = metrics.roc_auc_score(top_scores_label, top_scores) if any(top_scores_label) else 0.5

        print({"rr@1": rr1, "rr@5": rr5, "rr@10": rr10, "mrr": mrr, "auc_roc": auc_roc,
              "Throughput": len(dataset.duplicateIds)/(end_time - start_time)})
        wandb.log({"rr@1": rr1, "rr@5": rr5, "rr@10": rr10, "mrr": mrr, "auc_roc": auc_roc,
               "Throughput": len(dataset.duplicateIds)/(end_time - start_time)})
