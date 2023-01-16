
from typing import Dict, Any, List, Tuple

from sklearn import metrics

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.S3M import S3M
from trainers.trainer import MLTrainer
from data.bug_report_database import BugReportDatabase, Frame
from data.report_dataset import ReportDataset
from data.embed_map import PackedSequence, MetaData, get_packed_seq_from_frames


class S3MTrainer(MLTrainer):
    def __init__(self, report_db: BugReportDatabase, batch_size: int, embed_dim: int, nb_reports: int, nb_epochs: int, hid_dim: int, dropout: float) -> None:
        model = S3M(embed_dim=embed_dim, hid_dim=hid_dim, dropout=dropout)
        super().__init__(report_db, model, batch_size, embed_dim, nb_reports, nb_epochs)

        # for training
        self.loss_fn = nn.CrossEntropyLoss()

        # for metrics
        self.embed_metadata: List[MetaData] = []
        self.embeddings: List[torch.Tensor] = []

    def _train_step(self, batch: Tuple[int, Tuple[PackedSequence, Tuple[List[PackedSequence], List[PackedSequence]]]], optimizer: torch.optim.Optimizer) -> None:
        self.model.train()
        self.embed_layer.train()

        b, (queries, (good_reports, bad_reports)) = batch
        device = self.available_device

        queries_emb = self._embed_packed_seq(queries.to(device=device))
        predictions: List[torch.Tensor] = []
        for good_report, bad_report in zip(good_reports, bad_reports):
            good_emb = self._embed_packed_seq(good_report.to(device=device))
            bad_emb = self._embed_packed_seq(bad_report.to(device=device))
            pred_good = self.model(queries_emb, good_emb)
            pred_bad = self.model(queries_emb, bad_emb)

            predictions.append(
                (pred_good-pred_bad).view(pred_good.shape[0], -1))

        cat_pred = torch.cat(predictions, dim=0)
        target = torch.ones(
            (cat_pred.shape[0]), dtype=torch.long, device=device)

        loss = self.loss_fn(cat_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print({"batch": b, "loss": loss})
        self.model.eval()
        self.embed_layer.eval()

    def _prepare_candidates(self, dataset: ReportDataset) -> None:
        assert type(self.model) is S3M
        device = self.available_device

        # clear current values
        self.embed_metadata.clear()
        self.embeddings.clear()

        batch_size: int = self.batch_size * 4 # we can have bigger batch sizes as there are no auto-grad graphs

        possible_stacks: List[Tuple[MetaData, List[Frame]]] = [(MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts),
                                                                stack.frames_lists[0]) for report in self.report_db.report_list if report.creation_ts <= dataset.end_ts for stack_idx, stack in enumerate(report.stacktraces)]
        with torch.no_grad():
            for i in tqdm(range(0, len(possible_stacks), batch_size)):
                can_batch = possible_stacks[i:i+batch_size]
                can_md: List[MetaData] = [ md for md, _ in can_batch ]
                can_frames: List[List[Frame]] = [ frames for _, frames in can_batch ]

                embedding = self.model.encoder(self._embed_packed_seq(get_packed_seq_from_frames(list(can_frames), report_db=self.report_db).to(device=device))).detach().cpu()
                for idx, md in enumerate(can_md):
                    self.embed_metadata.append(md)
                    self.embeddings.append(embedding[idx])


    def _run_metrics(self, dataset: ReportDataset) -> Tuple[List[int], Tuple[List[float], List[int]]]:
        assert type(self.model) is S3M
        rank_results: List[int] = []
        top_scores: List[float] = []
        top_scores_label: List[int] = []
        batch_size: int = self.batch_size * 4 # we can have bigger batch sizes as there are no auto-grad graphs
        for duplicate_id in tqdm(dataset.duplicateIds[len(dataset.duplicateIds)//2:]):
            dup = self.report_db[duplicate_id]
            assert dup.dup_id is not None

            candidates: List[Tuple[MetaData, torch.Tensor]] = [ (self.embed_metadata[id], embed) for id, embed in enumerate(self.embeddings) if self.embed_metadata[id].timestamp <= dup.creation_ts and self.embed_metadata[id].bug_id != dup.bug_id ]

            same_bucket_can: List[int] = [ idx for idx, (can, _) in enumerate(candidates) if dup.dup_id == can.master_id]
            results: List[float] = [0.] * len(candidates)
            for q_idx in range(len(dup.stacktraces)) :
                stack = dup.stacktraces[q_idx]
                query : torch.Tensor = self.model.encoder(self._embed_packed_seq(get_packed_seq_from_frames([stack.frames_lists[0]], report_db=self.report_db)))

                for i in range(0, len(candidates), batch_size):
                    can_batch = candidates[i:i+batch_size]
                    can_tensors: List[torch.Tensor] = [ tensor for _, tensor in can_batch ]

                    q_tensor = query.expand((len(can_batch), -1))
                    candidates_tensors: torch.Tensor = torch.stack(can_tensors, dim=0)
                
                    scores = self.model.classifier(q_tensor, candidates_tensors).reshape((-1,2))

                    for j in range(scores.shape[0]):
                        results[i + j] = max(results[i + j], scores[j,1].detach().item())
            np_same_bucket_can = np.array(same_bucket_can)

            sorted_cans = np.argsort(results)[::-1]
            rank = min(np.argsort(sorted_cans)[np_same_bucket_can])

            rank_results.append(rank)

            top_scores.append(results[sorted_cans[0]])
            top_scores_label.append(1 if rank == 0 else 0)

        # return metrics
        return rank_results, (top_scores, top_scores_label)
