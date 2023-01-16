
from typing import Dict, Any, List, Tuple, Set

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

import faiss

from models.S3M import LSTMEncoder
from trainers.trainer import MLTrainer
from data.bug_report_database import BugReportDatabase, Frame
from data.report_dataset import ReportDataset
from data.embed_map import PackedSequence, MetaData, get_packed_seq_from_frames


def remove_dups(arr : List[Tuple[int, float]] ) -> List[Tuple[int, float]] :
        item_set : Set[int] = set()
        result : List[Tuple[int, float]] = []

        for item, score in arr :
            if item not in item_set :
                result.append((item, score))
                item_set.add(item)

        return result


class FactsTrainer(MLTrainer):
    def __init__(self, report_db: BugReportDatabase, batch_size: int, embed_dim: int, nb_reports: int, nb_epochs: int, hid_dim: int, dropout: float) -> None:
        model = LSTMEncoder(embed_dim=embed_dim, hid_dim=hid_dim, dropout=dropout)
        super().__init__(report_db, model, batch_size, embed_dim, nb_reports, nb_epochs)

        # for training
        ## triplet Cosine
        #cosine_dist = lambda lhs, rhs : F.cosine_similarity(lhs, rhs)
        #self.loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_dist, margin=0.5)
        
        ## Normal Cosine
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
        self.loss_fn = lambda issues, bad_reports, good_reports : self.cosine_loss( torch.cat([issues, issues], dim=0), torch.cat([good_reports, bad_reports], dim=0), torch.cat([torch.ones(good_reports.shape[0]), -1 * torch.ones(bad_reports.shape[0])],dim=0).to(device=good_reports.device) )
        # Euclidean
        #self.loss_fn = nn.TripletMarginLoss()

        wandb.watch(models = [self.model, self.embed_layer], criterion= self.loss_fn, log_freq=20, log_graph=True)

        # for metrics
        self.embed_metadata: List[MetaData] = []
        # Cosine
        # self.faiss_index : faiss.Index = faiss.IndexIDMap(faiss.IndexFlatIP(hid_dim))
        #self.faiss_index : faiss.Index = faiss.IndexIDMap(faiss.IndexFlatL2(hid_dim))
    	## HNSW
        self.hnsw = faiss.IndexHNSWFlat(hid_dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index : faiss.Index = faiss.IndexIDMap(self.hnsw)
        #self.faiss_index : faiss.Index = faiss.IndexIDMap(faiss.IndexHNSWFlat(hid_dim, 32))

        ## IVF
        # self.faiss_index : faiss.Index = faiss.IndexIDMap(faiss.IndexIVFFlat(faiss.IndexFlatIP(hid_dim), hid_dim, 64, faiss.METRIC_INNER_PRODUCT ))


        ## Euclidean
        # self.faiss_index : faiss.Index = faiss.IndexIDMap(faiss.IndexFlatL2(hid_dim))

    def _train_step(self, batch: Tuple[int, Tuple[PackedSequence, Tuple[List[PackedSequence], List[PackedSequence]]]], optimizer: torch.optim.Optimizer) -> None:
        self.model.train()
        self.embed_layer.train()

        b, (queries, (good_reports, bad_reports)) = batch
        device = self.available_device

        queries_emb = self._embed_packed_seq(queries.to(device=device))

        issue_tensor_list : List[torch.Tensor] = []
        good_report_tensor_list : List[torch.Tensor] = []
        bad_report_tensor_list : List[torch.Tensor] = []
        issue_embed: torch.Tensor = self.model(queries_emb)
        for good_report, bad_report in zip(good_reports, bad_reports) :
            issue_tensor_list.append(issue_embed)
            good_report_tensor_list.append(self.model(self._embed_packed_seq(good_report.to(device=device))))
            bad_report_tensor_list.append(self.model(self._embed_packed_seq(bad_report.to(device=device))))

        # Cosine
        loss = self.loss_fn(torch.cat(issue_tensor_list, dim=0),torch.cat(bad_report_tensor_list, dim=0), torch.cat(good_report_tensor_list, dim=0))
        #loss = self.loss_fn(torch.cat(issue_tensor_list, dim=0), torch.cat(good_report_tensor_list, dim=0), torch.cat(bad_report_tensor_list, dim=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print({"batch": b, "loss": loss})
        self.model.eval()
        self.embed_layer.eval()

    def _prepare_candidates(self, dataset: ReportDataset) -> None:
        assert type(self.model) is LSTMEncoder
        device = self.available_device

        # clear current values
        self.embed_metadata.clear()
        self.faiss_index.reset()

        batch_size: int = self.batch_size * 4 # we can have bigger batch sizes as there are no auto-grad graphs

        possible_stacks: List[Tuple[MetaData, List[Frame]]] = [(MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts),
                                                                stack.frames_lists[0]) for report in self.report_db.report_list if report.creation_ts <= dataset.end_ts for stack_idx, stack in enumerate(report.stacktraces)]
        with torch.no_grad():
            for i in tqdm(range(0, len(possible_stacks), batch_size)):
                can_batch = possible_stacks[i:i+batch_size]
                can_md: List[MetaData] = [ md for md, _ in can_batch ]
                can_frames: List[List[Frame]] = [ frames for _, frames in can_batch ]

                embedding = F.normalize(self.model(self._embed_packed_seq(get_packed_seq_from_frames(list(can_frames), report_db=self.report_db).to(device=device))), dim=-1).detach().cpu()
                #embedding = self.model(self._embed_packed_seq(get_packed_seq_from_frames(list(can_frames), report_db=self.report_db).to(device=device))).detach().cpu()
                
                start_idx= len(self.embed_metadata)

                self.faiss_index.add_with_ids(embedding.numpy(), np.arange(start= start_idx, stop= start_idx + len(can_batch), step= 1, dtype= np.int64 ))
                self.embed_metadata.extend(can_md)
        self.hnsw.reorder_links()
    def _run_metrics(self, dataset: ReportDataset) -> Tuple[List[int], Tuple[List[float], List[int]]]:
        assert type(self.model) is LSTMEncoder
        rank_results: List[int] = []
        top_scores: List[float] = []
        top_scores_label: List[int] = []
        # batch_size: int = self.batch_size * 4 # we can have bigger batch sizes as there are no auto-grad graphs
        for duplicate_id in tqdm(dataset.duplicateIds[len(dataset.duplicateIds)//2:]):
            dup = self.report_db[duplicate_id]
            assert dup.dup_id is not None

            queries : torch.Tensor = F.normalize(self.model(self._embed_packed_seq(get_packed_seq_from_frames([stack.frames_lists[0] for stack in dup.stacktraces], report_db=self.report_db))), dim=-1).detach()
            #queries : torch.Tensor = self.model(self._embed_packed_seq(get_packed_seq_from_frames([stack.frames_lists[0] for stack in dup.stacktraces], report_db=self.report_db))).detach()
            results, search_ids = self.faiss_index.search(queries.numpy(), k=100)

            flatten_scores = results.flatten()
            best_idxs = np.argsort(flatten_scores)[::-1]
            #best_idxs = np.argsort(flatten_scores)

            best_cans = [ (id, score) for id, score in remove_dups(zip(list(search_ids.flatten()[best_idxs]), list(flatten_scores[best_idxs]))) if self.embed_metadata[id].bug_id != dup.bug_id  ]
            np_same_bucket_can = np.array([idx for idx, (id, _) in enumerate(best_cans) if self.embed_metadata[id].master_id == dup.dup_id])

            if len(np_same_bucket_can) == 0 :
                rank_results.append(99999)
                continue

            rank = min(np_same_bucket_can)

            rank_results.append(rank)

            top_scores.append(best_cans[0][1])
            top_scores_label.append(1 if rank == 0 else 0)

        # return metrics
        return rank_results, (top_scores, top_scores_label)
