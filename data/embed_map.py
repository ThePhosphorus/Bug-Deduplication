from collections import defaultdict
from abc import abstractmethod
from typing import List, Tuple, DefaultDict, Set

import numpy as np
from tqdm import tqdm

import faiss
import torch
import torch.nn.functional as F

from data.bug_report_database import BugReport, BugReportDatabase, Frame, StackTrace

from models.S3M import S3M

PackedSequence = torch.nn.utils.rnn.PackedSequence
RELAX : bool = False

def get_packed_seq_from_frames(frames: List[List[Frame]], report_db: BugReportDatabase) -> PackedSequence :
    lengths: List[int] = [len(frame_list) for frame_list in frames ]

    batch_size : int = len(frames)
    max_length : int  = max(lengths)

    tensor: torch.Tensor = torch.zeros((batch_size, max_length), dtype=torch.long)

    for idx, frame_list in enumerate(frames):
        tensor[idx] = report_db.vocab.turn_to_id_tensor(frame_list, max_length)

    return torch.nn.utils.rnn.pack_padded_sequence(tensor, torch.Tensor(lengths), batch_first=True, enforce_sorted=False)


class MetaData(object) :
        def __init__(self, bug_id : int, stack_idx : int, master_id: int, timestamp: int) -> None:
            self.bug_id: int = bug_id
            self.stack_idx : int = stack_idx
            self.master_id : int = master_id
            self.timestamp : int = timestamp

class EmbedMap(object) :
    def __init__(self, report_db: BugReportDatabase, embedding_batch_size: int, device: torch.device) -> None:
        self.report_db: BugReportDatabase = report_db
        self.embedding_batch_size: int = embedding_batch_size
        self.device: torch.device = device
        self.duplicates: Set[int] = set()


    def add_duplicates(self, duplicates: List[int]) -> None:
        for dup in duplicates:
            self.duplicates.add(dup)
    
    @abstractmethod
    def compute_bad_candidates(self, model: S3M, override_bad_candidates : bool = True) -> None:
        raise NotImplementedError

class MemoryEmbedMap(EmbedMap):
    def __init__(self, report_db: BugReportDatabase, embedding_batch_size: int, device: torch.device) -> None:
        super().__init__(report_db, embedding_batch_size, device)

    def compute_bad_candidates(self, model: S3M, override_bad_candidates : bool = True) -> None:
        print("Computing bad candidates")
        # Reset emeddings in memory
        embeddings_metadata : List[MetaData] = []
        embeddings : List[torch.Tensor] = []
        id_by_master: DefaultDict[int, List[int]] = defaultdict(list)
        # set the model in eval to be sure
        model.eval()
        with torch.no_grad(): # we don't want a grad tree for each tensor
            if RELAX :
                possible_stacks : List[Tuple[ MetaData , List[Frame]]] = [ ( MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts), stack.frames_lists[0] ) for report in self.report_db.report_list if len(self.report_db.master_set[report.dup_id or report.bug_id]) > 1 for stack_idx, stack in enumerate(report.stacktraces)]
            else :
                possible_stacks : List[Tuple[ MetaData , List[Frame]]] = [ ( MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts), stack.frames_lists[0] ) for report in self.report_db.report_list for stack_idx, stack in enumerate(report.stacktraces)]
            print("generating embeddings")
            for i in tqdm(range(0, len(possible_stacks), self.embedding_batch_size)):
                can_batch = possible_stacks[i:i+self.embedding_batch_size]
                can_md: List[MetaData] = [ md for md, _ in can_batch ]
                can_frames: List[List[Frame]] = [ frames for _, frames in can_batch ]

                embedding = model.encoder(get_packed_seq_from_frames(can_frames, report_db=self.report_db).to(device=self.device)).detach().cpu()
                for idx, md in enumerate(can_md):
                    id = len(embeddings)
                    id_by_master[md.master_id].append(id)
                    embeddings_metadata.append(md)
                    embeddings.append(embedding[idx])

            print("Computing similarity")
            for duplicate_id in tqdm(self.duplicates):
                dup = self.report_db[duplicate_id]
                assert dup.dup_id is not None

                candidates_bug_id : List[int] = [ can_id for can_id in id_by_master if  dup.creation_ts - self.report_db.master_set_newest[can_id] <= (350 * 24 * 3600) ]

                if dup.dup_id not in candidates_bug_id:
                    continue
                candidates: List[Tuple[MetaData, torch.Tensor]] = [ (embeddings_metadata[id], embeddings[id]) for can_id in candidates_bug_id for id in id_by_master[can_id] if embeddings_metadata[id].timestamp <= dup.creation_ts and embeddings_metadata[id].bug_id != dup.bug_id ]

                for q_idx in range(len(dup.stacktraces)) :
                    stack = dup.stacktraces[q_idx]
                    query : torch.Tensor = model.encoder(get_packed_seq_from_frames([stack.frames_lists[0]], report_db=self.report_db).to(device=self.device))

                    score_per_can: List[Tuple[MetaData, float]] = []
                    for i in range(0, len(candidates), self.embedding_batch_size):
                        can_batch = candidates[i:i+self.embedding_batch_size]
                        can_mds: List[MetaData] = [ md for md, _ in can_batch ]
                        can_tensors: List[torch.Tensor] = [ tensor[None, :] for _, tensor in can_batch ]

                        q_tensor = query.expand((len(can_batch), -1))
                        candidates_tensors: torch.Tensor = torch.cat(can_tensors, dim=0)
                    
                        scores = model.classifier(q_tensor, candidates_tensors.to(device=self.device)).reshape((-1,2))

                        for j in range(scores.shape[0]):
                            score_per_can.append((can_mds[j], scores[j,1].detach().item()))

                    predictions = sorted(score_per_can, key= lambda x : x[1], reverse=True)
                    prediction_ids = [ pred for pred, _ in predictions]

                    if override_bad_candidates :
                        stack.candidates = []
                    stack.prediction = None

                    for idx, pred in enumerate(prediction_ids) :
                        if pred.master_id == dup.dup_id :
                            if stack.prediction is None or stack.prediction > (idx + 1): 
                                stack.prediction = (idx + 1)
                        elif override_bad_candidates and len(stack.candidates) <= 50 :
                            stack.candidates.append((pred.bug_id, pred.stack_idx))


class CosineFaissEmbedMap(EmbedMap) :
    def __init__(self, report_db: BugReportDatabase, embedding_batch_size: int, device: torch.device) -> None:
        super().__init__(report_db, embedding_batch_size, device)

    def compute_bad_candidates(self, model: S3M, override_bad_candidates : bool = True) -> None:
        print("Computing bad candidates")
        # set the model in eval to be sure
        model.eval()
        with torch.no_grad(): # we don't want a grad tree for each tensor
            res = faiss.StandardGpuResources()
            cpu_faiss_index : faiss.Index = faiss.IndexIDMap(faiss.IndexFlatIP(model.encoder.hidden_dim * 2))
            #faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_faiss_index)
            faiss_index = cpu_faiss_index
            if RELAX :
                possible_stacks : List[Tuple[ MetaData , List[Frame]]] = [ ( MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts), stack.frames_lists[0] ) for report in self.report_db.report_list if len(self.report_db.master_set[report.dup_id or report.bug_id]) > 1 for stack_idx, stack in enumerate(report.stacktraces)]
            else :
                possible_stacks : List[Tuple[ MetaData , List[Frame]]] = [ ( MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts), stack.frames_lists[0] ) for report in self.report_db.report_list for stack_idx, stack in enumerate(report.stacktraces)]
            print("generating embeddings")
            for i in tqdm(range(0, len(possible_stacks), self.embedding_batch_size)):
                can_batch = possible_stacks[i:i+self.embedding_batch_size]
                can_frames: List[List[Frame]] = [ frames for _, frames in can_batch ]

                embedding = F.normalize( model.encoder(get_packed_seq_from_frames(can_frames, report_db=self.report_db).to(device=self.device)), dim=-1)
                faiss_index.add_with_ids(embedding.detach().cpu().numpy(), np.arange(start= i, stop= i + len(can_batch), step= 1, dtype= np.int64 ))

            print("Computing similarity")
            duplicate_reports : List[BugReport] = [ self.report_db[duplicate_id] for duplicate_id in self.duplicates ]
            duplicate_queries : List[Tuple[BugReport, StackTrace, int]] = [ (dup, stack, idx) for dup in duplicate_reports 
                if dup.dup_id is not None and dup.creation_ts - self.report_db.master_set_newest[dup.dup_id] <= (350 * 24 * 3600) # window
                for idx, stack in enumerate(dup.stacktraces) ]

            for i in tqdm(range(0, len(duplicate_queries), self.embedding_batch_size)):
                query_batch = duplicate_queries[i:i+self.embedding_batch_size]
                q_tensor : torch.Tensor = model.encoder(get_packed_seq_from_frames([ q_stack.frames_lists[0] for _, q_stack, _ in query_batch], report_db=self.report_db).to(device=self.device))
                norm_q = F.normalize(q_tensor, dim=-1).detach().cpu().numpy()
                
                search_embed, search_results = faiss_index.search(norm_q, k=50)

                for j in range(search_results.shape[0]) : # for each query stack
                    query_obj, _, stack_idx = query_batch[j]
                    query = self.report_db[query_obj.bug_id]
                    stack = query.stacktraces[stack_idx]
                    if override_bad_candidates :
                            stack.candidates = []
                    stack.prediction = None
                    offset: int = 0

                    for k in range(search_results.shape[1]) : # for each predicted candidate
                        candidate = search_results[j,k]
                        can_md, cStack = possible_stacks[candidate]

                        # Check if candidate is valid
                        if can_md.timestamp > query.creation_ts or can_md.bug_id == query.bug_id :
                            offset += 1
                            continue

                        if can_md.master_id == query.dup_id : # good candidate
                            idx = k + 1 - offset

                            model.encoder(get_packed_seq_from_frames([stack.frames_lists[0], cStack], report_db=self.report_db).to(device=self.device))

                            if stack.prediction is None or stack.prediction > idx: 
                                stack.prediction = idx
                        elif override_bad_candidates : # bad candidate 
                            assert stack.candidates is not None
                            stack.candidates.append((can_md.bug_id, can_md.stack_idx))
            #del faiss_index
            #del cpu_faiss_index


class EuclideanFaissEmbedMap(EmbedMap) :
    def __init__(self, report_db: BugReportDatabase, embedding_batch_size: int, device: torch.device) -> None:
        super().__init__(report_db, embedding_batch_size, device)

    def compute_bad_candidates(self, model: S3M, override_bad_candidates : bool = True) -> None:
        print("Computing bad candidates")
        # set the model in eval to be sure
        model.eval()
        with torch.no_grad(): # we don't want a grad tree for each tensor
            faiss_index : faiss.Index = faiss.IndexIDMap(faiss.IndexFlatL2(model.encoder.hidden_dim * 2))
            if RELAX :
                possible_stacks : List[Tuple[ MetaData , List[Frame]]] = [ ( MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts), stack.frames_lists[0] ) for report in self.report_db.report_list if len(self.report_db.master_set[report.dup_id or report.bug_id]) > 1 for stack_idx, stack in enumerate(report.stacktraces)]
            else :
                possible_stacks : List[Tuple[ MetaData , List[Frame]]] = [ ( MetaData(report.bug_id, stack_idx, report.dup_id or report.bug_id, report.creation_ts), stack.frames_lists[0] ) for report in self.report_db.report_list for stack_idx, stack in enumerate(report.stacktraces)]
            print("generating embeddings")
            for i in tqdm(range(0, len(possible_stacks), self.embedding_batch_size)):
                can_batch = possible_stacks[i:i+self.embedding_batch_size]
                can_frames: List[List[Frame]] = [ frames for _, frames in can_batch ]

                embedding = model.encoder(get_packed_seq_from_frames(can_frames, report_db=self.report_db).to(device=self.device))
                faiss_index.add_with_ids(embedding.detach().cpu().numpy(), np.arange(start= i, stop= i + len(can_batch), step= 1, dtype= np.int64 ))

            print("Computing similarity")
            duplicate_reports : List[BugReport] = [ self.report_db[duplicate_id] for duplicate_id in self.duplicates ]
            duplicate_queries : List[Tuple[BugReport, StackTrace, int]] = [ (dup, stack, idx) for dup in duplicate_reports 
                if dup.dup_id is not None and dup.creation_ts - self.report_db.master_set_newest[dup.dup_id] <= (350 * 24 * 3600) # window
                for idx, stack in enumerate(dup.stacktraces) ]

            for i in tqdm(range(0, len(duplicate_queries), self.embedding_batch_size)):
                query_batch = duplicate_queries[i:i+self.embedding_batch_size]
                q_tensor : torch.Tensor = model.encoder(get_packed_seq_from_frames([ q_stack.frames_lists[0] for _, q_stack, _ in query_batch], report_db=self.report_db).to(device=self.device))
                norm_q = q_tensor.detach().cpu().numpy()
                
                _, search_results = faiss_index.search(norm_q, k=50)

                for j in range(search_results.shape[0]) : # for each query stack
                    query_obj, _, stack_idx = query_batch[j]
                    query = self.report_db[query_obj.bug_id]
                    stack = query.stacktraces[stack_idx]
                    offset: int = 0

                    for k in range(search_results.shape[1]) : # for each predicted candidate
                        candidate = search_results[j,k]
                        can_md, _ = possible_stacks[candidate]
                        if override_bad_candidates :
                            stack.candidates = []
                        stack.prediction = None

                        # Check if candidate is valid
                        if can_md.timestamp > query.creation_ts or can_md.bug_id == query.bug_id :
                            offset += 1
                            continue

                        if can_md.master_id == query.dup_id : # good candidate
                            idx = k + 1 - offset
                            if stack.prediction is None or stack.prediction > idx: 
                                stack.prediction = idx
                        elif override_bad_candidates : # bad candidate 
                            assert stack.candidates is not None
                            stack.candidates.append((can_md.bug_id, can_md.stack_idx))
