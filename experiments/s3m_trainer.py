import random
import argparse
from typing import List, Tuple, Generator
import os 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from data.embed_map import EmbedMap, CosineFaissEmbedMap, EuclideanFaissEmbedMap, PackedSequence, get_packed_seq_from_frames, MemoryEmbedMap
from models.S3M import S3M

import wandb

import pickle

from data.bug_report_database import BugReportDatabase, Frame, StackTrace
from data.report_dataset import ReportDataset

RELAX: bool = False
DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE: torch.device = torch.device('cpu')

def generate_batches(report_db: BugReportDatabase, report_dataset: ReportDataset, nb_reports: int, batch_size: int) -> Generator[Tuple[PackedSequence, Tuple[List[PackedSequence], List[PackedSequence]]], None, None] :
    duplicates = report_dataset.duplicateIds
    random.shuffle(duplicates)
    assert report_db[duplicates[0]].stacktrace.candidates is not None # Should never be None (lerch at 0 and auto-sync after)
    master_sets = report_db.master_set
    
    for i in range(0, len(duplicates), batch_size):
        dup_batch = duplicates[i:i+batch_size]

        queries : List[List[Frame]] = []
        good_reports_frame_lists: List[List[List[Frame]]] = []
        bad_reports_frame_lists: List[List[List[Frame]]] = []
        for duplicate_id in dup_batch:
            # Get master
            duplicate = report_db[duplicate_id]
            assert duplicate.dup_id is not None
            master_set = master_sets[duplicate.dup_id]

            query : StackTrace = random.choice(duplicate.stacktraces)
            assert query.candidates is not None

            queries.append(query.frames_lists[0])

            possible_good_reports: List[List[Frame]] = [ stack.frames_lists[0] for candidate in master_set if candidate != duplicate_id and report_db[candidate].creation_ts <= duplicate.creation_ts for stack in report_db[candidate].stacktraces]

            good_reports : List[List[Frame]] = random.choices(possible_good_reports, k=nb_reports)

            bad_reports: List[List[Frame]] = []
            if len(query.candidates) > 0 :
                bad_reports = [report_db[can_report_id].stacktraces[can_stack].frames_lists[0]  for can_report_id, can_stack in random.choices(query.candidates, k=nb_reports)]
            else :
                 while len(bad_reports) != nb_reports :
                    random_stack = random.choice(report_db.report_list)

                    if (random_stack.dup_id == duplicate.dup_id or random_stack.bug_id == duplicate.dup_id or  random_stack.dup_id == duplicate.bug_id or random_stack.bug_id == duplicate.bug_id) :
                        continue
                    bad_reports.append(random.choice(random_stack.stacktraces).frames_lists[0])
            good_reports_frame_lists.append(good_reports)
            bad_reports_frame_lists.append(bad_reports)
        
        yield (get_packed_seq_from_frames(queries, report_db), # Issues
         ([get_packed_seq_from_frames([ frame_lists[item] for frame_lists in good_reports_frame_lists], report_db) for item in range(nb_reports)], # good reports
          [get_packed_seq_from_frames([ frame_lists[item] for frame_lists in bad_reports_frame_lists], report_db) for item in range(nb_reports)])) # Bad reports

def compute_metrics(report_db: BugReportDatabase, test_dataset:ReportDataset, metric: str = "test") :
    # Compute Map Recall rate, RR@1, RR@5, RR@10
    duplicates = test_dataset.duplicateIds

    recall_rate_pos: List[int] = []
    bad : int = 0
    for dup_id in duplicates :
        pred_idx = min([stacktrace.prediction or 999999 for stacktrace in report_db[dup_id].stacktraces])
        if RELAX and pred_idx == 999999 :
            bad += 1
            continue
        recall_rate_pos.append(pred_idx)

    nb_pred = len(recall_rate_pos)
    rr_pos = np.array(recall_rate_pos)
    mrr = np.mean(1./rr_pos)
    hits_per_k : List[int] = [ 0 for _ in range(20) ]

    for pos in recall_rate_pos :
        if pos <= 20:
            hits_per_k[pos-1] += 1

    wandb.log({f"{metric}.mrr": mrr, f"{metric}.rr@1":sum(hits_per_k[:1])/nb_pred , f"{metric}.rr@5":sum(hits_per_k[:5])/nb_pred, f"{metric}.rr@10":sum(hits_per_k[:10])/nb_pred, f"{metric}.bad": bad/len(duplicates)})
    print({f"{metric}.mrr": mrr, f"{metric}.rr@1":sum(hits_per_k[:1])/nb_pred , f"{metric}.rr@5":sum(hits_per_k[:5])/nb_pred, f"{metric}.rr@10":sum(hits_per_k[:10])/nb_pred, f"{metric}.bad": bad/len(duplicates)})

def validate(model: S3M, loss_fn: nn.CrossEntropyLoss, all_val: bool, report_db: BugReportDatabase, val_dataset: ReportDataset, nb_reports: int, mode: int ) :
    # run validation
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        loss_count = 0
        total_accuracy = 0.
        total_precision = 0.
        total_recall = 0.
        nb_elems = 0

        for b, (issue, (good_reports, bad_reports)) in enumerate( generate_batches(report_db, val_dataset,nb_reports=nb_reports, batch_size= batch_size)):
            if not all_val and b >= 20 : 
                break
            nb_TN = 0
            nb_FP = 0
            nb_FN = 0
            nb_TP = 0

            if mode == 2 :
                issue_tensor_list : List[torch.Tensor] = []
                good_report_tensor_list : List[torch.Tensor] = []
                bad_report_tensor_list : List[torch.Tensor] = []
                issue_embed: torch.Tensor = model.encoder(issue.to(device=DEVICE))
                for good_report, bad_report in zip(good_reports, bad_reports) :
                    issue_tensor_list.append(issue_embed)
                    good_report_tensor_list.append(model.encoder(good_report.to(device=DEVICE)))
                    bad_report_tensor_list.append(model.encoder(bad_report.to(device=DEVICE)))

                issue_tensor = torch.cat(issue_tensor_list, dim=0)
                good_tensor = torch.cat(good_report_tensor_list, dim=0)
                bad_tensor = torch.cat(bad_report_tensor_list, dim=0)
                loss = loss_fn(issue_tensor,good_tensor,bad_tensor)
                total_loss += loss.detach().item()
                loss_count += 1
                pred = torch.sign(torch.sum(torch.abs(good_tensor - issue_tensor) - torch.abs(bad_tensor - issue_tensor), dim=1)).detach().cpu().to(dtype=torch.long) + 1
                indexed = torch.bincount(pred, minlength=3)
                nb_TP += indexed[2]
                nb_FN += indexed[0]
            elif mode == 1:
                issue_tensor_list : List[torch.Tensor] = []
                good_report_tensor_list : List[torch.Tensor] = []
                bad_report_tensor_list : List[torch.Tensor] = []
                issue_embed: torch.Tensor = model.encoder(issue.to(device=DEVICE))
                for good_report, bad_report in zip(good_reports, bad_reports) :
                    issue_tensor_list.append(issue_embed)
                    good_report_tensor_list.append(model.encoder(good_report.to(device=DEVICE)))
                    bad_report_tensor_list.append(model.encoder(bad_report.to(device=DEVICE)))

                issue_tensor = torch.cat(issue_tensor_list, dim=0)
                good_tensor = torch.cat(good_report_tensor_list, dim=0)
                bad_tensor = torch.cat(bad_report_tensor_list, dim=0)
                loss = loss_fn(issue_tensor,bad_tensor, good_tensor)
                total_loss += loss.detach().item()
                loss_count += 1
                pred = torch.sign(torch.sum(torch.abs(bad_tensor - issue_tensor) - torch.abs(good_tensor - issue_tensor), dim=1)).detach().cpu().to(dtype=torch.long) + 1
                indexed = torch.bincount(pred, minlength=3)
                nb_TP += indexed[2]
                nb_FN += indexed[0]
            else:
                predictions: List[torch.Tensor] = []
                for good_report, bad_report in zip(good_reports, bad_reports) :
                    pred_good = model(issue.to(device=DEVICE), good_report.to(device=DEVICE))
                    pred_bad = model(issue.to(device=DEVICE), bad_report.to(device=DEVICE))

                    predictions.append((pred_good-pred_bad).view(pred_good.shape[0],-1))
                
                cat_pred = torch.cat(predictions,dim=0)
                target = torch.tensor([1] * cat_pred.shape[0]).to(device=DEVICE)

                loss = loss_fn(cat_pred,target)
                total_loss += loss.detach().item()
                loss_count += 1

                class_pred = cat_pred.argmax(dim=1).flatten()
                indexed = torch.bincount(2 * target + class_pred, minlength=4).to(torch.device('cpu'))
                nb_TN = indexed[0].detach().item()
                nb_FP = indexed[1].detach().item()
                nb_FN = indexed[2].detach().item()
                nb_TP = indexed[3].detach().item()
            eps=1e-10
            accuracy = (nb_TP + nb_TN) / (nb_TP + nb_TN + nb_FP + nb_FN +eps)
            precision = (nb_TP) / (nb_TP + nb_FP + eps)
            recall = (nb_TP) / (nb_TP + nb_FN + eps)

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            nb_elems += 1

        wandb.log({'val.loss':total_loss/loss_count, 'val.acc':total_accuracy/nb_elems, 'val.prec':total_precision/nb_elems,"val.recall": total_recall/nb_elems}, commit=False)
        print("val.loss", total_loss/loss_count )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug_dataset', action="store", type=str, required=True, help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('--remove_recursion', action="store_true" ,help="Remove recursion from stack traces")
    parser.add_argument('--input_dim', action="store", type=int, default=50,required=False ,help="Input embedding size")
    parser.add_argument('--hid_dim', action="store", type=int, default=200,required=False ,help="Bi-LSTM hidden layer size")
    parser.add_argument('--dropout', action="store", type=float, default=0.1,required=False ,help="Neural Net Dropout value for classifier")
    parser.add_argument('--feature_number', action="store", type=int, default=4,required=False ,help="Number of features")
    parser.add_argument('--nb_epochs', action="store", type=int, default=100,required=False ,help="Number of epochs to train")
    parser.add_argument('--batch_size', action="store", type=int, default=64,required=False ,help="Size of batches for training")
    parser.add_argument("--trim_len", type=int, default=0, required=False, help="Trim length for S3M method")
    parser.add_argument("--nb_reports_per_issue", type=int, default=4, required=False, help="Number of reports to run per issue")
    parser.add_argument("--metrics_report_rate", type=int, default=20, required=False, help="The rate (in epochs) at witch to run metrics")
    parser.add_argument('--use_cosine', action="store_true" ,help="Use cosine classification instead of FFNN")
    parser.add_argument('--use_euclidean', action="store_true" ,help="Use cosine classification instead of FFNN")
    parser.add_argument('--precompute_only', action="store_true" ,help="Only prepare the dataset")
    parser.add_argument('--lr', action="store", default=1e-3 ,help="learning rate")

    args = parser.parse_args()

    if args.use_cosine and args.use_euclidean :
        print("Error: can't use cosine and euclidean at the same time, Euclidean will be used")
    mode: int = 0
    if args.use_euclidean :
        mode = 2
    elif args.use_cosine :
        mode = 1
    
    if args.bug_dataset[-2:] == ".P" :
        with open(args.bug_dataset, "rb") as file:
            report_db, (train_dataset, val_dataset, test_dataset) = pickle.load(file)
    else :
        report_db = BugReportDatabase.from_json(args.bug_dataset, remove_recursion=args.remove_recursion, trim_len=args.trim_len)
        train_dataset, val_dataset, test_dataset = ReportDataset.split(report_db=report_db)
        report_db.gen_candidates(train_dataset.duplicateIds)
        report_db.gen_candidates(val_dataset.duplicateIds)
        report_db.gen_candidates(test_dataset.duplicateIds)
        with open(f"./out/{os.path.basename(args.bug_dataset)}.P", "wb") as file:
            pickle.dump((report_db, (train_dataset, val_dataset, test_dataset)), file)

    if args.precompute_only :
        exit(0)

    wandb.init(project="bug_dedup", entity="spratai", name="S3M Faiss", group="auto-sync FAISS", config=args, mode="offline")
    wandb.config.device=str(DEVICE)
    print("This will be running on:", DEVICE)

    model: S3M = S3M(vocab_size= len(report_db.vocab),input_dim=args.input_dim,hid_dim=args.hid_dim, dropout=args.dropout, feature_number=args.feature_number, use_classifier=mode == 0)
    
    cosine_dist = lambda lhs, rhs : F.cosine_similarity(lhs, rhs)
    LossFuncs: List[nn.Module] = [nn.CrossEntropyLoss(), nn.TripletMarginWithDistanceLoss(distance_function=cosine_dist, margin=0.5), nn.TripletMarginLoss()]
    loss_fn =  LossFuncs[mode]
    optimizer = Adam(model.parameters(), lr=float(args.lr), weight_decay=0.0)

    model.to(device=DEVICE)

    batch_size = args.batch_size
    nb_reports = args.nb_epochs

    EmbedMaps: List[EmbedMap] = [MemoryEmbedMap, CosineFaissEmbedMap, EuclideanFaissEmbedMap]

    embed_map : EmbedMap = EmbedMaps[mode](report_db, batch_size * 2, DEVICE)
    embed_map.add_duplicates(train_dataset.duplicateIds)
    embed_map.add_duplicates(val_dataset.duplicateIds)
    embed_map.add_duplicates(test_dataset.duplicateIds)
    # Training
    wandb.watch(models=model,criterion=loss_fn, log_freq=100)
    for e in range(nb_reports):
        print (f"Epoch {e + 1} on {nb_reports}")
        if mode != 0 :
            embed_map.compute_bad_candidates(model, override_bad_candidates= e != 0 and e % args.metrics_report_rate == 0)
            compute_metrics(report_db=report_db, test_dataset=train_dataset, metric="train")
            compute_metrics(report_db=report_db, test_dataset=val_dataset, metric="val")
            compute_metrics(report_db=report_db, test_dataset=test_dataset, metric="test")
        elif e != 0 and e % args.metrics_report_rate == 0:
            embed_map.compute_bad_candidates(model)
            compute_metrics(report_db=report_db, test_dataset=train_dataset, metric="train")
            compute_metrics(report_db=report_db, test_dataset=val_dataset, metric="val")
            compute_metrics(report_db=report_db, test_dataset=test_dataset, metric="test")
            # torch.save(model, f"./out/S3M.{e}.pt")

        count: int = 0
        nb_batches = len(train_dataset.duplicateIds) // batch_size
        report_rate = max( nb_batches // 10, 10)

        for b, (issue, (good_reports, bad_reports)) in enumerate( generate_batches(report_db, train_dataset,nb_reports=args.nb_reports_per_issue, batch_size= batch_size)):
            model.train()
            if mode == 2: # euclidean
                issue_tensor_list : List[torch.Tensor] = []
                good_report_tensor_list : List[torch.Tensor] = []
                bad_report_tensor_list : List[torch.Tensor] = []
                issue_embed: torch.Tensor = model.encoder(issue.to(device=DEVICE))
                for good_report, bad_report in zip(good_reports, bad_reports) :
                    issue_tensor_list.append(issue_embed)
                    good_report_tensor_list.append(model.encoder(good_report.to(device=DEVICE)))
                    bad_report_tensor_list.append(model.encoder(bad_report.to(device=DEVICE)))

                optimizer.zero_grad()
                loss = loss_fn(torch.cat(issue_tensor_list, dim=0),torch.cat(good_report_tensor_list, dim=0),torch.cat(bad_report_tensor_list, dim=0))
                loss.backward()

            elif mode == 1 :
                issue_tensor_list : List[torch.Tensor] = []
                good_report_tensor_list : List[torch.Tensor] = []
                bad_report_tensor_list : List[torch.Tensor] = []
                issue_embed: torch.Tensor = model.encoder(issue.to(device=DEVICE))
                for good_report, bad_report in zip(good_reports, bad_reports) :
                    issue_tensor_list.append(issue_embed)
                    good_report_tensor_list.append(model.encoder(good_report.to(device=DEVICE)))
                    bad_report_tensor_list.append(model.encoder(bad_report.to(device=DEVICE)))

                loss = loss_fn(torch.cat(issue_tensor_list, dim=0),torch.cat(bad_report_tensor_list, dim=0), torch.cat(good_report_tensor_list, dim=0))
                optimizer.zero_grad()
                loss.backward()

            else :
                predictions: List[torch.Tensor] = []
                for good_report, bad_report in zip(good_reports, bad_reports) :
                    pred_good = model(issue.to(device=DEVICE), good_report.to(device=DEVICE))
                    pred_bad = model(issue.to(device=DEVICE), bad_report.to(device=DEVICE))

                    predictions.append((pred_good-pred_bad).view(pred_good.shape[0],-1))
                
                cat_pred = torch.cat(predictions,dim=0)
                target = torch.ones((cat_pred.shape[0]), dtype=torch.long, device=DEVICE)

                optimizer.zero_grad()
                loss = loss_fn(cat_pred,target)
                loss.backward()

            optimizer.step()

            # if b % report_rate == 0:    
            #     validate(model = model, loss_fn = loss_fn, all_val=False, report_db= report_db, val_dataset=val_dataset, nb_reports=args.nb_reports_per_issue, mode = mode)
            print({"train.loss": loss, "batch": b, "epoch": e})
            wandb.log({"train.loss": loss, "batch": b, "epoch": e}, commit=True)

        print(f"finished epoch {e} with {(nb_batches)} batches")
        validate(model = model, loss_fn = loss_fn, all_val=True, report_db= report_db, val_dataset=val_dataset, nb_reports=args.nb_reports_per_issue, mode = mode)
        wandb.log({"epoch": e, "batch": nb_batches}, commit=True)
