import argparse
from typing import Dict, Type, Union

from trainers.s3m import S3MTrainer
from trainers.facts import FactsTrainer

from data.bug_report_database import BugReportDatabase
from data.report_dataset import ReportDataset

import wandb

method_map : Dict[str, Union[Type[S3MTrainer], Type[FactsTrainer]]] = {
    "s3m" : S3MTrainer,
    "facts" : FactsTrainer
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug_dataset', action="store", type=str, required=True, help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('--method', action="store", type=str, required=True, help="method to use", choices=list(method_map.keys()))
    parser.add_argument('--trim_len', action= "store", type=int, help="trimming length of function names (only for java datasets)", default= 0)
    parser.add_argument('--remove_recursion', action= "store_true", help="recursion should be removed")

    parser.add_argument('--batch_size', action= "store", type=int, help="Batch size during training", default= 64)
    parser.add_argument('--embed_dim', action= "store", type=int, help="Size of frame embedding size", default=50)
    parser.add_argument('--nb_reports', action= "store", type=int, help="Number of good and bad reports to use during training", default= 8)
    parser.add_argument('--nb_epochs', action= "store", type=int, help="Number of epochs to train for", default= 8)
    parser.add_argument('--hid_dim', action= "store", type=int, help="Size of stack embedding size", default= 100)
    parser.add_argument('--dropout', action= "store", type=float, help="Dropout value used during training", default=0.1)

    
    args = parser.parse_args()

    wandb.init(project="bug_dedup_thesis", entity="spratai", group="test", config=args)
   
    wandb.config.num_layers = 1

    wandb.define_metric("rr@1", summary="mean", goal="maximize")
    wandb.define_metric("rr@5", summary="mean", goal="maximize")
    wandb.define_metric("rr@10", summary="mean", goal="maximize")
    wandb.define_metric("mrr", summary="mean", goal="maximize")
    wandb.define_metric("auc_roc", summary="mean", goal="maximize")
    wandb.define_metric("Throughput", summary="mean", goal="minimize")

    report_db = BugReportDatabase.from_json(args.bug_dataset, remove_recursion=args.remove_recursion, trim_len=args.trim_len)
    trainer = method_map[args.method](report_db= report_db, batch_size = args.batch_size, embed_dim = args.embed_dim, nb_reports = args.nb_reports, nb_epochs = args.nb_epochs, hid_dim = args.hid_dim, dropout = args.dropout)
    for dataset in ReportDataset.progressive(report_db=report_db) :
        trainer.compute(dataset= dataset)
        trainer.fit(dataset=dataset)
