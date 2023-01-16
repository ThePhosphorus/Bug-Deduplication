import argparse
from typing import Dict, Type

from trainers.trainer import ClassicTrainer
from trainers.tracesim import TracesimTrainer
from trainers.brodie05 import Brodie05Trainer
from trainers.fast import FaSTTrainer

from data.bug_report_database import BugReportDatabase
from data.report_dataset import ReportDataset

import wandb

method_map : Dict[str, Type[ClassicTrainer]] = {
    "brodie05" : Brodie05Trainer,
    "tracesim" : TracesimTrainer,
    "fast" : FaSTTrainer,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug_dataset', action="store", type=str, required=True, help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('--method', action="store", type=str, required=True, help="method to use", choices=list(method_map.keys()))
    parser.add_argument('--incremental', action="store_true", help="Use past value as starting point for training")
    parser.add_argument('--trim_len', action= "store", type=int, help="trimming length of function names (only for java datasets)", default= 0)
    parser.add_argument('--remove_recursion', action= "store_true", help="recursion should be removed")

    args = parser.parse_args()

    wandb.init(project="bug_dedup_thesis", entity="spratai", group="test", config=args)

    wandb.define_metric("rr@1", summary="mean", goal="maximize")
    wandb.define_metric("rr@5", summary="mean", goal="maximize")
    wandb.define_metric("rr@10", summary="mean", goal="maximize")
    wandb.define_metric("mrr", summary="mean", goal="maximize")
    wandb.define_metric("auc_roc", summary="mean", goal="maximize")
    wandb.define_metric("Throughput", summary="mean", goal="minimize")

    report_db = BugReportDatabase.from_json(args.bug_dataset, remove_recursion=args.remove_recursion, trim_len=args.trim_len)
    trainer = method_map[args.method](report_db= report_db, is_incremental= args.incremental)
    for dataset in ReportDataset.progressive(report_db=report_db) :
        trainer.compute(dataset= dataset)
        trainer.fit(dataset=dataset)
