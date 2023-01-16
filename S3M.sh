#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --account=def-aloise
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16G
#SBATCH --job-name=S3M
#SBATCH --output=/home/phos/projects/def-aloise/phos/logs/S3M/%x_%a-%j.out

#SBATCH --mail-user=adem.aber-aouni@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

BASE_DIR="${HOME}/projects/def-aloise/phos/bug_deduplication_stack_traces"


export PYTHONPATH=${PYTHONPATH}:${BASE_DIR}
source ${BASE_DIR}/modules.sh

cd ${BASE_DIR}
#python experiments/s3m_trainer.py --bug_dataset=./out/netbeans_stacktraces_trim_0_no_recursion.json.P --trim_len=0 --input_dim=50 --hid_dim=100 --dropout=0.1 --feature_number=3 --nb_epochs=51 --batch_size=128 --nb_reports_per_issue=8 --metrics_report_rate=1
python experiments/s3m_trainer.py --bug_dataset=./out/netbeans_stacktraces_trim_0_with_recursion.json.P --trim_len=0 --input_dim=50 --hid_dim=100 --dropout=0.5 --nb_epochs=101 --batch_size=128 --nb_reports_per_issue=16 --metrics_report_rate=5 --lr 0.01 --use_cosine
