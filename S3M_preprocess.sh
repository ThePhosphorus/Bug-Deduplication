#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --account=def-aloise
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --job-name=S3M_Preprocess
#SBATCH --output=/home/phos/projects/def-aloise/phos/logs/S3M/%x_%a-%j.out

#SBATCH --mail-user=adem.aber-aouni@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

BASE_DIR="${HOME}/projects/def-aloise/phos/bug_deduplication_stack_traces"


export PYTHONPATH=${PYTHONPATH}:${BASE_DIR}
source ${BASE_DIR}/modules.sh

cd ${BASE_DIR}
python experiments/s3m_trainer.py --bug_dataset=./datasets/netbeans/netbeans_stacktraces.json --precompute_only --trim_len=0 --remove_recursion

