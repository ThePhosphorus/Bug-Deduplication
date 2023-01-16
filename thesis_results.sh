#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --account=def-aloise
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --job-name=thesis_compute_CPU
#SBATCH --output=/home/phos/projects/def-aloise/phos/logs/thesis/%x-%j.out

#SBATCH --mail-user=adem.aber-aouni@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

BASE_DIR="${HOME}/projects/def-aloise/phos/bug_deduplication_stack_traces"


export PYTHONPATH=${PYTHONPATH}:${BASE_DIR}
source ${BASE_DIR}/modules.sh

cd ${BASE_DIR}
#python ./experiments/test.py --bug_dataset=./datasets/netbeans/netbeans_stacktraces.json --method=fast --incremental
#python ./experiments/test.py --bug_dataset=./datasets/eclipse/eclipse_stacktraces.json --method=fast --incremental --remove_recursion
python ./experiments/test.py --bug_dataset=./datasets/campbell_dataset/reports.json --method=fast --incremental --remove_recursion
#python ./experiments/test.py --bug_dataset=./datasets/gnome/gnome_stacktraces.json --method=fast --incremental --remove_recursion
