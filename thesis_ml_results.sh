#!/bin/bash

#SBATCH --time=7:00:00
#SBATCH --account=def-aloise
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:p100:1
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
#python ./experiments/ml_test.py --bug_dataset=./datasets/netbeans/netbeans_stacktraces.json --method=facts --trim_len=1 --remove_recursion --batch_size=256 --nb_reports=8 --nb_epochs=4 --hid_dim=128 --embed_dim=64 --dropout=0.1
#python ./experiments/ml_test.py --bug_dataset=./datasets/eclipse/eclipse_stacktraces.json --method=facts --trim_len=1 --remove_recursion --batch_size=256 --nb_reports=8 --nb_epochs=4 --hid_dim=128 --embed_dim=64 --dropout=0.1
python ./experiments/ml_test.py --bug_dataset=./datasets/campbell_dataset/reports.json --method=facts --trim_len=1 --remove_recursion --batch_size=256 --nb_reports=8 --nb_epochs=4 --hid_dim=128 --embed_dim=64 --dropout=0.1
#python ./experiments/ml_test.py --bug_dataset=./datasets/gnome/gnome_stacktraces.json --method=facts --trim_len=1 --remove_recursion --batch_size=512 --nb_reports=8 --nb_epochs=4 --hid_dim=64 --embed_dim=128 --dropout=0.1
