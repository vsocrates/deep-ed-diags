#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=abd_pain_sweep
#SBATCH --out="abdpainsweep_%x-%j.out"
#SBATCH --mem=90G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL


# do something
echo "I'm echoing to stdout"
echo "I'm echoing to stderr" 1>&2
echo "My JobID is ${SLURM_JOBID}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s)"

module restore condacuda101
source activate factcheck_env
cd /home/vs428/Documents/deep-ed-diags

date=$(date '+%Y-%m-%d-%H%M%S')


python ./lit_simple_nn_runner_multilabel.py

