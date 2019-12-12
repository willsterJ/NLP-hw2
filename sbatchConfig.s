#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=test
#SBATCH --output=slurm_%j.out

module load python3/intel/3.6.3
module load pytorch/python3.6/0.3.0_4

python3 ./script.py --model custom
