#!/bin/bash
#SBATCH --job-name=process_classify_wos         # Job name
#SBATCH --output=/n/netscratch/siacus_lab/Lab/logs/process_classify_wos_%j.out    # Standard output log
#SBATCH --error=//n/netscratch/siacus_lab/Lab/logs/process_classify_wos_%j.out     # Error log file
#SBATCH --ntasks=1                        # Number of tasks (1 task for this script)
#SBATCH --cpus-per-task=4                 # Request 2 CPU core
#SBATCH --mem=40G                         # Request 25GB of memory
#SBATCH --time=11:59:00                   # Set a time limit of 12 hours
#SBATCH --partition=gpu_test              # Partition name for GPU jobs
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --mail-type=ALL                   # This command would send an email when the job advances.


# Load necessary Python module
module load nvhpc/23.7-fasrc01
module load cuda/12.2.0-fasrc01 
module load gcc/12.2.0-fasrc01
module load cmake


# Trap to requeue the job if interrupted
#trap 'scontrol requeue $SLURM_JOB_ID' TERM INT HUP

# Directly specify the Python executable from the gpu environment
/n/home11/siacus/miniconda3/envs/jago/bin/python classify_wos.py 

