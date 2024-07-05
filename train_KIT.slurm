#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Mar 15 2023 17:24:35 GMT+1100

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="mld"

# The project ID which this job should run under:
#SBATCH --account="punim1485"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Use this email address:
#SBATCH --mail-user=lingxiao1@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-50:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

cd /data/gpfs/projects/punim1485/lingxiao1/mdm/motion-latent-diffusion

module load CUDA/12.2.0
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate mld

python -m train --cfg configs/config_vae_kit.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s