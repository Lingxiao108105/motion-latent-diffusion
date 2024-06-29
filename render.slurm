#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Mar 15 2023 17:24:35 GMT+1100

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="mld_render"

# The project ID which this job should run under:
#SBATCH --account="punim1485"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4

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
#SBATCH --time=0-04:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

module load CUDA/12.2.0
module load gcc/11.2.0
module load anaconda3/2021.11
eval "$(conda shell.bash hook)"
conda activate mld


# python -m fit --dir results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2023-06-19-04-33-08 --save_folder results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2023-06-19-04-33-08/out --cuda True
# python -m fit --dir results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01 --save_folder results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/out --cuda True
# /data/gpfs/projects/punim1485/lingxiao1/mdm/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=/data/gpfs/projects/punim1485/lingxiao1/mdm/motion-latent-diffusion/results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01 --mode=video --joint_type=HumanML3D


# python -m fit --dir results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01 --save_folder results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/out --cuda True
srun -n 4 python -m fit --dir ./vedios --save_folder ./vedios --cuda True
# /data/gpfs/projects/punim1485/lingxiao1/mdm/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=/data/gpfs/projects/punim1485/lingxiao1/mdm/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000475000_seed10_the_person_walked_forward_and_is_picking_up_his_toolbox/smpl/ --mode=video --joint_type=HumanML3D
srun -n 4 /data/gpfs/projects/punim1485/lingxiao1/mdm/blender-2.93.18-linux-x64/blender --background --python myRender.py -- --cfg=./configs/render_mld.yaml --dir=./vedios --mode=video --joint_type=HumanML3D


# /data/gpfs/projects/punim1485/lingxiao1/mdm/motion-diffusion-model/save/my_humanml_trans_enc_512_my_train3/samples_my_humanml_trans_enc_512_my_train3_000500000_seed10_the_person_walked_forward_and_is_picking_up_his_toolbox/results.npy

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s