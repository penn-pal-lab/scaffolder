#!/usr/bin/env bash
## dj-partition settings
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --array=0-3
#SBATCH --mem=50G 
#SBATCH --exclude=kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,dj-2080ti-0.grasp.maas

# batch partition settings
##SBATCH --partition=batch
##SBATCH --gres=gpu:1
##SBATCH --qos=normal
##SBATCH --time=24:00:00
##SBATCH --array=5-9
##SBATCH --mem=27G 

QOS="dj-med"
if [[ "$QOS" == "dj-med" ]]; then
    TIMEOUT=11h
elif [[ "$QOS" == "dj-high" ]]; then
    TIMEOUT=23h
else
    TIMEOUT=23h
fi

MODEL_SIZE="large"

EXPERIMENT="asym-handpenrgb-explore-DecodeAll-500kbuffer-${MODEL_SIZE}"; 

export WANDB_RUN_GROUP=$EXPERIMENT; 
export WANDB__SERVICE_WAIT=600;
export MUJOCO_GL=egl; 
export CUDA_VISIBLE_DEVICES=0; 
export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES;

SEED=$SLURM_ARRAY_TASK_ID
conda activate dreamerv3

CONFIG="gymnasium_asymhandpenrgb,${MODEL_SIZE}"

timeout $TIMEOUT python -u dreamerv3/train.py --logdir ~/logdir/${EXPERIMENT}_s$SEED --configs $CONFIG --seed $SEED >> "${EXPERIMENT}_s${SEED}.out"


if [[ $? == 124 ]]; then 
  echo "Asking slurm to requeue this job.\n" >> "${EXPERIMENT}_s${SEED}.out"
  scontrol requeue $SLURM_JOB_ID
fi
exit 0