#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="sphere"
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=16G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1 
#SBATCH --array=1-27 ##loop over \sigma^2 values


cd /storage/homefs/ch19g182/Python/inflation_deflation/main/experiments


nvcc --version
nvidia-smi

python main_cluster.py  -c configs/sphere_correlated_NID.config