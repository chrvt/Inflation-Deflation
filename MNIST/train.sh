#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="MNIST"
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1  

cd /storage/homefs/ch19g182/Python/manifold-flow/experiments

###Inflation-Deflation
#python train.py --modelname uniform_deq --sig2 0 --noise_type uniform --dataset mnist --algorithm flow --modellatentdim 8 --outerlayers 5 --innerlayers 5 --lineartransform permutation --outertransform rq-coupling --innertransform rq-coupling --splinerange 10.0 --splinebins 11 --dropout 0.0 --epochs 100  --scheduler_restart 100 --batchsize 100 --lr 3.0e-4 --msefactor 1. --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 5.0 --validationsplit 0.1 --dir /storage/homefs/ch19g182/Python/manifold-flow
#python train.py --modelname gaussian_deq_001 --sig2 0.01 --noise_type gaussian --dataset mnist --algorithm flow --modellatentdim 8 --outerlayers 5 --innerlayers 5 --lineartransform permutation --outertransform rq-coupling --innertransform rq-coupling --splinerange 10.0 --splinebins 11 --dropout 0.0 --epochs 100  --scheduler_restart 100 --batchsize 100 --lr 3.0e-4 --msefactor 1. --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 5.0 --validationsplit 0.1 --dir /storage/homefs/ch19g182/Python/manifold-flow
#python train.py --modelname no_deq --sig2 0.0 --noise_type non --dataset mnist --algorithm flow --modellatentdim 8 --outerlayers 5 --innerlayers 5 --lineartransform permutation --outertransform rq-coupling --innertransform rq-coupling --splinerange 10.0 --splinebins 11 --dropout 0.0 --epochs 100  --scheduler_restart 100 --batchsize 100 --lr 3.0e-4 --msefactor 1. --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 5.0 --validationsplit 0.1 --dir /storage/homefs/ch19g182/Python/manifold-flow


###mflow
python train.py --modelname no_deq --sig2 0.0 --noise_type non --sequential --dataset mnist --algorithm mf --modellatentdim 8 --outerlayers 10 --innerlayers 10 --lineartransform permutation --outertransform rq-coupling --innertransform rq-coupling --splinerange 10.0 --splinebins 11 --dropout 0.0 --epochs 100  --scheduler_restart 100 --batchsize 100 --lr 3.0e-4 --msefactor 1. --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 5.0 --validationsplit 0.1 --dir /storage/homefs/ch19g182/Python/manifold-flow
#python train.py --modelname uniform --sig2 0.0 --noise_type uniform --sequential --dataset mnist --algorithm mf --modellatentdim 8 --outerlayers 14 --innerlayers 6 --lineartransform permutation --outertransform rq-coupling --innertransform rq-coupling --splinerange 10.0 --splinebins 11 --dropout 0.0 --epochs 100  --scheduler_restart 100 --batchsize 100 --lr 3.0e-4 --msefactor 1. --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 5.0 --validationsplit 0.1 --dir /storage/homefs/ch19g182/manifold-flow
#python train.py --modelname gaussian_deq_small --sig2 0.1 --noise_type gaussian --sequential --dataset mnist --algorithm mf --modellatentdim 8 --outerlayers 10 --innerlayers 10 --lineartransform permutation --outertransform rq-coupling --innertransform rq-coupling --splinerange 10.0 --splinebins 11 --dropout 0.0 --epochs 100  --scheduler_restart 100 --batchsize 100 --lr 3.0e-4 --msefactor 1. --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 5.0 --validationsplit 0.1 --dir /storage/homefs/ch19g182/Python/manifold-flow