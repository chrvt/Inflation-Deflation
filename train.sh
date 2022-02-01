#!/usr/bin/env bash
python main_cluster.py --dataset thin_spiral --latent_distribution exponential --noise_type gaussian --data_dim 2 --hidden_dim 210 --n_hidden 6 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --calculate_KS --n_gradient_steps 50000 --cuda 0 --seed 0 --sig2 10.0
