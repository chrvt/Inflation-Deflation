# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:46:49 2022
Simulating Sigma bounds:
    - for constant curvature manifolds, sigma bounds are analytical
    - else, sigma bound is estimated using samples
    - 2 options: maximum (taking maximum over all), average (taking average over all)
@author: horvat
"""

import sys
import os
import numpy as np


from datasets import load_simulator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Which potential function to approximate.')
parser.add_argument("--latent_distribution", type=str, default=None, help="Latent distribution (for datasets where that is variable)")
parser.add_argument("--noise_type", type=str, default="gaussian", help="Noise type: gaussian, normal (if possible)")
parser.add_argument('--sig2', type=float, default='0.0', help='Noise magnitude')
args = parser.parse_args()


args.dataset = 'hyperboloid'
args.latent_distribution = 'unimodal'

save_path = os.path.join(r'D:\PROJECTS\Inflation_deflation\results',args.dataset,args.latent_distribution)
##------------------------------------
def calculate_nearest_neighbor(X,n_):
    dist_ij = np.zeros([n_,n_])
    dist_min = np.zeros(n_) + 1000
    for i in range(n_):
        for j in range(i):
            dist_ij[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            dist_ij[j,i] = dist_ij[i,j]
            dist_ij[i,i] = 1000
    
    for k in range(n_):
        dist_min[k] = np.min(dist_ij[k,:])

    # import pdb
    # pdb.set_trace() 
           
    return np.mean(dist_min)
            
            
            
simulator = load_simulator(args)
#Hyperparameters
N = 10000 #N_samples

v, theta = simulator._draw_z(N)

#upper bound
bound = 8/(np.cosh(v)**2+np.sinh(v)**2) #2/((36*np.sin(theta-v-np.pi)**2 + 6*np.cos(theta-v-np.pi))*(np.cosh(v)**2+2*np.sinh(v)**2))
avg_bound = np.mean(bound)
max_bound = np.max(bound)

upper_bounds = [avg_bound,max_bound]

#lower bound: average nearest neighbor
data =  simulator._transform_z_to_x(v,theta)
lower_bound = calculate_nearest_neighbor(data,N)

sigma_bounds = {'lower_bound': lower_bound, 'upper_bounds': upper_bounds}

np.save(os.path.join(save_path,'sigma_bounds.npy'),sigma_bounds)


##To Do: for all simulator.calculate_bound, lower bound 1 time enough, embed in img generator?