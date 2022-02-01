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
from pathlib import Path

from datasets import load_simulator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Which potential function to approximate.')
parser.add_argument("--latent_distribution", type=str, default=None, help="Latent distribution (for datasets where that is variable)")
parser.add_argument("--noise_type", type=str, default="gaussian", help="Noise type: gaussian, normal (if possible)")
parser.add_argument('--sig2', type=float, default='0.0', help='Noise magnitude')
args = parser.parse_args()

manifolds = ['swiss_roll','sphere','torus','hyperboloid','thin_spiral','spheroid','stiefel'] # ['torus','hyperboloid',,'thin_spiral',,'spheroid']
latents = ['mixture','unimodal','correlated','exponential']
save = True

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
import pdb

count_mani = -1
for manifold in manifolds:
    print('manifold ',manifold)
    for latent_distribution in latents:
        
        args.dataset = manifold
        args.latent_distribution = latent_distribution

        save_path = os.path.join(r'D:\PROJECTS\Inflation_deflation\results',args.dataset,args.latent_distribution)
        
        if not Path(save_path).is_dir():
            continue                
        print('--latent ',latent_distribution)
               
        simulator = load_simulator(args)
        #Hyperparameters
        N = 10000 #N_samples
        
        if manifold not in ['thin_spiral','stiefel']:
            z1, z2 = simulator._draw_z(N)
        else: 
            z1 = simulator._draw_z(N)
            z2 = None
        
        
        #upper bound
        bound = simulator.calculate_sigma_bound(z1,z2)
        gauss = simulator.calculate_gauss_curvature(z1,z2)
        # print(' --gauss min',1/np.min(np.abs(gauss)))
        # print(' --gauss mean',1/np.mean(np.abs(gauss)))
        # print(' --gauss max',1/np.max(np.abs(gauss)))
        # print('latent',latent_distribution)
        # print('bound',bound)
        #8/(np.cosh(v)**2+np.sinh(v)**2) #2/((36*np.sin(theta-v-np.pi)**2 + 6*np.cos(theta-v-np.pi))*(np.cosh(v)**2+2*np.sinh(v)**2))
        min_bound = np.min(np.abs(bound))
        avg_bound = np.mean(np.abs(bound))
        max_bound = np.max(np.abs(bound))
        # print('  --min_bound ',min_bound)
        # print('  --avg_bound ',avg_bound)
        # print('  --max_bound ',max_bound)
        
        if np.mean(gauss)>0:
            up = np.stack((np.abs(bound),1/np.abs(gauss)),axis=1)
            upper = np.min(up,axis=1)
        else:
            upper = np.abs(bound)
        # pdb.set_trace()
        # print('just gauss', np.mean( 1/np.abs(gauss)))
        print('average bound', np.mean(upper))
        # print('maximum bound', np.max(upper))
        
        upper_bounds = [avg_bound,max_bound]
        
        #lower bound: average nearest neighbor
        if manifold not in ['thin_spiral','stiefel']:
            data =  simulator._transform_z_to_x(z1,z2,mode='test')
        else: 
            data =  simulator._transform_z_to_x(z1,mode='test')
            
        # lower_bound = calculate_nearest_neighbor(data,N)
        # print('    --lower_bound ',lower_bound)
        
        sigma_path = os.path.join(save_path,'sigma_bounds.npy')
        sigma_bounds = np.load(sigma_path,allow_pickle=True).item()
        print('gauss in loaded file ',sigma_bounds['gauss_curvature'])
        sigma_bounds['gauss_curvature'] = np.mean(upper) #np.mean(np.abs(gauss))
        
        #sigma_bounds = {'lower_bound': lower_bound, 'upper_bounds': upper_bounds, 'gauss_curvature': np.mean(gauss)}
        
        # import pdb
        # pdb.set_trace() 
        if save:
            np.save(os.path.join(save_path,'sigma_bounds.npy'),sigma_bounds)  #'sigma_bounds.npy'
            

##To Do: for all simulator.calculate_bound, lower bound 1 time enough, embed in img generator?