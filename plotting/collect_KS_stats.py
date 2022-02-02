# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:20:52 2021
KS aggregator
@author: horvat
"""
import numpy as np
import os, re
from shutil import copyfile

root_dir = r'./results/stiefel'  #swiss_roll, torus, sphere, spheroid, thin_spiral, hyperboloid


for density in ['mixture','correlated','unimodal','exponential']: # ,'mixture' 'exponential', 'unimodal' 'unimodal', 'exponential',
    print(density)
    data_path = os.path.join(root_dir,density) # 
    
    for noise_type in ['gaussian','normal']:
        KS_star = np.inf
        seed_star = -1
        sig_star = -1
        
        print('-',noise_type)
        rootdir = os.path.join(data_path,noise_type)
        
        sigmas = [0.0,1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07,1e-06,5e-06,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0,2.0, 3.0, 4.0,  6.0 , 8.0, 10.0]
        KS_stats_ = np.zeros([len(sigmas),3]) -1
        
        for subdir, dirs, files in os.walk(rootdir):
            model_dir = os.path.join(rootdir,subdir)
            if model_dir != rootdir and model_dir[-4:] != 'logs':
                # print('modeldir',model_dir)
                for subdir, dirs, files in os.walk(model_dir):
                    seed = int(model_dir[-1])  
                    sig2 = float(re.search('sig2_(.*)_seed',model_dir).group(1))
                    for file in files:
                        file_path = os.path.join(model_dir,file)
                        # if file[-3:] == 'pdf':
                        #       os.remove(file_path)
                        if file == 'KS_final.npy':
                            KS = np.load(file_path)
                            KS_stats_[sigmas.index(sig2),seed] = KS
                            if KS < KS_star:
                                KS_star = KS
                                seed_star = seed
                                sig_star = sig2
        print(KS_stats_)
        KS_stats = np.mean(KS_stats_,axis=1)
        KS_std = np.std(KS_stats_,axis=1)
        #move density_flow.
        # sig_star = sigmas[np.argmin(KS_stats)]
        model_string = 'sig2_'+str(sig_star)+'_seed_'+str(seed_star)
        density_dir = os.path.join(rootdir,model_string)
        density_flow_dir = os.path.join(density_dir,'density_flow.npy')
        density_target_dir = os.path.join(data_path,noise_type+'_density_flow.npy')
        copyfile(density_flow_dir, density_target_dir)
        # print('-- best KS=',np.min(KS_stats),' for sig2=',sig_star) KS_star
        print('-- best KS=', KS_star,' for sig2=',sig_star)
        np.save(os.path.join(data_path,'KS_'+noise_type)+'_mean.npy',KS_stats)
        np.save(os.path.join(data_path,'KS_'+noise_type)+'_std.npy',KS_std)
        np.save(os.path.join(data_path,'stars_'+noise_type)+'.npy',[sig_star,KS_star])
        del KS_star, seed_star, sig_star, KS_stats, KS_std, model_string, density_dir, density_flow_dir, density_target_dir
                    
        



