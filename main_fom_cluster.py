"""
Implementation of Inflation/Deflation method based on Block Neural Autoregressive Flow
http://arxiv.org/abs/1904.04676
"""
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
import sys

import math
import numpy as np
import os
import time
import argparse
import pprint
from functools import partial
from scipy.special import gamma
#import matplotlib
#matplotlib.use('Agg')
import matplotlib
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from tqdm import tqdm
import pdb

from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.special import i0
from scipy import integrate

from datasets import load_simulator, SIMULATORS
from models import BlockNeuralAutoregressiveFlow as BNAF
from plotting import plt_latent_fom as plot_latent

from utils import load_checkpoint

from torch.utils.data import DataLoader
# from utils import create_filename

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# general
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
parser.add_argument('--calculate_KS', action='store_true', help='Caclulates KS_test at the end of the training.')
parser.add_argument('--restore_file', action='store_true', help='Restore model.')
parser.add_argument('--debug', action='store_true', help='Debug mode: for more infos')
#parser.add_argument('--output_dir', default='results\{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--output_dir', default='./results')  #.format(os.path.splitext(__file__)[0]))
parser.add_argument('--cuda', default=0, type=int, help='Which GPU to run on.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
# target density
parser.add_argument('--dataset', type=str, help='Which potential function to approximate.')
parser.add_argument("--latent_distribution", type=str, default=None, help="Latent distribution (for datasets where that is variable)")
# model parameters
parser.add_argument('--data_dim', type=int, default=3, help='Dimension of the data.')
parser.add_argument('--hidden_dim', type=int, default=210, help='Dimensions of hidden layers.')
parser.add_argument('--n_hidden', type=int, default=3, help='Number of hidden layers.')
# training parameters
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--n_gradient_steps', type=int, default=2000, help='Number of steps to train.')
parser.add_argument('--batch_size', type=int, default=200, help='Training batch size.')
parser.add_argument('--lr', type=float, default=1e-1, help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay.')
parser.add_argument('--lr_patience', type=float, default=2000, help='Number of steps before decaying learning rate.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to save model and samples.')
parser.add_argument("--noise_type", type=str, default="gaussian", help="Noise type: gaussian, normal (if possible)")
# For the general model we have to set up the potential
parser.add_argument('--optim', type=str, default='adam', help='Which optimizer to use?')
parser.add_argument('--sig2', type=float, default='0.0', help='Noise magnitude')
parser.add_argument('--mc_samples', type=int, default='1', help='amount of MC samples for noise')


def calculate_KS_stats(args,model,simulator,precision=100,tag='test'):
    logger.info("Start calculating KS statistics")
    model.eval()
    if args.dataset == 'von_Mises_circle':
        prec = precision #precision for integrals
        CDF_original = torch.zeros(prec)
        CDF_model = torch.zeros(prec) 
        for k in range(prec):
            b = -np.pi*(prec-1-k)/(prec-1) + np.pi*k/(prec-1)
            z = torch.linspace(-np.pi,b,1000)
            dens = integrand_circle(z,model,args.datadim, args.pieepsilon**2) 
            CDF_model[k] = torch.trapz(dens, z)
            log_prob = torch.tensor(simulator._log_density(z.cpu().numpy()))
            CDF_original[k] = torch.trapz(torch.exp(log_prob),z)
        CDF_diff = torch.abs(CDF_model-CDF_original)
        KS_test = torch.max(CDF_diff).cpu().detach().cpu().numpy() 
        logger.info("KS statistics: %s", KS_test)
    elif args.dataset in ['torus','swiss_roll','sphere','hyperboloid']:     
        n_pts = precision
        data, latent, true_probs, jacobians, multiplier = simulator.generate_grid(n_pts,mode='data_space')
        
        # print('jacobian',jacobians.shapes)
        
        u1, u2 = latent[0], latent[1]  #theta phi for torus,sphere; u,v for swissroll
        dx = u1[1]-u1[0]      #for integration
        dy = u2[1]-u2[0]      #for integration
        
        # data = simulator._transform_x_to_z(data)
        u1_mesh, u2_mesh = np.meshgrid(u1, u2 )
        latent = np.stack((u1_mesh.flatten(), u2_mesh.flatten()), axis=1)
        # pdb.set_trace()
        # latent = np.concatenate([u1_mesh.reshape([n_pts,1]),u2_mesh.reshape([n_pts,1])],axis=1)
        # pdb.set_trace()
        xx = torch.tensor(latent).to(args.device, torch.float)
        logprobs = []
        with torch.no_grad():
            model.eval()
            for xx_k in xx.split(args.batch_size, dim=0):
                z_k, logdet_k = model(xx_k)	
                logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
            logprobs = torch.cat(logprobs, 0) 
            
            probs_flow = torch.exp(logprobs) #s+0.5*np.log(2*np.pi*args.sig2))  #gaussian noise assumption
            # pdb.set_trace()
            probs_flow = probs_flow.view(n_pts,n_pts)
            
            density_flow = probs_flow #* torch.abs(torch.tensor(jacobians)).to(args.device, torch.float) 
            print('true probs integral ',true_probs.sum()*dx*dy)
            print('flow probs integral ',density_flow.sum()*dx*dy)
            # import pdb
            # pdb.set_trace() .T
            diff = density_flow - torch.tensor(true_probs).view(n_pts,int(n_pts*multiplier)).to(args.device, torch.float) 
            CDF_diff = torch.zeros([n_pts,n_pts*multiplier])
            for i in range(n_pts):
                for j in range(n_pts*multiplier):
                    summe = torch.sum(diff[0:i,0:j])
                    CDF_diff[i,j] = torch.abs(summe)
            
            KS_test = (torch.max(CDF_diff)*(dx*dy)).cpu().detach().cpu().numpy() 
            logger.info("KS statistics: %s", KS_test)   
        
    elif args.dataset in ['thin_spiral']:
        def integrand(x, model, simulator):
            data = simulator._transform_z_to_x(x,mode='test')
            xx_ = torch.tensor(data).to(args.device, torch.float)
            model.eval()
            with torch.no_grad():
                z_, logdet_ = model(xx_)	
                log_prob = torch.sum(model.base_dist.log_prob(z_)+logdet_,dim=1) +0.5*np.log(2*np.pi*args.sig2)  
                
                c1 = 540 * 2* np.pi / 360
                r = np.sqrt(x) * c1
                jacobians = ((1+r**2)/r**2) * c1**4 / 36   
                density = torch.exp(log_prob) * torch.sqrt(torch.tensor(jacobians).to(args.device, torch.float))
            return density
            
        prec = precision #precision for integrals
        CDF_original = torch.zeros(prec)
        CDF_model = torch.zeros(prec) 
        
        a_, b_ = 0, 2.5
        dx = (b_ - a_)/1000
        dy = 1
        for k in range(1,prec+1):
            b = a_*(prec-k)/(prec) + b_*k/(prec)
            
            z_np = np.linspace(a_,b,1000+1)[1:] #.to(args.device, torch.float)
            z_torch = torch.tensor(z_np).to(args.device, torch.float)
            
            dens = integrand(z_np,model,simulator) 
            CDF_model[k-1] = torch.trapz(dens, z_torch)
            true_probs = torch.tensor(simulator._density(np.abs(z_np))).to(args.device, torch.float)
            CDF_original[k-1] = torch.trapz(true_probs,z_torch)
        
        print('true probs integral ',true_probs.sum()*dx*dy)
        print('flow probs integral ',dens.sum()*dx*dy)  
        
        CDF_diff = torch.abs(CDF_model-CDF_original)
        KS_test = torch.max(CDF_diff).cpu().detach().cpu().numpy() 
        logger.info("KS statistics: %s", KS_test)
        # data, latent, true_probs, jacobians, multiplier = simulator.generate_grid(n_pts,mode='data_space')
        

        # latent_test = simulator.load_latent(train=False,dataset_dir=create_filename("dataset", None, args))
        
        
        # order = np.argsort(latent_test)
        # latent_test = latent_test[order] #sort: lowest to highest
        # z = np.sqrt(latent_test) * 540 * (2 * np.pi) / 360 
        # d1x = - np.cos(z) * z     #d/dz = -cos(z) +sin(z)z    --> ||grad||^2 = cos^2 - cos sin z + sin^2 z^2 +
        # d1y =   np.sin(z) * z     #d/dz =  sin(z) +cos(z)z   --->             sin^2 + cos sin z + cos^2 z^2
        # x = np.stack([ d1x,  d1y], axis=1) / 3  #      
        
        # x = torch.tensor(data).to(args.device, torch.float)
        # logprobs =  []
        # # with torch.no_grad():
        # model.eval()
        # params_ = None
        # step = 0
        # with torch.no_grad():
        #     model.eval()
        #     for xx_k in xx.split(args.batch_size, dim=0):
        #         z_k, logdet_k = model(xx_k)	
        #         logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
        #     logprobs = torch.cat(logprobs, 0) 
            
        #     density_flow = torch.exp(logprobs+0.5*np.log(2*np.pi*args.sig2)) * torch.abs(torch.tensor(jacobians))
            
            
        #     logger.info("Calculated latent probs, KS not implemented")  
        #     np.save(create_filename("results", "latent_probs", args), logprobs.detach().cpu().numpy())
            
    elif args.dataset in ['two_thin_spirals']:
        def integrand(x, model, simulator):
            data = np.sign(x).reshape([x.shape[0],1]) * simulator._transform_z_to_x(np.abs(x),mode='test')
            xx_ = torch.tensor(data).to(args.device, torch.float)
            model.eval()
            with torch.no_grad():
                z_, logdet_ = model(xx_)	
                log_prob = torch.sum(model.base_dist.log_prob(z_)+logdet_,dim=1) +0.5*np.log(2*np.pi*args.sig2)  
                
                c1 = 540 * 2* np.pi / 360
                r = np.sqrt(np.abs(x)) * c1
                r[r==0]=1/10000
                jacobians = ((1+r**2)/r**2) * c1**4 / 36   
                
                density = torch.exp(log_prob) * torch.sqrt(torch.tensor(jacobians).to(args.device, torch.float))
            return density
            
        prec = 100 #precision for integrals
        CDF_original = torch.zeros(prec)
        CDF_model = torch.zeros(prec) 
        
        a_, b_ = -2.5, 2.5
        dx = (b_ - a_)/1000
        dy = 1
        for sign in range(2):
            for k in range(1,prec+1):
                b = a_*(prec-k)/(prec) + b_*k/(prec)
                
                z_np = np.linspace(a_,b,1000+1)[1:] #.to(args.device, torch.float)
                z_torch = torch.tensor(z_np).to(args.device, torch.float)
                
                dens = integrand(z_np,model,simulator) 
                CDF_model[k-1] = torch.trapz(dens, z_torch)
                true_probs = torch.tensor(0.5*simulator._density(z_np)).to(args.device, torch.float)
                CDF_original[k-1] = torch.trapz(true_probs,z_torch)
        
        
        
        print('true probs integral ',true_probs.sum()*dx*dy)
        print('flow probs integral ',dens.sum()*dx*dy)  
        
        CDF_diff = torch.abs(CDF_model-CDF_original)
        KS_test = torch.max(CDF_diff).cpu().detach().cpu().numpy() 
        logger.info("KS statistics: %s", KS_test)
    
    
    else:
        KS_test = 0
        logger.info("KS not implemented for %s dataset", args.dataset)  
        
    np.save(os.path.join(args.output_dir, 'KS_'+tag+'.npy'), KS_test)
    np.save(os.path.join(args.output_dir, 'density_flow.npy'), density_flow.detach().cpu().numpy())

def compute_kl_pq_loss(model, batch):
    """ Compute BNAF eq 2 & 16:
    KL(p||q_fwd) where q_fwd is the forward flow transform (log_q_fwd = log_q_base + logdet), p is the target distribution.
    Returns the minimization objective for density estimation (NLL under the flow since the entropy of the target dist is fixed wrt the optimization) """
    z_, logdet_ = model(batch)
    log_probs = torch.sum(model.base_dist.log_prob(z_)+ logdet_, dim=1) 
    return -log_probs.mean(0)  
    # return -log_probs/mc_samples   


# --------------------
# Validating
# --------------------
from torch.utils.data import Dataset

class NumpyValidationSet(Dataset):
    def __init__(self, x, device='cpu', dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.x = torch.from_numpy(x)

    def __getitem__(self, index):
        x = self.x[index, ...]
        return x.to(self.device,self.dtype)

    def __len__(self):
        return self.x.shape[0]
    
with torch.no_grad():   
    def validate_flow(model, val_loader, loss_fn):
        losses_val = 0
        
        for batch_data in val_loader:
            args.step += 1
            model.eval()
            
            batch_loss = loss_fn(model, batch_data)  
            
            losses_val += batch_loss.item()
        return losses_val/len(val_loader)



# --------------------
# Training
# --------------------

def train_flow(model, simulator, loss_fn, optimizer, scheduler, args, double_precision=False):
    losses = []
    best_loss = np.inf
    dtype = torch.double if double_precision else torch.float
    
    z1, z2 = simulator._draw_z(1000)
    z = np.concatenate([z1.reshape([1000,1]),z2.reshape([1000,1])],axis=1)
    
    validation_latents = z #.sample(1000)
    # validation_latents = simulator._transform_x_to_z(validation_set)
    validation_set = NumpyValidationSet(validation_latents,device=args.device,dtype=dtype)
    
    val_loader = DataLoader(
    validation_set,
    shuffle=True,
    batch_size=args.batch_size,
    # pin_memory=self.run_on_gpu,
    #num_workers=n_workers,
            )
    
    # pdb.set_trace() 
    with tqdm(total=args.n_gradient_steps, desc='Start step {}; Training for {} steps'.format(args.step,args.n_gradient_steps)) as pbar:
        for step in range(args.step+1,args.n_gradient_steps):
            args.step += 1
            
            model.train()
            #batch = simulator.sample(args.batch_size) 
            #latent = simulator._transform_x_to_z(batch)  #(simulator._draw_z(args.batch_size)).reshape([args.batch_size,1])._transform_x_to_z(batch)
            
            z1, z2 = simulator._draw_z(args.batch_size)
            latent = np.concatenate([z1.reshape([args.batch_size,1]),z2.reshape([args.batch_size,1])],axis=1)
            # latent = (simulator._draw_z(args.batch_size)).reshape([args.batch_size,simulator.latent_dist()])
            
            z = torch.from_numpy(latent).to(args.device, dtype)
            
            loss = loss_fn(model, z)  
            
            # if torch.isnan(loss).any() or torch.isinf(loss).any():
            #     import pdb
            #     pdb.set_trace()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.optim == 'adam':
                scheduler.step(loss) #x
            
            pbar.set_postfix(loss = '{:.3f}'.format(loss.item()))
            pbar.update()
            
            if step %10000 == 0:
                # scheduler.step()
                # save model
                checkpoint = {'step': args.step,
                              'state_dict': model.state_dict(),
                              'optimizer' : optimizer.state_dict(),
                              'scheduler' : scheduler.state_dict()}
                torch.save(checkpoint , os.path.join(args.output_dir, 'checkpoint.pt'))
                # calculate_KS_stats(args,model,simulator)
                
            if step %100 == 0 and step > 900: #40000:
                val_loss =  validate_flow(model, val_loader, loss_fn)
                    
                if val_loss < best_loss:
                    best_loss = val_loss
                    # save model
                    checkpoint = {'step': args.step,
                                  'state_dict': model.state_dict()}
                    torch.save(checkpoint , os.path.join(args.output_dir, 'checkpoint_best.pt'))
                    # if args.calculate_KS:
                    #     calculate_KS_stats(args,model,simulator)
                
                
            # if step%1000 == 0 or args.step == 1:
            #     # pdb.set_trace()
            #     plot_latent(args.output_dir,simulator,model,i_epoch=step,n_grid=100,dtype=dtype,device=args.device)
            #     if args.calculate_KS:
            #         calculate_KS_stats(args,model,simulator)

#to do:  KS statistics


if __name__ == '__main__':
    warnings.simplefilter("once")
    args = parser.parse_args()
    
    
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")
    
    param_string = 'sig2_'+str(args.sig2)+'_seed_'+str(args.seed)
    
    original_output_dir = os.path.join(args.output_dir, args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.latent_distribution,args.noise_type, param_string) 
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)
   
    simulator = load_simulator(args)
    loss_fn = compute_kl_pq_loss

    model = BNAF(simulator.latent_dim(), args.n_hidden, args.hidden_dim, use_batch_norm = False).to(args.device)
    
    # save settings
    config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__)) + \
             'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
             'Model:\n{}'.format(model)
    config_path = os.path.join(args.output_dir, 'config.txt')
    if not os.path.exists(config_path):
        with open(config_path, 'a') as f:
            print(config, file=f)
    
    if args.train:
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_gradient_steps)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.lr_patience, verbose=True)
        elif args.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.lr_patience, verbose=True)
        else:
            raise RuntimeError('Invalid `optimizer`.')
        if args.restore_file:
            
            from utils import load_checkpoint
            model, optimizer, scheduler, args.step = load_checkpoint(args.output_dir,model,optimizer,scheduler)
            # pdb.set_trace()
            # optim_checkpoint = torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device)
            # optimizer.load_state_dict(optim_checkpoint['optimizer'])
            # scheduler.load_state_dict(optim_checkpoint['scheduler'])
        train_flow(model, simulator, loss_fn, optimizer, scheduler, args)

    logger.info("Final evaluation: ")
    model, optimizer, scheduler, args.step = load_checkpoint(args.output_dir,model,best=True)
    if args.plot:
        plot_latent(args.output_dir,simulator,model,i_epoch=1337,n_grid=500,dtype=torch.float,device=args.device)
    calculate_KS_stats(args,model,simulator,precision=500,tag='final')
    logger.info("All done...have an amazing day!")
    
