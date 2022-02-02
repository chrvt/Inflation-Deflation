import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import ListedColormap

from basic_units import radians

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.gridspec as gridspec
import os

from utils import make_grid, set_ticks_and_labels, no_ticks

def plt_errorbar(xplt,yplt,yerr,label=None,lw=2,c='k',marker='o',alpha=0.3,ls=None):
    ax.plot(xplt,yplt,lw=lw,c=c,marker=marker,ls=ls,label=label)
    ax.fill_between(xplt,yplt-yerr,yplt+yerr,color=c,alpha=alpha)

# import matplotlib as mpl
# upper = mpl.cm.jet(np.arange(256))
# # set lower part: 1 * 256/4 entries
# # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
# lower = np.ones((int(256/4),4)) * 0.8
# # - modify the first three columns (RGB):
# #   range linearly between white (1,1,1) and the first color of the upper colormap
# for i in range(3):
#   lower[:,i] = np.linspace(0.8, upper[0,i], lower.shape[0])
# lower[0:8,:]=1
# # combine parts of colormap
# cmap = np.vstack(( lower, upper ))
# # convert to matplotlib colormap
# cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

# def normalize(inputs_):
#     max_input = np.max(inputs_)
#     inputs_norm = inputs_/max_input  #<---to [0,1]
#     return inputs_norm*max_value #<---to [0,max_value]

import matplotlib
from matplotlib.colors import Normalize as norm
norm = matplotlib.colors.Normalize()
    
def plt_latent(latent, probs, title, x_shift=.5, title_shift = 1.0):
    ax.pcolormesh(latent[0], latent[1], probs, cmap = plt.cm.jet  , shading='nearest')  #,shading='auto') #,xunits=radians,yunits=radians)   
    ax.set_title(title, x = x_shift, y=title_shift)

def plt_latent_spiral(latent, probs, title, x_shift=.5, title_shift = 1.0, color = 'blue',label=None, linewidth=1):
    ax.plot(latent, probs, c=color,label=label,linewidth=linewidth)  #,shading='auto') #,xunits=radians,yunits=radians)   
    ax.set_title(title, x = x_shift, y=title_shift)            

    
base_path = r'./results'
# plt.close()
save = True  #false
cheat = False
manifold = 'stiefel'  # 'swiss_roll' 'hyperboloid'  torus, sphere, swiss_roll spheroid
latent_distribution = 'mixture' #'mixture' ''correlated', 'exponential', 'unimodal'

data_path = os.path.join(base_path,manifold,latent_distribution)

stars_gauss = np.load(os.path.join(data_path,'stars_gaussian.npy'))
stars_normal = np.load(os.path.join(data_path,'stars_normal.npy'))
    
baseline = 'NID '+r'$\sigma^2=$' + str(stars_normal[0])  #str(0.01)  #
FG_title = 'IID '+r'$\sigma^2=$' +str(stars_gauss[0]) #+r'$\sigma^2=0.01$'


save_path = r'./figures'

#--------
SMALL_SIZE = 10
MEDIUM_SIZE = 30
BIGGER_SIZE = 40
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
 #--------
 
    
###########################################
####    GRID   ############################
n_pts = 1000
# if manifold == 'thin_spiral':
#     n_pts = 1000
# else: 
#     n_pts = 500
latent, latent_mesh, data, probs_true, dx, dy =  make_grid(n_pts,manifold,latent_distribution)

outer = gridspec.GridSpec(1, 4, wspace=0.2, hspace=0.3)
fig = plt.figure(figsize=(42, 10))
fig.clf()

#######################################################
####    target latent density  ########################
# ax = fig.add_subplot(outer[0,0]) 
# if manifold in ['thin_spiral','stiefel']:
#     plt_latent_spiral(latent, probs_true, title=r'$\pi(z)$')
# else:
#     plt_latent(latent, probs_true, title=r'$\pi(z)$')
# ax.set_xlabel(r'$z_1$')     
# # max_value = probs_true.max()
# set_ticks_and_labels(manifold,ax,only_x=False)

###########################################
####    normal noise   ########################
ax = fig.add_subplot(outer[0, 0]) 
if manifold == 'sphere':
    probs_normal = np.load(os.path.join(data_path,'normal_density_flow.npy')).T
else:
    probs_normal = np.load(r'D:\PROJECTS\Inflation_deflation\results\stiefel\mixture\normal_density_flow.npy')[:-1] #001
    #np.load(r'D:\PROJECTS\Inflation_deflation\results\stiefel\mixture\normal\sig2_0.01_seed_1\density_flow.npy')[:-1] #np.load(os.path.join(data_path,'normal_density_flow.npy'))[:-1]
  # =np.load(os.path.join(data_path,'sphere_3D/mixture/4mix_FOM/50K_MC_samples1/FOM_density_flow.npy'))#
if cheat:
    probs_normal[probs_normal>np.max(probs_true)] = np.max(probs_true)
if manifold in ['thin_spiral','stiefel']:
    plt_latent_spiral(latent, probs_normal, title=baseline, color='orange',label='NID', linewidth=2.5)
    plt_latent_spiral(latent, probs_true, title=baseline,label='true')
    ax.legend()
    ax.legend(loc=1)
    set_ticks_and_labels(manifold,ax,only_x=False)
else:
    plt_latent(latent, probs_normal, title=baseline)
    set_ticks_and_labels(manifold,ax,only_x=True)
ax.set_xlabel(r'$z_1$') 

###########################################
####    full Gaussian   ########################
ax= fig.add_subplot(outer[0, 1]) 
if manifold == 'sphere':
    probs_gauss = np.load(os.path.join(data_path,'gaussian_density_flow.npy')).T
else:
    probs_gauss = np.load(os.path.join(data_path,'gaussian_density_flow.npy'))[:-1]
if cheat:
    probs_gauss[probs_gauss>np.max(probs_true)] = np.max(probs_true)
if manifold in ['thin_spiral','stiefel']:
    plt_latent_spiral(latent, probs_gauss, title=FG_title, color='orange',label='IID',linewidth=2.5)
    plt_latent_spiral(latent, probs_true, title=FG_title, label='true')
    ax.legend()
    ax.legend(loc=1)
    set_ticks_and_labels(manifold,ax,only_x=True)
else:
    plt_latent(latent, probs_gauss, title=FG_title)
    set_ticks_and_labels(manifold,ax,only_x=True)
ax.set_xlabel(r'$z_1$') 

#######################################################
# ####    target data density  ##########################
# if manifold == 'sphere':
#     ax = fig.add_subplot(outer[1,0],projection="mollweide")
#     latent_ = [latent[0]-np.pi, latent[1]-np.pi/2]
#     plt_latent(latent_, probs_true, title=r'$p^*(x)$', title_shift=1.03)
#     no_ticks(ax)
# elif manifold == 'torus':
#     ax = fig.add_subplot(outer[1,0],projection='3d') 
#     ax.set_zlim(-3,3)
#     import matplotlib
#     from matplotlib.colors import Normalize as norm
#     norm = matplotlib.colors.Normalize()
#     ax.plot_surface(data[:,1].reshape([n_pts,n_pts]), data[:,0].reshape([n_pts,n_pts]), data[:,2].reshape([n_pts,n_pts]), facecolors = plt.cm.jet(norm(probs_true)), antialiased=True) 
#     ax.set_title(r'$p^*(x)$', y=.8)
#     no_ticks(ax,threeD=True)
#     ax.margins(x=-0.49, y=-0.49) 
# elif manifold == 'swiss_roll':
#     ax = fig.add_subplot(outer[1,0],projection='3d') 
#     ax.scatter(data[:,0], data[:,1], data[:,2], c=probs_true, cmap=plt.cm.jet) 
#     ax.set_title(r'$p^*(x)$', y=0.95)
#     no_ticks(ax,threeD=True)
#     ax.margins(x=-0.49, y=-0.49) 
# elif manifold == 'hyperboloid':
#     import torch
#     def to_poincare(x):
#         return torch.div(x[..., 1:], (1+x[...,0]).reshape(x.shape[0], 1))  
#     ax = fig.add_subplot(outer[1,0]) 
#     xy_poincare = to_poincare(torch.from_numpy(data).float()) 
#     x = xy_poincare[:,0].cpu().numpy().reshape(n_pts, n_pts)
#     y = xy_poincare[:,1].cpu().numpy().reshape(n_pts, n_pts)
#     plt_latent([x,y], probs_true, title=r'$p^*(x)$', x_shift=0.1, title_shift=.95)
#     #ax.pcolormesh(y, x, probs_true, cmap=plt.cm.jet, shading='auto')  
#     ax.axis('off')
#     no_ticks(ax,threeD=False)
 
# elif manifold == 'spheroid':
#     ax = fig.add_subplot(outer[1,0],projection='3d') 
#     import matplotlib
#     from matplotlib.colors import Normalize as norm
#     norm = matplotlib.colors.Normalize()
#     # probs_true[probs_true <1e-01] = 1e-01
#     ax.plot_surface(data[:,0].reshape([n_pts,n_pts]), data[:,1].reshape([n_pts,n_pts]), data[:,2].reshape([n_pts,n_pts]), facecolors = plt.cm.jet(norm(probs_true.reshape([n_pts,n_pts]))), antialiased=True)
#     ax.set_title(r'$p^*(x)$', y=.9,x=0)
#     no_ticks(ax,threeD=True)
#     if latent_distribution == 'correlated':
#         ax.view_init(-30,25)
    
# elif manifold == 'thin_spiral':
#     ax = fig.add_subplot(outer[1,0]) 
#     # create colormap
#     # ---------------
    
#     # create a colormap that consists of
#     # - 1/5 : custom colormap, ranging from white to the first color of the colormap
#     # - 4/5 : existing colormap
    
#     # set upper part: 4 * 256/4 entries
#     upper = mpl.cm.jet(np.arange(256))
#     # set lower part: 1 * 256/4 entries
#     # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
#     lower = np.ones((int(256/4),4)) * 0.8
#     # - modify the first three columns (RGB):
#     #   range linearly between white (1,1,1) and the first color of the upper colormap
#     for i in range(3):
#       lower[:,i] = np.linspace(0.8, upper[0,i], lower.shape[0])
#     lower[0:8,:]=1
#     # combine parts of colormap
#     cmap = np.vstack(( lower, upper ))
#     # convert to matplotlib colormap
#     cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
    
#     c1 = 540 * 2* np.pi / 360
#     r = np.sqrt(latent) * c1
#     jac_det = ((1+r**2)/r**2) * c1**4 / 36   
    
#     p_max = np.max(probs_true  / np.sqrt(jac_det))  #we are going from latent to data space, therefore divide by jac_det
#     normalize = matplotlib.colors.Normalize(vmin=0, vmax=p_max,clip=True)  #10 is max value if p*(x)
#     probs_data = normalize(probs_true  / np.sqrt(jac_det) )
#     for i in range(n_pts-1):
#         # z2 = x[i,0]**2+x[i,1]**2
#         p_ = probs_data[i] 
#         ax.plot(data[i:i+2,0], data[i:i+2,1], color=cmap(p_),linewidth=2)
#     plt.title(r'$p^*(x)$',x=0.2,y=0.8)
#     ax.axis('off')
        



################################################
####    KS plot         ########################
 
total = 27  #max 15 es fehlen: 0.00005, 0.0k003, 0.001, 0.01
x_axe = np.array([0.0,1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07,1e-06,5e-06,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0,2.0, 3.0, 4.0,  6.0 , 8.0, 10.0])
    
ax = fig.add_subplot(outer[0,2:5]) 

losses_gaussian_avg = np.load(os.path.join(data_path,'KS_gaussian_mean.npy'))   #'sphere__KS_total_gaussian_avg.npy')
losses_gaussian_std = np.load(os.path.join(data_path,'KS_gaussian_std.npy')) #np.zeros(len(losses_gaussian_avg))
#losses_gaussian_std = np.load(os.path.join(data_path,'sphere_KS_plots/sphere_KS_total_gaussian_std.npy'))
losses_normal_avg = np.load(os.path.join(data_path,'KS_normal_mean.npy'))
losses_normal_std = np.load(os.path.join(data_path,'KS_normal_std.npy')) #np.zeros(len(losses_normal_avg))
# losses_normal_std = np.load(os.path.join(data_path,'sphere_KS_plots/sphere_KS_total_normal_std.npy')) 

plt_errorbar(x_axe,losses_normal_avg,losses_normal_std,c='blue', label='NID')    #.flatten()
plt_errorbar(x_axe,losses_gaussian_avg,losses_gaussian_std,c='red', label='IID') #.flatten()
 
#FOM baseline
FOM_path = os.path.join(data_path,'FOM','sig2_0.0_seed_11')
KS_FOM = np.load(os.path.join(FOM_path,'KS_final.npy')) #np.array(0.001) #
ax.plot(x_axe, np.ones(total)*KS_FOM , linestyle='solid', color='black', linewidth=1.5, label='FOM')

##table
clust_data = np.zeros((3,1))
clust_data[0,0] = round(stars_normal[1],5)
clust_data[1,0] = round(stars_gauss[1],5)
clust_data[2,0] = round(KS_FOM.item(),5)
collabel=[r'KS*']
rowlabel=("NS", "FG", "FOM") #,colours=['blue','red','black'],
ax.table(cellText=clust_data,cellLoc='left',rowLabels=rowlabel,colLabels=collabel,loc='center',colWidths=[0.1]).scale(1, 4)

ax.legend()
ax.legend(loc=1)

unit = 0.5
y_tick = np.arange(0, 1+unit, unit)
y_label = [r"0", r"$0.5$", r"$1$"]
ax.set_yticks(y_tick*1)
ax.set_yticklabels(y_label)

ax.set_xlim(2e-09,10)
ax.set_ylim(-0.05,1.1)
ax.margins(0.05)
ax.set_xscale('log')
ax.set_ylabel('KS',rotation=0)  
ax.yaxis.set_label_coords(-0.03,1.0)  
ax.xaxis.set_label_coords(0.5,-0.06)  
# ax.yaxis.set_label_coords(-0.03,0.9)
ax.set_xlabel(r'$\sigma^2$')
 
print('data_path', data_path)
sigma_path = os.path.join(data_path,'sigma_bounds.npy')
if os.path.exists(sigma_path): #Path(sigma_path).is_dir():
    # print('sigma bounds!')
    sigma_bounds = np.load(sigma_path,allow_pickle=True).item()
    
    if manifold not in ['thin_spiral']:
        lower_bound = sigma_bounds['lower_bound']**2 / 3 / 1#/ 2
    else: 
        lower_bound = sigma_bounds['lower_bound']**2 / 2 / 1 #/ 2
        
    upper_bound_ = sigma_bounds['upper_bounds'][1]  #0...average, 1...maximum
    
    upper_bound = sigma_bounds['gauss_curvature'] #**2
    print('gauss', upper_bound)
    
    # if gauss == 0:
    #     upper_bound = upper_bound_
    # else:
    #     upper_bound = np.min([upper_bound_,gauss])
    # import pdb
    # pdb.set_trace()
    
    ax.axvline(x=lower_bound, linestyle = 'dashed',linewidth=1.5, color = 'black') 
    ax.axvline(x=upper_bound, linestyle = 'dashed',linewidth=1.5, color = 'black') 
    ax.set_xscale('log')
else: 
    print('sigma bounds not found')
    ax.set_xscale('log')
    # continue
            
            
if save:
    plt.savefig( os.path.join(save_path,manifold+'_'+latent_distribution+'.png') )  #s,bbox_inches='tight') 001
print('All done! You are awesome <3') #+str(d)+' painted')