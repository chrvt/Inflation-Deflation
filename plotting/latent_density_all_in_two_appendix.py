import numpy as np
from pathlib import Path
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
 

import matplotlib
from matplotlib.colors import Normalize as norm
norm = matplotlib.colors.Normalize()
    
def plt_latent(latent, probs, title, x_shift=.5, title_shift = 1.0):
    ax.pcolormesh(latent[0], latent[1], probs, cmap = plt.cm.jet  , shading='nearest')  #,shading='auto') #,xunits=radians,yunits=radians)   
    ax.set_title(title, x = x_shift, y=title_shift)

def plt_latent_spiral(latent, probs, title, x_shift=.5, title_shift = 1.0, color = 'blue',label=None, linewidth=3.5,linestyle='solid'):
    ax.plot(latent, probs, c=color,label=label,linewidth=linewidth,linestyle=linestyle) #,shading='auto') #,xunits=radians,yunits=radians)   
    ax.set_title(title, x = x_shift, y=title_shift)            
       

    
base_path = r'./results'
save_path = r'./figures'
# plt.close()
save = True  #false
cheat = False

manifolds1 = ['sphere','torus','hyperboloid']
manifolds2 = ['thin_spiral','swiss_roll','spheroid']
latents = ['mixture','correlated','exponential','unimodal']
# manifold = 'thin_spiral'  # 'swiss_roll' 'hyperboloid'  torus, sphere, swiss_roll spheroid

count_mani = -1
for manifolds in [manifolds1,manifolds2]:
    axes = []
    count = -1
    count_mani += 1
    if count_mani ==0:
        outer = gridspec.GridSpec(6, 5, wspace=0.2, hspace=0.3)
        axes_shape = [6,5]
        fig = plt.figure(figsize=(52, 70))
    else: 
        outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.3)
        axes_shape = [5,5]
        fig = plt.figure(figsize=(52, 60))
    fig.clf()
    
    for manifold in manifolds:
        
        for latent_distribution in latents:
            
            data_path = os.path.join(base_path,manifold,latent_distribution)
            if Path(data_path).is_dir():
                count += 1
                stars_gauss = np.load(os.path.join(data_path,'stars_gaussian.npy'))
                stars_normal = np.load(os.path.join(data_path,'stars_normal.npy'))
            else: 
                continue
    
        
            baseline = 'NID with '+r'$\sigma^2=$' +str(stars_normal[0])
            FG_title = 'IID with '+r'$\sigma^2=$' +str(stars_gauss[0]) #+r'$\sigma^2=0.01$'
            
                
            ###########################################
            ####    GRID   ############################
            if manifold == 'thin_spiral':
                n_pts = 1000
            else: 
                n_pts = 500
            latent, latent_mesh, data, probs_true, dx, dy =  make_grid(n_pts,manifold,latent_distribution)
            
            
            #######################################################
            ####    target data density  ##########################
            if manifold == 'sphere':
                ax = fig.add_subplot(outer[count,0],projection="mollweide")
                latent_ = [latent[0]-np.pi, latent[1]-np.pi/2]
            
                
                ax.pcolormesh(latent_[0], latent_[1], probs_true, cmap = plt.cm.jet  , shading='nearest')  #,shading='auto') #,xunits=radians,yunits=radians)   
                if latent_distribution in ['mixture']:
                    ax.set_title('Sphere $p^*(x)$', x = 0.5, y=1.4, fontsize=50)
                    # ax.text(-30,1.6,r'$p^*(x)$',rotation=0,fontsize=40) 
                
                no_ticks(ax)
            elif manifold == 'torus':
                ax = fig.add_subplot(outer[count,0],projection='3d') 
                ax.set_zlim(-3,3)
                import matplotlib
                from matplotlib.colors import Normalize as norm
                norm = matplotlib.colors.Normalize()
                ax.plot_surface(data[:,1].reshape([n_pts,n_pts]), data[:,0].reshape([n_pts,n_pts]), data[:,2].reshape([n_pts,n_pts]), facecolors = plt.cm.jet(norm(probs_true)), antialiased=True) 
                
                if latent_distribution in ['mixture']:
                    ax.text(-1.3,-0.7,6.5,'Torus $p^*(x)$',rotation=0,fontsize=50,fontstyle='italic') 
                    # ax.set_title(r'$p^*(x)$', y=.8,x=0.19)
                    
                no_ticks(ax,threeD=True)
                ax.margins(x=-0.49, y=-0.49) 
            elif manifold == 'swiss_roll':
                ax = fig.add_subplot(outer[count,0],projection='3d') 
                ax.scatter(data[:,0], data[:,1], data[:,2], c=probs_true, cmap=plt.cm.jet) 
                
                
                if latent_distribution in ['mixture']:
                    ax.text(-2,-2,40,r'Swiss Roll $p^*(x)$',rotation=0,fontsize=50,fontstyle='italic') 
                    # ax.set_title(r'$p^*(x)$', y=0.80,x=0.19)
               
                    
                no_ticks(ax,threeD=True)
                
                ax.margins(x=-0.49, y=-0.49) 
            elif manifold == 'hyperboloid':
                import torch
                def to_poincare(x):
                    return torch.div(x[..., 1:], (1+x[...,0]).reshape(x.shape[0], 1))  
                ax = fig.add_subplot(outer[count,0]) 
                xy_poincare = to_poincare(torch.from_numpy(data).float()) 
                x = xy_poincare[:,0].cpu().numpy().reshape(n_pts, n_pts)
                y = xy_poincare[:,1].cpu().numpy().reshape(n_pts, n_pts)
                
                
                if latent_distribution in ['correlated']:
                    plt_latent([x,y], probs_true, title=r'', x_shift=0.1, title_shift=.9)
                    ax.text(-0.6,0.8,r'Hyperboloid $p^*(x)$',rotation=0,fontsize=50,fontstyle='italic') 
                else:
                    plt_latent([x,y], probs_true, title=r'', x_shift=0.1, title_shift=.9)
                    
                ax.axis('off')
                no_ticks(ax,threeD=False)
             
            elif manifold == 'spheroid':
                ax = fig.add_subplot(outer[count,0],projection='3d') 
                import matplotlib
                from matplotlib.colors import Normalize as norm
                norm = matplotlib.colors.Normalize()
                # probs_true[probs_true <1e-01] = 1e-01
                ax.plot_surface(data[:,0].reshape([n_pts,n_pts]), data[:,1].reshape([n_pts,n_pts]), data[:,2].reshape([n_pts,n_pts]), facecolors = plt.cm.jet(norm(probs_true.reshape([n_pts,n_pts]))), antialiased=True)
                
                
                if latent_distribution in ['mixture']:
                    ax.text(-3,-2,6.6,r'Spheroid $p^*(x)$',rotation=0,fontsize=50,fontstyle='italic') 
                    # ax.set_title(r'$p^*(x)$', y=0.92,x=0.3)
                    
                no_ticks(ax,threeD=True)
                if latent_distribution == 'correlated':
                    ax.view_init(-30,25)
                
            elif manifold == 'thin_spiral':
                ax = fig.add_subplot(outer[count,0]) 
                # create colormap
                # ---------------
                
                # create a colormap that consists of
                # - 1/5 : custom colormap, ranging from white to the first color of the colormap
                # - 4/5 : existing colormap
                
                # set upper part: 4 * 256/4 entries
                upper = mpl.cm.jet(np.arange(256))
                # set lower part: 1 * 256/4 entries
                # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
                lower = np.ones((int(256/4),4)) * 0.8
                # - modify the first three columns (RGB):
                #   range linearly between white (1,1,1) and the first color of the upper colormap
                for i in range(3):
                  lower[:,i] = np.linspace(0.8, upper[0,i], lower.shape[0])
                lower[0:8,:]=1
                # combine parts of colormap
                cmap = np.vstack(( lower, upper ))
                # convert to matplotlib colormap
                cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
                
                c1 = 540 * 2* np.pi / 360
                r = np.sqrt(latent) * c1
                jac_det = ((1+r**2)/r**2) * c1**4 / 36   
                
                p_max = np.max(probs_true  / np.sqrt(jac_det))  #we are going from latent to data space, therefore divide by jac_det
                normalize = matplotlib.colors.Normalize(vmin=0, vmax=p_max,clip=True)  #10 is max value if p*(x)
                probs_data = normalize(probs_true  / np.sqrt(jac_det) )
                for i in range(n_pts-1):
                    # z2 = x[i,0]**2+x[i,1]**2
                    p_ = probs_data[i] 
                    ax.plot(data[i:i+2,0], data[i:i+2,1], color=cmap(p_),linewidth=2)
                    
                # plt.title(r'$p^*(x)$',x=0.3,y=0.75)
                ax.text(-2.6,5,r'Thin Spiral $p^*(x)$',rotation=0,fontsize=50,fontstyle='italic') 
                
                ax.axis('off')
            
            axes += [ax]
            
            #######################################################
            ####    target latent density  ########################
            ax = fig.add_subplot(outer[count,1]) 
            axes += [ax]
            if manifold == 'thin_spiral':
                plt_latent_spiral(latent, probs_true, title=r'$\pi_u(u)$')
            else:
                plt_latent(latent, probs_true, title=r'$\pi_u(u)$')
            ax.set_xlabel(r'$u_1$')     
            # max_value = probs_true.max()
            set_ticks_and_labels(manifold,ax,only_x=False)
            
            ###########################################
            ####    normal noise   ########################
            ax = fig.add_subplot(outer[count, 2]) 
            axes += [ax]
            if manifold == 'sphere':
                probs_normal = np.load(os.path.join(data_path,'normal_density_flow.npy')).T
            else:
                probs_normal = np.load(os.path.join(data_path,'normal_density_flow.npy'))
              # =np.load(os.path.join(data_path,'sphere_3D/mixture/4mix_FOM/50K_MC_samples1/FOM_density_flow.npy'))#
            if cheat:
                probs_normal[probs_normal>np.max(probs_true)] = np.max(probs_true)
            if manifold == 'thin_spiral':
                plt_latent_spiral(latent, probs_normal, title=baseline, color='orange',label='NS')
                plt_latent_spiral(latent, probs_true, title=baseline,label='true')
                ax.legend()
                ax.legend(loc=1)
                set_ticks_and_labels(manifold,ax,only_x=True)
            else:
                plt_latent(latent, probs_normal, title=baseline)
                set_ticks_and_labels(manifold,ax,only_x=True)
            ax.set_xlabel(r'$u_1$') 
            
            
            ###########################################
            ####    full Gaussian   ########################
            ax= fig.add_subplot(outer[count, 3]) 
            axes += [ax]
            if manifold == 'sphere':
                probs_gauss = np.load(os.path.join(data_path,'gaussian_density_flow.npy')).T
            else:
                probs_gauss = np.load(os.path.join(data_path,'gaussian_density_flow.npy'))
            if cheat:
                probs_gauss[probs_gauss>np.max(probs_true)] = np.max(probs_true)
            if manifold == 'thin_spiral':
                plt_latent_spiral(latent, probs_gauss, title=FG_title, color='orange',label='FG')
                plt_latent_spiral(latent, probs_true, title=FG_title, label='true')
                ax.legend()
                ax.legend(loc=1)
            else:
                plt_latent(latent, probs_gauss, title=FG_title)
                set_ticks_and_labels(manifold,ax,only_x=True)
            ax.set_xlabel(r'$u_1$') 

            ###########################################
            ####    FOM   ########################
            ax= fig.add_subplot(outer[count, 4]) 
            axes += [ax]
            
            FOM_path = os.path.join(data_path,'FOM','sig2_0.0_seed_11')
            probs_fom = np.load(os.path.join(FOM_path,'density_flow.npy'))
            
                
            if manifold == 'sphere':
                probs_fom = np.load(os.path.join(FOM_path,'density_flow.npy')).T
            else:
                probs_fom = np.load(os.path.join(FOM_path,'density_flow.npy'))     
               
            if manifold == 'thin_spiral':  
                plt_latent_spiral(latent, probs_true, title='FOM')
            else:
                plt_latent(latent, probs_fom, title='FOM')
            set_ticks_and_labels(manifold,ax,only_x=True)
            # if cheat:
            #     probs_gauss[probs_gauss>np.max(probs_true)] = np.max(probs_true)

            # else:
            #     plt_latent(latent, probs_gauss, title=FG_title)
            #     set_ticks_and_labels(manifold,ax,only_x=True)
            ax.set_xlabel(r'$u_1$')             
          
    import matplotlib.transforms as mtrans
    axes = np.array(axes)
    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes)), mtrans.Bbox).reshape(axes.shape)
    
    #Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes))).reshape(axes_shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes))).reshape(axes_shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)
    
    # Draw a horizontal lines at those coordinates
    if count_mani == 0:
        count_ax = 0
    else: count_ax = 1
    for y in ys:
        count_ax += 1
        if count_ax % 2 == 0:
            line = plt.Line2D([0.13,0.9],[y,y], transform=fig.transFigure, color="black")
            fig.add_artist(line)
    
    
    if save:
        plt.savefig( os.path.join(save_path,'all_in_one_appendix'+str(count_mani)+'.png') )  #s,bbox_inches='tight')
    print('All done! You are awesome <3') #+str(d)+' painted')