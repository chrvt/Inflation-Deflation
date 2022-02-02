import numpy as np
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize as norm
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
SMALL_SIZE = 30
MEDIUM_SIZE = 50
BIGGER_SIZE = 60
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
for manifolds in [manifolds1,manifolds2]: #manifolds1,
    axes = []
    count = -1
    count_mani += 1
    if count_mani ==0:
        outer = gridspec.GridSpec(6, 5, wspace=0.2, hspace=0.3)
        axes_shape = [6,4]
        fig = plt.figure(figsize=(72, 90))
    else: 
        outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.3)
        axes_shape = [5,4]
        fig = plt.figure(figsize=(72, 90))
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
    
        
            baseline = 'NID '+r'$\sigma^2=$' +str(stars_normal[0])
            FG_title = 'IID '+r'$\sigma^2=$' +str(stars_gauss[0]) #+r'$\sigma^2=0.01$'
            
                
            ###########################################
            ####    GRID   ############################
            if manifold == 'thin_spiral':
                n_pts = 1000
            else: 
                n_pts = 500
            latent, latent_mesh, data, probs_true, dx, dy =  make_grid(n_pts,manifold,latent_distribution)
            
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
                plt_latent_spiral(latent, probs_normal, title=baseline, color='orange',label='NID')
                plt_latent_spiral(latent, probs_true, title=baseline,label='true',linestyle='dotted')
                ax.legend()
                ax.legend(loc=1)
                set_ticks_and_labels(manifold,ax,only_x=True)
            else:
                plt_latent(latent, probs_normal, title=baseline)
                set_ticks_and_labels(manifold,ax,only_x=True)
            ax.set_xlabel(r'$u_1$') 
            
            
            ###########################################
            ####    full Gaussian   ########################
            # ax= fig.add_subplot(outer[0, 2]) 
            # if manifold == 'sphere':
            #     probs_gauss = np.load(os.path.join(data_path,'gaussian_density_flow.npy')).T
            # else:
            #     probs_gauss = np.load(os.path.join(data_path,'gaussian_density_flow.npy'))
            # if cheat:
            #     probs_gauss[probs_gauss>np.max(probs_true)] = np.max(probs_true)
            # if manifold == 'thin_spiral':
            #     plt_latent_spiral(latent, probs_gauss, title=FG_title, color='orange',label='FG')
            #     plt_latent_spiral(latent, probs_true, title=FG_title, label='true')
            #     ax.legend()
            #     ax.legend(loc=1)
            # else:
            #     plt_latent(latent, probs_gauss, title=FG_title)
            #     set_ticks_and_labels(manifold,ax,only_x=True)
            # ax.set_xlabel(r'$u_1$') 
            
            #######################################################
            ####    target data density  ##########################
            # if manifold == 'sphessre':
            #     ax = fig.add_subplot(outer[count,0],projection="3d")
            #     latent_ = [latent[0]-np.pi, latent[1]-np.pi/2]
                
            #     norm = matplotlib.colors.Normalize()
            #     ax.plot_surface(X[:,0].reshape([n_grid,n_grid]), X[:,1].reshape([n_grid,n_grid]), X[:,2].reshape([n_grid,n_grid]), facecolors = plt.cm.jet(norm(probs)), antialiased=True)  # rstride=100, cstride=100,    , edgecolors='w'
            #     ax.view_init(0,180)
            #     ax.pcolormesh(latent_[0], latent_[1], probs_true, cmap = plt.cm.jet  , shading='nearest')  #,shading='auto') #,xunits=radians,yunits=radians)   
            #     if latent_distribution in ['mixture']:
            #         ax.set_title('Sphere $p^*(x)$', x = 0.5, y=1.4, fontsize=70)
            #         # ax.text(-30,1.6,r'$p^*(x)$',rotation=0,fontsize=40) 
                
            #     no_ticks(ax)
            if manifold in ['sphere','torus']:
                ax = fig.add_subplot(outer[count,0],projection='3d') 
                if manifold == 'torus': 
                    ax.set_zlim(-3,3)
                else: ax.set_zlim(-1,1)
                import matplotlib
                norm = matplotlib.colors.Normalize()
                ax.plot_surface(data[:,1].reshape([n_pts,n_pts]), data[:,0].reshape([n_pts,n_pts]), data[:,2].reshape([n_pts,n_pts]), facecolors = plt.cm.jet(norm(probs_true)), antialiased=True) 
                            
                if latent_distribution in ['mixture']:
                    if manifold == 'torus':
                        ax.text(-1.3,-0.7,6.5,'Torus',rotation=0,fontsize=70,fontstyle='italic') 
                    elif manifold == 'sphere':
                        ax.view_init(0,180)
                        ax.set_title('Sphere', x = 0.5, y=1., fontsize=70)
                    # ax.set_title(r'$p^*(x)$', y=.8,x=0.19)
                  
                no_ticks(ax,threeD=True)
                ax.margins(x=-0.49, y=-0.49) 
            elif manifold == 'swiss_roll':
                ax = fig.add_subplot(outer[count,0],projection='3d') 
                ax.scatter(data[:,0], data[:,1], data[:,2], c=probs_true, cmap=plt.cm.jet) 
                
                
                if latent_distribution in ['mixture']:
                    ax.text(-2,-2,40,r'Swiss Roll',rotation=0,fontsize=70,fontstyle='italic') 
                    # ax.set_title(r'$p^*(x)$', y=0.80,x=0.19)
               
                    
                no_ticks(ax,threeD=True)
                
                ax.margins(x=-0.49, y=-0.49) 
            elif manifold == 'hyperboloid':
                # import torch
                # def to_poincare(x):
                #     return torch.div(x[..., 1:], (1+x[...,0]).reshape(x.shape[0], 1))  
                ax = fig.add_subplot(outer[count,0], projection='3d') 
                
                norm = matplotlib.colors.Normalize()
                ax.plot_surface(data[:,2].reshape([n_pts,n_pts]), data[:,1].reshape([n_pts,n_pts]), data[:,0].reshape([n_pts,n_pts]), facecolors = plt.cm.jet(norm(probs_true).reshape([n_pts,n_pts])), antialiased=True) 

                # ax.scatter(data[:,0], data[:,1], data[:,2], c=probs, cmap=plt.cm.jet) 
                
                # xy_poincare = to_poincare(torch.from_numpy(data).float()) 
                # x = xy_poincare[:,0].cpu().numpy().reshape(n_pts, n_pts)
                # y = xy_poincare[:,1].cpu().numpy().reshape(n_pts, n_pts)
                
                if latent_distribution in ['correlated']:
                    ax.set_title('Hyperboloid', x = 0.5, y=1., fontsize=70)
                    # ax.text(-0.6,0.8,6,r'Hyperboloid $p^*(x)$',rotation=0,fontsize=70,fontstyle='italic') 
                    # plt_latent([x,y], probs_true, title='', x_shift=0.1, title_shift=.9)
                # else:
                    # plt_latent([x,y], probs_true, title='', x_shift=0.1, title_shift=.9)
                    
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
                    ax.text(-3,-2,6.6,r'$(\mathbb{HS})^2$',rotation=0,fontsize=70,fontstyle='italic') 
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
                ax.text(-2.6,5,r'Thin Spiral',rotation=0,fontsize=70,fontstyle='italic') 
                
                ax.axis('off')
            
            axes += [ax]
            
            
            
            ################################################
            ####    KS plot         ########################
             
            total = 27  #max 15 es fehlen: 0.00005, 0.0k003, 0.001, 0.01
            x_axe = np.array([0.0,1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07,1e-06,5e-06,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0,2.0, 3.0, 4.0,  6.0 , 8.0, 10.0])
                
            ax = fig.add_subplot(outer[count,3:5]) 
            axes += [ax]
            
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
            KS_FOM = np.load(os.path.join(FOM_path,'KS_final.npy')) #np.array(0.001) 
            ax.plot(x_axe, np.ones(total)*KS_FOM , linestyle='solid', color='black', linewidth=1.5, label='FOM')
            
            ##table
            clust_data = np.zeros((3,1))
            clust_data[0,0] = round(stars_normal[1],5)
            clust_data[1,0] = round(stars_gauss[1],5)
            clust_data[2,0] = round(KS_FOM.item(),5)
            collabel=[r'KS*']
            rowlabel=("NID", "IID", "FOM") #,colours=['blue','red','black'],
            ax.table(cellText=clust_data,cellLoc='left',rowLabels=rowlabel,colLabels=collabel,loc='center',colWidths=[0.12]).scale(1.5, 4.5)
            
             #FOM and sigma bounds for D=2 only
            """
            sigmas = np.load('D:/PROJECTS/normal_noise/sigmas_sphere.npy')
            ax.axvline(x=sigmas[count-1,0], linestyle = 'dashed',linewidth=1.5, color = 'black')
            #plt.axvline(x=sigmas[count-1,1], linestyle = 'dashed', color = 'black')    
            ax.axvline(x=1.0444, linestyle = 'dashed',linewidth=1.5, color = 'black')       
            """ 
            
            

            # sigmas =  np.load(os.path.join(data_path,'sphere_KS_plots/sigmas_sphere.npy'))  #np.load('D:/PROJECTS/normal_noise/sphere_KS_plots/sigmas_sphere.npy')
            # r_1, my_bound , JP_bound = sigmas[0], sigmas[1], sigmas[2]
            # ax.axvline(x=r_1, linestyle = 'dashed',linewidth=1.5, color = 'black') 
            # ax.axvline(x=JP_bound, linestyle = 'dashed',linewidth=1.5, color = 'black')   
            
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
            ax.set_ylabel('KS',rotation=0)  
            ax.yaxis.set_label_coords(-0.03,1.0)  
            ax.xaxis.set_label_coords(0.5,-0.06)  
            # ax.yaxis.set_label_coords(-0.03,0.9)
            
            ax.set_xlabel(r'$\sigma^2$')
            
            
            #SIGMA bounds
            # import pdb
            # pdb.set_trace()
            # print('data_path', data_path)
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
                ax.set_xscale('log')
                continue
            #fig.add_subplot(ax)
            # plt.show()
            # fig.tight_layout()
          
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
        plt.savefig( os.path.join(save_path,'all_in_one'+str(count_mani)+'.png') )  #s,bbox_inches='tight')
    print('All done! You are awesome <3') #+str(d)+' painted')