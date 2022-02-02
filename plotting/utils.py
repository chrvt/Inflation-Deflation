""" making grid for plot depending on manifold and latent distribution """
import numpy as np
from scipy.special import i0


def true_density(data, manifold, latent_distribution):
    if manifold == 'sphere':
        theta = data[1]
        phi = data[0]
        if latent_distribution == 'mixture':
            kappa = 6.0
            mu11 = 1*np.pi/4       
            mu12 = np.pi/2             #northpole
            mu21 = 3*np.pi/4 
            mu22 = 4*np.pi/3           #southpole
            mu31 = 3*np.pi/4 
            mu32 = np.pi/2 
            mu41 = np.pi/4
            mu42 = 4*np.pi/3 
            probs = 1/4* (2*np.exp(kappa*np.cos(2* (theta-mu31))) * np.exp(kappa*np.cos(phi-mu32)) *(1/(2*np.pi*i0(kappa))**2)
                 +2*np.exp(kappa*np.cos(2* (theta-mu11))) * np.exp(kappa*np.cos(phi-mu12)) *(1/(2*np.pi*i0(kappa))**2)   
                 +2*np.exp(kappa*np.cos(2* (theta-mu21))) * np.exp(kappa*np.cos(phi-mu22)) *(1/(2*np.pi*i0(kappa))**2)  
                 +2*np.exp(kappa*np.cos(2* (theta-mu41))) * np.exp(kappa*np.cos(phi-mu42)) *(1/(2*np.pi*i0(kappa))**2)
                 )
            return probs
        
        elif latent_distribution == 'correlated':
            kappa = 6.0
            mu11, kappa11 = 0, kappa        
            mu12, kappa12 = np.pi/2 , kappa             #northpole
            mu21, kappa21 = np.pi , kappa 
            mu22, kappa22 = 3*np.pi/2  , kappa          #southpole
            mu31, kappa31 = np.pi/2 , kappa
            mu32, kappa32 = 0 , kappa
            mu41, kappa41 = np.pi/2,  50
            mu42, kappa42 = np.pi , kappa
            
            prob = (1/(2*np.pi*i0(kappa41))) * 2*np.exp(kappa41*np.cos(2*(theta-mu41))) / (2*np.pi)
            probs = 1/3 * (prob 
                         +2*np.exp(kappa11*np.cos(2* (theta-mu11))) * np.exp(kappa12*np.cos(phi-mu12)) *(1/(2*np.pi*i0(kappa))**2)   
                         +2*np.exp(kappa21*np.cos(2* (theta-mu21))) * np.exp(kappa22*np.cos(phi-mu22)) *(1/(2*np.pi*i0(kappa))**2)   
                         ) 
            return probs
        
        elif latent_distribution == 'unimodal':
            kappa, mu1, mu2  = 6, np.pi/2, np.pi
            probs = 2*np.exp(kappa*np.cos(2* (theta-mu1))) * np.exp(kappa*np.cos(phi-mu2)) *(1/(2*np.pi*i0(kappa))**2)  
            return probs
        
    elif manifold =='torus':
        theta = data[1]
        phi = data[0]
        if latent_distribution == 'mixture':
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = 2, 0.21, 2.85, 1.89, 6.18, 3.77, 1.56 
            probs = 0.3*   ( np.exp(kappa*np.cos(theta-mu11)+kappa*np.cos(phi-mu12))
                          +  np.exp(kappa*np.cos(theta-mu21)+kappa*np.cos(phi-mu22))
                          +  np.exp(kappa*np.cos(theta-mu31)+kappa*np.cos(phi-mu32)) )*(1/(2*np.pi*i0(kappa))**2) 
            return probs
        
        elif latent_distribution == 'correlated':
            kappa, mu = 2, 1.94
            probs = 1/(2*np.pi) * np.exp(kappa*np.cos(phi+theta-mu)) *(1/(2*np.pi*i0(kappa))**1)
            return probs
        
        elif latent_distribution == 'unimodal':
            mu1, mu2, kappa = 4.18-np.pi, 5.96-np.pi, 2
            probs = np.exp(kappa*np.cos(phi-mu1) + kappa*np.cos(theta-mu2))*(1/(2*np.pi*i0(kappa))**2) 
            return probs
        
    elif manifold =='swiss_roll':
        u_ = data[0]
        v_ = data[1]
        if latent_distribution == 'mixture':
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = 2, 0.1, 0.1, 0.5, 0.8, 0.8, 0.8 
            probs =  0.3*(np.exp(kappa*np.cos((u_-mu11)*(2*np.pi))+kappa*np.cos((v_-mu12)*(2*np.pi)))
                        + np.exp(kappa*np.cos((u_-mu21)*(2*np.pi))+kappa*np.cos((v_-mu22)*(2*np.pi)))
                        + np.exp(kappa*np.cos((u_-mu31)*(2*np.pi))+kappa*np.cos((v_-mu32)*(2*np.pi))))*(1/(2*np.pi*i0(kappa))**2) 
        elif latent_distribution == 'correlated':
            def _translate(x): #from [0,1] to [-pi,pi]
                return 2*np.pi*x-np.pi
            def _translate_inverse(u): #from [-pi,pi] to [0,1]  
                return (u+np.pi)/(2*np.pi)
    
            kappa, mu = 5, 0.0
            probs = 2*np.pi * np.exp((kappa)*np.cos(_translate(v_) - _translate(u_))) *(1/(2*np.pi*i0(kappa))**1) 
            
        elif latent_distribution == 'unimodal':
            mu1, mu2, kappa = 0.5, 0.9, 2 
            probs = np.exp(kappa*np.cos((u_-mu1)*(2*np.pi)) + kappa*np.cos((v_-mu2)*(2*np.pi)))  
        return probs
    
    elif manifold == 'hyperboloid':
        v = data[0]
        theta = data[1]
        if latent_distribution == 'mixture':
            kappa, mu11, mu12  = 6, np.pi/2, 3*np.pi/2
            probs_theta = 0.5*( np.exp(kappa*np.cos((theta-mu11))) + np.exp(kappa*np.cos(theta-mu12)) ) *(1/(2*np.pi*i0(kappa)))
            probs_v = np.zeros(*v.shape)
            for i in range(len(v)):
                if 1.0<=v[i] <= 1.5:
                    probs_v[i]=1/2
            probs = probs_v * probs_theta 
        elif latent_distribution == 'correlated':
            kappa = 6
            probs_theta = 1*( np.exp(kappa*np.cos((theta-v+np.pi)))) *(1/(2*np.pi*i0(kappa)))
            #1*( np.exp(kappa*np.cos((theta-v+np.pi)))) *(1/(2*np.pi*i0(kappa)))
            probs_v = np.ones(*v.shape)/2
            probs = probs_v * probs_theta 
        elif latent_distribution == 'unimodal':
            sig2 = 1 
            probs_theta = 1/(2*np.pi)
            probs_v =  2*np.exp(-v/0.5) #np.sqrt(2*np.pi*sig2)*np.exp(-v**2/(2*sig2))
            probs = probs_v * probs_theta   
        return probs
            
    elif manifold =='thin_spiral':
        z = data[0]
        from scipy.stats import expon
        probs= expon.pdf(z,scale=0.3) 
                
        return probs
    
    elif manifold == 'spheroid':
        z1, z2 = data[0], data[1]
        idx_hyp = np.where(z1 < 0 )[0]
        idx_sph = np.where(z1 >=0 )[0] 
        probs = np.zeros(len(z1))
        
        if latent_distribution == 'mixture':
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 =6.0,  0, np.pi, -0.5, np.pi/2, 0.5, np.pi/2
            probs = 1/3* (np.exp(kappa*np.cos((z1-mu11))) * np.exp(kappa*np.cos(z2-mu12)) *(1/(2*np.pi*i0(kappa))**2)
                         +np.exp(kappa*np.cos((z1-mu21))) * np.exp(kappa*np.cos(z2-mu22)) *(1/(2*np.pi*i0(kappa))**2)   
                         +np.exp(kappa*np.cos((z1-mu31))) * np.exp(kappa*np.cos(z2-mu32)) *(1/(2*np.pi*i0(kappa))**2)
                         )
        elif latent_distribution == 'correlated':
            from scipy.stats import expon
            _scale, _kappa = 0.4, 10
            theta, phi = z1[idx_sph], z2[idx_sph]
            probs_phi =  np.exp((_kappa)*np.cos(phi- np.pi + theta)) *(1/(2*np.pi*i0(_kappa))) #10/(2*np.pi)#n
            probs_theta = expon.pdf(theta,scale=_scale)
            
            probs[idx_sph] = 0.5 * probs_theta * probs_phi 
            
            v, psi = z1[idx_hyp], z2[idx_hyp]
            probs_theta = 1*( np.exp(_kappa*np.cos(psi-np.pi-np.abs(v)))) *(1/(2*np.pi*i0(_kappa)))
            probs_v = expon.pdf(np.abs(v),scale=_scale) 
            
            probs[idx_hyp] = 0.5 * probs_v * probs_theta
        
        return probs
    
    elif manifold == 'stiefel':
        theta_ = data[0]
        kappa, mu1, mu2, mu3, mu4 = 6.0, 0, -np.pi/2, np.pi/2, np.pi
        probs = 1/4* (np.exp(kappa*np.cos(theta_-mu1)) + np.exp(kappa*np.cos(theta_-mu2))
                     +np.exp(kappa*np.cos(theta_-mu3)) + np.exp(kappa*np.cos(theta_-mu4))
                     ) * (1/(2*np.pi*i0(kappa)))    
        return probs
        
        
def _transform_u_to_hyperboloid(v,psi,sign=1):
    a,b,c = 1, 1, 1 
    x = -a*(np.cosh(np.abs(v)))*np.cos(psi)
    y = -b*(np.cosh(np.abs(v)))*np.sin(psi)
    z = sign * c*np.sinh(np.abs(v))
    samples = np.stack([x,y,z], axis=1) 
    return samples

def _transform_u_to_sphere(theta,phi):
    c, a = 0, 1
    x = (c + a*np.cos(theta+np.pi)) * np.cos(phi)
    y = (c + a*np.cos(theta+np.pi)) * np.sin(phi)
    z = a * np.sin(theta+np.pi)
    x = np.stack([x,y,z], axis=1) 
    return x  



def make_grid(n_pts,manifold='sphere',latent_distr = 'mixture'):
    # print('manifold ',manifold == 'torus')
    if manifold == 'sphere':
        theta = np.linspace(0, np.pi, n_pts+1)[1:]
        dx = theta[1]-theta[0]
        
        phi = np.linspace(0, 2*np.pi, n_pts)
        dy = phi[1]-phi[0]
        
        latent = [phi,theta]
        phi_mesh, theta_mesh = np.meshgrid(phi,theta)
        latent_mesh = [phi_mesh,theta_mesh]
        
        grid = np.stack((phi_mesh.flatten(), theta_mesh.flatten()), axis=1)
        
        probs_true = true_density([grid[:,0],grid[:,1]],manifold,latent_distr).reshape([n_pts,n_pts])

        a, c = 1, 0
        d1x = (c + a*np.sin(theta_mesh)) * np.cos(phi_mesh)
        d1y = (c + a*np.sin(theta_mesh)) * np.sin(phi_mesh)
        d1z = (a * np.cos(theta_mesh))
        data = np.stack([d1x, d1y, d1z], axis=1) 
    
    elif manifold == 'torus':
        theta = np.linspace(-np.pi, np.pi, n_pts) #+1  [1:n+1]
        dx = theta[1]-theta[0]
        
        phi = np.linspace(-np.pi, np.pi, n_pts) #+1    [1:n+1]
        dy = phi[1]-phi[0]
        
        latent = [theta,phi]
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)
        latent_mesh = [theta_mesh,phi_mesh]
        
        grid = np.stack((phi_mesh.flatten(),theta_mesh.flatten()), axis=1)
        probs_true = true_density([grid[:,0],grid[:,1]],manifold,latent_distr).reshape([n_pts,n_pts])
        
        a, c = 0.6, 1
        d1x = (c + a*np.cos(theta_mesh)) * np.cos(phi_mesh)
        d1y = (c + a*np.cos(theta_mesh)) * np.sin(phi_mesh)
        d1z = (a * np.sin(theta_mesh))
        data = np.stack([d1x, d1y, d1z], axis=1) 
        # jacobians = np.abs(a*(c+a*np.cos(grid[:,0])))
        
    elif manifold == 'swiss_roll':
        multiplier = 1
        # u_ = np.linspace(0, 1, n_pts)
        u_ = (np.linspace(0, 1, n_pts)).reshape([1,n_pts]) #np.random.rand(1, n_samples)
        v_ = (np.linspace(0, 1, int(n_pts*multiplier))).reshape([1,int(n_pts*multiplier)]) 
        dx = u_[0,1]-u_[0,0]
        dy = v_[0,1]-v_[0,0]
        
        latent = [u_,v_]
        
        u_mesh, v_mesh = np.meshgrid(u_, v_)
        latent_mesh = [u_mesh,v_mesh]
        latent = latent_mesh
        
        grid = np.stack((u_mesh.flatten(), v_mesh.flatten()), axis=1)
        probs_true = true_density([u_mesh,v_mesh],manifold,latent_distr)
        
        u = grid[:,0].reshape([1,-1])
        v = grid[:,1].reshape([1,-1])
        
        t = 1.5 * np.pi * (1 + 2 * v)
        x = t * np.cos(t)
        y = 21 * u
        z = t * np.sin(t)
        
        x = np.concatenate([x, y, z])
        data=x.T
        
        t = 3/2*np.pi*(1+2*v_mesh)
        jacobians = 3*np.pi*21*np.sqrt(1+t**2)
        
    elif manifold == 'thin_spiral':
        u_ = np.linspace(0, 2.5, n_pts+1)[1:]
        dx = u_[1]-u_[0]
        dy = 1
        probs_true = true_density([u_,None],manifold,latent_distr)
        latent = u_
        latent_mesh = u_ 
        u_trans = np.sqrt(u_) * 540 * (2 * np.pi) / 360
        d1x = - np.cos(u_trans) * u_trans 
        d1y =   np.sin(u_trans) * u_trans 
        data = np.stack([ d1x,  d1y], axis=1) / 3
        
        
    elif manifold =='hyperboloid':
        theta = np.linspace(0, 2*np.pi, n_pts) #+1  [1:n+1]
        v = np.linspace(0, 2, n_pts) #+1    [1:n+1]
        dx = theta[1]-theta[0]
        dy = v[1]-v[0]
        
        latent = [v,theta]
        v_mesh, theta_mesh = np.meshgrid(v, theta)
        latent_mesh = [v_mesh,theta_mesh]
        
        grid = np.stack((v_mesh.flatten(), theta_mesh.flatten()), axis=1)
        probs_true = true_density([grid[:,0],grid[:,1]],manifold,latent_distr).reshape([n_pts,n_pts])
        
        a,b,c = 1,1,1 #0.1, 0.1, 0.11
        x = a*(np.sinh(grid[:,0]))*np.cos(grid[:,1])
        y = b*(np.sinh(grid[:,0]))*np.sin(grid[:,1])
        z = 1*c*np.cosh(grid[:,0])
        data = np.stack([z,y,x], axis=1) 
        # jacobians = np.sqrt( ( a**2 * np.sinh(v_mesh)**2 + c**2 * np.cosh(v_mesh)**2 ) * a**2 * np.cosh(v_mesh)**2 )
      
    elif manifold =='spheroid':
        if latent_distr == 'correlated':
            z1_a = np.linspace(-2, 0, int(n_pts/2)) #[:-1]  
            z1_b = np.linspace(0, 2, int(n_pts/2)+1)[1:]  #np.pi/2
            z1_  = np.concatenate([z1_a,z1_b])
        elif latent_distr == 'mixture':
            z1_ = np.linspace(-2, 1.7,n_pts) 
        
        z2_ = np.linspace(0, 2*np.pi, n_pts)
        
        dx = z1_[1]-z1_[0]
        dy = z2_[1]-z2_[0]
        
        latent = [z1_,z2_]
        z1_mesh, z2_mesh = np.meshgrid(z1_,z2_)
        latent_mesh = [z1_mesh,z2_mesh]
        
        grid = np.stack((z1_mesh.flatten(), z2_mesh.flatten()), axis=1)
        probs_true = true_density([grid[:,0],grid[:,1]],manifold,latent_distr).reshape([n_pts,n_pts])
        
        idx_hyp = np.where(grid[:,0] < 0 )[0]
        idx_sph = np.where(grid[:,0] >=0 )[0]       
        
        data = np.zeros([len(grid[:,0]),3])
        
        data[idx_sph,:] = _transform_u_to_sphere(grid[idx_sph,0],grid[idx_sph,1])
        data[idx_hyp,:] = _transform_u_to_hyperboloid(grid[idx_hyp,0],grid[idx_hyp,1])
        
        # data = np.stack([z,y,x], axis=1) 
       
    elif manifold == 'stiefel':
        theta = np.linspace(-np.pi, np.pi, n_pts)
        latent = theta
        
        dx = theta[1]-theta[0]
        dy = 1
        
        probs_true = true_density([theta,None],manifold,latent_distr)
        
        x1 = np.cos(theta)
        x2 = np.sin(theta)
        sign =  1 #(-1)**np.random.binomial(size=len(theta), n=1, p= 0.5)
        data = np.stack([x1,x2,-sign*x2,sign*x1], axis=1) 
        
        jacobians = 2
        
    
    return latent, None, data, probs_true, dx, dy


def set_ticks_and_labels(manifold,ax,only_x=True):
    if manifold == 'sphere':
        unit   = 2
        x_tick = np.arange(0, 2+unit, unit)
        x_label = [r"0", r"$2\pi$"]
        ax.set_xticks(x_tick*np.pi)
        ax.set_xticklabels(x_label)
        ax.set_xlabel(r'$u_1$')
        
        
        if only_x:
            ax.set_yticks([])
        else:
            ax.set_ylabel(r'$u_2$')
            unit   = 1
            y_tick = np.arange(0, 1+unit, unit)
            y_label = [r"",r"$\pi$"] 
            ax.set_yticks(y_tick*np.pi)
            ax.set_yticklabels(y_label)
            
    elif manifold == 'torus':
        unit   = 2
        x_tick = np.arange(-1, 1+unit, unit)
        x_label = [r"0", r"$2\pi$"]
        ax.set_xticks(x_tick*np.pi)
        ax.set_xticklabels(x_label)
        ax.set_xlabel(r'$u_1$')
        
        
        if only_x:
            ax.set_yticks([])
        else:
            ax.set_ylabel(r'$u_2$')
            unit   = 2
            y_tick = np.arange(-1, 1+unit, unit)
            y_label = [r"", r"$2\pi$"]
            ax.set_yticks(y_tick*np.pi)
            ax.set_yticklabels(y_label)
            
    elif manifold == 'swiss_roll':
        unit   = 1
        x_tick = np.arange(0, 1+unit, unit)
        x_label = [r"0", r"$1$"]
        ax.set_xticks(x_tick*1)
        ax.set_xticklabels(x_label)
        ax.set_xlabel(r'$u_1$')
        
        if only_x:
            ax.set_yticks([])
        else:
            unit   = 1
            y_tick = np.arange(0, 1+unit, unit)
            y_label = [r"", r"$1$"]
            ax.set_yticks(y_tick*1)
            ax.set_yticklabels(y_label)
            ax.set_ylabel(r'$u_2$')
            
    elif manifold == 'hyperboloid':
        unit   = 2
        x_tick = np.arange(0, 2+unit, unit)
        x_label = [r"", r"$2$"]
        ax.set_xticks(x_tick*1)
        ax.set_xticklabels(x_label)
        ax.set_xlabel(r'$u_1$')
        
        if only_x:
            ax.set_yticks([])
        else:
            ax.set_ylabel(r'$u_2$')
            unit   = 2
            y_tick = np.arange(0, 2+unit, unit)
            y_label = [r"0", r"$2\pi$"]
            ax.set_yticks(y_tick*np.pi)
            ax.set_yticklabels(y_label)  
    elif manifold == 'spheroid':
        unit   = 1
        x_tick = np.arange(-2, 2+unit, unit)
        x_label = [r"-2", r"$-1$",r"", r"$1$",r""]
        ax.set_xticks(x_tick*1)
        ax.set_xticklabels(x_label)
        ax.set_xlabel(r'$u_1$')
        ax.set_xlim(-2,1.6)
        if only_x:
            ax.set_yticks([])
        else:
            ax.set_ylabel(r'$u_2$')
            unit   = 2
            y_tick = np.arange(0, 2+unit, unit)
            y_label = [r"0", r"$2\pi$"]
            ax.set_yticks(y_tick*np.pi)
            ax.set_yticklabels(y_label)
    
    elif manifold == 'stiefel':
        unit = np.pi/2
        x_tick = np.arange(-np.pi, np.pi+unit, unit)
        x_label = [r"$-\pi$", r"",r"", r"",r"$\pi$"]
        ax.set_xticks(x_tick*1)
        ax.set_xticklabels(x_label)
        ax.set_xlabel(r'$u_1$')
        
        if only_x:
            ax.set_yticks([])
        else:
            unit   = 0.05
            y_tick = np.arange(0, 0.25+unit, unit)
            y_label = [r"",r"0.05",r"",r"",r"",r"0.25"] # r"",r"", r"$1$"
            ax.set_yticks(y_tick*1)
            ax.set_yticklabels(y_label)
            ax.set_ylim(0.04,0.26)
        # ax.set_yticks([])
        # if only_x:
        #     ax.set_yticks([])
        # else:
            # ax.set_ylabel(r'$u_2$')
            # unit   = 2
            # y_tick = np.arange(0, 2+unit, unit)
            # y_label = [r"0", r"$2\pi$"]
            # ax.set_yticks(y_tick*np.pi)
            # ax.set_yticklabels(y_label)
        
    ax.xaxis.set_label_coords(0.5,-0.03)
    if not only_x:
        ax.set_ylabel(r'$u_2$') 
        ax.yaxis.set_label_coords(-0.03,0.5) 
            
def no_ticks(ax,threeD=False):
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    if threeD:
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
    

            
        
    
        
    