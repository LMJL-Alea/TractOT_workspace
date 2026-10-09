import numpy as np

def add_noise_weights(w,mask,eps):
    """
    Applied Gaussian noise to the weights of the MCM image over the mask.

    Parameters
    ----------
    w : numpy array
        Weights of the mcm
        Is shape is size_img x nb_compartments 
        
    mask : numpy array
        mask where applied the noise
        Is shape is size_img

    eps : float 
        Quantity of noise to apply.
        
    Returns
    -------
    w_noisy : numpy array 
        Obtained noisy weights
        Is shape is size_img x nb_compartments
    """
    w_noisy=np.where(mask[:,:,:,None],w+abs(np.random.normal(0,eps,w.shape)),w)
    return np.where(mask[:,:,:,None],w_noisy/w_noisy.sum(-1)[:,:,:,None],w)
    
def add_noise_eigvals(D,mask,eps):
    """
    Applied Gaussian noise to the eigenvalues of the MCM anisotropic covariates over the mask.

    Parameters
    ----------
    D : numpy array
        Eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension

    mask : numpy array
        mask where applied the noise
        Is shape is size_img

    eps : float 
        Quantity of noise to apply.
        
    Returns
    -------
    D_noisy : numpy array 
        Obtained noisy eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension
    """
    D_noisy=np.where(mask[:,:,:,None,None],D+np.random.normal(0,eps,D.shape),D)
    return abs(D_noisy)
    
def add_noise_rotation(cov,mask,eps):
    """
    Applied rotation (det=1) to the anisotropic covariates of the MCM, over the mask.

    Parameters
    ----------
    cov : numpy array
        Covariates matrices of the aniso compartments of the MCM
        Is shape is size_img x nb_compartments_aniso x dimension x dimension

    mask : numpy array
        mask where applied the noise
        Is shape is size_img

    eps : float 
        Quantity of noise to apply.
        
    Returns
    -------
    cov_noisy : numpy array
        Obtained noisy covariates matrices of the aniso compartments of the MCM
        Is shape is size_img x nb_compartments_aniso x dimension x dimension
    """
    d=3
    
    theta=np.random.normal(0,eps,cov.shape[:-2])
    axis=np.random.randint(0,d,cov.shape[:-2])
    
    R=np.tile(np.eye(d),cov.shape[:-2]+(1,1,))
    
    mask_axis_rotation=np.einsum('...i,...j->...ij',(np.tile(np.arange(d),cov.shape[:-2]+(1,))!=axis[:,:,:,:,None]),np.tile(np.arange(d),cov.shape[:-2]+(1,))!=axis[:,:,:,:,None])
    
    R_22=np.zeros(cov.shape[:-2]+(d-1,d-1,))
    R_22[:,:,:,:,0,0]=np.cos(theta)
    R_22[:,:,:,:,1,1]=np.cos(theta)
    R_22[:,:,:,:,0,1]=-np.sin(theta)
    R_22[:,:,:,:,1,0]=np.sin(theta)
    
    R[mask_axis_rotation]=R_22.reshape(-1)
    cov_noisy=np.where(mask[:,:,:,None,None,None],np.einsum('...ij,...jk,...kl->...il',R,cov,R.transpose((0,1,2,3,5,4))),cov)
    return cov_noisy
