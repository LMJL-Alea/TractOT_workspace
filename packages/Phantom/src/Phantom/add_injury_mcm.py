import numpy as np

def reduce_signal(D,mask_mostcolinear,l):
    """
    Perform isotropification of the most colinear compartment over the mask.
    i.e. compute the optimal transport geodesic between covariate and closest isotropic. 

    Parameters
    ----------
    D : numpy array
        Eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension

    mask_mostcolinear : numpy array 
        Mask of most colinear compartment
        Is shape is size_img x nb_compartments_aniso

    l : float > 1
        Parameter to scale the eigenvalues 
        
    Returns
    -------
    D_low_signal : numpy array 
        Obtained eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension
    """
    D_low_signal=np.where(mask_mostcolinear[:,:,:,:,None],D/l,D)

    return D_low_signal

def isotropification(D,mask_mostcolinear,t):
    """
    Perform isotropification of the most colinear compartment over the mask.
    i.e. compute the optimal transport geodesic between covariate and closest isotropic. 

    Parameters
    ----------
    D : numpy array
        Eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension

    mask_mostcolinear : numpy array 
        Mask of most colinear compartment
        Is shape is size_img x nb_compartments_aniso

    t : float 
        Parameter of the geodesic between no isotropification (t=0) and full isotropification (t=1).
        
    Returns
    -------
    D_isotropificated : numpy array 
        Obtained eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension
    """

    D_mean=np.mean(D,axis=-1)
    #D_isotropificated=np.where(mask_mostcolinear[:,:,:,:,None],(1-t)*D+t*D_mean[:,:,:,:,None],D)
    D_isotropificated=np.where(mask_mostcolinear[:,:,:,:,None],(t**2)*D_mean[:,:,:,:,None]+((1-t)**2)*D+t*(1-t)*np.sqrt(D)*np.sqrt(D_mean[:,:,:,:,None]),D)

    return D_isotropificated

def colinear_to_free_water(mcm_w,mask,mask_and_colinear,w_free_water):
    """
    Transfer weight of the most colinear compartment to a new isotropic compartment, over the mask.

    Parameters
    ----------
    mcm_w : numpy array
        weights of the MCM
        Is shape is size_img x nb_compartments

    mask : numpy array
        mask where applied the injury
        Is shape is size_img

    mask_and_colinear : numpy array 
        Mask of most colinear compartment and the mask
        Is shape is size_img x nb_compartments_aniso

    w_free_water : float 
        Quantity of weight to transfer from most_colinear to free water compartment.
        
    Returns
    -------
    mcm_w : numpy array 
        Obtained weights of the aniso
        Is shape is size_img x nb_compartments
    """
    mcm_w[:,:,:,1:][mask_and_colinear]=mcm_w[:,:,:,1:][mask_and_colinear]-w_free_water
    mcm_w[:,:,:,1:][mask_and_colinear]=np.maximum(mcm_w[:,:,:,1:][mask_and_colinear],0)
    mcm_w[:,:,:,0][np.array(mask_and_colinear.sum(-1),dtype=bool)]=mcm_w[:,:,:,0][np.array(mask_and_colinear.sum(-1),dtype=bool)]+w_free_water
    mcm_w=np.where(mask[:,:,:,None],mcm_w/mcm_w.sum(-1)[:,:,:,None],mcm_w)

    return mcm_w

def colinear_to_reduced_signal(D,mcm_w,mask,mask_and_colinear,l,w_reduced_signal):
    """
    Transfer weight of the most colinear compartment to a new reduced signal compartment, over the mask.

    Parameters
    ----------
    D : numpy array
        Eigenvalues of the covariates matrices of the MCM
        Is shape is size_img x nb_compartments x dim

    mcm_w : numpy array
        weights of the MCM
        Is shape is size_img x nb_compartments

    mask : numpy array
        mask where applied the injury
        Is shape is size_img

    mask_and_colinear : numpy array 
        Mask of most colinear compartment and the mask
        Is shape is size_img x nb_compartments_aniso

    l : float > 1
        Parameter to scale the eigenvalues, to produce the new reduced signal compartment

    w_reduced_signal : float 
        Quantity of weight to transfer from most_colinear to the new reduced signal compartment.
        
    Returns
    -------
    mcm_w : numpy array 
        Obtained weights of the new old compartments
        Is shape is size_img x nb_compartment
    
    D_reduced_signal : numpy array
        Obtained eigenvalues of the covariate matrix of the new compartment
        Is shape is size_img x 1 x dim

    mcm_w_reduced_signal : numpy array 
        Obtained weights of the new reduced signal compartment
        Is shape is size_img x 1
    """
    d=D.shape[-1]
    size_img=mcm_w.shape[:-1]
    
    mcm_w_reduced_signal=np.zeros(size_img)
    D_reduced_signal=np.zeros(size_img+(1,d,))

    mcm_w[:,:,:,1:][mask_and_colinear]=mcm_w[:,:,:,1:][mask_and_colinear]-w_reduced_signal
    mcm_w[:,:,:,1:][mask_and_colinear]=np.maximum(mcm_w[:,:,:,1:][mask_and_colinear],0)

    mcm_w_reduced_signal[mask]=w_reduced_signal
    D_reduced_signal[np.array(mask_and_colinear.sum(-1),dtype=bool),0]=D[mask_and_colinear]/l

    return mcm_w,D_reduced_signal,mcm_w_reduced_signal.reshape(mcm_w_reduced_signal.shape+(1,))

def colinear_to_isotropification(D,mcm_w,mask,mask_and_colinear,w_isotropification):
    """
    Transfer weight of the most colinear compartment to a new isotropic compartment, over the mask.

    Parameters
    ----------
    D : numpy array
        Eigen values of the covariates matrices of the MCM
        Is shape is size_img x nb_compartments x dim

    mcm_w : numpy array
        weights of the MCM
        Is shape is size_img x nb_compartments

    mask : numpy array
        mask where applied the injury
        Is shape is size_img

    mask_and_and_colinear : numpy array 
        Mask of most colinear compartment and the mask
        Is shape is size_img x nb_compartments_aniso

    w_isotropifiaction : float 
        Quantity of weight to transfer from most_colinear to the new isotropic compartment.
        
    Returns
    -------
    mcm_w : numpy array 
        Obtained weights of the old compartments
        Is shape is size_img x nb_compartment
    
    D_isotropification : numpy array
        Obtained eigenvalues of the covariate matrix of the new compartment
        Is shape is size_img x 1 x dim

    mcm_w_isotropification : numpy array 
        Obtained weights of the new iso compartment
        Is shape is size_img x 1
    """
    d=D.shape[-1]
    size_img=mcm_w.shape[:-1]
    
    mcm_w_isotropification=np.zeros(size_img+(1,))
    D_isotropification=np.zeros(size_img+(1,d,))

    mcm_w[:,:,:,1:][mask_and_colinear]=mcm_w[:,:,:,1:][mask_and_colinear]-w_isotropification
    mcm_w[:,:,:,1:][mask_and_colinear]=np.maximum(mcm_w[:,:,:,1:][mask_and_colinear],0)

    mcm_w_isotropification[mask,0]=w_isotropification
    D_isotropification[np.array(mask_and_colinear.sum(-1),dtype=bool),0]=np.tile(D[mask_and_colinear].mean(-1),d).reshape(-1,d)

    return mcm_w,D_isotropification,mcm_w_isotropification
    

