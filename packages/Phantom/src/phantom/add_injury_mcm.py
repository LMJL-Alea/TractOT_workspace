import numpy as np

def reduce_signal(D,mask_and_colinear,l):
    """
    Perform isotropification of the most colinear compartment over the mask.
    i.e. compute the optimal transport geodesic between covariate and closest isotropic. 

    Parameters
    ----------
    D : numpy array
        Eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension

    mask_and_colinear : numpy array 
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
    
    D_low_signal=np.where(mask_and_colinear[:,:,:,:,None],D/l,D)

    return D_low_signal

def isotropification(D,mask_and_colinear,t):
    """
    Perform isotropification of the most colinear compartment over the mask.
    i.e. compute the optimal transport geodesic between covariate and closest isotropic. 

    Parameters
    ----------
    D : numpy array
        Eigenvalues of the MCM aniso covariates
        Is shape is size_img x nb_compartments_aniso x dimension

    mask_and_colinear : numpy array 
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

    D_max=np.max(D,axis=-1)
    #D_isotropificated=np.where(mask_and_colinear[:,:,:,:,None],(1-t)*D+t*D_max[:,:,:,:,None],D)
    D_isotropificated=np.where(mask_and_colinear[:,:,:,:,None],(t**2)*D_max[:,:,:,:,None]+((1-t)**2)*D+t*(1-t)*np.sqrt(D)*np.sqrt(D_max[:,:,:,:,None]),D)

    return D_isotropificated

def colinear_to_free_water(mcm_w,mask_and_colinear,w_free_water):
    """
    Transfer weight of the most colinear compartment to a new isotropic compartment, over the mask.

    Parameters
    ----------
    mcm_w : numpy array
        weights of the MCM
        Is shape is size_img x nb_compartments

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
    w=np.minimum(w_free_water,mcm_w[:,:,:,1:][mask_and_colinear])
    mcm_w[:,:,:,1:][mask_and_colinear]=mcm_w[:,:,:,1:][mask_and_colinear]-w
    mcm_w[:,:,:,0][np.array(mask_and_colinear.sum(-1),dtype=bool)]=mcm_w[:,:,:,0][np.array(mask_and_colinear.sum(-1),dtype=bool)]+w

    return mcm_w

def colinear_to_reduced_signal(D,mcm_w,mask_and_colinear,i,w_reduced_signal):
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

    mask_and_colinear : numpy array 
        Mask of most colinear compartment and the mask
        Is shape is size_img x nb_compartments_aniso

    i : float > 1
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

    w=np.minimum(w_reduced_signal,mcm_w[:,:,:,1:][mask_and_colinear])

    mcm_w_reduced_signal=np.zeros(size_img)
    D_reduced_signal=np.zeros(size_img+(d,))

    mcm_w[:,:,:,1:][mask_and_colinear]=mcm_w[:,:,:,1:][mask_and_colinear]-w
    mcm_w[:,:,:,1:][mask_and_colinear]=np.maximum(mcm_w[:,:,:,1:][mask_and_colinear],0)

    mcm_w_reduced_signal[np.array(mask_and_colinear.sum(-1),dtype=bool)]=w
    D_reduced_signal[np.array(mask_and_colinear.sum(-1),dtype=bool)]=D[mask_and_colinear]/i
    #print(D_reduced_signal[np.array(mask_and_colinear.sum(-1),dtype=bool),0][:3])
    #print(D[mask_and_colinear][:3])

    return mcm_w,D_reduced_signal,mcm_w_reduced_signal.reshape(mcm_w_reduced_signal.shape+(1,))

def colinear_to_isotropification(D,mcm_w,mask_and_colinear,w_isotropification):
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

    w=np.minimum(w_isotropification,mcm_w[:,:,:,1:][mask_and_colinear])

    mcm_w_isotropification=np.zeros(size_img+(1,))
    D_isotropification=np.zeros(size_img+(d,))

    mcm_w[:,:,:,1:][mask_and_colinear]=mcm_w[:,:,:,1:][mask_and_colinear]-w

    mcm_w_isotropification[np.array(mask_and_colinear.sum(-1),dtype=bool),0]=w
    D_isotropification[np.array(mask_and_colinear.sum(-1),dtype=bool)]=np.repeat(D[mask_and_colinear].max(-1),d).reshape(-1,d)

    return mcm_w,D_isotropification,mcm_w_isotropification
    

