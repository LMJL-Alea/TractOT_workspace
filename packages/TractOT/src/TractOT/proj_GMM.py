import numpy as np
from TractOT.distances import distances_aniso_along,distances_aniso
from TractOT.utils import sqrtms


def proj_GMM_along(k2,ksize,w1,S1,max_itr=10,eps=1e-7,init='plusplus_weighted'):
    """
    Reduce the number of compartment of a Gaussian Mixture

    This function reduce the number of compartment of an array of Gaussian mixture

    Parameters
    ----------
    k2 : numpy array or scalar
        An array of positive integer defining the number of non null compartment of the target mixtures
    ksize : scalar
        A positive scalar for the output size of the Gaussian mixture (some weights and covariates can be null)
    w1 : numpy array
        An array of positive weights for the input Gaussian mixture. 
        Is shape is nb_MAPxnb_compartment_input
    S1: numpy array
        An array of covariate matrices for the input Gaussian mixture. 
        Is shape is nb_MAPxnb_compartment_inputxdxd
    max_itr: positive integer
        Maximum number of iteration before the algorithm stop
        Default is 10
    eps: positive scalar
        minimum difference between two iterates to make the algorithm stop
        Default is 1e-7
        
    Returns
    -------
    w2 : numpy array
        An array of positive weights for the output Gaussian mixture. Is shape is nb_MAPxksize
        Note that w2 has k2 non null value
    S2: numpy array
        An array of covariate matrices for the output Gaussian mixture. Is shape is nb_MAPxk_sizexdxd
        Note that S2 has k2 non null covariate matrices
    """
    print(S1.shape,ksize,k2)
    
    nb_MAP=w1.shape[0]
    d=S1.shape[-1]

    #Flexible nb of compartments
    if np.array(k2).shape == (): #Should find a better test
        k2=np.int32(k2*np.ones(nb_MAP))
        
    #Initialisation
    if init=='random':
        S2=init_random(S1,k2,ksize)
    if init=='most weighted':
        S2=init_most_weighted(S1,w1,k2,ksize)
    if init=='plusplus':
        S2=init_plusplus(S1,k2,ksize)
    if init=='plusplus_weighted':
        S2=init_plusplus_weighted(S1,w1,k2,ksize)
        
    for i in range(max_itr):
        P=update_P(k2,w1,S1,S2)
        
        S2_new=update_S(P,w1,S1,S2,eps)
        
        #Check convergence
        diff=np.linalg.norm(S2_new-S2)
        if diff <eps:
            S2 = S2_new
            break
        S2 = S2_new

    w2=update_w(P)
    return w2,S2
    
def update_P(k2,w1,S1,S2):
    ksize=S2.shape[1]
    nb_MAP,k1=S1.shape[0],S1.shape[1]
    C=distances_aniso_along(S1,S2)
    idxs=np.zeros((nb_MAP,k1))
    for i in range(nb_MAP):
        idxs[i]=np.argmin(C[i,:,:k2[i]],-1)
    non_zero=idxs[:,:,None]==np.arange(ksize)
    return w1[:,:,None]*non_zero
    
def update_w(P):
    return np.sum(P,1)

def update_S(P,w1,S1,S2,eps):
    ksize=P.shape[-1]
    return gaussians_barycenter_fixpoint(np.repeat(S1[:,:,None,:,:],ksize,axis=2), P, max_itr=50, eps=eps,Sigma_init=S2)

def gaussians_barycenter_fixpoint(S, P, max_itr=50, eps=1e-7,Sigma_init=None):
    """
    Compute barycenters of gausians by fix point algo

    Parameters
    ----------
    S: numpy array
        An array of covariate matrices 
        Is shape is nb_MAPxnb_compartment_inputxdxd

    P: numpy array
        An array of centroid assignment for the covariates 
        Is shape is nb_MAPxnb_compartment_input

    max_itr: positive integer
        Maximum number of iteration before the algorithm stop
        Default is 10
    eps: positive scalar
        minimum difference between two iterates to make the algorithm stop
        Default is 1e-7

    Sigma_init: numpy array
        A array of warm start for the centroids
        Is shape is nb_MAPxksizexdxd
    
    Returns
    -------
    Sigma: numpy array
        The computed centroids
        Is shape is nb_MAPxksizexdxd
    """

    Pweight=np.divide(P,np.sum(P,1)[:,None,:],where=(np.sum(P,1)!=0)[:,None,:],out=np.zeros(P.shape))
    
    if Sigma_init is None:
        Sigma=np.sum(sqrtms(S)*Pweight[:,:,:,None,None],1)
    else:
        Sigma=Sigma_init
        
    for i in range(max_itr): #scale with d not with K
        Sigma_sq=sqrtms(Sigma)
        Sigma_new = np.sum(sqrtms(np.einsum('m...ij,m...jk,m...kl->m...il',Sigma_sq,S,Sigma_sq))*Pweight[:,:,:,None,None],1)
        
        # check convergence
        diff = np.linalg.norm(Sigma - Sigma_new)
        if diff <= eps:
            Sigma = Sigma_new
            break
            
        Sigma = Sigma_new
    return Sigma

#Initialization of the k-means algo

def init_random(S1,k2,ksize):
    nb_MAP,d=S1.shape[0],S1.shape[-1]
    S2=np.zeros((nb_MAP,ksize,d,d))
    for i in range(nb_MAP):
        S2[i,:k2[i]]=np.copy(S1[i,:k2[i]])
    return S2

def init_most_weighted(S1,w1,k2,ksize):
    nb_MAP,d=S1.shape[0],S1.shape[-1]
    u=np.argsort(w1)[:,::-1]
    w1=np.take_along_axis(w1,u,1)
    S1=np.take_along_axis(S1,u[:,:,None,None],1)
    S2=np.zeros((nb_MAP,ksize,d,d))
    for i in range(nb_MAP):
        S2[i,:k2[i]]=np.copy(S1[i,:k2[i]])
    return S2
    
def init_plusplus(S1,k2,ksize):
    nb_MAP,d=S1.shape[0],S1.shape[-1]
    S2=np.zeros((nb_MAP,ksize,d,d))
    for i in range(nb_MAP):
        S2[i,:min(1,k2[i])]=np.copy(S1[i,:min(1,k2[i])])
        for j in range(1,k2[i]):
            #S=S[i][np.linalg.norm(S[i],axis=(-2,-1))!=0] #ensure that S1[i] is not null
            C=distances_aniso(S1[i],S2[i,:j]) 
            p=np.min(C,1)/np.sum(np.min(C,1)) 
            idx=np.random.choice(np.arange(S1.shape[-3]),p=p)
            S2[i,j]=np.copy(S1[i,idx])
    return S2

def init_plusplus_weighted(S1,w1,k2,ksize):
    nb_MAP,d=S1.shape[0],S1.shape[-1]
    u=np.argsort(w1)[:,::-1]
    w1=np.take_along_axis(w1,u,1)
    S1=np.take_along_axis(S1,u[:,:,None,None],1)
    S2=np.zeros((nb_MAP,ksize,d,d))
    for i in range(nb_MAP):
        S2[i,:min(1,k2[i])]=np.copy(S1[i,:min(1,k2[i])])
        for j in range(1,k2[i]):
            #S=S[i][np.linalg.norm(S[i],axis=(-2,-1))!=0] #ensure that S1[i] is not null
            C=distances_aniso(S1[i],S2[i,:j]) 
            p=np.min(C,1)*w1[i]
            p/=p.sum()
            idx=np.random.choice(np.arange(S1.shape[-3]),p=p)
            S2[i,j]=np.copy(S1[i,idx])
    return S2