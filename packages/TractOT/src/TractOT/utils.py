import numpy as np
import scipy as sc
    
def sqrtms(M):
    """
    Compute the square roots of symmetric matrices in the last two dimension

    Parameters
    ----------
    M : numpy array
        input matrice, last two dimension must be equal

    Returns
    -------
    N : numpy array
        output square root matrice
    """
    Val, Vec = np.linalg.eigh(M)
    Val=np.maximum(Val,0)
    Val = np.sqrt(Val)
    Q = np.einsum("...jk,...k->...jk", Vec, Val)
    return np.einsum("...jk,...kl->...jl", Q, np.swapaxes(Vec, -1, -2))
    
def sqrtm(M,sym=False):
    """
    Compute the square roots of square matrices in the last two dimension

    Parameters
    ----------
    M : numpy array
        input matrice, last two dimension must be equal

    Returns
    -------
    N : numpy array
        output square root matrice
    """
    Val, Vec = np.linalg.eig(M)
    Val=np.maximum(Val,0)
    Val = np.sqrt(Val)
    Q = np.einsum("...jk,...k->...jk", Vec, Val)
    return np.einsum("...jk,...kl->...jl", Q, np.swapaxes(Vec, -1, -2))

  
def random_MASs(nb_MAS,nb_MAP,k,d=3):
    """
    Sample a random MAS with equal number of MAP

    Parameters
    ----------
    nb_MAS : int
        Number of output MAS

    nb_MAP : int
        Number of output MAP for each MAS

    k : int
        Number of compartment for each MAP

    d : int
        dimension

    Returns
    -------
    m : numpy array
        spatial information of the MAS
    
    wI : numpy array
        weight of the iso compartment

    I : numpy array
        value of the iso compartment

    w : numpy array
        weight of the aniso compartments

    S : numpy array
        covariates matrices of the aniso compartments
    """
    m=np.random.normal(size=(nb_MAS,nb_MAP,d))
    
    w=abs(np.random.standard_normal((nb_MAS,nb_MAP,k+1)))
    w/=np.sum(w,-1)[:,:,None]
    wI=w[:,:,0]
    w=w[:,:,1:]

    I=abs(np.random.standard_normal((nb_MAS,nb_MAP)))
    S=np.zeros((nb_MAS,nb_MAP,k,d,d))
    for i in range(nb_MAS):
        for j in range(nb_MAP):
            S[i,j]=random_covs(k,d)

    return m,wI,I,w,S
    
def random_mixture(k,d):
    """
    Sample random mixture
    """
    return random_covs(k,d),random_weight(k)
    
def random_covs(k,d):
    """
    Sample random covariates
    """
    S=np.zeros((k,d,d))
    for i in range(k):
        U=np.diag(np.random.uniform(0,10,d))
        P=sc.stats.ortho_group.rvs(d)
        S[i,:,:]=P@U@(P.T)
    return S

def random_weight(k):
    """
    Sample random weight
    """
    w=abs(np.random.standard_normal(k))
    return w/sum(w)   


def concat_w_MAP(wI,w):
    """
    Concatenate the weight of the iso and the aniso compartments of a MAP
    """
    w_stack=np.zeros(w.shape[0]+1)
    w_stack[0]=wI
    w_stack[1:]=w
    return w_stack

def concat_w_MAS(wI,w):
    """
    Concatenate the weight of the iso and the aniso compartments of a MAS
    """
    w_stack=np.zeros((w.shape[0],w.shape[1]+1))
    w_stack[:,0]=wI
    w_stack[:,1:]=w
    return w_stack
    
def compute_ksize(MAT):
    """
    Compute the number of compartment for a MAT, when covariates matrices are flatten
    """
    return int((MAT.shape[1]-8)/7)
    
    
####################################
# Covariances matrices to 1D array #
# 1D array to covariances matrices #
####################################

def cov_to_array_MAS(S,d=3):
    """
    Flatten covariance matrix into array of 6 elements

    Parameters
    ----------
    S : numpy array
        Covariates matrices
        Is shape is nb_MAP x nb_compartment x d x d

    d : int
        dimension

    Returns
    -------
    l : numpy array 
        flatten covariates matrices
        Is shape is nb_MAP x (nb_compartments x 6)
    """
    nb_pts,ksize=S.shape[0],S.shape[1]
    l=np.zeros((nb_pts,ksize,6))
    l[:,:,0]=S[:,:,0,0]
    l[:,:,1:3]=S[:,:,:2,1]
    l[:,:,3:]=S[:,:,:,-1]
    return l.reshape((nb_pts,-1))

def cov_to_array_MAT(S,d=3):
    """
    Flatten covariance matrix into array of 6 elements

    Parameters
    ----------
    S : numpy array
        Covariates matrices
        Is shape is nb_MAS x nb_MAP x nb_compartment x d x d

    d : int
        dimension

    Returns
    -------
    l : numpy array 
        flatten covariates matrices
        Is shape is (nb_MAS x nb_MAP) x (nb_compartments x 6)
    """

    nb_MAS,nb_pts,ksize=S.shape[0],S.shape[1],S.shape[2]
    l=np.zeros((nb_MAS,nb_pts,6*ksize))
    for i in range(nb_MAS):
        l[i]=cov_to_array_MAS(S[i])
    return l.reshape((nb_MAS*nb_pts,-1))
    
def array_to_cov_MAS(l,d=3):
    #Convert an array of kx6 elements into the corresponding covariances matrices
    """
    Inflating flattened covariance matrix

    Parameters
    ----------
    l : numpy array
        Covariates matrices
        Is shape is nb_MAP x (nb_compartment x 6)

    d : int
        dimension

    Returns
    -------
    S : numpy array 
        flatten covariates matrices
        Is shape is nb_MAP x nb_compartments x d x d
    """
    ksize=int((l.shape[1])/6)
    nb_pts=l.shape[0]
    
    lr=l.reshape((nb_pts,ksize,6))
    S=np.zeros((nb_pts,ksize,d,d)) 

    S[:,:,0,0]=lr[:,:,0]
    S[:,:,:2,1]=lr[:,:,1:3]
    S[:,:,:,2]=lr[:,:,3:]
    S[:,:,1,0]=lr[:,:,1]
    S[:,:,2,:2]=lr[:,:,3:5]
    return S
    
def array_to_cov_neighbor(l,d=3):
    """
    Inflating flattened covariance matrix.
    For 3 x 3 x 3 neighbors

    Parameters
    ----------
    l : numpy array
        Covariates matrices
        Is shape is nb_MAP x 3 x 3 x 3 x 6

    d : int
        dimension

    Returns
    -------
    S : numpy array 
        flatten covariates matrices
        Is shape is nb_MAP x 3 x 3 x 3 x d x d
    """
    
    nb_pts=l.shape[0]
    d=l.shape[1]
    S=np.zeros((nb_pts,3,3,3,d,d))

    S[:,:,:,:,0,0]=l[:,:,:,:,0]
    S[:,:,:,:,:2,1]=l[:,:,:,:,1:3]
    S[:,:,:,:,:,2]=l[:,:,:,:,3:]
    S[:,:,:,:,1,0]=l[:,:,:,:,1]
    S[:,:,:,:,2,:2]=l[:,:,:,:,3:5]
    return S

def most_colinear_compartment(m,S):
    """
    Given a streamline and a set of covariates matrices return a mask of indices.

    Parameters
    ----------
    m : numpy array
        streamline
        Is shape is nb_pts x 3

    S : int
        covariates matrices
        Is shape is nb_pts x nb_compartment x 3 x 3

    Returns
    -------

    S : numpy array 
        indices of the most colinear index
        Is shape is nb_pts
    """

    dir_m=np.zeros(m.shape)
    dir_m[1:-1]=m[2:]-m[:-2]
    dir_m[0]=m[1]-m[0]
    dir_m[-1]=m[-1]-m[-2]
    dir_m/=np.linalg.norm(dir_m,axis=1)[:,None]

    val,vec=np.linalg.eigh(S)
    largest_vec=np.multiply(vec[:,:,:,-1],0,where=(val[:,:,-1]==0)[:,:,None],out=vec[:,:,:,-1]) #avoid eigenvector of null matrix
    return np.argmin(np.arccos(abs(np.clip(np.sum(dir_m[:,None]*largest_vec,-1),-1.0,1.0))),1)
