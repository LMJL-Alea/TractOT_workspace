import numpy as np

from TractOT.utils import array_to_cov_neighbor, cov_to_array_MAS,most_colinear_compartment,array_to_cov_MAS
from TractOT.proj_GMM import proj_GMM_along
from TractOT.distances import distance_confidence_along

def add_mcm_lift(streamline_coordinate,origin_tract,step_tract,mcm_w,mcm_I,mcm_cov,mcm_coordinate,d=3):
    """
    Add MCM to every points of a streamline to build MAS. For every points all MCM of the neighbors voxels are averaged.

    Parameters
    ----------
    streamline_coordinate : numpy array
        streamline coordinate
        Is shape is nb_ptsxd
        
    origin_tract : numpy array
        origin of the tract, from dipy header
        Is shape is d

    step_tract : numpy array
        step of the tract, from dipy header
        Is shape is d
        
    mcm_w : numpy array
        Weights image of the MCM
        Is shape is nb_compartment x size_img

    mcm_I : numpy array
        Value of the isotropic compartment
        Is shape is size_img

    mcm_cov : list of numpy array
        covariance matrices of the anisotropic compartments
        Is shape is nb_compartment_anisotropic x 6 x size_img

    mcm_coordinate: numpy array
        Coordinate of the image
        Is shape is d x size_img

    d: int
        Dimension
        
    Returns
    -------
    wIf : numpy array
        weight of the isotropic compartment along the MAS
        Is shape is nb_pts

    If : numpy array
        value of the isotropic compartment along the MAS
        Is shape is nb_pts

    wf : numpy array
        weights of the anisotropic compartments along the MAS
        Is shape is nb_pts x nb_compartments_aniso

    Sf : numpy array
        Covariates matrices of the anisotropic compartments along the MAS
        Is shape is nb_pts x (nb_compartments_aniso x 6)

    conf : numpy array
        Confidence indicator. When no lift conf=1 
        Is shape is nb_pts

    idx_col : numpy array
        Indices of the most colinear compartment along the MAS
        Is shape is nb_pts 
    """
    nb_pts=streamline_coordinate.shape[0]
    ksize=mcm_w.shape[0]-1 #shape of the array aka maximum number of compartment
    
    #Compute neigbor kernel
    step=np.array([0,1,-1])
    step_i, step_j, step_k = np.meshgrid(step, step, step, indexing='ij')
    kernel_neighbor = np.stack([step_i, step_j, step_k], axis=-1)[None]
    
    #Compute closest voxel from tract coordinate then compute its neigbord
    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
    #if np.all((streamline_coordinate[0]*step_tract-mcm_coordinate[:,idx[0,0],idx[0,1],idx[0,2]]*step_img)>1.25):
    #    print("mauvais voxel",Flush=True)
    idx_neighbor=idx[:,None,None,None,:]+kernel_neighbor
        
    #Compute distance to each neigbor
    voxels_neighbor=mcm_coordinate[:,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]].transpose(1,0,2,3,4)
    D=np.sum((voxels_neighbor-streamline_coordinate[:,:,None,None,None])**2,1)
    
    #Get weights of compartements and value of iso compartment 
    I=mcm_I[idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]]
    wI=mcm_w[0,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]]
    w=mcm_w[1:,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]].transpose(1,2,3,4,0)
    kmax=np.maximum(np.max(np.count_nonzero(w,axis=-1),axis=(1,2,3)),1) #Number of non null compartment  
    
    #Compute weights of neigbor wrt to distance
    w_dist=1/D
    mask=np.ones((nb_pts,3,3,3))*(np.sum(w,-1)+wI>(1-1e-7)) #Mask for neighbor voxels out of the image    
    w_dist=w_dist*mask
    for i in range(nb_pts):    
        if w_dist[i].sum((0,1,2)) != 0: #Avoid diving by 0
            w_dist[i]/=w_dist[i].sum((0,1,2))
       
    # Interpolation of iso
    If=np.sum(I*w_dist,(1,2,3))
    wIf=np.sum(wI*w_dist,(1,2,3))
    
    #Get aniso compartment
    S=np.zeros((nb_pts,3,3,3,ksize,d,d))
    for i in range(0,ksize):
        S[:,:,:,:,i]=array_to_cov_neighbor(mcm_cov[i][:,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]].transpose(1,2,3,4,0))
    
    #Interpolation of aniso
    w=w*w_dist[:,:,:,:,None]
    wf,Sf=proj_GMM_along(kmax,ksize,w.reshape(nb_pts,-1),S.reshape(nb_pts,-1,d,d),max_itr=50,eps=1e-7)
    
    #Compute distance target source
    D=distance_confidence_along(w.reshape(nb_pts,-1),wf,S.reshape(nb_pts,-1,d,d),Sf)
    conf=1/(D+1e-8)
    conf[conf==1e8]=np.min(conf[conf!=1e8]) #set conf for MAP with no aniso compartment

    #Compute most colinear compartment
    idx_col=most_colinear_compartment(streamline_coordinate,Sf)
    return wIf,If,wf,cov_to_array_MAS(Sf),conf,idx_col
    
    
def add_mcm_nolift(streamline_coordinate,origin_tract,step_tract,mcm_w,mcm_I,mcm_cov,mcm_coordinate,d=3):
    """
    Add MCM to every points of a streamline to build MAS.

    Parameters
    ----------
    streamline_coordinate : numpy array
        streamline coordinate
        Is shape is nb_ptsxd
        
    origin_tract : numpy array
        origin of the tract, from dipy header
        Is shape is d

    step_tract : numpy array
        step of the tract, from dipy header
        Is shape is d
        
    mcm_w : numpy array
        Weights image of the MCM
        Is shape is nb_compartment x size_img

    mcm_I : numpy array
        Value of the isotropic compartment
        Is shape is size_img

    mcm_cov : list of numpy array
        covariance matrices of the anisotropic compartments
        Is shape is nb_compartment_anisotropic x 6 x size_img

    mcm_coordinate: numpy array
        Coordinate of the image
        Is shape is d x size_img

    d: int
        Dimension
        
    Returns
    -------
    wIf : numpy array
        weight of the isotropic compartment along the MAS
        Is shape is nb_pts

    If : numpy array
        value of the isotropic compartment along the MAS
        Is shape is nb_pts

    wf : numpy array
        weights of the anisotropic compartments along the MAS
        Is shape is nb_pts x nb_compartments_aniso

    Sf : numpy array
        Covariates matrices of the anisotropic compartments along the MAS
        Is shape is nb_pts x (nb_compartments_aniso x 6)

    conf : numpy array
        Confidence indicator. When no lift conf=1 
        Is shape is nb_pts

    idx_col : numpy array
        Indices of the most colinear compartment along the MAS
        Is shape is nb_pts 
    """

    nb_pts=streamline_coordinate.shape[0]
    ksize=mcm_w.shape[0]-1 #shape of the array aka maximum number of compartment
    
    #Compute closest voxel from tract coordinate then compute its neigbord
    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
    #if np.all((streamline_coordinate[0]*step_tract-mcm_coordinate[:,idx[0,0],idx[0,1],idx[0,2]]*step_img)>1.25):
    #    print("mauvais voxel",Flush=True)
        
    #Get weights of compartements and value of iso compartment 
    If=mcm_I[idx[:,0],idx[:,1],idx[:,2]]
    wIf=mcm_w[0,idx[:,0],idx[:,1],idx[:,2]]
    wf=mcm_w[1:,idx[:,0],idx[:,1],idx[:,2]].transpose(1,0)
    
    #Get aniso compartment
    Sf=np.zeros((nb_pts,ksize,d,d))
    for i in range(0,ksize):
        Sf[:,i]=array_to_cov_MAS(mcm_cov[i][:,idx[:,0],idx[:,1],idx[:,2]].transpose(1,0))[:,0]

    #Compute most colinear compartment
    idx_col=most_colinear_compartment(streamline_coordinate,Sf)
    conf=np.ones(nb_pts)
    return wIf,If,wf,cov_to_array_MAS(Sf),conf,idx_col
    