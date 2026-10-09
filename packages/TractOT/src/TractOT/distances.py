import numpy as np
import ot


from TractOT.utils import sqrtms,concat_w_MAP,concat_w_MAS

def distances_aniso(S1,S2):
    """
    Compute the matrix of Bures/Wasserstein distance between two sets of covariates matrices

    Parameters
    ----------
    Si: numpy array
        An array of k_i covariate matrices 
        Is shape is k_ix3x3
        i=1,2
        
    Returns 
    -------
    numpy array
    Matrix of pairwise Bures/Wasserstein distances between two covariates matrices
    Is shape is k_1xk_2
        
    """
    sq1=sqrtms(S1)
    C=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_C=sqrtms(C)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)[:,None]
    tr2 = np.trace(S2,axis1=-2,axis2=-1)[None,:]  # (1, k_t)
    return np.maximum(tr1 + tr2 -2*np.trace(sq_C,axis1=-2,axis2=-1),0)


def distances_aniso_along(S1,S2): 
    """
    Compute the matrix of Bures/Wasserstein distance between two sets of covariates matrices along their first dimension

    Parameters
    ----------
    Si: numpy array
        An array of nxk_i covariate matrices 
        Is shape is nxk_ix3x3
        i=1,2
        
    Returns
    -------
    numpy array
    Matrix of pairwise Bures/Wasserstein distances between two covariates matrices along n
    Is shape is nxk_1xk_2
        
    """
    sq1=sqrtms(S1)
    C=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_C=sqrtms(C)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)[:,:,None]
    tr2 = np.trace(S2,axis1=-2,axis2=-1)[:,None,:]  # (1, k_t)
    return np.maximum(tr1 + tr2 -2*np.trace(sq_C,axis1=-2,axis2=-1),0)
    
def distances_iso_aniso(I1,I2,S1,S2):  
    """
    Compute the matrix of distances between two set of iso and aniso covariates matrices

    Parameters
    ----------
    Ii : numpy array
        value of the isotropic compartment
        Is shape is 1
        i=1,2
         
    Si: numpy array
        An array of k_i covariate matrices 
        Is shape is k_ix3x3
        i=1,2
        
    Returns
    -------
    numpy array
    Matrix of pairwise distances between the two sets of iso and aniso covariates matrices
    Is shape is (1+k_1)x(1+k_2)
        
    """
    k1,d=S1.shape[0],S1.shape[-1]
    k2=S2.shape[0]
    C=np.zeros((k1+1,k2+1))
    
    sq1=sqrtms(S1)
    sq2=sqrtms(S2)
    A=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_A=sqrtms(A)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)
    tr2 = np.trace(S2,axis1=-2,axis2=-1) 

    C[1:,1:]=tr1[:,None] + tr2[None,:] -2*np.trace(sq_A,axis1=-2,axis2=-1)

    C[0,0]=d*(I1+I2-2*np.sqrt(I1*I2))
    C[0,1:]=d*I1+tr2-2*np.sqrt(I1)*np.trace(sq2,axis1=-2,axis2=-1)
    C[1:,0]=d*I2+tr1-2*np.sqrt(I2)*np.trace(sq1,axis1=-2,axis2=-1)
    return np.maximum(C,0)
    
def distances_iso_aniso_along(I1,I2,S1,S2):
    """
    Compute the matrix of Bures/Wasserstein distances between two set of iso and aniso covariates matrix along their first dimension

    Parameters
    ----------
    Ii : numpy array
        value of the isotropic compartment along first dimension
        Is shape is nx1
        i=1,2
         
    Si: numpy array
        An array of k_i covariate matrices along first dimension
        Is shape is nxk_ix3x3
        i=1,2
        
    Returns
    -------
    numpy array
    Matrix of pairwise distances between the two set of iso and aniso covariates matrices along n
    Is shape is nx(1+k_1)x(1+k_2)
        
    """
    
    nb_pts,k1,d=S1.shape[0],S1.shape[1],S1.shape[-1]
    k2=S2.shape[1]
    C=np.zeros((nb_pts,k1+1,k2+1))
    
    sq1=sqrtms(S1)
    sq2=sqrtms(S2)
    A=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_A=sqrtms(A)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)
    tr2 = np.trace(S2,axis1=-2,axis2=-1)
    C[:,1:,1:]=np.maximum(tr1[:,:,None] + tr2[:,None,:] -2*np.trace(sq_A,axis1=-2,axis2=-1),0)

    C[:,0,0]=d*(I1+I2-2*np.sqrt(I1*I2))
    C[:,0,1:]=d*I1[:,None]+tr2-2*np.sqrt(I1)[:,None]*np.trace(sq2,axis1=-2,axis2=-1)
    C[:,1:,0]=d*I2[:,None]+tr1-2*np.sqrt(I2)[:,None]*np.trace(sq1,axis1=-2,axis2=-1)
    return np.maximum(C,0)
    
def distances_spatial_iso_aniso(m1,m2,I1,I2,S1,S2,alpha=.5):
    """
    Compute the matrix of Bures/Wasserstein distances between two set of iso and aniso covariates matrices + spatial distance 

    Parameters
    ----------
    mi : numpy array
        spatial vector
        Is shape 3
        i=1,2
        
    Ii : numpy array
        value of the isotropic compartment
        Is shape is 1
        i=1,2
         
    Si: numpy array
        An array of k_i covariate matrices 
        Is shape is k_ix3x3
        i=1,2
        
    alpha: real number between 0 and 1
        Alpha is a trade off between spatial distance (alpha=1) and covariates distances (alpha=0)        
    
    Returns
    -------
    numpy array
    Matrix of (1-alpha)xdistances between two set of iso and aniso covariates matrices + alphaxspatial distance
    Is shape is (k_1+1)x(k_2+1)
        
    """
    return alpha*np.sum((m1-m2)**2)+(1-alpha)*distances_iso_aniso(I1,I2,S1,S2)
    
def distances_spatial_iso_aniso_along(m1,m2,I1,I2,S1,S2,alpha=.5):
    """
    Compute the matrix of Bures/Wasserstein distances between two set of iso and aniso covariates matrices + spatial distance along their first dimension

    Parameters
    ----------
    mi : numpy array
        spatial vector
        Is shape nx3
        i=1,2
        
    Ii : numpy array
        value of the isotropic compartment
        Is shape is nx1
        i=1,2
         
    Si: numpy array
        An array of k_i covariate matrices 
        Is shape is nxk_ix3x3
        i=1,2
        
    alpha: real number between 0 and 1
        Alpha is a trade off between spatial distance (alpha=1) and covariates distances (alpha=0)        
    
    Returns
    -------
    numpy array
    Matrix of (1-alpha)xdistances between two set of iso and aniso covariates matrices + alphaxspatial distance along n
    Is shape is nx(k_1+1)x(k_2+1)
        
    """
    return alpha*np.sum((m1-m2)**2,1)[:,None,None]+(1-alpha)*distances_iso_aniso_along(I1,I2,S1,S2)
    
def distance_MAP(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5):
    """
    Compute the distance between two MAPs, aka two Gaussians mixture with (at least) one isotropic compartment

    Parameters
    ----------
    mi : numpy array
        spatial information of the i-th MAP
        Is shape is 3
        i=1,2
        
    wIi : numpy array
        weight of the isotropic compartment of the i-th MAP
        Is shape 1
        i=1,2
         
    Ii : numpy array
        value of the isotropic compartment of the i-th MAP
        Is shape is 1
        i=1,2
        
    wi : numpy array
        weight of the anisotropics compartments of the i-th MAP
        Is shape is nb_compartment_i
        i=1,2
         
    Si: numpy array
        An array of covariate matrices for anisotropic compartments of the i-th MAP 
        Is shape is nb_compartment_ix3x3
        i=1,2
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0)
    
        
    Returns
    -------
    positive scalar
    MW_alpha distance between the two MAP 
        
    """
    C=distances_spatial_iso_aniso(m1,m2,I1,I2,S1,S2,alpha)
    
    w1_stack=concat_w_MAP(wI1,w1)
    w2_stack=concat_w_MAP(wI2,w2) 
    
    return ot.emd2(w1_stack,w2_stack,C)
    

    
def distance_MAS_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5): 
    """
    Compute the distance between two augmented MAS

    Parameters
    ----------
    mi : numpy array
        spatial information of the i-th streamline
        Is shape is nb_MAPx3
        i=1,2
        
    wIi : numpy array
        weight of the isotropic compartment of the i-th streamline
        Is shape is nb_MAP
        i=1,2
         
    Ii : numpy array
        value of the isotropic compartment of the i-th streamline
        Is shape is nb_MAP
        i=1,2
        
    wi : numpy array
        weight of the anisotropics compartments of the i-th streamline
        Is shape is nb_MAPxnb_compartment_i
        i=1,2
         
    Si: numpy array
        An array of covariate matrices for anisotropic compartments of the i-th streamline 
        Is shape is nb_MAPxnb_compartment_ix3x3
        i=1,2
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0)
    
        
    Returns
    -------
    d : positive scalar
        Return the average distance between the MAP 
        
    """
    C=distances_spatial_iso_aniso_along(m1,m2,I1,I2,S1,S2,alpha)
    nb_MAP=C.shape[0]

    D=0
    w1_stack=concat_w_MAS(wI1,w1)
    w2_stack=concat_w_MAS(wI2,w2)    
    for i in range(nb_MAP):
        #if np.sum(w1_stack) != 0 and np.sum(w2_stack)!=0: #for voxel out of image
        D+=ot.emd2(w1_stack[i],w2_stack[i],C[i])
    return D/nb_MAP
    
def distances_MAS_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=0.5):
    """
    Compute the matrix of distance between two sets of MAS

    Parameters
    ----------
    mi : numpy array
        spatial information of the i-th set of streamlines
        Is shape is nb_MASxnb_MAPx3
        i=1,2
        
    wIi : numpy array
        weight of the isotropic compartment of the i-th set of streamlines
        Is shape is nb_MASxnb_MAP
        i=1,2
         
    Ii : numpy array
        value of the isotropic compartment of the i-th set of streamlines
        Is shape is nb_MASxnb_MAP
        i=1,2
        
    wi : numpy array
        weight of the anisotropics compartments of the i-th set of streamlines
        Is shape is nb_MASxnb_MAPxnb_compartment with nb_compartment fix for each streamlines
        i=1,2
         
    Si: numpy array
        An array of covariate matrices for anisotropic compartments of the i-th set of streamlines 
        Is shape is nb_MAPxnb_compartment_ix3x3 with nb_compartment fix for each streamlines
        i=1,2
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0)
    
        
    Returns
    -------
    C : numpy array
        Matrix of pairwise distances between two streamlines
        Is shape is nb_MAS1xnb_MAS2
        
    """  
    
    nb_MAS1,nb_MAS2=m1.shape[0],m2.shape[0]

    
    C=np.zeros((nb_MAS1,nb_MAS2))
    for i in range(nb_MAS1):
        for j in range(nb_MAS2):
            C[i,j]=distance_MAS_idx(m1[i],m2[j],wI1[i],wI2[j],I1[i],I2[j],w1[i],w2[j],S1[i],S2[j],alpha)

    return C
    
    
def distance_confidence_along(w1,w2,S1,S2):
    """
    Compute the inverse of the confidence score from two sets of covariates matrices along their first dimension

    Parameters
    ----------
    wi : numpy array
        weight of the anisotropics compartments of the i-th set
        Is shape is nxnb_compartment_i with nb_compartment_i fix for each set
        i=1,2
         
    Si: numpy array
        An array of covariate matrices for anisotropic compartments of the i-th set 
        Is shape is nxnb_compartment_ix3x3 with nb_compartment_i fix for each set
        i=1,2
    
    Returns
    -------
    D : numpy array
        Array of inverse confidence score along n
        Is shape is n
        
    """
    C=distances_aniso_along(S1,S2)
    nb_MAP=C.shape[0]
    D=np.zeros(nb_MAP)
    for i in range(nb_MAP):
        D[i]=ot.emd2(w1[i],w2[i],C[i])
    return D
    
    
def distances_MAS_svm(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,labels_svm1,labels_svm2,alpha=0.5,sum_or_max=None):
    """
    Compute the matrix of pairwise distances between two sets of MAS with possibly different number of MAP for each MAS

    Parameters
    ----------
    mi : list of numpy array
        spatial information of the i-th set of MAS
        List has length nb_MAS_i and each elements has shape nb_MAP_jx3 
        i=1,2 and j in [1,40]
        
    wIi : list of numpy array
        weight of the isotropic compartment of the i-th set of MAS
        List has length nb_MAS_i and each elements has shape nb_MAP_j
        i=1,2 and j in [1,40]
         
    Ii : list of numpy array
        value of the isotropic compartment of the i-th set of MAS
        List has length nb_MAS_i and each elements has shape nb_MAP_j 
        i=1,2 and j in [1,40]
        
    wi : list of numpy array
        weight of the anisotropics compartments of the i-th set of MAS
        List has length nb_MAS_i and each elements has shape nb_MAP_jxnb_compartment_i 
        i=1,2 and j in [1,40]
         
    Si: list of numpy array
        An array of covariate matrices for anisotropic compartments of the i-th set of MAS
        List has length nb_MAS_i and each elements has shape nb_MAP_jxnb_compartment_ix3x3 
        i=1,2 and j in [1,40]
        
    labels_svmi : list of numpy array
        Labels of each MAP of the i-th set of MAS 
        List has length nb_MAS_i and each elements has shape nb_MAP_j 
        i=1,2 and j in [1,40]
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0)
    
    sum_or_max: string
        sum_or_max indicates if the distance along a MAS should be maximum or sum of MAP
        
    Returns
    -------
    C : numpy array
        Matrix of pairwise MW_alpha distances between two MAS
        Is shape is nb_MAS1xnb_MAS2
        
    C_nb_MAP : numpy array
        Number of MAP used to compute the distances between the two map. That is the intersection of the two set of labels
        Is shape is nb_MAS1xnb_MAS2
        
    """   
    L1,L2=len(m1),len(m2)
    C=np.zeros((L1,L2))
    C_nb_MAP=np.zeros((L1,L2))
    
    for l1 in range(L1):
        for l2 in range(L2):
            labels_svm_intersect=np.intersect1d(labels_svm1[l1],labels_svm2[l2])
            cost=[]
            for idx in labels_svm_intersect:
                if alpha==1:
                    cost+=[np.sum((m1[l1][labels_svm1[l1]==idx][0]-m2[l2][labels_svm2[l2]==idx][0])**2)]
                else:
                    cost+=[distance_MAP(m1[l1][labels_svm1[l1]==idx][0],m2[l2][labels_svm2[l2]==idx][0],
                        wI1[l1][labels_svm1[l1]==idx][0],wI2[l2][labels_svm2[l2]==idx][0],
                        I1[l1][labels_svm1[l1]==idx][0],I2[l2][labels_svm2[l2]==idx][0],
                        w1[l1][labels_svm1[l1]==idx][0],w2[l2][labels_svm2[l2]==idx][0],
                        S1[l1][labels_svm1[l1]==idx][0],S2[l2][labels_svm2[l2]==idx][0],
                        alpha=alpha)]
            if sum_or_max=='sum':
                C[l1,l2]=sum(cost)
            if sum_or_max=='max':
                C[l1,l2]=max(cost)
            C_nb_MAP[l1,l2]=labels_svm_intersect.shape[0]
    return C,C_nb_MAP #C//C_nb_MAP
    
def distance_cores_svm(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,labels_svm1,labels_svm2,alpha=0.5,beta=None,sum_or_max='sum'):    
    """
    Compute the distance between two sets of MAS with same number of MAS put possibly different number of MAP for each MAS

    Parameters
    ----------
    mi : list of numpy array
        spatial information of the i-th set of MAS
        List has length nb_MAS and each elements has shape nb_MAP_jx3 
        i=1,2 and j in [1,40]
        
    wIi : list of numpy array
        weight of the isotropic compartment of the i-th set of MAS
        List has length nb_MAS and each elements has shape nb_MAP_j
        i=1,2 and j in [1,40]
         
    Ii : list of numpy array
        value of the isotropic compartment of the i-th set of MAS
        List has length nb_MAS and each elements has shape nb_MAP_j 
        i=1,2 and j in [1,40]
        
    wi : list of numpy array
        weight of the anisotropics compartments of the i-th set of MAS
        List has length nb_MAS and each elements has shape nb_MAP_jxnb_compartment_i 
        i=1,2 and j in [1,40]
         
    Si: list of numpy array
        An array of covariate matrices for anisotropic compartments of the i-th set of MAS
        List has length nb_MAS and each elements has shape nb_MAP_jxnb_compartment_ix3x3 
        i=1,2 and j in [1,40]
        
    labels_svmi : list of numpy array
        Labels of each MAP of the i-th set of MAS 
        List has length nb_MAS and each elements has shape nb_MAP_j 
        i=1,2 and j in [1,40]
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0). Alpha acts on the cost matrix to compute cost  
        
    beta: real number between 0 and 1
        Beta is a trade off between comparing streamlines only with spatial information (beta=1)
        and only diffusion information (beta=0). Alpha acts on the cost matrix to compute transport plan
    
    sum_or_max: string
        sum_or_max indicates if the distance along a MAS should be maximum or sum of MAP
        
    Returns
    -------
    cost: positive number
        Distance between two sets of MAS, after optimal assignment of each MAS
        
    sigma: numpy array
        Optimal permutations between the two sets of MAS
        Is shape us nb_MAS
        
    C : numpy array
        Matrix of pairwise MW_alpha distances between two MAS
        Is shape is nb_MASxnb_MAS
        
    C_nb_MAP : numpy array
        Number of MAP used to compute the distances between the two map. That is the intersection of the two set of labels
        Is shape is nb_MASxnb_MAS
        
    """   
    if beta is None:
        beta=alpha
    #Compute assignement wrt to cost matrix computed with beta
    C,C_nb_MAP=distances_MAS_svm(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,labels_svm1,labels_svm2,alpha=beta,sum_or_max=sum_or_max)
    P=ot.emd([],[],C)
    sigma=np.nonzero(P)[1]
    
    #Compute OT distance wrt to cost matrix computed with alpha
    C,C_nb_MAP=distances_MAS_svm(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,labels_svm1,labels_svm2,alpha=alpha,sum_or_max=sum_or_max)
    C=C/P.shape[0] #renormalize
    
    if sum_or_max=='sum':
        cost=np.sum(C*P) #P sum to one
    if sum_or_max=='max':
        cost=np.max(C*P)
        
    return cost,sigma,C,C_nb_MAP  


def distances_FAS_svm(m1,m2,w1,w2,FA1,FA2,labels_svm1,labels_svm2,alpha=0.5,sum_or_max='sum'):
    """
    Compute the matrix of pairwise distances between two sets of FAS with possibly different number of FAP for each FAS

    Parameters
    ----------
    mi : list of numpy array
        spatial information of the i-th set of FAS
        List has length nb_FAS and each elements has shape nb_FAP_jx3 
        i=1,2 and j in [1,40]
        
    wi : list of numpy array
        weight of the FA probability for the i-th set of FAS
        List has length nb_FAS and each elements has shape nb_FAP_jxk
        i=1,2 and j in [1,40]
         
    FAi : list of numpy array
        values of the FA probability for the i-th set of FAS
        List has length nb_FAS and each elements has shape nb_FAP_jxk 
        i=1,2 and j in [1,40]
        
    labels_svmi : list of numpy array
        Labels of each MAP of the i-th set of MAS 
        List has length nb_MAS and each elements has shape nb_MAP_j 
        i=1,2 and j in [1,40]
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0). Alpha acts on the cost matrix to compute cost  
        
    beta: real number between 0 and 1
        Beta is a trade off between comparing streamlines only with spatial information (beta=1)
        and only diffusion information (beta=0). Alpha acts on the cost matrix to compute transport plan
    
    sum_or_max: string
        sum_or_max indicates if the distance along a FAS should be maximum or sum of FAP
        
        
    Returns
    -------
    C : numpy array
        Matrix of pairwise W distances between two FAS
        Is shape is nb_FASxnb_FAS
        
    C_nb_FAS : numpy array
        Number of FAP used to compute the distances between the two map. That is the intersection of the two set of labels
        Is shape is nb_FASxnb_FAS
        
    """   
    
    L1,L2=len(m1),len(m2)
    C=np.zeros((L1,L2))
    C_nb_FAP=np.zeros((L1,L2))
    
    for l1 in range(L1):
        for l2 in range(L2):
            labels_svm_intersect=np.intersect1d(labels_svm1[l1],labels_svm2[l2])
            cost=[]
            for idx in labels_svm_intersect:
                if alpha==1:
                    cost+=[np.sum((m1[l1][labels_svm1[l1]==idx][0]-m2[l2][labels_svm2[l2]==idx][0])**2)]
                else:
                    cost+=[alpha*np.sum((m1[l1][labels_svm1[l1]==idx][0]-m2[l2][labels_svm2[l2]==idx][0])**2)+
                       (1-alpha)*ot.emd2_1d(FA1[l1][labels_svm1[l1]==idx][0],FA2[l2][labels_svm2[l2]==idx][0],w1[l1][labels_svm1[l1]==idx][0],w2[l2][labels_svm2[l2]==idx][0],p=2)] #use the 1d ot
            if sum_or_max=='sum':
                C[l1,l2]=sum(cost)
            if sum_or_max=='max':
                C[l1,l2]=max(cost)
            C_nb_FAP[l1,l2]=labels_svm_intersect.shape[0]
    return C/C_nb_FAP,C_nb_FAP

def distance_coresFA_svm(m1,m2,w1,w2,FA1,FA2,labels_svm1,labels_svm2,alpha=0.5,beta=None,sum_or_max='sum'):    
    
    """
    Compute the distance between two sets of FAS with same number of FAS put possibly different number of FAP for each FAS

    Parameters
    ----------
    mi : list of numpy array
        spatial information of the i-th set of FAS
        List has length nb_FAS and each elements has shape nb_FAP_jx3 
        i=1,2 and j in [1,40]
        
    wi : list of numpy array
        weight of the FA probability for the i-th set of FAS
        List has length nb_FAS and each elements has shape nb_FAP_jxk
        i=1,2 and j in [1,40]
         
    FAi : list of numpy array
        values of the FA probability for the i-th set of FAS
        List has length nb_FAS and each elements has shape nb_FAP_jxk 
        i=1,2 and j in [1,40]
        
    labels_svmi : list of numpy array
        Labels of each MAP of the i-th set of MAS 
        List has length nb_MAS and each elements has shape nb_MAP_j 
        i=1,2 and j in [1,40]
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0). Alpha acts on the cost matrix to compute cost  
        
    beta: real number between 0 and 1
        Beta is a trade off between comparing streamlines only with spatial information (beta=1)
        and only diffusion information (beta=0). Alpha acts on the cost matrix to compute transport plan
    
    sum_or_max: string
        sum_or_max indicates if the distance along a FAS should be maximum or sum of FAP
        
        
    Returns
    -------
    cost: positive number
        Distance between two sets of FAS, after optimal assignment of each FAS
        
    sigma: numpy array
        Optimal permutations between the two sets of FAS
        Is shape us nb_FAS
        
    C : numpy array
        Matrix of pairwise W distances between two FAS
        Is shape is nb_FASxnb_FAS
        
    C_nb_FAS : numpy array
        Number of FAP used to compute the distances between the two map. That is the intersection of the two set of labels
        Is shape is nb_FASxnb_FAS
        
    """   


    if beta is None:
        beta=alpha
        
    #Compute assignement wrt to cost matrix computed with beta
    C,C_nb_FAP=distances_FAS_svm(m1,m2,w1,w2,FA1,FA2,labels_svm1,labels_svm2,alpha=beta)
    P=ot.emd([],[],C)
    sigma=np.nonzero(P)[1]
    
    #Compute OT distance wrt to cost matrix computed with alpha
    C,C_nb_FAP=distances_FAS_svm(m1,m2,w1,w2,FA1,FA2,labels_svm1,labels_svm2,alpha=alpha)
    C=C/P.shape[0] #renormalize
    
    if sum_or_max=='sum':
        cost=np.sum(C*P) #P sum to one
    if sum_or_max=='max':
        cost=np.max(C*P)
        
    return cost,sigma,C,C_nb_FAP 
    
'''
def distance_aniso(S1,S2):
    """
    Compute the Bures/Wasserstein distance between two covariate matrices

    Parameters
    ----------
    Si: numpy array
        Covariate matrices  
        Is shape 3x3
        i=1,2
    
    Returns
    -------
    positive scalar
    """
    sq1=sqrtms(S1)
    C=np.einsum("...kl,...lm,...mn->...kn",sq1,S2,sq1)
    sq_C=sqrtms(C)
    tr1 = np.trace(S1)
    tr2 = np.trace(S2) 
    return np.maximum(tr1 + tr2 -2*np.trace(sq_C),0)

def distances_aniso(S1,S2):
    """
    Compute the matrix of Bures/Wasserstein distance between two sets of covariates matrices

    Parameters
    ----------
    Si: numpy array
        An array of k_i covariate matrices 
        Is shape is k_ix3x3
        i=1,2
        
    Returns
    -------
    numpy array
    Matrix of pairwise Bures/Wasserstein distances between two covariates matrices
    Is shape is k_1xk_2
        
    """
    sq1=sqrtms(S1)
    C=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_C=sqrtms(C)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)[:,None]
    tr2 = np.trace(S2,axis1=-2,axis2=-1)[None,:]  # (1, k_t)
    return np.maximum(tr1 + tr2 -2*np.trace(sq_C,axis1=-2,axis2=-1),0)


def distance_MAP_along(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5):
    """
    Compute the distance between two MAPs, aka two Gaussians mixture with (at least) one isotropic compartment along their first dim

    Parameters
    ----------
    mi : numpy array
        spatial information of the i-th MAP
        Is shape is nx3
        i=1,2
        
    wIi : numpy array
        weight of the isotropic compartment of the i-th MAP
        Is shape n
        i=1,2
         
    Ii : numpy array
        value of the isotropic compartment of the i-th MAP
        Is shape is n
        i=1,2
        
    wi : numpy array
        weight of the anisotropics compartments of the i-th MAP
        Is shape is nxnb_compartment_i
        i=1,2
         
    Si: numpy array
        An array of covariate matrices for anisotropic compartments of the i-th MAP 
        Is shape nxis nb_compartment_ix3x3
        i=1,2
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0)
    
        
    Returns
    -------
    numpy array
    MW_alpha distances along nx2 MAP 
        
    """
    nb_MAP=m1.shape[0]

    C=distances_spatial_iso_aniso_along(m1,m2,I1,I2,S1,S2,alpha)
    D=np.zeros(nb_MAP)
    w1_stack=concat_w_MAS(wI1,w1)
    w2_stack=concat_w_MAS(wI2,w2)    
    for i in range(nb_MAP):
        D[i]=ot.emd2(w1_stack[i],w2_stack[i],C[i])
    return D



def distances_MAP(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5):
    """
    Compute the matrix of MW_alpha distances between two sets of MAP

    Parameters
    ----------
    mi : numpy array
        spatial information of the i-th MAP
        Is shape is nb_MAP_ix3
        i=1,2
        
    wIi : numpy array
        weight of the isotropic compartment of the i-th MAP
        Is shape is nb_MAP_i
        i=1,2
         
    Ii : numpy array
        value of the isotropic compartment of the i-th MAP
        Is shape is nb_MAP_i
        i=1,2
        
    wi : numpy array
        weight of the anisotropics compartments of the i-th MAP
        Is shape is nb_MAP_ixnb_compartment_i with nb_compartment fix for each set of MAP
        i=1,2
         
    Si: numpy array
        An array of covariate matrices for anisotropic compartments of the i-th MAP 
        Is shape is nb_MAPxnb_compartment_ix3x3 with nb_compartment fix for each set of MAP
        i=1,2
        
    alpha: real number between 0 and 1
        Alpha is a trade off between comparing streamlines only with spatial information (alpha=1)
        and only diffusion information (alpha=0)
    
        
    Returns
    -------
    numpy array
    Matrix of pairwise MW_alpha distances between the two sets of MAP
    Is shape is nb_MAP1xnb_MAP2
        
    """ 
    nb_MAPs1,nb_MAPs2=m1.shape[0],m2.shape[0]
    C=np.zeros((nb_MAPs1,nb_MAPs2))
    for i in range(nb_MAPs1):
        for j in range(nb_MAPs2):
            C[i,j]=distance_MAP(m1[i],m2[j],wI1[i],wI2[j],I1[i],I2[j],w1[i],w2[j],S1[i],S2[j],alpha)
    return C
    
def distance_MAS(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,wMAS1=None,wMAS2=None,alpha=.5):
    #m1,m2 in R(nb_MAP1,d),R(nb_MAP2,d)
    #wI1,wI2 in R(nb_MAP1),R(nb_MAP2,d)
    #I1,I2 in R(nb_MAP1),R(nb_MAP2,d)
    #w1 in R(nb_MAP1,k1) and w2 in R(nb_MAP2,k2)
    #S1 in R(nb_MAP1,k1,d,d) and S2 in R(nb_MAP2,k2,d,d)
    #return scalar

    if wMAS1 is None:
        wMAS1=np.ones(m1.shape[0])/m1.shape[0]
        
    if wMAS2 is None:
        wMAS2=np.ones(m2.shape[0])/m2.shape[0]
        
    C=distances_MAP(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha)
    return ot.emd2(wMAS1,wMAS2,C)
    
#from ICP import ICP_tracts    
def distance_MAT_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5,reflexion=False,lam1=0,lam2=0):
    #m1,m2 in R(nb_MAS,nb_MAP,d),R(nb_MAS,nb_MAP,d)
    #wI1,wI2 in R(nb_MAS,nb_MAP),R(nb_MAS,nb_MAP)
    #I1,I2 in R(nb_MAS,nb_MAP),R(nb_MAS,nb_MAP)
    #w1 in R(nb_MAS,nb_MAP,k1) and w2 in R(nb_MAS,nb_MAP,k2)
    #S1 in R(nb_MAS,nb_MAP,k1,d,d) and S2 in R(nb_MAS,nb_MAP,k2,d,d)
    #return scalar
    
    nb_MAS1=m1.shape[0]  
    nb_MAS=m1.shape[0]
    
    m1_new,m2_new,R,t,sigma=ICP_tracts(m1,m2,reflexion,lam1,lam2,max_itr=3)
    
    if reflexion:
        Ox=np.eye(3)
        Ox[0,0]=-1    
        S1_new=Ox@S1@Ox
    else:
        S1_new=S1
    
    w1_new=w1
    S1_new=R.T@S1_new@R
    wI1_new=wI1
    I1_new=I1 #iso invariant by rotation
    
    wI2_new=wI2[sigma]
    I2_new=I2[sigma]
    w2_new=w2[sigma]
    S2_new=S2[sigma]

    d=0
    for i in range(nb_MAS):
        d+=distance_MAS_idx(m1_new[i],m2_new[i],wI1_new[i],wI2_new[i],I1_new[i],I2_new[i],w1_new[i],w2_new[i],S1_new[i],S2_new[i],alpha=.5)
        d+=lam1*np.sum((np.eye(3)-R)**2)+lam2*np.sum(t**2)
    return d
'''