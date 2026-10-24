import numpy as np
from sklearn.cluster import KMeans
import pickle

from tractot.proj_GMM import proj_GMM_along
from tractot.utils import cov_to_array_MAS


#Avoid sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def average_SAS_per_labels(n_clusters,m,Scalar,labels_svm,labels_clus,k=1,antecedent=False):       
    """
    Compute cores SAS 

    This function compute cores SAS by k-means of scalar, given clusters
    At each SAP of the cores, SA is represented as a probability  

    Parameters
    ----------
    n_clusters : positive integer
        Number of target cores SAS

    m : numpy array 
        Euclidean coordinate of the SAS. 
        Is shape is nb_SASxnb_SAPx3

    Scalar : numpy array
        An array of scalar. 
        Is shape is nb_SASxnb_SAP

    labels_svm: numpy array
        Label of each SAP given by an SVM 
        Is shape is nb_SASxnb_SAP

    labels_clus: numpy array
        Label of each SAS
        Is shape is nb_SAS

    k: positive integer
        Number of scalar for each SAP of the output
    antecedent: Boolean
        Indices of the antecedent for the averages
        
    Returns
    Note: returns list since cores SAS might have different number of SAP
    -------
    cores_m : list of numpy array
        A list of array for the Euclidean coordinate of the cores SAS
        List has length n_clusters

    cores_w: list of numpy array
        A list of array for weights of the probability scalar of the cores SAS
        List has length n_clustersxnb_SAPxk

    cores_scalar: list of numpy array
        A list of array of the probability scalar of the cores SAS
        List has length n_clustersxnb_SAPxk

    idx_svm_l: list of numpy array
        A list with svm labels of each point of the cores SAS 

    idx_cluster: list of numpy array
        A list with cluster labels of each point of the cores SAS
        Note that each SAS as the label along its SAP

    size_cluster: list of numpy array
        A list containing the number of SAP that have been averaging to obtain each SAP of the cores SAS

    idx_antecedent: list of list
        A list containing the indices of the antecedent for each averaged SAP
    """
    nb_pts=m.shape[1]
        
    cores_m,cores_w,cores_scalar,idx_svm_l,idx_cluster,size_cluster,idx_antecedent=[],[],[],[],[],[],[]
    for i in range(n_clusters):
        print('l = ',i,end=' : ',flush=True)
        
        nb_pts_i=np.unique(labels_svm[labels_clus==i]).shape[0]
        idx_cluster_i=i*np.ones(nb_pts_i)
        
        cores_m_i=np.zeros((nb_pts_i,3))
        cores_w_i=np.zeros((nb_pts_i,k))
        cores_scalar_i=np.zeros((nb_pts_i,k))
        idx_svm_i=np.zeros(nb_pts_i)
        size_cluster_i=np.zeros(nb_pts_i)
        idx_antecedent_i=[]
        
        for j,idx_svm in enumerate(np.unique(labels_svm[labels_clus==i])):
            print(idx_svm,end=' ',flush=True)
            idx_svm_i[j]=idx_svm
            #print(size_cluster,end=' ')
            
            size_cluster_i[j]=(m[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]).shape[0]
            

            #spatial
            cores_m_i[j]=np.mean(m[labels_clus==i][labels_svm[labels_clus==i]==idx_svm],axis=0)
            
            #Scalar
            k_i=np.int8(min(k,size_cluster_i[j])) #Ensure that k<size_cluster_i
            kmeans = KMeans(n_clusters=k_i, random_state=0).fit(Scalar[labels_clus==i][labels_svm[labels_clus==i]==idx_svm].reshape(-1,1))
            kf=np.unique(kmeans.labels_).shape[0] #Deals with repetition of centroids
            cores_scalar_i[j,:kf]=kmeans.cluster_centers_[np.unique(kmeans.labels_),0]
            cores_w_i[j,:kf]=np.unique(kmeans.labels_, return_counts=True)[1]
            cores_w_i[j]/=np.sum(cores_w_i[j])
            
            idx_antecedent_i+=[np.logical_and((labels_clus==i)[:,None],labels_svm==idx_svm)]
  

        cores_m+=[cores_m_i]
        cores_w+=[cores_w_i]
        cores_scalar+=[cores_scalar_i]
        idx_cluster+=[idx_cluster_i]
        idx_svm_l+=[idx_svm_i]
        size_cluster+=[size_cluster_i]
        idx_antecedent+=[np.array(idx_antecedent_i,dtype=bool).reshape(nb_pts,-1)]
        print()
    
    if antecedent:
        return cores_m,cores_w,cores_scalar,idx_svm_l,idx_cluster,size_cluster,idx_antecedent
    else:
        return cores_m,cores_w,cores_scalar,idx_svm_l,idx_cluster,size_cluster




def average_MAS_per_labels(n_clusters,m,wI,I,w,S,conf,labels_svm,labels_clus,k2=None,antecedent=False):       
    """
    Compute cores MAS 

    This function compute cores MAS by averaging of MAS, given clusters  

    Parameters
    ----------
    n_clusters : positive integer
        Number of target cores MAS
    m : numpy array 
        Euclidean coordinate of the MAS. 
        Is shape is nb_MASxnb_MAPx3
    wI : numpy array
        An array of positive weights for the isotropic comparment. 
        Is shape is nb_MASxnb_MAP
    I: numpy array
        Array of the eigenvalues for the isotropic covariate
        Is shape is nb_MASxnb_MAP
    w: numpy array
        An array of positive weights for the anisotropic comparment.
        Is shape is nb_MASxnb_MAPxnb_compartement
    S: positive scalar
        An array of the covariates matrices for the anisotropic part
        Is shape is nb_MASxnb_MAPxnb_compartmentxdxd
    conf: numpy array
        Confidence score for the points of the MAS
        Is shape is nb_MASxnb_MAP
    labels_svm: numpy array
        Label of each MAP given by an SVM 
        Is shape is nb_MASxnb_MAP
    labels_clus: numpy array
        Label of each MAS
        Is shape is nb_MAS
    k2: positive integer
        Number of compartment for the output
        If k2 is None, the number of compartment is the maximum number of compartments of the antecedent 
    antecedent: Boolean
        Indices of the antecedent for the averages
        
    Returns
    Note returns list since cores MAS might have different number of MAP
    -------
    cores_m : list of numpy array
        A list of array for the Euclidean coordinate of the cores MAS
        List has length n_clusters
    cores_wI: list of numpy array
        A list of array for weights of the isotropic compartment of the cores MAS
        List has length n_clusters
    cores_I: list of numpy array
        A list of array for eigenvalue of the isotropic compartment of the cores MAS
        List has length n_clusters
    cores_w: list of numpy array
        A list of array for weights of the anisotropic compartment of the cores MAS
        List has length n_clusters
    cores_S: list of numpy array
        A list of array for the anisotropic covariates the cores MAS
        List has length n_clusters
    idx_svm_l: list of numpy array
        A list with svm labels of each point of the cores MAS 
    idx_cluster: list of numpy array
        A list with cluster labels of each point of the cores MAS
        Note that each MAS as the label along its MAP
    size_cluster: list of numpy array
        A list containg the number of MAP that have been averaging to obtain each MAP of the cores MAS
    idx_antecedent: list of list
        A list containg the indices of the antecedent for each averaged MAP
    """
    nb_pts=m.shape[1]
    if k2 is None:
        K_size=w.shape[-1]
    else:
        K_size=k2
        
    cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm_l,idx_cluster,size_cluster,idx_antecedent=[],[],[],[],[],[],[],[],[]
    
    for i in range(n_clusters):
        print('l = ',i,end=' : ',flush=True)
        
        nb_pts_i=np.unique(labels_svm[labels_clus==i]).shape[0]
        idx_cluster_i=i*np.ones(nb_pts_i)
        
        cores_m_i=np.zeros((nb_pts_i,3))
        cores_wI_i=np.zeros(nb_pts_i)
        cores_I_i=np.zeros(nb_pts_i)
        cores_w_i=np.zeros((nb_pts_i,K_size))
        cores_S_i=np.zeros((nb_pts_i,K_size,3,3))
        idx_svm_i=np.zeros(nb_pts_i)
        size_cluster_i=np.zeros(nb_pts_i)
        idx_antecedent_i=[]
        
        for j,idx_svm in enumerate(np.unique(labels_svm[labels_clus==i])):
            print(idx_svm,end=' ',flush=True)
            idx_svm_i[j]=idx_svm
            #print(size_cluster,end=' ')
            
            size_cluster_i[j]=(w[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]).shape[0]
            
            if k2 is None:
                k2=max(np.max(np.count_nonzero(w[labels_clus==i][labels_svm[labels_clus==i]==idx_svm],axis=1)),1) #prevent pb from MAP with all weight on iso
            #print('k2 ',k2)
            
            #Normalization of the score
            conf_idx_svm_clus=conf[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]
            
            #spatial
            cores_m_i[j]=np.mean(m[labels_clus==i][labels_svm[labels_clus==i]==idx_svm],axis=0)
            
            #iso
            cores_wI_i[j]=np.mean(wI[labels_clus==i][labels_svm[labels_clus==i]==idx_svm])
            cores_I_i[j]=np.mean(I[labels_clus==i][labels_svm[labels_clus==i]==idx_svm])
    
            #Aniso
            min_k2=min((w[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]*conf_idx_svm_clus[:,None]).shape[0],k2)
            cores_w_i[j][:k2],cores_S_i[j][:k2]=proj_GMM_along(min_k2,k2,
                                                   (w[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]*conf_idx_svm_clus[:,None]).reshape(1,-1),
                                                   S[labels_clus==i][labels_svm[labels_clus==i]==idx_svm].reshape(1,-1,3,3),
                                                   max_itr=30)                           
            if np.sum(cores_w_i[j]) != 0:
                cores_w_i[j]=(cores_w_i[j]/np.sum(cores_w_i[j]))*(1-cores_wI_i[j])
                
            idx_antecedent_i+=[np.logical_and((labels_clus==i)[:,None],labels_svm==idx_svm)]
            
        cores_m+=[cores_m_i]
        cores_wI+=[cores_wI_i]
        cores_I+=[cores_I_i]
        cores_w+=[cores_w_i]
        cores_S+=[cov_to_array_MAS(cores_S_i)]
        idx_cluster+=[idx_cluster_i]
        idx_svm_l+=[idx_svm_i]
        size_cluster+=[size_cluster_i]
        idx_antecedent+=[np.array(idx_antecedent_i,dtype=bool).reshape(nb_pts,-1)]
        print()
    if antecedent:
        return cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm_l,idx_cluster,size_cluster,idx_antecedent
    else: 
        return cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm_l,idx_cluster,size_cluster


def infer_labels_svm_endpoint(path_svm,n_clusters,m,random_state=0,nb_labels_m_endpoint=1):
    """
    Compute labels for MAS clustering by performing k-means over the end points of the MAS   

    Parameters
    ----------
    path_svm : string
        path to the stored svm
    n_clusters : positive integer 
        number of whished cluster
    m : numpy array 
        Euclidean coordinate of the MAS. 
        Is shape is nb_MASxnb_MAPx3
    random_state: positiveinteger
        seed for the k-mean algorithm
    nb_labels_m_endpoint: positive integer
        Number of labels to be consider endpoints of the tract
        
    Returns
    -------
    labels_svm : numpy array
        An array to labels each point of the MAS wrt to the svm parcellation
        Is shape is nb_MASxnb_MAP
    labels_clus: numpy array
        An array to label every MAS
        Is shape is nb_MAS
    """    
    nb_streamline,nb_pts=m.shape[0],m.shape[1]
    
    if path_svm is None:
        print("Infer label with index")
        labels_svm=np.repeat(np.arange(1,nb_pts+1)[None,:],nb_streamline,0)
    
    else:
        with open(path_svm, 'rb') as file:
            svc = pickle.load(file)
        labels_svm=svc.predict(X=m.reshape(-1,3)).reshape(nb_streamline,nb_pts)
        
    m_endpoint=m[:,-nb_labels_m_endpoint][labels_svm[:,-nb_labels_m_endpoint]>=nb_pts-nb_labels_m_endpoint+1]
    
    print('kmeans over ',m_endpoint.shape,' endpoints')
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(m_endpoint)

    m_centroid_endpoint=kmeans.cluster_centers_
    labels_clus=kmeans.predict(m[:,-1])
    return labels_svm,labels_clus