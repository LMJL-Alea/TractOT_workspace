import numpy as np
from tractot.utils import array_to_cov_MAS

def trx_to_numpy_MAT_array(MAT,nb_MAS=None,streamlineId=None,d=3):
    """
    Convert a Mixture Augmented Tractogram encoded as StatefulTractogram into numpy arrays. 
    All MAS must have the same number of MAP 
    
    Parameters
    ----------
    MAT : StatefulTractogram (Dipy)
        Mixture Augmented Tractogram
        
    nb_MAS : int
        Number of MAS to load

    streamlineId : list
        list of index of the MAS to load
        Is length is nb_MAS
        
    d : int
        Dimension
        
    Returns 
    -------
    m : numpy array
        spatial information of the MAS
        Is shape is nb_MAS x nb_MAP x d

    wI : numpy array
        weights of the isotropic compartment
        Is shape is nb_MAS x nb_MAP

    I : numpy array
        value of the isotropic compartment
        Is shape is nb_MAS x nb_MAP

    w : numpy array
        weights of the anisotropic compartments
        Is shape is nb_MAS x nb_MAP x nb_compartment_aniso

    S : numpy array
        Covariates matrices of the anisotropic compartments
        Is shape is nb_MAS x nb_MAP x nb_compartment_aniso

    conf : numpy array
        Confidence indicator 
        Is shape is nb_MAS x nb_MAP

    nb_MAP : int
        number of points for the MAS
    """
    
    if nb_MAS is None:
        nb_MAS=len(MAT.streamlines)
    
    nb_MAP=MAT.streamlines[0].shape[0]
    k=np.int32(MAT.data_per_point["aniso"][0].shape[1]/6)

    m=np.zeros((nb_MAS,nb_MAP,d))
    wI=np.zeros((nb_MAS,nb_MAP))
    I=np.zeros((nb_MAS,nb_MAP))
    w=np.zeros((nb_MAS,nb_MAP,k))
    S=np.zeros((nb_MAS,nb_MAP,k,d,d))
    conf=np.zeros((nb_MAS,nb_MAP))
    
    for i in range(0,nb_MAS):
        if streamlineId is None:
            m[i] = MAT.streamlines[i]
            wI[i] = MAT.data_per_point["weight iso"][i].reshape(-1)
            I[i] = MAT.data_per_point["iso"][i].reshape(-1)
            w[i] = MAT.data_per_point["weights aniso"][i]
            S[i] = array_to_cov_MAS(MAT.data_per_point["aniso"][i])
            conf[i] = MAT.data_per_point["confidence"][i].reshape(-1)
        else:
            m[i] = MAT.streamlines[streamlineId[i]]            
            wI[i] = MAT.data_per_point["weight iso"][streamlineId[i]].reshape(-1)
            I[i] = MAT.data_per_point["iso"][streamlineId[i]].reshape(-1)
            w[i] = MAT.data_per_point["weights aniso"][streamlineId[i]]
            S[i] = array_to_cov_MAS(MAT.data_per_point["aniso"][streamlineId[i]])
            conf[i] = MAT.data_per_point["confidence"][streamlineId[i]].reshape(-1)
    return m,wI,I,w,S,conf,nb_MAP 


def trx_to_numpy_MAT_list(MAT,nb_MAT=None):
    """
    Convert a Mixture Augmented Tractogram as StatefulTractogram into list of numpy arrays. 
    Every MAS can have different number of MAP 
    Used for cores MAS
    
    Parameters
    ----------
    MAT : StatefulTractogram (Dipy)
        Mixture Augmented Tractogram
        
    nb_MAS : int
        Number of MAS to load
        
    Returns 
    -------
    m : list
        spatial information of the MAS
        Is shape is nb_MAS_i x nb_MAP_i x d

    wI : list
        weights of the isotropic compartment
        Is shape is nb_MAS_i x nb_MAP_i

    I : list
        value of the isotropic compartment
        Is shape is nb_MAS_i x nb_MAP_i

    w : list
        weights of the anisotropic compartments
        Is shape is nb_MAS_i x nb_MAP_i x nb_compartment_aniso

    S : list
        Covariates matrices of the anisotropic compartments
        Is shape is nb_MAS_i x nb_MAP_i x nb_compartment_aniso

    labels_svm : list 
        svm label assigned to each MAP 
        Is shape is nb_MAS_i x nb_MAP_i

    nb_MAP : int
        number of points for the MAS
    """
    
    if nb_MAT is None:
        nb_MAT=len(MAT.streamlines)
    
    m,wI,I,w,S,labels_svm,nb_MAP=[],[],[],[],[],[],[]
    
    for i in range(nb_MAT):
        m+=[MAT.streamlines[i]]
        wI+=[MAT.data_per_point["weight iso"][i].reshape(-1)]
        I+=[MAT.data_per_point["iso"][i].reshape(-1)]
        w+=[MAT.data_per_point["weights aniso"][i]]
        S+=[array_to_cov_MAS(MAT.data_per_point["aniso"][i])]
        labels_svm+=[MAT.data_per_point["svm"][i].reshape(-1)]
        nb_MAP+=[MAT.streamlines[i].shape[0]]
        
    return m,wI,I,w,S,labels_svm,nb_MAP



def trx_to_numpy_FAT_array(FAT,nb_FAS=None,streamlineId=None,d=3):
    """
    Convert a FA Augmented Tractogram as StatefulTractogram into numpy arrays. 
    All FAS must have the same number of FAP 
    
    Parameters
    ----------
    FAT : StatefulTractogram (Dipy)
        FA Augmented Tractogram
        
    nb_FAS : int
        Number of FAS to load

    streamlineId : list
        list of index of the FAS to load
        Is length is nb_FAS
        
    d : int
        Dimension
        
    Returns 
    -------
    m : numpy array
        spatial information of the FAS
        Is shape is nb_FAS x nb_FAP x d

    FA: numpy array
        FA of the FAS
        Is shape is nb_FAS x nb_FAP

    nb_FAP : int
        number of points for the FAS
    """
    
    if nb_FAS is None:
        nb_FAS=len(FAT.streamlines)
    
    nb_FAP=FAT.streamlines[0].shape[0]

    m=np.zeros((nb_FAS,nb_FAP,d))
    FA=np.zeros((nb_FAS,nb_FAP))
    
    for i in range(0,nb_FAS):
        if streamlineId is None:
            m[i] = FAT.streamlines[i]
            FA[i] = FAT.data_per_point["FA"][i].reshape(-1)
        else:
            m[i] = FAT.streamlines[streamlineId[i]]            
            FA[i] = FAT.data_per_point["FA"][streamlineId[i]].reshape(-1)

    return m,FA,nb_FAP
    
    
def trx_to_numpy_FAT_list(FAT,nb_FAS=None):
    """
    Convert a FA Augmented Tractogram as StatefulTractogram into list of numpy arrays. 
    Every FAS can have different number of FAP 
    Used for cores FAS
    
    Parameters
    ----------
    FAT : StatefulTractogram (Dipy)
        FA Augmented Tractogram
        
    nb_FAS : int
        Number of FAS to load
        
    Returns 
    -------
    m : list
        spatial information of the FAS
        Is shape is nb_FAS_i x nb_FAP_i x d

    w : list
        weights of the FA centroids
        Is shape is nb_FAS_i x nb_FAP_i x nb_centroid

    FA : list
        centroids of FA of the FAS
        Is shape is nb_FAS x nb_FAP x nb_centroids

    labels_svm : list 
        svm label assigned to each FAP 
        Is shape is nb_FAS_i x nb_FAP_i

    nb_FAP : int
        number of points for the FAS
    """
    if nb_FAS is None:
        nb_FAS=len(FAT.streamlines)
    
    m,w,FA,labels_svm,nb_FAP=[],[],[],[],[]
    for i in range(nb_FAS):
        m+=[FAT.streamlines[i]]
        w+=[FAT.data_per_point['weights'][i]]
        FA+=[FAT.data_per_point['FA'][i]]
        labels_svm+=[FAT.data_per_point['svm'][i].reshape(-1)]
        nb_FAP+=[FAT.streamlines[i].shape[0]]

    return m,w,FA,labels_svm,nb_FAP

"""
def vtk_to_numpy_tract_list(tract,nb_streamline=1,streamlineId=None,d=3):
    #Transform a .vtk file of tract into a list of numpy array 
    nb_pts=[]
    m=[]
    for i in range(0,nb_streamline):
        if streamlineId is None:
            streamline = tract.streamlines[i]
        else:
            streamline = tract.streamlines[streamlineId[i]]
        
        nb_pts+=[streamline.shape[0]]    
        m+=[streamline]
    return m,nb_pts

def vtk_to_numpy_tract_array(tract,nb_streamline=1,streamlineId=None,d=3):
    #Transform a .vtk file of tract into numpy array
    #Every streamline must have the same nb of points
    
    nb_pts=tract.streamlines[0].shape[0]
    m=np.zeros((nb_streamline,nb_pts,d))
    
    for i in range(0,nb_streamline):
        if streamlineId is None:
            m[i] = tract.streamlines[i]
        else:
            m[i] = tract.streamlines[streamlineId[i]]
    
    return m,nb_pts
"""
