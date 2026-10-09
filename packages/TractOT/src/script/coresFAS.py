"""
Compute cores FAS.
Cluster FAS in L group and then average over spatial and FA information
"""
import argparse
import numpy as np
from dipy.io.streamline import load_tractogram,save_tractogram
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('-t','--augmented_tractogram_path', type=str,required=True,help="path of input augmented tractogram")
parser.add_argument('-o','--output_path_cores_FAS',type=str,required=True,help="output path of obtained cores FAS in .trx")
parser.add_argument('-s','--svm_path',type=str,required=False,help="path of the learned svm")
parser.add_argument('-L','--number_of_cores_FAS',type=int,required=True,help="number of cores FAS")
parser.add_argument('-k','--number_of_dirac_cores_FAS',type=int,required=True,help="number of dirac at each FAP of cores FAS")
parser.add_argument('-c','--most_colinear',type=str,required=True,help="If true, compute FA of the most colinear compartment, then augmented_tractogram must be a MAS")
parser.add_argument('-e','--seed',type=int,required=False,help="seed")

args = parser.parse_args()
augmented_tractogram_path=args.augmented_tractogram_path
output_path_cores_FAS=args.output_path_cores_FAS
svm_path=args.svm_path
L=args.number_of_cores_FAS
k=args.number_of_dirac_cores_FAS
most_colinear=args.most_colinear
seed=args.seed

from TractOT.load_to_numpy import trx_to_numpy_FAT_array,trx_to_numpy_MAT_array
from TractOT.build_cores import infer_labels_svm_endpoint
from TractOT.build_cores import average_FAS_per_labels

FAS = load_tractogram(augmented_tractogram_path, reference='same',bbox_valid_check=False) #can be FAS or MAS

if most_colinear == "True":
    m,_,_,_,S,_,_=trx_to_numpy_MAT_array(FAS)
    nb_pts=S.shape[1]
    colinear=np.array(FAS.data_per_point["colinear"])
    S_colinear=np.zeros(S.shape[:-3]+(3,3,))
    for i in range(len(FAS.streamlines)):
        S_colinear[:,:]=S[i,np.arange(nb_pts),colinear[i,np.arange(nb_pts),0]]
        
    D=np.linalg.eigh(S_colinear)[0]
    MD=D.mean(-1)
    sum_sq_D=(D**2).sum(-1)
    FA=np.sqrt(2/3*((((D-MD[:,:,None])**2).sum(-1))/sum_sq_D))  
    
else:
    m,FA,_=trx_to_numpy_FAT_array(FAS)

print("Predict parcellation",flush=True)
labels_svm,labels_clus=infer_labels_svm_endpoint(svm_path,L,m,random_state=seed)

cores_m,cores_w,cores_FA,idx_svm,idx_cluster,size_cluster=average_FAS_per_labels(L,m,FA,labels_svm,labels_clus,k=k,antecedant=False) 

cores=FAS.from_sft(cores_m,FAS)
dico={}
dico['weights']=cores_w
dico['FA']=cores_FA
dico['svm']=idx_svm
dico['cluster']=idx_cluster
dico['size cluster']=size_cluster

cores.data_per_point=dico
save_tractogram(cores,output_path_cores_FAS,bbox_valid_check=False)
print('',flush=True)