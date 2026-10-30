"""
Compute cores SAS.
Cluster SAS in L group and then average over spatial and scalar information
"""
import argparse
import numpy as np
from dipy.io.streamline import load_tractogram,save_tractogram

from tractot.load_to_numpy import trx_to_numpy_SAT_array,trx_to_numpy_MAT_array
from tractot.build_cores import infer_labels_svm_endpoint, average_SAS_per_labels

parser = argparse.ArgumentParser()
parser.add_argument('-t','--augmented_tractogram_path', type=str,required=True,help="path of input augmented tractogram")
parser.add_argument('-o','--output_path_cores_SAS',type=str,required=True,help="output path of obtained cores SAS in .trx")
parser.add_argument('-s','--svm_path',type=str,required=False,help="path of the learned svm")
parser.add_argument('-L','--number_of_cores_SAS',type=int,required=True,help="number of cores SAS")
parser.add_argument('-k','--number_of_dirac_cores_SAS',type=int,required=True,help="number of dirac at each SAP of cores SAS")
parser.add_argument('-c','--most_colinear',type=str,required=True,help="If true, compute Scalar of the most colinear compartment, then augmented_tractogram must be a MAS")
parser.add_argument('-e','--seed',type=int,required=False,help="seed")
parser.add_argument('-a','--antecedent',type=str,required=True,help="return antecedent of the average or not")

args = parser.parse_args()
augmented_tractogram_path=args.augmented_tractogram_path
output_path_cores_SAS=args.output_path_cores_SAS
svm_path=args.svm_path
L=args.number_of_cores_SAS
k=args.number_of_dirac_cores_SAS
most_colinear=args.most_colinear
seed=args.seed
antecedent=args.antecedent

if antecedent == "True":
    antecedent = True
else:
    antecedent = False

SAS = load_tractogram(augmented_tractogram_path, reference='same',bbox_valid_check=False) #can be a MAS if most_colinear == True

if most_colinear == "True":
    m,_,_,_,S,_,_=trx_to_numpy_MAT_array(SAS)
    nb_pts=S.shape[1]
    colinear=np.array(SAS.data_per_point["colinear"])
    S_colinear=np.zeros(S.shape[:-3]+(3,3,))
    for i in range(len(SAS.streamlines)):
        S_colinear[:,:]=S[i,np.arange(nb_pts),colinear[i,np.arange(nb_pts),0]]
        
    D=np.linalg.eigh(S_colinear)[0]
    MD=D.mean(-1)
    sum_sq_D=(D**2).sum(-1)
    Scalar=np.sqrt(2/3*((((D-MD[:,:,None])**2).sum(-1))/sum_sq_D))  
    
else:
    m,Scalar,_=trx_to_numpy_SAT_array(SAS)

print("Predict parcellation",flush=True)
labels_svm,labels_clus=infer_labels_svm_endpoint(svm_path,L,m,random_state=seed)


if antecedent:
    cores_m,cores_w,cores_Scalar,idx_svm,idx_cluster,size_cluster,idx_antecedent=average_SAS_per_labels(L,m,Scalar,labels_svm,labels_clus,k=k,antecedent=True) 
else:
    cores_m,cores_w,cores_Scalar,idx_svm,idx_cluster,size_cluster=average_SAS_per_labels(L,m,Scalar,labels_svm,labels_clus,k=k,antecedent=False) 



cores=SAS.from_sft(cores_m,SAS)
dico={}
dico['weights']=cores_w
dico['scalar']=cores_Scalar
dico['svm']=idx_svm
dico['cluster']=idx_cluster
dico['size cluster']=size_cluster
if antecedent:
    dico['antecedent']=idx_antecedent

cores.data_per_point=dico
save_tractogram(cores,output_path_cores_SAS,bbox_valid_check=False)
print('',flush=True)