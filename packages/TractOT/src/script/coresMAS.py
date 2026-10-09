"""
Compute cores MAS
MAP of the core MAS are average of MAP after clustering of MAS and svm parcellation
"""

import argparse
import numpy as np
from dipy.io.streamline import load_tractogram,save_tractogram
from sklearn.cluster import KMeans

from TractOT.load_to_numpy import trx_to_numpy_MAT_array
from TractOT.build_cores import average_MAS_per_labels,infer_labels_svm_endpoint

parser = argparse.ArgumentParser()
parser.add_argument('-t','--augmented_tractogram_path', type=str,required=True,help="path to the input augmented tractogram")
parser.add_argument('-o','--output_path_cores_MAS',type=str,required=True,help="output path of the obtained cores MAS in .trx")
parser.add_argument('-s','--svm_path',type=str,required=False,help="path of the learned svm")
parser.add_argument('-L','--number_of_cores_MAS',type=int,required=True,help="number of cores MAS")
parser.add_argument('-k','--number_of_compartment_cores_MAS',type=str,required=False,help="number of compartment at each MAP of cores MAS, if None takes the greater number of non null compartment over the averages MAP")
parser.add_argument('-c','--most_colinear',type=str,required=True,help="if true, average over the most colinear compartment")
parser.add_argument('-e','--seed',type=int,required=False,help="seed")

args = parser.parse_args()
augmented_tractogram_path=args.augmented_tractogram_path
output_path_cores_MAS=args.output_path_cores_MAS
svm_path=args.svm_path
L=args.number_of_cores_MAS
k2=args.number_of_compartment_cores_MAS
seed=args.seed
most_colinear=args.most_colinear

if k2=='None':
    k2=None
else:
    k2=int(k2)

MAS = load_tractogram(augmented_tractogram_path, reference='same',bbox_valid_check=False)
#print(output_path_cores_MAS,flush='True')
nb_pts=40
nb_streamline=len(MAS.streamlines)

m,wI,I,w,S,conf,_=trx_to_numpy_MAT_array(MAS,nb_streamline)
#print(m.shape,wI.shape,I.shape,w.shape,S.shape,conf.shape)

print("Predict parcellation",flush=True)
labels_svm,labels_clus=infer_labels_svm_endpoint(svm_path,L,m,random_state=seed,nb_labels_m_endpoint=1)
#print(labels_svm.shape,labels_clus.shape)

if most_colinear == "True":
    nb_pts=w.shape[1]
    colinear=np.array(MAS.data_per_point["colinear"])
    w_colinear=np.zeros(w.shape[:-1]+(1,))
    S_colinear=np.zeros(S.shape[:-3]+(1,3,3,))
    for i in range(nb_streamline):
        w_colinear[:,:,0]=w[i,np.arange(nb_pts),colinear[i,np.arange(nb_pts),0]]
        S_colinear[:,:,0]=S[i,np.arange(nb_pts),colinear[i,np.arange(nb_pts),0]]
    cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm,idx_cluster,size_cluster=average_MAS_per_labels(L,m,wI,I,w_colinear,S_colinear,conf,labels_svm,labels_clus,k2=k2)
else:
    cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm,idx_cluster,size_cluster=average_MAS_per_labels(L,m,wI,I,w,S,conf,labels_svm,labels_clus,k2=k2)
    
#print(len(cores_m),len(cores_wI),len(cores_I),len(cores_w),len(cores_S),len(idx_svm),len(idx_cluster),len(size_cluster))
#print(cores_m[0].shape,cores_wI[0].shape,cores_I[0].shape,cores_w[0].shape,cores_S[0].shape,idx_svm[0].shape,idx_cluster[0].shape,size_cluster[0].shape)


cores=MAS.from_sft(cores_m,MAS)
dico={}
dico['weight iso']=cores_wI
dico['iso']=cores_I
dico['weights aniso']=cores_w
dico['aniso']=cores_S
dico['svm']=idx_svm
dico['cluster']=idx_cluster
dico['size cluster']=size_cluster
cores.data_per_point=dico


save_tractogram(cores,output_path_cores_MAS,bbox_valid_check=False)
    
print('',flush=True)