"""
From a set of MAS, construct a MAS with 1 compartment which correspond to the most colinear one
"""


import argparse
import numpy as np
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.io.utils import Space
from tractot.utils import array_to_cov_MAS, cov_to_array_MAS


parser = argparse.ArgumentParser()
parser.add_argument('-t','--augmented_tractogram_path', type=str,required=True,help="path of input augmented tractogram")
parser.add_argument('-o','--output_path_colinear_MAS',type=str,required=True,help="output path of obtained MAS in .trx")

args = parser.parse_args()
augmented_tractogram_path=args.augmented_tractogram_path
output_path_colinear_MAS=args.output_path_colinear_MAS


#Load
MAS = load_tractogram(augmented_tractogram_path, reference='same',to_space=Space.LPSMM)
nb_MAS=len(MAS.streamlines)
nb_MAP=MAS.streamlines[0].shape[0] 


w2=[]
S2=[]
k=np.int32(MAS.data_per_point["aniso"][0].shape[1]/6)
S=np.zeros((nb_MAS,nb_MAP,k,3,3))
w = MAS.data_per_point["weights aniso"]

for i in range(nb_MAS):
    colinear=np.int16(MAS.data_per_point['colinear'][i]).reshape(-1)
    w2+=[w[i][np.arange(nb_MAP),colinear].reshape(-1,1)]
    S = (array_to_cov_MAS(MAS.data_per_point["aniso"][i])[np.arange(nb_MAP),colinear]).reshape(-1,1,3,3)
    S2+=[cov_to_array_MAS(S)]
    
MAS.data_per_point['weights aniso']=w2
MAS.data_per_point['aniso']=S2

save_tractogram(MAS,output_path_colinear_MAS)
print('',flush=True)