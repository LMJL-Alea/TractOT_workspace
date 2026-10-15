"""
Compute the most colinear compartment given a mcm and a set of streamlines

"""

import numpy as np
import nrrd
import argparse
from dipy.io.streamline import load_tractogram
from dipy.io.utils import Space
from tractot.utils import most_colinear_compartment,array_to_cov_MAS

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input folder's mcm")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained noisy mcm")
parser.add_argument('-t','--tractogram_path',type=str,required=True,help="tractogram path")
parser.add_argument('-n','--name_tractogram',type=str,required=True,help="tractogram name")
parser.add_argument('-r','--reference_path',type=str,required=True,help="reference path")
parser.add_argument('-m','--mcm_name',type=str,required=True,help="name of the mcm")


args = parser.parse_args()
path=args.path
output_path=args.output_path
tract_path=args.tractogram_path
name_tract=args.name_tractogram
reference_path=args.reference_path
mcm_name=args.mcm_name


mcm_w,header=nrrd.read(path+mcm_name+''
'_weights.nrrd')
sizes=header['sizes']
sizes[0]=3
header["sizes"]=sizes
ksize=mcm_w.shape[0]-1 #shape of the array aka maximum number of compartment
    
#Load aniso compartment
mcm_cov=[]
for i in range(ksize):
    mcm_cov+=[nrrd.read(path+mcm_name+'_'+str(i+1)+'.nrrd')[0]]

#Load tract
tract = load_tractogram(
            tract_path, 
            reference=reference_path,to_space=Space.LPSMM,
            bbox_valid_check=False,
            trk_header_check=False)
origin_tract=tract.space_attributes[0][:3,-1]
step_tract=np.diag(tract.space_attributes[0])[:3]

nb_streamline=len(tract.streamlines)
            
img_most_colinear=np.zeros((ksize,)+mcm_w.shape[1:])

print(nb_streamline,end='->',flush=True)
for si in range(nb_streamline): #nb of streamlines
    streamline_coordinate=tract.streamlines[si] #Get coordinate of the streamline
    nb_pts=streamline_coordinate.shape[0]
    
    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
          
    #Get aniso compartment
    Sf=np.zeros((nb_pts,ksize,3,3))
    for i in range(0,ksize):
        Sf[:,i]=array_to_cov_MAS(mcm_cov[i][:,idx[:,0],idx[:,1],idx[:,2]].transpose(1,0))[:,0]
    
    
    #Compute most colinear compartment
    idx_col=most_colinear_compartment(streamline_coordinate,Sf)

    img_most_colinear[:,idx[:,0],idx[:,1],idx[:,2]]=0 # Fix to avoid having several compartments
    idx,index=np.unique(idx,axis=0,return_index=True) # Not ideal, should be update
    img_most_colinear[idx_col[index],idx[:,0],idx[:,1],idx[:,2]]=1
    
nrrd.write(output_path+'most_colinear_'+name_tract+'.nrrd',img_most_colinear,header=header)


