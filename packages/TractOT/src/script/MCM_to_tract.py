"""
Associate MCM images with a streamline to build Mixture Augmented Streamlines (MAS)
It can lift the neighbors voxels to ensure robustness, average is done in an optimal transport fashion.
"""


#Add MCM with an average over neigbhor voxels
# average in the sense of optimal tranpost metric
import numpy as np
import nrrd
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.io.utils import Space

from tractot.add_mcm_to_tract import add_mcm_lift,add_mcm_nolift


parser = argparse.ArgumentParser()
parser.add_argument('-t','--tractogram_path', type=str,required=True,help="path to input streamlines")
parser.add_argument('-o','--output_path_tractogram',type=str,required=True,help="output path of obtained augmented tractogram in .trx")
parser.add_argument('-c','--mcm_path',type=str,required=True,help="path of the mcm images")
parser.add_argument('-w','--weights_path',type=str,required=True,help="path of the weights image")
parser.add_argument('-s','--coordinate_path',type=str,required=True,help="path to the image of coordinate")
parser.add_argument('-r','--reference_path',type=str,required=False,help="path to the reference image")
parser.add_argument('-l','--lift',type=str,required=str,help="Lifting the neighbor voxels, True or False")

args = parser.parse_args()
tractogram_path=args.tractogram_path
output_path_tractogram=args.output_path_tractogram
mcm_path=args.mcm_path
weights_path=args.weights_path
coordinate_path=args.coordinate_path
lift=args.lift

if lift =='True':
    lift=True
if lift=='False':
    lift=False

if args.reference_path:
    reference_path=args.reference_path
else:
    reference='same'



d=3

#Load coordinate of mcm image
mcm_coordinate,header= nrrd.read(coordinate_path)
origin_img=header['space origin']
step_img=np.diag(header['space directions'][1:])

#Load weights of compartements and value of iso compartment
mcm_w,_=nrrd.read(weights_path)    
mcm_I,_=nrrd.read(mcm_path+'0.nrrd')
ksize=mcm_w.shape[0]-1 #shape of the array aka maximum number of compartment
    
#Load aniso compartment
mcm_cov=[]
for i in range(ksize):
    mcm_cov+=[nrrd.read(mcm_path+str(i+1)+'.nrrd')[0]]

    
#Load tract
tract = load_tractogram(
            tractogram_path, 
            reference=reference_path,to_space=Space.LPSMM,
            bbox_valid_check=False)
            
origin_tract=tract.space_attributes[0][:3,-1]
step_tract=np.diag(tract.space_attributes[0])[:3]

nb_streamline=len(tract.streamlines)
            
list_m,list_wI,list_I,list_w,list_S,list_conf,list_col=[],[],[],[],[],[],[]
            
print(nb_streamline,end='->',flush=True)
for si in range(nb_streamline): #nb of streamlines
    if si%10==0:
        print(" ",si,end=' ',flush=True)
    streamline_coordinate=tract.streamlines[si] #Get coordinate of the streamline

    if lift == True:
        wI,I,w,S,conf,idx_col=add_mcm_lift(streamline_coordinate,origin_tract,step_tract,mcm_w,mcm_I,mcm_cov,mcm_coordinate)
    else:
        wI,I,w,S,conf,idx_col=add_mcm_nolift(streamline_coordinate,origin_tract,step_tract,mcm_w,mcm_I,mcm_cov,mcm_coordinate)
         
    list_m+=[streamline_coordinate]
    list_wI+=[wI]
    list_I+=[I]
    list_w+=[w]
    list_S+=[S]
    list_conf+=[conf]
    list_col+=[idx_col]
 
#Normalize confidence 
max_confidence=0
for i in range(len(list_conf)):
    if max(list_conf[i])>max_confidence:
        max_confidence=max(list_conf[i])
for i in range(len(list_conf)):
    list_conf[i]/=max_confidence 
 
tract.streamlines=list_m
dico={}
dico['weight iso']=list_wI
dico['iso']=list_I
dico['weights aniso']=list_w
dico['aniso']=list_S
dico['confidence']=list_conf
dico['colinear']=list_col
tract.data_per_point=dico
save_tractogram(tract,output_path_tractogram,bbox_valid_check=False)
print('',flush=True)