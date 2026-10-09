"""
Associate a FA image with a streamline to build FA Augmented Streamlines (FAS)
So far, it takes only the FA of the voxel where the streamlines lie (no lift)
"""

import numpy as np
import nrrd
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.io.utils import Space

parser = argparse.ArgumentParser()
parser.add_argument('-t','--tractogram_path', type=str,required=True,help="path to the input tractogram")
parser.add_argument('-o','--output_path_tractogram',type=str,required=True,help="output path of obtained augmented tractogram in .trx")
parser.add_argument('-f','--FA_path',type=str,required=True,help="path to the FA image")
parser.add_argument('-s','--coordinate_path',type=str,required=True,help="path to the image of coordinate")
parser.add_argument('-r','--reference_path',type=str,required=False,help="path to the reference image")

args = parser.parse_args()
tractogram_path=args.tractogram_path
output_path_tractogram=args.output_path_tractogram
FA_path=args.FA_path
coordinate_path=args.coordinate_path

if args.reference_path:
    reference_path=args.reference_path
else:
    reference='same'


d=3


#Load coordinate of mcm image
FA_coordinate,header= nrrd.read(coordinate_path)
origin_img=header['space origin']
step_img=np.diag(header['space directions'][1:])

#Load FA image
FA_img,_=nrrd.read(FA_path)    
    

    
#Load tract
tract = load_tractogram(
            tractogram_path, 
            reference=reference_path,to_space=Space.LPSMM,
            bbox_valid_check=False,
            trk_header_check=False)
origin_tract=tract.space_attributes[0][:3,-1]
step_tract=np.diag(tract.space_attributes[0])[:3]

nb_streamline=len(tract.streamlines)
            
list_m,list_FA=[],[]
     
R=np.diag(np.array([1,1,1]))
     
print(nb_streamline,end='->',flush=True)
for si in range(0,nb_streamline): #nb of streamlines
#for si in range(1,3):
    if si%50==0:
        print(" ",si,end=' ',flush=True)
    streamline_coordinate=tract.streamlines[si]#@R #Get coordinate of the streamline
    nb_pts=streamline_coordinate.shape[0]
    
    #Compute closest voxel from tract coordinate then compute its neigbord
    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
    #if np.all((streamline_coordinate[0]*step_tract-mcm_coordinate[:,idx[0,0],idx[0,1],idx[0,2]]*step_img)>1.25):
    #    print("mauvais voxel",Flush=True)
        
    
    #Get FA
    FAf=FA_img[idx[:,0],idx[:,1],idx[:,2]]
    
    list_m+=[streamline_coordinate]
    list_FA+=[FAf]

print((np.array(list_FA)==0).sum())
tract.streamlines=list_m
dico={}
dico['FA']=list_FA
tract.data_per_point=dico
save_tractogram(tract,output_path_tractogram,bbox_valid_check=False)
print('',flush=True)