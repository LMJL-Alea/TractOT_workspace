"""
Compute the bundle mask, given a set of streamlines.
"""


import numpy as np
from scipy.ndimage import binary_dilation
from dipy.io.streamline import load_tractogram
import nrrd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--tract_path', type=str,required=True,help="path of input tract")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained mask")
parser.add_argument('-e','--header_path',type=str,required=True,help="header")
parser.add_argument('-c','--coordinate_path',type=str,required=True,help="coordinate")
parser.add_argument('-d','--dilation',type=int,required=True,help="dilation of mask")

args = parser.parse_args()
tract_path=args.tract_path
output_path=args.output_path
header_path=args.header_path
coordinate_path=args.coordinate_path
dilation=args.dilation

print("Start computing masks for tracts")

coordinate=nrrd.read(coordinate_path)[0]
header=nrrd.read(header_path)[1]
    

tract=load_tractogram(tract_path,reference="same",
                    bbox_valid_check=False,
                    trk_header_check=False)
                    
origin_tract=tract.space_attributes[0][:3,-1]
step_tract=np.diag(tract.space_attributes[0])[:3]
#step_tract=step_tract*2

mask=np.zeros(coordinate.shape[1:])
mask_start=np.zeros(coordinate.shape[1:])
mask_end=np.zeros(coordinate.shape[1:])

nb_streamline=len(tract)
for si in range(0,nb_streamline): #nb of streamlines
    streamline_coordinate=tract.streamlines[si]
    streamline_coordinate_start=tract.streamlines[si][:8]
    streamline_coordinate_end=tract.streamlines[si][-8:]

    nb_pts=streamline_coordinate.shape[0]

    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
    idx_start=np.int32(np.round((streamline_coordinate_start-origin_tract)/step_tract))
    idx_end=np.int32(np.round((streamline_coordinate_end-origin_tract)/step_tract))
    
    mask[idx[:,0],idx[:,1],idx[:,2]]=1
    mask_start[idx_start[:,0],idx_start[:,1],idx_start[:,2]]=1
    mask_end[idx_end[:,0],idx_end[:,1],idx_end[:,2]]=1


mask = binary_dilation(mask, iterations=dilation).astype(np.uint8)
mask_start = binary_dilation(mask_start, iterations=dilation).astype(np.uint8)
mask_end = binary_dilation(mask_end, iterations=dilation).astype(np.uint8)

nrrd.write(output_path+'.nrrd',mask,header=header)
nrrd.write(output_path+'_start.nrrd',mask_start,header=header)
nrrd.write(output_path+'_end.nrrd',mask_end,header=header)
