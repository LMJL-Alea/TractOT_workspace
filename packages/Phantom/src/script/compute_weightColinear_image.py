"""
Compute the image of most colinear compartment weight
"""

import numpy as np
import nrrd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input weight")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained weights of Free Water")
parser.add_argument('-c','--most_colinear_compartment_path',type=str,required=True,help="most colinear compartment")

print("Compute weight colinear compartment")

args = parser.parse_args()
path=args.path
output_path=args.output_path
most_colinear_compartment_path=args.most_colinear_compartment_path

#Load mask and weights
mask_mostcolinear=np.bool(nrrd.read(most_colinear_compartment_path)[0])
mcm_w,header_w=nrrd.read(path)

w_colinear=np.zeros(mcm_w.shape[1:])
w_colinear[np.array(mask_mostcolinear.sum(0),dtype=bool)]=mcm_w[1:4][mask_mostcolinear]

header_w["sizes"]=np.array([mcm_w.shape[1],mcm_w.shape[2],mcm_w.shape[3]])
header_w["space directions"]=header_w['space directions'][1:]
header_w["kinds"]=header_w['kinds'][1:]
nrrd.write(output_path, w_colinear,header=header_w)