"""
Simulate injury.
Injury is applied to the most colinear compartment
"""

import numpy as np
import nrrd
import argparse

from phantom.add_injury_mcm import colinear_to_free_water

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input folder's mcm")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained noisy mcm")
parser.add_argument('-m','--mask_path',type=str,required=True,help="mask where applied noise")
parser.add_argument('-c','--most_colinear_compartment_path',type=str,required=True,help="most colinear compartment")
parser.add_argument('-w','--weight_injury',nargs='+',type=float,required=True,help="quantity weight to from most colinear compartment to free water compartment")
parser.add_argument('-n1','--mcm_name_input',type=str,required=True,help="name of the input mcm")
parser.add_argument('-n2','--mcm_name_output',type=str,required=True,help="name of the output mcm")


args = parser.parse_args()
path=args.path
output_path=args.output_path
mask_path=args.mask_path
most_colinear_compartment_path=args.most_colinear_compartment_path
w_injury=args.weight_injury
mcm_name_input=args.mcm_name_input
mcm_name_output=args.mcm_name_output

#Load mask
mask=np.bool(nrrd.read(mask_path)[0])
mask_mostcolinear=np.bool(nrrd.read(most_colinear_compartment_path)[0]).transpose((1,2,3,0))
mask_and_colinear=np.logical_and(mask[:,:,:,None],mask_mostcolinear)

#Load weights
mcm_w,header_w=nrrd.read(path+mcm_name_input+'_weights.nrrd')
mcm_w=mcm_w.transpose((1,2,3,0))
ksize=mcm_w.shape[-1]-1 #shape of the array aka maximum number of compartment


#Transfer weights to new compartment
mcm_w=colinear_to_free_water(mcm_w,mask_and_colinear,w_injury)

#Write
mcm_w=mcm_w.transpose((3,0,1,2))
nrrd.write(output_path+mcm_name_output+'_weights.nrrd', mcm_w,header=header_w)