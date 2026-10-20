"""
Compute the "and" and "or" mask of two bundles 
"""

import numpy as np
import nrrd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--masks',nargs='+',type=str,required=True,help="path masks")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained noisy mcm")


args = parser.parse_args()
masks=args.masks
output_path=args.output_path

print("Start computing and/or masks")

mask1=nrrd.read(masks[0])[0]
mask2,header=nrrd.read(masks[1])

mask_and=np.int8(np.logical_and(mask1,mask2))
mask_or=np.int8(np.logical_or(mask1,mask2))

nrrd.write(output_path+'_and.nrrd',mask_and,header=header)
nrrd.write(output_path+'_or.nrrd',mask_or,header=header)
