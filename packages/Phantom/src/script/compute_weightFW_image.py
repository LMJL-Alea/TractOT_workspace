"""
Compute the image of Free Water weight
"""

import numpy as np
import nrrd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input weight")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained weights of Free Water")

print("Compute weight free water compartment")

args = parser.parse_args()
path=args.path
output_path=args.output_path

mcm_w,header_w=nrrd.read(path)
w_FW=mcm_w[0]

header_w["sizes"]=np.array([mcm_w.shape[1],mcm_w.shape[2],mcm_w.shape[3]])
header_w["space directions"]=header_w['space directions'][1:]
header_w["kinds"]=header_w['kinds'][1:]

nrrd.write(output_path, w_FW,header=header_w)