"""
Compute the image of Mean Diffusivity (MD)
"""

import numpy as np
import nrrd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input DTI")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained MD image")

print("Compute MD")

args = parser.parse_args()
path=args.path
output_path=args.output_path

DTI,header=nrrd.read(path)

MD=(DTI[0,:,:,:]+DTI[2,:,:,:]+DTI[5,:,:,:])/3


header["sizes"]=np.array([DTI.shape[1],DTI.shape[2],DTI.shape[3]])
header["space directions"]=header['space directions'][1:]
header["kinds"]=header['kinds'][1:]

nrrd.write(output_path, MD,header=header)