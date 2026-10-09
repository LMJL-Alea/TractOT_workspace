"""
Compute Tract Orientation Mapping (TOM).
"""

import nrrd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--peaks',type=str,required=True,help="peaks image")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained noisy mcm")
parser.add_argument('-m','--mask_path',type=str,required=True,help="mask of the tract")
parser.add_argument('-g','--general_direction_path', type=str,required=True,help="general direction of the tract")

args = parser.parse_args()
peaks=args.peaks
output_path=args.output_path
mask_path=args.mask_path
general_direction_path=args.general_direction_path

# Compute TOM
peaks=nib.load(peaks).get_fdata()
peaks=np.where(abs(peaks)<1e-5,0,peaks)

mask,_=nrrd.read(mask_path)

general_dir=np.loadtxt(general_direction_path)

peaks_masked=peaks[mask!=0]
dirs=np.zeros((3,peaks_masked.shape[0],3))
dirs[0]=peaks_masked[:,:3]
dirs[1]=peaks_masked[:,3:6]
dirs[2]=peaks_masked[:,6:]
dirs/=np.linalg.norm(dirs,axis=-1)[:,:,None]

idx=np.argmin(np.arccos(abs(np.clip(np.sum(dirs*general_dir[None,None],-1),-1.0,1.0))),0)
most_colinear_dir=dirs[idx,np.arange(peaks_masked.shape[0])]

TOM=np.zeros((peaks.shape[:-1]+(3,)))
TOM[mask!=0]=most_colinear_dir
TOM=np.where(np.isnan(TOM),0,TOM)
TOM=TOM.transpose((3,0,1,2))

_,header2=nrrd.read("/home/gui/Documents/Post-doc/data/Disco/100/04-AlignedMCM/MCM_avg_aligned/MCM_avg_aligned_1.nrrd")
header2["sizes"]=np.array([40,40,40,3])

nrrd.write(output_path,TOM,header=header2)

