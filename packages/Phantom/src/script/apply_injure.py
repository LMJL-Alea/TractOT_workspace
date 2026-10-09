"""
Simulate injury.
Injury is applied to the most colinear compartment
"""

import numpy as np
import nrrd
import ot
import argparse

from TractOT.distances import distances_iso_aniso_along
from Phantom.add_injury_mcm import colinear_to_isotropification,colinear_to_free_water,colinear_to_reduced_signal,isotropification,reduce_signal

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input folder's mcm")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained noisy mcm")
parser.add_argument('-m','--mask_path',type=str,required=True,help="mask where applied noise")
parser.add_argument('-c','--most_colinear_compartment_path',type=str,required=True,help="most colinear compartment")
parser.add_argument('-t','--t',type=float,required=True,help="t for isotropification")
parser.add_argument('-l','--lowering',nargs='+',type=float,required=True,help="scales to lowering diffusion intensity")
parser.add_argument('-w','--w_injury',nargs='+',type=float,required=True,help="quantity of weights to transfer to the free water,reduced signal and isotropificated compartments")

args = parser.parse_args()
path=args.path
output_path=args.output_path
mask_path=args.mask_path
most_colinear_compartment_path=args.most_colinear_compartment_path
t=args.t
low=args.lowering
w_injury=args.w_injury



mask=np.bool(nrrd.read(mask_path)[0])
mask_mostcolinear=np.bool(nrrd.read(most_colinear_compartment_path)[0]).transpose((1,2,3,0))

mcm_w,header_w=nrrd.read(path+'MCM_avg_aligned_weights.nrrd')
mcm_w=mcm_w.transpose((1,2,3,0))
ksize=mcm_w.shape[-1]-1 #shape of the array aka maximum number of compartment
size_img=mcm_w.shape[:-1]
d=3

mcm_cov=np.zeros(mcm_w.shape[:-1]+(ksize,d,d,))
for i in range(ksize):
    compartment_i,header_cov=nrrd.read(path+'MCM_avg_aligned_'+str(i+1)+'.nrrd')
    compartment_i=compartment_i.transpose(1,2,3,0)
    mcm_cov[:,:,:,i,0,0]=compartment_i[:,:,:,0]
    mcm_cov[:,:,:,i,:2,1]=compartment_i[:,:,:,1:3]
    mcm_cov[:,:,:,i,:,2]=compartment_i[:,:,:,3:]
    mcm_cov[:,:,:,i,1,0]=compartment_i[:,:,:,1]
    mcm_cov[:,:,:,i,2,:2]=compartment_i[:,:,:,3:5]


mask_and_colinear=np.logical_and(mask[:,:,:,None],mask_mostcolinear)

D,V=np.linalg.eigh(mcm_cov)

#Modify the most colinear compartment
D_reduced_signal=isotropification(D,mask_mostcolinear,t)
D_reduced_signal_isotropification=reduce_signal(D_reduced_signal,mask_mostcolinear,low[0])

mcm_cov_isotropificated_reduced_signal=np.where(mask[:,:,:,None,None,None],np.einsum('...ij,...j,...jk->...ik',V,D_reduced_signal_isotropification,V.transpose((0,1,2,3,5,4))),mcm_cov)



#Transfer weights to new compartment
mcm_w=colinear_to_free_water(mcm_w,mask,mask_and_colinear,w_injury[0])
mcm_w,D_new_reduced_signal,mcm_w_reduced_signal=colinear_to_reduced_signal(D,mcm_w,mask,mask_and_colinear,low[1],w_injury[1])
mcm_w,D_new_isotropification,mcm_w_isotropification=colinear_to_isotropification(D,mcm_w,mask,mask_and_colinear,w_injury[2])

mcm_cov_new_reduced_signal=np.zeros(size_img+(1,d,d))
mcm_cov_new_isotropification=np.zeros(size_img+(1,d,d))

mcm_cov_new_reduced_signal[np.array(mask_and_colinear.sum(-1),dtype=bool)]=np.einsum('...ij,...j,...jk->...ik',V[mask_and_colinear].reshape(-1,1,3,3),D_new_reduced_signal[np.array(mask_and_colinear.sum(-1),dtype=bool)],V[mask_and_colinear].transpose((0,2,1)).reshape(-1,1,3,3))

I=np.tile(np.eye(d),D_new_isotropification.shape[:-1]+(1,1,))
mcm_cov_new_isotropification=np.einsum('...ij,...j->...ij',I,D_new_isotropification)

if w_injury[1] != 0:
    ksize+=1
    mcm_w=np.concatenate((mcm_w,mcm_w_reduced_signal),axis=3)
    mcm_cov=np.concatenate((mcm_cov_isotropificated_reduced_signal,mcm_cov_new_reduced_signal),axis=3)
    
if w_injury[2] != 0:
    ksize+=1
    mcm_w=np.concatenate((mcm_w,mcm_w_isotropification),axis=3)
    mcm_cov=np.concatenate((mcm_cov,mcm_cov_new_isotropification),axis=3)

mcm_w=np.where(np.array(mask_and_colinear.sum(-1),dtype=bool)[:,:,:,None],mcm_w/mcm_w.sum(-1)[:,:,:,None],mcm_w)

#Write
for i in range(ksize):
    compartment_i=np.zeros((6,)+mcm_cov.shape[:-3])
    compartment_i[0]=mcm_cov[:,:,:,i,0,0]
    compartment_i[1]=mcm_cov[:,:,:,i,0,1]
    compartment_i[2]=mcm_cov[:,:,:,i,1,1]
    compartment_i[3]=mcm_cov[:,:,:,i,0,2]
    compartment_i[4]=mcm_cov[:,:,:,i,1,2]
    compartment_i[5]=mcm_cov[:,:,:,i,2,2]
    nrrd.write(output_path+'MCM_avg_aligned_'+str(i+1)+'.nrrd', compartment_i,header=header_cov)

mcm_w=mcm_w.transpose((3,0,1,2))
nrrd.write(output_path+'MCM_avg_aligned_weights.nrrd', mcm_w,header=header_w)