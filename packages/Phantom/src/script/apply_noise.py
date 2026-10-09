"""
Add noise to a MCM.
"""


import numpy as np
import nrrd
import ot
import argparse
import shutil

from Phantom.add_noise_mcm import add_noise_weights,add_noise_eigvals,add_noise_rotation

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input folder's mcm")
parser.add_argument('-g','--original_path', type=str,required=True,help="path of original folder's mcm")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained noisy mcm")
parser.add_argument('-m','--mask_path',type=str,required=True,help="mask where applied noise")
parser.add_argument('-e','--epsilon',nargs='+',type=float,required=True,help="3 epsilon for noise")

args = parser.parse_args()
path=args.path
output_path=args.output_path
mask_path=args.mask_path
eps=args.epsilon
original_path=args.original_path


from Phantom.add_noise_mcm import add_noise_weights,add_noise_eigvals,add_noise_rotation
from TractOT.distances import distances_iso_aniso_along

mask=np.bool(nrrd.read(mask_path)[0])
mcm_w,header_w=nrrd.read(path+'MCM_avg_aligned_weights.nrrd')
mcm_w=mcm_w.transpose((1,2,3,0))
ksize=mcm_w.shape[-1]-1 #shape of the array aka maximum number of compartment

I,header_I=nrrd.read(path+'MCM_avg_aligned_'+str(0)+'.nrrd')

mcm_cov=np.zeros(mcm_w.shape[:-1]+(ksize,3,3,))
for i in range(ksize):
    compartment_i,header_cov=nrrd.read(path+'MCM_avg_aligned_'+str(i+1)+'.nrrd')
    compartment_i=compartment_i.transpose(1,2,3,0)
    mcm_cov[:,:,:,i,0,0]=compartment_i[:,:,:,0]
    mcm_cov[:,:,:,i,:2,1]=compartment_i[:,:,:,1:3]
    mcm_cov[:,:,:,i,:,2]=compartment_i[:,:,:,3:]
    mcm_cov[:,:,:,i,1,0]=compartment_i[:,:,:,1]
    mcm_cov[:,:,:,i,2,:2]=compartment_i[:,:,:,3:5]

#mask= mcm_w.sum(-1) !=0

#noise weights
mcm_w_noisy = add_noise_weights(mcm_w,mask,eps[0])

D,V=np.linalg.eigh(mcm_cov)

#noise eigenvalues
D_noisy=add_noise_eigvals(D,mask,eps[1])
mcm_cov_noisy_eigvals=np.einsum('...ij,...j,...jk->...ik',V,D_noisy,V.transpose((0,1,2,3,5,4)))

#noise rotation
mcm_cov_noisy=add_noise_rotation(mcm_cov_noisy_eigvals,mask,eps[2])
    
#Write
nrrd.write(output_path+'MCM_avg_aligned/MCM_avg_aligned_weights.nrrd',mcm_w_noisy.transpose((3,0,1,2)),header=header_w)
nrrd.write(output_path+'MCM_avg_aligned/MCM_avg_aligned_0.nrrd',I,header=header_I)

for i in range(ksize):
    compartment_i=np.zeros((6,)+mcm_cov.shape[:-3])
    compartment_i[0]=mcm_cov_noisy[:,:,:,i,0,0]
    compartment_i[1]=mcm_cov_noisy[:,:,:,i,0,1]
    compartment_i[2]=mcm_cov_noisy[:,:,:,i,1,1]
    compartment_i[3]=mcm_cov_noisy[:,:,:,i,0,2]
    compartment_i[4]=mcm_cov_noisy[:,:,:,i,1,2]
    compartment_i[5]=mcm_cov_noisy[:,:,:,i,2,2]
    nrrd.write(output_path+'MCM_avg_aligned/MCM_avg_aligned_'+str(i+1)+'.nrrd', compartment_i,header=header_cov)
    
shutil.copyfile(original_path+'MCM_avg_aligned_'+str(ksize)+'cpt_aniso.mcm',output_path+'MCM_avg_aligned.mcm')


"""
print("Compute distance to original",flush=True)
mcm_w_original=nrrd.read(original_path+'MCM_avg_aligned/MCM_avg_aligned_weights.nrrd')[0].transpose((1,2,3,0))
mcm_cov_original=np.zeros(mcm_w.shape[:-1]+(ksize,3,3,))
for i in range(ksize):
    compartment_i=nrrd.read(original_path+'MCM_avg_aligned/MCM_avg_aligned_'+str(i+1)+'.nrrd')[0]
    compartment_i=compartment_i.transpose(1,2,3,0)
    mcm_cov_original[:,:,:,i,0,0]=compartment_i[:,:,:,0]
    mcm_cov_original[:,:,:,i,:2,1]=compartment_i[:,:,:,1:3]
    mcm_cov_original[:,:,:,i,:,2]=compartment_i[:,:,:,3:]
    mcm_cov_original[:,:,:,i,1,0]=compartment_i[:,:,:,1]
    mcm_cov_original[:,:,:,i,2,:2]=compartment_i[:,:,:,3:5]

C=distances_iso_aniso_along(I[mask],I[mask],mcm_cov_original[mask],mcm_cov_noisy[mask])

distances_1d_array=np.zeros(C.shape[0])
for i in range(C.shape[0]):
    distances_1d_array[i]=ot.emd2(mcm_w_original[mask][i],mcm_w_noisy[mask][i],C[i])
    
distances_nrrd=np.zeros(mcm_cov.shape[:-3])
distances_nrrd[mask]=distances_1d_array

nrrd.write(output_path+'MCM_avg_aligned/MCM_avg_aligned_distance_iso_to_original.nrrd',distances_nrrd,header=header_I)
"""

