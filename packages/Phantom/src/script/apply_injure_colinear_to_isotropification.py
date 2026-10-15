"""
Simulate injury.
Injury is applied to the most colinear compartment
"""

import numpy as np
import nrrd
import argparse

from phantom.add_injury_mcm import colinear_to_isotropification

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path of input folder's mcm")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path of obtained noisy mcm")
parser.add_argument('-m','--mask_path',type=str,required=True,help="mask where applied noise")
parser.add_argument('-c','--most_colinear_compartment_path',type=str,required=True,help="most colinear compartment")
parser.add_argument('-w','--weight_injury',nargs='+',type=float,required=True,help="quantity weight to from most colinear compartment to a new isotropic compartment")
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

#Load weights and covariances
mcm_w,header_w=nrrd.read(path+mcm_name_input+'_weights.nrrd')
mcm_w=mcm_w.transpose((1,2,3,0))
ksize=mcm_w.shape[-1]-1 #shape of the array aka maximum number of compartment
size_img=mcm_w.shape[:-1]
d=3

mcm_cov=np.zeros(mcm_w.shape[:-1]+(ksize,d,d,))
for i in range(ksize):
    compartment_i,header_cov=nrrd.read(path+mcm_name_input+'_'+str(i+1)+'.nrrd')
    compartment_i=compartment_i.transpose(1,2,3,0)
    mcm_cov[:,:,:,i,0,0]=compartment_i[:,:,:,0]
    mcm_cov[:,:,:,i,:2,1]=compartment_i[:,:,:,1:3]
    mcm_cov[:,:,:,i,:,2]=compartment_i[:,:,:,3:]
    mcm_cov[:,:,:,i,1,0]=compartment_i[:,:,:,1]
    mcm_cov[:,:,:,i,2,:2]=compartment_i[:,:,:,3:5]


D,V=np.linalg.eigh(mcm_cov)

#Transfer weights to new compartment
mcm_w,D_new_isotropification,mcm_w_isotropification=colinear_to_isotropification(D,mcm_w,mask_and_colinear,w_injury)

mcm_cov_new_isotropification=np.zeros(size_img+(d,d))

I=np.tile(np.eye(d),D_new_isotropification.shape[:-1]+(1,))
mcm_cov_new_isotropification=np.einsum('...ij,...j->...ij',I,D_new_isotropification)

mcm_w=np.concatenate((mcm_w,mcm_w_isotropification),axis=3)

#Write
new_compartment=np.zeros((6,)+mcm_cov.shape[:-3])
new_compartment[0]=mcm_cov_new_isotropification[:,:,:,0,0]
new_compartment[1]=mcm_cov_new_isotropification[:,:,:,0,1]
new_compartment[2]=mcm_cov_new_isotropification[:,:,:,1,1]
new_compartment[3]=mcm_cov_new_isotropification[:,:,:,0,2]
new_compartment[4]=mcm_cov_new_isotropification[:,:,:,1,2]
new_compartment[5]=mcm_cov_new_isotropification[:,:,:,2,2]
nrrd.write(output_path+mcm_name_output+'_'+str(ksize+1)+'.nrrd', new_compartment,header=header_cov)

mcm_w=mcm_w.transpose((3,0,1,2))
header_w['sizes'][0]+=1
nrrd.write(output_path+mcm_name_output+'_weights.nrrd', mcm_w,header=header_w)