"""
Perform tractography "à la TractSeg"
"""

import nrrd
import numpy as np
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.io.utils import Space
from tractseg.libs import tractseg_prob_tracking

#If you want to perform tractography for toy+- and Disco example, you might change a little bit the hyperparameters of the original tractseg_prob_tracking.py tractography module. Notably the step size

parser = argparse.ArgumentParser()
parser.add_argument('-t','--TOM',type=str,required=True,help="TOM")
parser.add_argument('-m','--mask',type=str,required=True,help="bundle mask mask")
parser.add_argument('-s','--start_mask',type=str,required=True,help="start mask")
parser.add_argument('-e','--end_mask',type=str,required=True,help="end mask")
parser.add_argument('-n','--number_of_streamline',type=int,required=True,help="number of streamline")
parser.add_argument('-r','--reference_tract',type=str,required=True,help="model of tract")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")



args = parser.parse_args()
TOM_path=args.TOM
mask_path=args.mask
start_mask_path=args.start_mask
end_mask_path=args.end_mask
nb_streamline=args.number_of_streamline
reference_tract_path=args.reference_tract
output_path=args.output_path


TOM,header=nrrd.read(TOM_path)
TOM=TOM.transpose((1,2,3,0))

bundle_mask=nrrd.read(mask_path)[0]
start_mask=nrrd.read(start_mask_path)[0]
end_mask=nrrd.read(end_mask_path)[0]

new_streamlines=tractseg_prob_tracking.track(TOM, max_nr_fibers=nb_streamline, smooth=False, compress=0.1, bundle_mask=bundle_mask,
            start_mask=start_mask, end_mask=end_mask, tracking_uncertainties=None, dilation=0,
            next_step_displacement_std=0.001, nr_cpus=-1, affine=np.eye(4), spacing=1, verbose=True)   
            
reference_streamline = load_tractogram(reference_tract_path,
                      reference='same',bbox_valid_check=False,to_space=Space.LPSMM)
                      
                      
new_streamlines=reference_streamline.from_sft(new_streamlines,reference_streamline)
save_tractogram(new_streamlines,output_path,bbox_valid_check=False)
