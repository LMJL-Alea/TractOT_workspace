"""
down_sampling_spatial

Given a set of streamlines, reduce the number of streamlines.
Meaningful for the CST, to avoid density bias. 

"""

import numpy as np
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram

from preprocessing.down_sampling import down_sampling_tract

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-r','--reference_path',type=str,required=True,help="reference image or .trk")
parser.add_argument('-c','--nb_step', type=int,required=True,help="number of steps of the grid")

args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
nb_step=args.nb_step

tract = load_tractogram(
            input_path,
            reference=reference)

nb_streamline=len(tract.streamlines)
m1=np.array(tract.streamlines)

m2=down_sampling_tract(nb_step,m1,upper=-1)
tract.streamlines=m2
print("Obtained tracto has ",m2.shape[0],' streamlines',flush=True)
save_tractogram(tract,output_path)

