"""
Given a set of streamlines, set the number of points each streamlines.
Points are determined to be distributed over a uniform grid

"""

import argparse
from dipy.io.streamline import load_tractogram,save_tractogram

from preprocessing.new_grid import newgrid_streamline

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-r','--reference_path',type=str,required=True,help="reference image or .trk")
parser.add_argument('-n','--nb_pts', type=int,required=True,help="whished number of points on each streamlines")

args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
nb_pts=args.nb_pts

tract = load_tractogram(
            input_path,
            reference=reference,
            bbox_valid_check=False)
nb_streamline=len(tract.streamlines)

m=[]
for i in range(nb_streamline):
    m+=[newgrid_streamline(tract.streamlines[i],nb_pts=nb_pts)]

tract.streamlines=m

save_tractogram(tract,output_path,bbox_valid_check=False) 

