#Convert trk to vtk
import numpy as np
import argparse
from dipy.io.streamline import load_tractogram
from dipy.io.utils import Space
import argparse

from tractot.save_vtk import save_vtk

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path

print(input_path)

tract = load_tractogram(
            input_path,
            reference='same',
            bbox_valid_check=False,to_space=Space.RASMM)

dico={}
for name,value in tract.data_per_point.items():
    dico[name]=value
save_vtk(streamlines=tract.streamlines, output_vtk_path=output_path,scalar_dict=dico,to_lps=False)

