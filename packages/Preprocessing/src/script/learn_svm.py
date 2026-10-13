"""
Learn a svm to partition a bundle
"""

import numpy as np
import argparse
from dipy.io.streamline import load_tractogram
from sklearn.svm import SVC
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-r','--reference_path',type=str,required=True,help="reference image or .trk")
parser.add_argument('-n','--nb_streamlines_train', type=int,required=True,help="number of streamlines to train the SVM")
parser.add_argument('-g','--gamma', type=float,required=True,help="hyperparameter svm")
parser.add_argument('-c','--C', type=float,required=True,help="hyperparameter svm")
parser.add_argument('-k','--kernel', type=str,required=True,help="kernel type")

args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
n_train=args.nb_streamlines_train
gamma=args.gamma
C=args.C
kernel=args.kernel


tract_train = load_tractogram(
            input_path,
            reference=reference,
            bbox_valid_check=False)

n_train=min(n_train,len(tract_train.streamlines))
m_train,_=np.array(tract_train.streamlines[:n_train])
nb_pts=m_train.shape[1]

samples=m_train.reshape(-1,3).tolist()
classes=np.repeat(np.arange(1,nb_pts+1)[None,:],n_train,0).reshape(-1).tolist()

print()
print("train SVM",flush=True)
svc = SVC(C=C,gamma=gamma, kernel=kernel)
svc.fit(X=samples, y=classes)

print("save SVM",flush=True)
with open(output_path, 'wb') as file:
    pickle.dump(svc, file)
