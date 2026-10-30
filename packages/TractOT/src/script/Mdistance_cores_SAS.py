"""
Compute the matrix of distances between two sets of core SAS.
Matrix is computed wrt a set of dataset 
"""

import numpy as np
from dipy.io.streamline import load_tractogram
import argparse
import pandas as pd
import os.path


from tractot.distances import distance_coresSAS_svm
from tractot.load_to_numpy import trx_to_numpy_SAT_list


parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path to the dataset")
parser.add_argument('-o','--output_path', type=str,required=True,help="output path for distance matrix")
parser.add_argument('-n','--name_cores', type=str,required=True,help="name of the cores")
parser.add_argument('-T','--name_tract',type=str,required=True,help="name of the tract")
parser.add_argument('-L','--numbers_cores',type=str,required=True,help="number of cores SAS")
parser.add_argument('-k','--number_of_compartment_cores',type=str,required=True,help="number of compartment at each SAP of cores")
parser.add_argument('-d1','--dataset_1', type=str,required=True,help="first dataset")
parser.add_argument('-d2','--dataset_2', type=str,required=True,help="second dataset")
parser.add_argument('-s1','--subjects1', nargs='+', type=str,required=True,help='name of the subject for first data set')
parser.add_argument('-s2','--subjects2', nargs='+', type=str,required=True,help='name of the subject for second data set')
parser.add_argument('-a','--alpha', type=float,required=True,help='alpha parameter, trade off spatial/diffusion in distance')
parser.add_argument('-b','--beta', type=int,required=True,help='beta parameter, trade off spatial/diffusion in assignment')
parser.add_argument('-s','--sum_or_max',type=str,required=True,help="sum or max when comparing SAP along streamline and sets of cores")

args = parser.parse_args()
path=args.path
d1=args.dataset_1
d2=args.dataset_2
subjects1=args.subjects1
subjects2=args.subjects2
TRACT=args.name_tract
L=args.numbers_cores
alpha=args.alpha
beta=args.beta
output_path=args.output_path
name=args.name_cores
k=args.number_of_compartment_cores
sum_or_max=args.sum_or_max

print(subjects1)
    
# Order dataset to avoid redundance
if d1>d2: #'HCP_105'<'PPMI/Control','PPMI/Control'<'PPMI/PD','PPMI/PD'<'PPMI/Prodromal'
    d1,d2=d2,d1
    subjects1,subjects2=subjects2,subjects1


path1=path+str(d1)
path2=path+str(d2)
if "/" in d1:
    d1=d1.replace("/","_")
if "/" in d2:
    d2=d2.replace("/","_")

#Load all subject of the data
all_s1=pd.read_csv(path1+'/000000/all_subject.csv')["Subject ID"]
all_s2=pd.read_csv(path2+'/000000/all_subject.csv')["Subject ID"]

M = pd.DataFrame(np.zeros((len(all_s1),len(all_s2))), index=all_s1, columns=all_s2)


#If M does not exist yet save it, otherwise load it
if os.path.isfile(output_path+name+'_'+TRACT+'_L'+L+'_k'+k+'_'+d1+'_'+d2+'_alpha'+str(alpha)+'_beta'+str(beta)+'_'+sum_or_max+'.csv'):
    print('file already exist, loading it')
    M = pd.read_csv(output_path+name+'_'+TRACT+'_L'+L+'_k'+k+'_'+d1+'_'+d2+'_alpha'+str(alpha)+'_beta'+str(beta)+'_'+sum_or_max+'.csv',index_col=0)
else:  
    print('file does not exist writing it')
    M.to_csv(output_path+name+'_'+TRACT+'_L'+L+'_k'+k+'_'+d1+'_'+d2+'_alpha'+str(alpha)+'_beta'+str(beta)+'_'+sum_or_max+'.csv')
    M = pd.read_csv(output_path+name+'_'+TRACT+'_L'+L+'_k'+k+'_'+d1+'_'+d2+'_alpha'+str(alpha)+'_beta'+str(beta)+'_'+sum_or_max+'.csv',index_col=0)
M.index = M.index.astype(str)
M.columns = M.columns.astype(str)



print(d1,' vs ',d2,' start')

#Compute distance patients
if d1==d2:
    print('same dataset -> symmetric matrice')
    for i,s1 in enumerate(subjects1):
        #print(s1,end=': ',flush=True)
        print(i,end=' ',flush=True)
        #print(path1+"/"+s1+'/06-AugmentedTractsOT/7'+name+'_'+TRACT+'_L'+L+'_k'+k+'.trx')
        cores1 = load_tractogram(path1+"/"+s1+'/06-AugmentedTractsOT/7'+name+'_'+TRACT+'_L'+L+'_k'+k+'.trx', reference='same',bbox_valid_check=False)
        m1,w1,Scalar1,labels_svm1,_=trx_to_numpy_SAT_list(cores1)
        for j,s2 in enumerate(subjects2):
            if s1<s2:#Lexicographic order
                #print(path2+"/"+s2+'/06-AugmentedTractsOT/7'+name+'_'+TRACT+'_L'+L+'_k'+k+'.trx')
                #print(s2,end=' ',flush=True)
                if M.at[s1, s2] != 0:
                    print('Overwritting a distance',end=' ',flush=True)
                cores2 = load_tractogram(path2+"/"+s2+'/06-AugmentedTractsOT/7'+name+'_'+TRACT+'_L'+L+'_k'+k+'.trx', reference='same',bbox_valid_check=False)
                m2,w2,Scalar2,labels_svm2,_=trx_to_numpy_SAT_list(cores2)
                cost,_,_,_=distance_coresSAS_svm(m1,m2,w1,w2,Scalar1,Scalar2,labels_svm1,labels_svm2,alpha=alpha,beta=beta,sum_or_max=sum_or_max)
                M.at[s1, s2] = cost
                M.at[s2, s1] = cost
                #print(C_nb_pts.min())
        #print()

else:
    for i,s1 in enumerate(subjects1):
        #print(s1,end=': ',flush=True)
        print(i,end=' ',flush=True)
        cores1 = load_tractogram(path1+"/"+s1+'/06-AugmentedTractsOT/7'+name+'_'+TRACT+'_L'+L+'_k'+k+'.trx', reference='same',bbox_valid_check=False)
        m1,w1,Scalar1,labels_svm1,_=trx_to_numpy_SAT_list(cores1)
        for j,s2 in enumerate(subjects2):
                #print(s2,end=' ',flush=True)
                if M.at[s1, s2] != 0:
                    print('Overwritting a distance',end=' ',flush=True)
                cores2 = load_tractogram(path2+"/"+s2+'/06-AugmentedTractsOT/7'+name+'_'+TRACT+'_L'+L+'_k'+k+'.trx', reference='same',bbox_valid_check=False)
                m2,w2,Scalar2,labels_svm2,_=trx_to_numpy_SAT_list(cores2)                  
                cost,_,_,_=distance_coresSAS_svm(m1,m2,w1,w2,Scalar1,Scalar2,labels_svm1,labels_svm2,alpha=alpha,beta=beta,sum_or_max=sum_or_max)
                M.at[s1, s2] = cost
                #print(C_nb_pts.min())
        #print()

print('Writting result')
M.to_csv(output_path+name+'_'+TRACT+'_L'+L+'_k'+k+'_'+d1+'_'+d2+'_alpha'+str(alpha)+'_beta'+str(beta)+'_'+sum_or_max+'.csv')
print(d1,' vs ',d2,' done')
