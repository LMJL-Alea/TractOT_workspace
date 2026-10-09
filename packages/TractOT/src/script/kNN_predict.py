"""
Perform k-NN prediction of the label based on matrix of distance between two datasets 
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def list_of_str(arg):
    #print(arg.split(' '))
    return list(map(str, arg.split(' ')))

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str,required=True,help="path to distances matrices")
parser.add_argument('-d','--datasets',nargs='+',type=str,required=True,help="datasets")
parser.add_argument('-s','--subjects',nargs='+', type=list_of_str,required=True,help="list of subjects")
parser.add_argument('-n','--name', type=str,required=True,help="name of the cores MAS")
parser.add_argument('-o','--output_path',type=str,required=True,help="Output path for outcome")
parser.add_argument('-a','--alphas',nargs='+',type=float,required=True,help="alpha parameter")
parser.add_argument('-t','--tracts',nargs='+',type=str,required=True,help="alpha parameter")
parser.add_argument('-b','--beta',type=str,required=True,help="beta parameter")
parser.add_argument('-k','--number_compartment',type=str,required=True,help="number of compartment")
parser.add_argument('-L','--number_core',type=str,required=True,help="number of cores")
parser.add_argument('-m','--sum_or_max',type=str,required=True,help="sum or max")



args = parser.parse_args()
path=args.path
name=args.name
list_d=args.datasets
list_subjects=args.subjects
output_path=args.output_path
alphas=args.alphas
beta=args.beta
K=args.number_compartment
L=args.number_core
sum_or_max=args.sum_or_max
TRACTS=args.tracts

print(list_d)
print(list_subjects)

n_seed=10000
n_neighbors=3
train_size=0.75

nb_dataset=len(list_d)


list_lensum=[0]
for i in range(len(list_subjects)):
    list_lensum+=[list_lensum[i]+len(list_subjects[i])]

#Load matrices distance
M=np.zeros((len(TRACTS),len(alphas),list_lensum[-1],list_lensum[-1]))
for l,TRACT in enumerate(TRACTS):
    for k,alpha in enumerate(alphas): 
        for i in range(len(list_d)):
            for j in range(i,len(list_d)):
                #print(list_d[i],list_d[j])
                M_temp=pd.read_csv(path+name+'_'+TRACT+'_L'+L+'_k'+K+'_'+list_d[i]+'_'+list_d[j]+'_alpha'+str(alpha)+'_beta'+beta+'_'+sum_or_max+'.csv',index_col=0)
                M_temp.index = M_temp.index.astype(str)
                M_temp.columns = M_temp.columns.astype(str)
                print(l,alpha,i,j,end=' :')
                print(np.sum((M_temp.loc[list_subjects[i],list_subjects[j]]==0)))
                M[l,k,list_lensum[i]:list_lensum[i+1],list_lensum[j]:list_lensum[j+1]]= M_temp.loc[list_subjects[i],list_subjects[j]]
                if i!=j:
                    M[l,k,list_lensum[j]:list_lensum[j+1],list_lensum[i]:list_lensum[i+1]]= M_temp.loc[list_subjects[i],list_subjects[j]].T
                
#Prediction test
y=[]
for i in range(nb_dataset):
    y+=len(list_subjects[i])*[i]
print('y',y,flush=True)
X=np.arange(M.shape[2])

res=np.zeros((M.shape[0],M.shape[1]))
for i in range(len(TRACTS)):
    print(i,end=': ')
    for j in range(len(alphas)):
        for n in range(n_seed):
            X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=train_size)
            neigh = KNeighborsClassifier(metric='precomputed',n_neighbors=n_neighbors)
            neigh.fit(M[i,j,X_train][:,X_train], y_train)
            res[i,j]+=sum(neigh.predict(M[i,j,X_test][:,X_train])==y_test)
        res[i,j]/=(n_seed*len(y_test))
        print(j,end=' ',flush=True)
    print()

for i in range(nb_dataset):
    output_path+=list_d[i]

print(res,flush=True)
np.savetxt(output_path+'_'+name+'_L'+L+'_k'+K+'_beta'+beta+'_'+sum_or_max+'_s'+str(n_seed)+'_n'+str(n_neighbors)+'_t'+str(train_size)+'.txt',res)