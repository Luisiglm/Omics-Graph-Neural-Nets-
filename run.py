# Script to Run the Experiments

import re
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import Preprocess
import layers
import models
import survival_definitions

# read gene names.
file_loc = 'F:\\genes_sans_bars.txt'
lineList = [line.rstrip('\n') for line in open(file_loc)]
names = list()
for i in range(len(lineList)):
    div = re.split(' ', lineList[i])
    names.append(div[0])

# read pathways

di = 'F:\\Pathways\\'
files_pways =  os.listdir(di)

paths_list = []

adj = np.zeros((len(names),len(names)))

pgenes = []
posgenes = []

for i in files_pways:
    adj_ = [line.rstrip('\n') for line in open('F:\\Pathways\\'+i)]
    for j in range(len(adj_)):
        div = re.split('\t', adj_[j])
        if (names.count(div[0])>0) and (names.count(div[1])>0):
            pos_i = names.index(div[0])
            pos_j = names.index(div[1])
            if pgenes.count(div[0])==0:
                pgenes.append(div[0])
                posgenes.append(pos_i)
            if pgenes.count(div[1])==0:
                pgenes.append(div[1])
                posgenes.append(pos_j)
            if div[2]=='inh':
                w = -1.0
            else:
                w = 1
            adj[pos_i,pos_j] = w
            #adj[pos_j,pos_i] = 1


#sort posgenes.
            
posgenes_sorted = posgenes.copy()
posgenes_sorted.sort()

adj_pways = adj[posgenes_sorted,:]
adj_pways = adj_pways[:,posgenes_sorted]

# reorder pgenes according to posgenes.
pgenes_sort = []
for i in range(len(posgenes_sorted)):
    idx = posgenes_sorted[i]
    idx_2 = posgenes.index(idx)
    pgenes_sort.append(pgenes[idx_2])

di = 'F:\\Datasets Firehose\\'
folders = list()

# get all folders.
for i in os.walk(di):
    print(i[0])
    folders.append(i[0])

# now we will go through the folders and remove from list which include gdac

folders_keep = list()

for i in folders:
    a = re.search("gdac", i)
    if (a == None):
        folders_keep.append(i)
# pop the first object

folders_keep.pop(0)

# select major cancer types.

#maj_can = [1,2,5,13,14,15,17,18,19,21,24,25,27,31,33] # deleted 22

maj_can = [1,2,3,5,9,13,14,15,16,17,18,19,21,24,27,31,33]
# do a for loop and use exp_mut_surv function to read the data.

builts = []# make sure all the genome builts are correct.

for i in range(len(maj_can)):
    print(folders_keep[maj_can[i]])
    pos_i, ns_i, all_1_i, all_2_i, vt_i, built, exp_i,age_i, drugs_i, os_i = Preprocess.exp_mut_surv(folders_keep[maj_can[i]], pgenes_sort)
    # concatenate all mutation data in the third axis.
    mut_i = np.concatenate((all_1_i, all_2_i),axis = 2)
    mut_i = np.concatenate((mut_i, vt_i,), axis = 2)
    # concatenate all datasets.
    if i == 0:
        mut = mut_i + 0. # copy mut_i
        pos = pos_i + 0.
        ns = ns_i + 0.
        builts.append(built)# make sure all genome builts are the same.
        exp = exp_i + 0. # copy exp_i
        age = age_i + 0. #  copy age
        drugs = drugs_i.copy() # copy drugs_i
        osurv = os_i + 0. #  copy overall survival
        cancers = np.zeros((mut.shape[0],len(maj_can)))
        cancers[:,i] = 1.  
    else:
        mut = np.concatenate((mut, mut_i), axis = 0)
        pos = np.concatenate((pos, pos_i), axis = 0)
        ns = np.concatenate((ns, ns_i),axis =0)
        exp = np.concatenate((exp, exp_i),axis = 0)
        age = np.concatenate((age, age_i),axis = 0)
        for j in range(len(drugs_i)):
            drugs.append(drugs_i[j])
        osurv = np.concatenate((osurv, os_i),axis = 0)
        cancers_i = np.zeros((mut_i.shape[0],len(maj_can)))
        cancers_i[:,i] = 1.
        cancers = np.concatenate((cancers, cancers_i), axis = 0)
        builts.append(built)# make sure all genome builts are the same.
 
# reshape exp.

exp = exp.reshape((exp.shape[0],exp.shape[1],1))
# log transform exp.
# if there are zeros make them a small number.
#exp[exp==0.] = 1e-5

#exp = np.log(exp)

# sample         

n = exp.shape[0] 

ntrain = int(np.round(0.8*(n)))
nval = int(np.round(0.1*(n)))
ntest= int(np.round(0.1*(n)))


# set a random seed.

np.random.seed(2021)

reshuffle_sub = np.random.choice(n, size = n, replace = False)


train = reshuffle_sub[0:ntrain]
val = reshuffle_sub[ntrain+1:(ntrain+nval)]
test = reshuffle_sub[(ntrain+nval+1):n]
    
# now we will encode the positions
pos_train, min_max = Preprocess.pos_normal(pos[train,:])
ns_train = ns[train,:]

ns_train = ns_train.reshape((ns_train.shape[0],ns_train.shape[1],1))

encode_train = Preprocess.mut_encoding(pos_train, 50)


mut_train = mut[train,:,:]

xs_train = np.concatenate((encode_train,mut_train), axis = 2)

xs_train = np.concatenate((xs_train, ns_train),axis = 2)

exp_train = Preprocess.normalize(exp[train,:,:],axis = 1)

xs_train = np.concatenate((xs_train,exp_train), axis = 2)

can_train = cancers[train,:]

file_loc = 'F:\\Outputs_pathways.txt'
lineList = [line.rstrip('\n') for line in open(file_loc)]
mask = np.zeros((len(posgenes),1))
for i in range(len(lineList)):
    div = lineList[i]
    if pgenes_sort.count(div)>0:
        mask[pgenes_sort.index(div),0] = 1.


# classify cancers using expression data alone.



gt_cl = models.gat_k_model_msk([xs_train.shape[1],xs_train.shape[2]],adj_pways,'relu',can_train.shape[1],mask,hops = 2,units = 100, act_out = 'softmax')
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum = 0.01, nesterov = True)
gt_cl.compile(optimizer = opt, loss= 'categorical_crossentropy', metrics = 'accuracy')
gt_cl.fit(xs_train, can_train, epochs= 100)

pos_val = Preprocess.pos_normal_val(pos[val,:], min_max)

ns_val = ns[val,:]

encode_val = Preprocess.mut_encoding(pos_val, 50)

mut_val = mut[val,:,:]

xs_val = np.concatenate((encode_val,mut_val), axis = 2)

ns_val = ns_val.reshape((ns_val.shape[0],ns_val.shape[1],1))

xs_val = np.concatenate((xs_val,ns_val), axis = 2)

exp_val = Preprocess.normalize(exp[val,:,:],axis = 1)

xs_val = np.concatenate((xs_val,exp_val), axis = 2)

can_val = cancers[val,:]

gt_cl.evaluate(xs_val, can_val)

f = gt_cl.layers[0](xs_val)
f, at = gt_cl.layers[1](f)
at = at.numpy()

BRAF_i = pgenes_sort.index('BRAF')
KRAS_i = pgenes_sort.index('KRAS')
RAF1_i = pgenes_sort.index('RAF1')

BRAF_V600e = np.where(pos[val,BRAF_i] == 140453136)[0]
## BRAF mutant
np.mean(at[BRAF_V600e,KRAS_i,BRAF_i])

## BRAF WT
BRAF_WT = np.where(pos[val,BRAF_i] == 0)[0]
KRAS_WT = np.where(pos[val,KRAS_i] == 0)[0]
KRAS_mut = np.where(pos[val,KRAS_i] != 0)[0]

np.mean(at[BRAF_WT,KRAS_i,BRAF_i])

## KRAS Mutant
import scipy.stats

scipy.stats.ttest_ind(at[BRAF_WT,KRAS_i,BRAF_i],at[BRAF_V600e,KRAS_i,BRAF_i])

np.mean(at[KRAS_WT,KRAS_i,BRAF_i])
np.mean(at[KRAS_mut,KRAS_i,BRAF_i])

scipy.stats.ttest_ind(at[KRAS_WT,KRAS_i,BRAF_i],at[KRAS_mut,KRAS_i,BRAF_i])


np.var(at[np.where(ns_val[:,BRAF_i]==1)[0],KRAS_i,BRAF_i])
##KRAS mutant
np.mean(at[np.where(ns_val[:,KRAS_i]==1)[0],KRAS_i,BRAF_i])
np.var(at[np.where(ns_val[:,KRAS_i]==1)[0],KRAS_i,BRAF_i])
## KRAS and BRAF WT
np.mean(at[np.where(ns_val[:,KRAS_i]+ns_val[:,BRAF_i]==0)[0],KRAS_i,BRAF_i])
np.var(at[np.where(ns_val[:,KRAS_i]+ns_val[:,BRAF_i]==0)[0],KRAS_i,BRAF_i])
## BRAF mutant
np.mean(at[np.where(ns_val[:,BRAF_i]==1)[0],BRAF_i,RAF1_i])
np.var(at[np.where(ns_val[:,BRAF_i]==1[0]),BRAF_i,RAF1_i])
