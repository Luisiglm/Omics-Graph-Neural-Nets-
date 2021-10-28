# Script to Run the Experiments

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

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


tp_idxs = read_mut_top_idxs('F:\\')
for i in range(len(tp_idxs)):
    tp_idxs[i] = tp_idxs[i]-1


# First for loop to read the mutation data. 
for i in range(len(folders_keep)):
    print(folders_keep[i])
    pos_mat = read_num_array(folders_keep[i],'pos_mat.txt')
    mut_type = read_num_array(folders_keep[i],'mut_types.txt')
    mut_mat = read_num_array(folders_keep[i],'mut_mat.txt')
    ref_file  = folders_keep[i]+'\\'+'ref_allele.txt'
    ref_al = [(line.strip()).split() for line in open(ref_file)]
    tum_file  = folders_keep[i]+'\\'+'tum_allele.txt'
    tum_al = [(line.strip()).split() for line in open(tum_file)]
    subst = substitution(ref_al,tum_al)
     # now let's concatenate.
    if (i ==0):
        mut = mut_mat + 0.
        pos = pos_mat +0.
        mut_t  = mut_type + 0.
        subs = subst + 0.
    else:
        mut = np.concatenate((mut,mut_mat), axis = 0)
        pos = np.concatenate((pos,pos_mat),axis = 0)
        mut_t = np.concatenate((mut_t, mut_type), axis = 0)
        subs = np.concatenate((subs,subst),axis = 0)


# select the genes we want to  focus on!
def read_phos_subt():
    file_ = 'F:/phos_table.txt'
    lineList = [line.rstrip('\n') for line in open(file_)]
    n = len(lineList)
    xx = list()
    for i in range(n):
        div = re.split(' ', lineList[i])
        xx.append(int(div[1])-1)
    return(xx)


# Select the indexes of the genes we want to focus on

phos_sub = read_phos_subt()
# sort list
phos_sub.sort() # for convenience?

un_tp_idxs = uni_top_phos_sub(tp_idxs, phos_sub)


# keep ony un_tp_idxs

pos = pos[:,un_tp_idxs]
mut = mut[:,un_tp_idxs]
subs = subs[:,un_tp_idxs,:]
# scale the positions.

pos = pos_normal(pos)
# encode the positions into sine and cosine terms.
encoding = mut_encoding(pos,mut, 50)


# for loop to read gene expression values and survival and create the cancer type matrix.
for i in range(len(folders_keep)):
    print(folders_keep[i])
    exp_i = np.transpose(read_num_array(folders_keep[i], 'exp_mat.txt'))
    exp_i = exp_i[:,un_tp_idxs]
    surv_i = read_surv(folders_keep[i], 'death_mat.txt')
    # now let's concatenate.
    if (i ==0):
        exp = exp_i + 0.
        cancers = np.zeros((exp.shape[0],len(folders_keep)))
        cancers[:,i] = 1
        survs = surv_i + 0.
    else:
        exp = np.concatenate((exp, exp_i), axis = 0)
        cancers_i = np.zeros((exp_i.shape[0],len(folders_keep)))
        cancers_i[:,i] = 1
        cancers = np.concatenate((cancers, cancers_i), axis = 0)
        if survs.shape[1]!= surv_i.shape[1]:
            # padding
            if survs.shape[1] > surv_i.shape[1]:
                missing = np.zeros((surv_i.shape[0],survs.shape[1]-surv_i.shape[1],2))
                ss = np.sum(surv_i[:,:,1],axis = 1)==surv_i.shape[1]
                missing[ss,:,1] = 1.
                surv_i = np.concatenate((surv_i, missing), axis = 1)
            else:
                missing = np.zeros((survs.shape[0],surv_i.shape[1]-survs.shape[1],2))
                ss = np.sum(survs[:,:,1],axis = 1)==survs.shape[1]
                missing[ss,:,1] = 1.
                survs = np.concatenate((survs, missing), axis = 1)
        survs = np.concatenate((survs, surv_i), axis = 0)

# Read the Reactome adjacency matrix.

Adj_ind = read_mat_indxs('F:\\react_indxs.txt')

adj = np.zeros((20531,20531))

for i in range(Adj_ind.shape[0]):
    adj[Adj_ind[i,0],Adj_ind[i,1]] = 1
    adj[Adj_ind[i,1],Adj_ind[i,0]] = 1

adj_sub = adj[:,un_tp_idxs]
adj_sub = adj_sub[un_tp_idxs,:]


# normalize exp.

exp = normalize(exp,1)

# select the most commonly diagnosed cancer types.

maj_can = [1,2,5,13,14,15,17,18,19,21,22,24,25,27,31,33]

cancers_maj = cancers[:,maj_can]

# add an extra dimension to gene expression. 
exp = np.reshape(exp, (exp.shape[0], exp.shape[1],1))

x = np.concatenate((exp,encoding), axis = 2)

x = np.concatenate((x,subs), axis = 2)

x_sub = x[np.sum(cancers_maj,axis=1)>0,:]

cancers_m = cancers_maj[np.sum(cancers_maj,axis=1)>0,:]

ntrain = int(np.round(0.8*(x_sub.shape[0])))
nval = int(np.round(0.1*(x_sub.shape[0])))
ntest= int(np.round(0.1*(x_sub.shape[0])))

n = x_sub.shape[0]
reshuffle_sub = np.random.choice(n, size = n, replace = False)


train = reshuffle_sub[0:ntrain]
val = reshuffle_sub[ntrain+1:(ntrain+nval)]
test = reshuffle_sub[(ntrain+nval+1):n]

xs_train = x_sub[train,:,:]
xs_val = x_sub[val,:,:]

can_train = cancers_m[train,:]

gt = gat_model([xs_train.shape[1],xs_train.shape[2]],adj_sub,'relu',can_train.shape[1],units = 100, act_out = 'softmax')
opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
gt.compile(optimizer = opt, loss= 'categorical_crossentropy', metrics = 'accuracy')
gt.fit(xs_train, can_train, epochs= 500)

can_val = cancers_m[val,:]
gt.evaluate(xs_val, can_val)


gte = gate_model([xs_train.shape[1],xs_train.shape[2]],adj_sub,'relu',can_train.shape[1],units = 100, act_out = 'softmax')
opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
gte.compile(optimizer = opt, loss= 'categorical_crossentropy', metrics = 'accuracy')
gte.fit(xs_train, can_train, epochs= 500)

can_val = cancers_m[val,:]
gte.evaluate(xs_val, can_val)


sgte = sign_gate_model([xs_train.shape[1],xs_train.shape[2]],adj_sub,'relu',can_train.shape[1],units = 100, act_out = 'softmax')
opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
sgte.compile(optimizer = opt, loss= 'categorical_crossentropy', metrics = 'accuracy')
sgte.fit(xs_train, can_train, epochs= 500)

can_val = cancers_m[val,:]
sgte.evaluate(xs_val, can_val)


sgte = sign_gate_attn_over([xs_train.shape[1],xs_train.shape[2]],adj_sub,'relu',can_train.shape[1],units = 100, act_out = 'softmax')
opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
sgte.compile(optimizer = opt, loss= 'categorical_crossentropy', metrics = 'accuracy')
sgte.fit(xs_train, can_train, epochs= 500)

can_val = cancers_m[val,:]
sgte.evaluate(xs_val, can_val)
