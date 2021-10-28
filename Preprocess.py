# Read Data from TCGA
# Functions to read matrices
# Functions to Encode Mutations
# Functions to Read Nucleotide Substitution

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


def mut_encoding(pos_mat, mut_types, dim):
    """ function to encode the mutation position into different cosine and sine terms.
        Args:
            pos_mat: a patients x genes numpy array with the position in the chromosome.
            mut_types: a patients x genes numpy with values from 1 to 4 if the mutation
                       is a SNP, DNP, DEL or INS respectively.
            dim: an int with the number of cosine and sine terms to calculate
        Returns:
            an numpy array patients x dims x 2 +4 with the cosine terms and the type.
    """
    # pre-allocate memory for sines and cosines.
    sines = np.zeros((pos_mat.shape[0], pos_mat.shape[1], dim))
    cosines = np.zeros((pos_mat.shape[0], pos_mat.shape[1],dim))
    # pre-allocate memory for mut_type matrix.
    m_tp = np.zeros((pos_mat.shape[0],pos_mat.shape[1],4))
    mask = 1*(pos_mat!=-100)
    for i in range(dim):
        sines[:,:,i] = np.sin(pos_mat*(i+1))*mask
        cosines[:,:,i] = np.cos(pos_mat*(i+1))*mask
    for i in range(4):
        m_tp[mut_types==(i+1),i] = 1
    # concatenate in the third axis.
    encode = np.concatenate((sines, cosines),axis = 2)
    encode = np.concatenate((encode,m_tp),axis = 2)
    pos_mat = np.reshape((pos_mat+(1-mask)*100), (pos_mat.shape[0],pos_mat.shape[1],1))
    encode = np.concatenate((encode,pos_mat),axis = 2)
    return(encode)



def pos_normal(pos):
    """ function to scale the mutation position to be between -pi and pi.
        Args:
            pos_mat: a patients x genes numpy array with the position in the chromosome.
        Returns:
            an numpy array patients x genes with the mutation positions. Sets 0 values to -100.
    """
    for i in range(pos.shape[1]):
        p = pos[pos[:,i]!=0,i]
        if p.shape[0]>0:
            # make sure the value is between -2pi and 2pi
            dist = np.max(p) - np.min(p)
            p = (p/dist)*(2*np.pi)
            # correct so that it goes from -pi to pi.
            pos[pos[:,i]==0,i] = -100.
            pos[pos[:,i]>0,i] = (p - np.min(p)-np.pi)
        else:
            pos[:,i] = -100.
    return(pos)

def substitution(ref_al, tum_al):
    """ function to create a matrix with the substitution nucleotides.
        Args:
            ref_al: a patients long list with genes string numpy array with the reference allele for each gene with mutation.
            tum_al: a patients long list with genes string numpy array with the tumor allele for each gene with mutation.
        Returns:
            an numpy array patients x genes x 8 with substitutions,
            The first four columns of the third dimension on the reference allele [T,C,G,A].
            The second four columns of the third dimension on the tumour allele [T,C,G,A].
    """
    # pre-allocate memory
    J = len(ref_al[0])
    res = np.zeros((len(ref_al),J,8))
    nucleotide = ["T","C","G","A"]
    for i in range(len(ref_al)):
        for j in range(J):
            if len(ref_al[i][j])==1:
                if nucleotide.count(ref_al[i][j])>0:
                    res[i,j,nucleotide.index(ref_al[i][j])] = 1.
            if len(tum_al[i][j])==1:
                if nucleotide.count(tum_al[i][j])>0:
                    res[i,j,nucleotide.index(tum_al[i][j])+4] = 1.
    return(res)
                 
        

def read_num_array(file_loc, file_name):
    """ function to read a nummeric matrix.
        Args:
            file_loc: The directory where the file is located.
            file_name: The name of the file. 
        Returns:
            an numpy array 
    """
    file_loc_exp = file_loc+'\\'+file_name
    lineList = [line.rstrip('\n') for line in open(file_loc_exp)]
    n = len(lineList)
    p = len(re.split(' ', lineList[0]))
    xx = np.zeros((n,p))
    for i in range(n):
        div = re.split(' ', lineList[i])
        for j in range(p):
            if (div[j]=="NA"):
                div[j] = 0.0
            xx[i,j] = float(div[j])
    return(xx)

def normalize(num_array, axis):
    """ function normalize a nummerical array.
        Args:
            num_array: a numpy array.
            axis: An integer with the axis to normalize the data. 
        Returns:
            a zero centred and unit variance numpy array along the axis specified.
    """
    if axis == 0:
        for i in range(num_array.shape[0]):
            if (np.var(num_array[i,:])>0):
                num_array[i,:] = (num_array[i,:] - np.mean(num_array[i,:]))/np.var(num_array[i,:])**.5
    elif axis == 1:
        for i in range(num_array.shape[1]):
            if (np.var(num_array[:,i])>0):
                num_array[:,i] = (num_array[:,i] - np.mean(num_array[:,i]))/np.var(num_array[:,i])**.5
    elif axis == 2:
        for i in range(num_array.shape[2]):
            if (np.var(num_array[:,:,i])>0):
                num_array[:,:,i] = (num_array[:,:,i] - np.mean(num_array[:,:,i]))/np.var(num_array[:,:,i])**.5
    return(num_array)


                                
def read_phos_data(file_loc):
    """ Function to read the phosphorilation data
        Args:
            file_loc: The directory where the file is located.
        Returns:
            a numpy array patients x genes x 5
    """
    # first read phos_subtable
    file_loc_table = file_loc+'\\phos_subtable.txt'
    # read then then phosphorylation file
    file_phos = file_loc + '\\phos_mat.txt'
    lineList = [line.rstrip('\n') for line in open(file_phos)]
    linetable = [line.rstrip('\n') for line in open(file_loc_table)]
    n = len(lineList)
    p = 20531 # no. of genes.
    k = 5 # third dimension. (5 is the max. no. of other phosphorylations)
    xx = np.zeros((n,p,k))
    for i in range(n):
        div = re.split(' ', lineList[i])
        # now, each column has a location specified in the linetable
        for j in range(len(linetable)):
            dj = re.split(' ', linetable[j])
            # it has length of two.
            if (div[j]=="NA"):
                div[j] = 0.0
            xx[i,int(dj[0])-1,int(dj[1])-1] = float(div[j])
    return(xx)

def read_phos_subt():
    file_ = 'F:/phos_table.txt'
    lineList = [line.rstrip('\n') for line in open(file_)]
    n = len(lineList)
    xx = list()
    for i in range(n):
        div = re.split(' ', lineList[i])
        xx.append(int(div[1])-1)
    return(xx)

def uni_top_phos_sub(top_idxs, phos_sub):
    xx = list() # pre-allocate memory for object.
    # initialize xx using top_idxs.
    for i in range(len(top_idxs)):
        xx.append(top_idxs[i])
    # now go through other list.
    for i in range(len(phos_sub)):
        if xx.count(phos_sub[i])==0:
            xx.append(phos_sub[i])
    xx.sort()
    return(xx)


def read_surv(file_loc, file_name):
    """ function to read a nummeric matrix.
        Args:
            file_loc: The directory where the file is located.
            file_name: The name of the file. 
        Returns:
            a 3d numpy array with patients x time x 2.
            The first column of the third dimension has 1 if the patient was seen alive.
            The second column of the third dimension has 1 if the patients was seen. That is not censored.
    """
    file_loc_death = file_loc+'\\' + file_name
    lineList = [line.rstrip('\n') for line in open(file_loc_death)]
    n = len(lineList)
    p = len(re.split(' ', lineList[0]))
    cens = np.zeros((n,p))
    bins = np.zeros((n,p))
    for i in range(n):
        div = re.split(' ', lineList[i])
        for j in range(p):
            if (div[j]=="NA"):
                div[j] = 0.0
            cens[i,j] = float(div[j])
            if int(div[j])>0:
                bins[i,0:j] = 1.
    cens_2 = np.zeros((n,p))
    cens_2[np.sum(cens,axis = 1)>0,:] = 1.
    # We have the death matrix now.
    file_loc_mxfup = file_loc+'\\maxfup.txt'
    mxfup = [line.rstrip('\n') for line in open(file_loc_mxfup)]
    for i in range(n):
        if int(mxfup[i])>0:
            bins[i,0:int(mxfup[i])] = 1
            cens_2[i,0:int(mxfup[i])] = 1
    y = np.zeros((n,p,2))
    y[:,:,1] = cens_2
    y[:,:,0] = bins
    return(y)


def read_mat_indxs(file_loc):
    # Read the Reactome file
    lineList = [line.rstrip('\n') for line in open(file_loc)]
    n = len(lineList)
    indixes = np.zeros((n,2),dtype = 'int64')
    for i in range(n):
        div = re.split(' ', lineList[i])
        indixes[i,0] = int(div[0])
        indixes[i,1] = int(div[1])
    return(indixes)
