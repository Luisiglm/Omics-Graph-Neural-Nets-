# Read Data from TCGA
# Functions to read matrices
# Functions to Encode Mutations
# Functions to Read Nucleotide Substitution
"""Preprocess.py."""

import re
import os
import numpy as np



def exp_mut_surv(folder_i, genes):
    """ Wrapper to obtain the mutation, gene expresion and clinical dataset of a set of genes form a folder.
        Args:
            folder_i: The folder with the TCGA cancer dataset.
            genes: a list with the HUGYO Symbol genes that we wish to focus on.
        Returns:
            pats: A list with the patients barcodes.
            pos_i: A patients x genes numpy array with the Start position of the mutation for each gene.
            ns_i: A patients x genes numpy array with a value of 1 if the gene has a nonsilent mutation.
            all_1_i: A patients x genes x 4 numpy array whose third axis represent the bases T, C, G, A, if a gene's allele 1 has either base there will be a value of 1.
            all_2_i: A patients x genes x 4 numpy array whose third axis represent the bases T, C, G, A, if a gene's allele 2 has either base there will be a value of 1.
            vt_i:  A patients x genes x 4 numpy array whose third axis represent the variant type SNP, DNP, INS and DEL with a value of 1 for each case.
            built: The chromosome built version. Expect value of 37.
            exp_i: A patients x genes numpy array with the TPKM RNA expression values.
            age_i: A patient x 1 numpy array with the age at diagnosis.
            drugs_i:  A list with the name of the drugs that a patient was prescribed.
            os_i: A patients x 2 numpy array where the first column has the time to the event or last followup and the column 2 a value of 1 if the patient died.
    """
    # get mutation data.
    pats, pos_i, ns_i, all_1_i, all_2_i, vt_i, built = mut_prep(folder_i, genes)
    print('muts done')
    # now with the expression data.
    exp_i, pat_exp_match, pat_input_match = exp_prep(genes, pats, 'F://')
    print('RNASeq done')
    # now let's do the clinical data.
    age_i, drugs_i, os_i = clin_prep(folder_i, pats)
    print('clin done')
    # return results.
    # make sure to keep only expression data that is complete.
    zeroes = np.where(np.sum(exp_i, axis=1) == 0.)
    # delete those observations from pos_i, ns_i, all_1_i, all_2_i, vt_i.
    pos_i = np.delete(pos_i, zeroes, axis=0)
    ns_i = np.delete(ns_i, zeroes, axis=0)
    all_1_i = np.delete(all_1_i, zeroes, axis=0)
    all_2_i = np.delete(all_2_i, zeroes, axis=0)
    vt_i = np.delete(vt_i, zeroes, axis=0)
    # delete from expression data.
    exp_i = np.delete(exp_i, zeroes, axis=0)
    # delete from age_i,  os_i
    age_i = np.delete(age_i, zeroes, axis=0)
    os_i = np.delete(os_i, zeroes, axis=0)
    for i in range(zeroes[0].shape[0]):
        zeroes[0][i] = zeroes[0][i] - i  # correct for deleted observations
        drugs_i.pop(zeroes[0][i])
    return (pos_i, ns_i, all_1_i, all_2_i, vt_i, built, exp_i, age_i, drugs_i, os_i)


def mut_prep(folder_i, genes):
    """ Wrapper to obtain the mutation of a set of genes form a folder.
        Args:
            folder_i: The folder with the TCGA cancer dataset.
            genes: a list with the HUGYO Symbol genes that we wish to focus on.
        Returns:
            pats: A list with the patients barcodes.
            pos_i: A patients x genes numpy array with the Start position of the mutation for each gene.
            ns_i: A patients x genes numpy array with a value of 1 if the gene has a nonsilent mutation.
            all_1_i: A patients x genes x 4 numpy array whose third axis represent the bases T, C, G, A, if a gene's allele 1 has either base there will be a value of 1.
            all_2_i: A patients x genes x 4 numpy array whose third axis represent the bases T, C, G, A, if a gene's allele 2 has either base there will be a value of 1.
            vt_i:  A patients x genes x 4 numpy array whose third axis represent the variant type SNP, DNP, INS and DEL with a value of 1 for each case.
            built: The chromosome built version. Expect value of 37.
    """
    # find mutation folder.
    mut_folder = find_mut_folder(folder_i)
    # get all files.
    mut_files = os.listdir(folder_i + '\\' + mut_folder)
    # delete the bleeding manifest.
    idx_mani = mut_files.index('MANIFEST.txt')
    mut_files.pop(idx_mani)
    # pre-allocate memory for data as a numpy array.
    pos_i = np.zeros((len(mut_files), len(genes)))
    ns_i = np.zeros((len(mut_files), len(genes)))
    all_1_i = np.zeros((len(mut_files), len(genes), 4))
    all_2_i = np.zeros((len(mut_files), len(genes), 4))
    vt_i = np.zeros((len(mut_files), len(genes), 5))
    # do a for loop to read each file in mut_files
    pats = []  # save each patient here.
    bases = ['T', 'C', 'G', 'A']
    vs = ['SNP', 'DNP', 'INS', 'DEL', 'TNP']
    # do a for loop for every patient in the files.
    for i in range(len(mut_files)):
        # get file name and store it in pats.
        pats.append(mut_files[i][0:12])
        # read file.
        mutread = [line.rstrip('\n') for line in open(folder_i + '\\' + mut_folder + '\\' + mut_files[i])]
        # great now check if the gene in each line is part of genes.
        div = re.split('\t', mutread[0])
        if div[0][0] == '#':
            mutread.pop(0)
            mutread.pop(0)
            mutread.pop(0)
            div = re.split('\t', mutread[0])
        name = div.index('Hugo_Symbol')
        if div.count('Start_position') == 0:
            pos_j = div.index('Start_Position')
        else:
            pos_j = div.index('Start_position')
        tum_al1 = div.index('Tumor_Seq_Allele1')
        tum_al2 = div.index('Tumor_Seq_Allele2')
        var_cl = div.index('Variant_Classification')
        var_type = div.index('Variant_Type')
        ncbi = div.index('NCBI_Build')
        for j in range(1, len(mutread)):
            div = re.split('\t', mutread[j])
            # check if gene is in the list!
            if genes.count(div[name]) > 0:
                pos_i[i, genes.index(div[name])] = float(div[pos_j])
                if div[var_cl] != 'Silent':
                    ns_i[i, genes.index(div[name])] = 1.
                if bases.count(div[tum_al1]) > 0:
                    all_1_i[i, genes.index(div[name]), bases.index(div[tum_al1])] = 1.
                if bases.count(div[tum_al2]) > 0:
                    all_2_i[i, genes.index(div[name]), bases.index(div[tum_al2])] = 1.
                vt_i[i, genes.index(div[name]), vs.index(div[var_type])] = 1.
        built = div[ncbi]
    return (pats, pos_i, ns_i, all_1_i, all_2_i, vt_i, built)


def find_mut_folder(folder_i):
    """ Function that matches the folder mutation within a folder.
        Args:
            folder_i: A string the path with the dataset folders.
        Returns:
            mut_folder: A string with the name of the mutation folder.

    """
    # Find all files within folder.
    files_folder = os.listdir(folder_i)
    # Read mutation data in folder_i.
    mut_folder = None
    mtch = False
    ii = 0
    # if we have the Raw Calls use them.
    while not mtch:
        mm = re.search('Raw_Calls', files_folder[ii])
        if mm != None:
            mut_folder = '%s' % files_folder[ii]
            mtch = True
        ii = ii + 1
        if ii == len(files_folder):
            mtch = True
    if mut_folder == None:
        ii = 0
        mtch = False
        while not mtch:
            mm = re.search('Packager_Calls', files_folder[ii])
            if mm != None:
                mut_folder = '%s' % files_folder[ii]
                mtch = True
            ii = ii + 1
            if ii == len(files_folder):
                mtch = True
    return (mut_folder)


def mut_encoding(pos_mat, dim):
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
    cosines = np.zeros((pos_mat.shape[0], pos_mat.shape[1], dim))
    # pre-allocate memory for mut_type matrix.
    mask = 1 * (pos_mat != -100)
    for i in range(dim):
        sines[:, :, i] = np.sin(pos_mat * (i + 1)) * mask
        cosines[:, :, i] = np.cos(pos_mat * (i + 1)) * mask
    encode = np.concatenate((sines, cosines), axis=2)
    pos_mat = np.reshape((pos_mat + (1 - mask) * 100), (pos_mat.shape[0], pos_mat.shape[1], 1))
    encode = np.concatenate((encode, pos_mat), axis=2)
    return (encode)


def depmap_read_exp(depmap_path, genes_list):
    # expression data first.
    exp_file = 'CCLE_expression.csv'
    # read file
    file_loc = depmap_path + exp_file
    lineList = [line.rstrip('\n') for line in open(file_loc)]
    exp_np = np.zeros((len(lineList) - 1, len(genes_list)))
    # find genes in file.
    gene_columns = re.split(',', lineList[0])
    # Brilliant shtuff lads
    for i in range(len(gene_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', gene_columns[i])
        if parenthesis != None:
            gene_columns[i] = gene_columns[i][0:parenthesis.start(0)]
    # go through gene list and get column indexes.
    cols_index = []
    for i in range(len(genes_list)):
        if gene_columns.count(genes_list[i]) > 0:
            cols_index.append(gene_columns.index(genes_list[i]))
        else:
            cols_index.append(None)
    # start filling in the values in the aray exp_np
    # record the names of the cell-lines.
    depmap_id = []
    for i in range(1, len(lineList)):
        div = re.split(',', lineList[i])
        depmap_id.append(div[0])
        for j in range(len(genes_list)):
            if cols_index[j] != None:
                exp_np[i - 1, j] = float(div[cols_index[j]])
    # the depmap_id will be useful to match the other files.
    # gene the gene_effect file.
    score_file = 'CRISPR_gene_effect.csv'
    file_loc = depmap_path + score_file
    lineList = [line.rstrip('\n') for line in open(file_loc)]
    # pre-allocate memory for crispr score.
    score = np.zeros((len(lineList) - 1, len(genes_list)))
    crispr_columns = re.split(',', lineList[0])
    for i in range(len(crispr_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', crispr_columns[i])
        if parenthesis != None:
            crispr_columns[i] = crispr_columns[i][0:parenthesis.start(0)]
    cols_crispr_index = []
    for i in range(len(genes_list)):
        if crispr_columns.count(genes_list[i]) > 0:
            cols_crispr_index.append(crispr_columns.index(genes_list[i]))
        else:
            cols_crispr_index.append(None)
    # we are going to go through the expression dataset and delete each gene in the crispr.
    No_crispr = len(genes_list) - cols_crispr_index.count(None)
    Obs = exp_np.shape[0] * No_crispr
    # pre-allocate memory for 'simulated' expression.
    exp_crispr = np.zeros((Obs, len(genes_list)))
    # scores
    scores_crispr = np.zeros((Obs,))
    ii = 0
    for i in range(1, len(lineList)):
        # read depmap_id.
        div = re.split(',', lineList[i])
        depmap_i = div[0]
        # match depmap id.
        if depmap_id.count(depmap_i) > 0:
            J = depmap_id.index(depmap_i)
            for j in range(len(genes_list)):
                if cols_crispr_index[j] != None:
                    exp_crispr[ii, :] = exp_np[J, :] + 0.
                    exp_crispr[ii, j] = 0.
                    if div[cols_crispr_index[j]] != '':
                        exp_crispr[ii, :] = exp_np[J, :] + 0.
                        exp_crispr[ii, j] = 0.
                        scores_crispr[ii] = div[cols_crispr_index[j]]
                        ii = ii + 1
    exp_crispr = exp_crispr[0:ii, :]
    scores_crispr = scores_crispr[0:ii]
    return (exp_np, exp_crispr, scores_crispr, depmap_id, crispr_columns, cols_crispr_index)


def pos_normal(pos):
    """ function to scale the mutation position to be between -pi and pi.
        Args:
            pos_mat: a patients x genes numpy array with the position in the chromosome.
        Returns:
            an numpy array patients x genes with the mutation positions. Sets 0 values to -100.
    """
    min_max = np.zeros((pos.shape[1], 3))
    pos_n = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[1]):
        p = pos[pos[:, i] != 0, i]
        if p.shape[0] > 0:
            # make sure the value is between -2pi and 2pi
            dist = np.max(p) - np.min(p)
            min_max[i, 0] = np.min(p)
            min_max[i, 1] = np.max(p)
            min_max[i, 2] = 1 / 2 * (min_max[i, 0] + min_max[i, 1])
            if dist > 0:
                p = ((p - min_max[i, 2]) / dist) * (2 * np.pi)
                # correct so that it goes from -pi to pi.
                pos_n[pos[:, i] == 0, i] = -100.
                pos_n[pos[:, i] > 0, i] = (p)  # - np.min(p)-np.pi)
            else:
                pos_n[pos[:, i] == 0, i] = -100.
                pos_n[pos[:, i] > 0, i] = 0.
        else:
            pos_n[:, i] = -100.
    return (pos_n, min_max)


def depmap_read_exp_mut(depmap_path, genes_list):
    # expression data first.
    exp_file = 'CCLE_expression.csv'
    # read file
    file_loc = depmap_path + exp_file
    lineList = [line.rstrip('\n') for line in open(file_loc)]
    exp_np = np.zeros((len(lineList) - 1, len(genes_list)))
    # find genes in file.
    gene_columns = re.split(',', lineList[0])
    # Brilliant shtuff lads
    for i in range(len(gene_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', gene_columns[i])
        if parenthesis != None:
            gene_columns[i] = gene_columns[i][0:parenthesis.start(0)]
    # go through gene list and get column indexes.
    cols_index = []
    for i in range(len(genes_list)):
        if gene_columns.count(genes_list[i]) > 0:
            cols_index.append(gene_columns.index(genes_list[i]))
        else:
            cols_index.append(None)
    # start filling in the values in the aray exp_np
    # record the names of the cell-lines.
    depmap_id = []
    for i in range(1, len(lineList)):
        div = re.split(',', lineList[i])
        depmap_id.append(div[0])
        for j in range(len(genes_list)):
            if cols_index[j] != None:
                exp_np[i - 1, j] = float(div[cols_index[j]])
    # the depmap_id will be useful to match the other files.
    # read mutation file.
    # in mutation file one of the columns has the Depmap id.
    mut_file = 'CCLE_mutations.csv'
    # read file
    file_loc = depmap_path + mut_file
    lineList = [line.rstrip('\n') for line in open(file_loc)]
    cols = re.split(',', lineList[0])
    depmap_id_col = cols.index('DepMap_ID')
    sp_col = cols.index('Start_position')
    vt_col = cols.index('Variant_Type')
    vc_col = cols.index('Variant_Classification')
    ra_col = cols.index('Reference_Allele')
    aa_col = cols.index('Alternate_Allele')
    pos_muts = np.zeros((exp_np.shape[0], exp_np.shape[1]))  # store mutation position.
    vtp = np.zeros((exp_np.shape[0], exp_np.shape[1], 5))  # variant type.
    subs = np.zeros((exp_np.shape[0], exp_np.shape[1], 8))
    ns = np.zeros((exp_np.shape[0], exp_np.shape[1]))
    vts = ['SNP', 'DNP', 'INS', 'DEL', 'TNP']
    bases = ['T', 'C', 'G', 'A']
    for i in range(1, len(lineList)):
        # separate the columns.
        div = re.split(',', lineList[i])
        # read Depmap ID
        depmap_i = div[depmap_id_col]
        gene_i = div[0]
        # check if it's part of our cohort.
        if depmap_id.count(depmap_i) > 0 and genes_list.count(gene_i) > 0:
            I = depmap_id.index(depmap_i)
            J = genes_list.index(gene_i)
            pos_muts[I, J] = float(div[sp_col])  # store start position.
            if vts.count(div[vt_col]) > 0:
                vtp[I, J, vts.index(div[vt_col])] = 1.
            if bases.count(div[ra_col]) > 0:
                subs[I, J, bases.index(div[ra_col])] = 1.
            if bases.count(div[aa_col]) > 0:
                subs[I, J, bases.index(div[aa_col]) + 4] = 1.
            if div[vc_col] != 'Silent':
                ns[I, J] = 1.
    # we have these mutation datasets we have to transform them into the same length of the scores.
    # gene the gene_effect file.
    score_file = 'CRISPR_gene_effect.csv'
    file_loc = depmap_path + score_file
    lineList = [line.rstrip('\n') for line in open(file_loc)]
    # pre-allocate memory for crispr score.
    score = np.zeros((len(lineList) - 1, len(genes_list)))
    crispr_columns = re.split(',', lineList[0])
    for i in range(len(crispr_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', crispr_columns[i])
        if parenthesis != None:
            crispr_columns[i] = crispr_columns[i][0:parenthesis.start(0)]
    cols_crispr_index = []
    for i in range(len(genes_list)):
        if crispr_columns.count(genes_list[i]) > 0:
            cols_crispr_index.append(crispr_columns.index(genes_list[i]))
        else:
            cols_crispr_index.append(None)
    # we are going to go through the expression dataset and delete each gene in the crispr.
    # pre-allocate memory for 'simulated' expression.
    scores_crispr = np.zeros((exp_np.shape[0], len(genes_list)))
    for i in range(1, len(lineList)):
        # read depmap_id.
        div = re.split(',', lineList[i])
        depmap_i = div[0]
        # match depmap id.
        if depmap_id.count(depmap_i) > 0:
            J = depmap_id.index(depmap_i)
            for j in range(len(genes_list)):
                if cols_crispr_index[j] != None:
                    if div[cols_crispr_index[j]] != '':
                        scores_crispr[i, j] = div[cols_crispr_index[j]]
    sum_crispr = np.sum(scores_crispr, axis=1)
    exp_np = exp_np[sum_crispr != 0, :]
    pos_muts = pos_muts[sum_crispr != 0, :]
    vtp = vtp[sum_crispr != 0, :]
    subs = subs[sum_crispr != 0, :]
    ns = ns[sum_crispr != 0, :]
    scores_crispr = scores_crispr[sum_crispr != 0, :]
    return (exp_np, scores_crispr, depmap_id, pos_muts, vtp, subs, ns)


def pos_normal_val(pos, min_max):
    """ function to scale the mutation position to be between -pi and pi.
        Args:
            pos_mat: a patients x genes numpy array with the position in the chromosome.
        Returns:
            an numpy array patients x genes with the mutation positions. Sets 0 values to -100.
    """
    pos_n = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[1]):
        p = pos[pos[:, i] != 0, i]
        if p.shape[0] > 0:
            # make sure the value is between -2pi and 2pi
            dist = min_max[i, 1] - min_max[i, 0]
            if dist > 0:
                p = ((p - min_max[i, 2]) / dist) * (2 * np.pi)
                # correct so that it goes from -pi to pi.
                pos_n[pos[:, i] == 0, i] = -100.
                pos_n[pos[:, i] > 0, i] = (p)
            else:
                # re-calculate distance
                if np.max(p) > min_max[i, 2]:
                    dist = np.max(p) - min_max[i, 2]
                else:
                    dist = min_max[i, 2] - np.min(p)
                if dist == 0:
                    pos_n[pos[:, i] == 0, i] = -100.
                    pos_n[pos[:, i] > 0, i] = 0.
                else:
                    p = ((p - min_max[i, 2]) / dist) * (np.pi)
                    # correct so that it goes from -pi to pi.
                    pos_n[pos[:, i] == 0, i] = -100.
                    pos_n[pos[:, i] > 0, i] = (p)
        else:
            pos_n[:, i] = -100.
    return (pos_n)


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
    res = np.zeros((len(ref_al), J, 8))
    nucleotide = ["T", "C", "G", "A"]
    for i in range(len(ref_al)):
        for j in range(J):
            if len(ref_al[i][j]) == 1:
                if nucleotide.count(ref_al[i][j]) > 0:
                    res[i, j, nucleotide.index(ref_al[i][j])] = 1.
            if len(tum_al[i][j]) == 1:
                if nucleotide.count(tum_al[i][j]) > 0:
                    res[i, j, nucleotide.index(tum_al[i][j]) + 4] = 1.
    return (res)


def read_num_array(file_loc, file_name):
    """ function to read a nummeric matrix.
        Args:
            file_loc: The directory where the file is located.
            file_name: The name of the file.
        Returns:
            an numpy array
    """
    file_loc_exp = file_loc + '\\' + file_name
    lineList = [line.rstrip('\n') for line in open(file_loc_exp)]
    n = len(lineList)
    p = len(re.split(' ', lineList[0]))
    xx = np.zeros((n, p))
    for i in range(n):
        div = re.split(' ', lineList[i])
        for j in range(p):
            if (div[j] == "NA"):
                div[j] = 0.0
            xx[i, j] = float(div[j])
    return (xx)


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
            if (np.var(num_array[i, :]) > 0):
                num_array[i, :] = (num_array[i, :] - np.mean(num_array[i, :])) / np.var(num_array[i, :]) ** .5
    elif axis == 1:
        for i in range(num_array.shape[1]):
            if (np.var(num_array[:, i]) > 0):
                if (len(num_array.shape) == 2):
                    num_array[:, i] = (num_array[:, i] - np.mean(num_array[:, i])) / np.var(num_array[:, i]) ** .5
                elif (len(num_array.shape) == 3):
                    for j in range(num_array.shape[2]):
                        num_array[:, i, j] = (num_array[:, i, j] - np.mean(num_array[:, i, j])) / np.var(
                            num_array[:, i, j]) ** .5
    elif axis == 2:
        for i in range(num_array.shape[2]):
            if (np.var(num_array[:, :, i]) > 0):
                num_array[:, :, i] = (num_array[:, :, i] - np.mean(num_array[:, :, i])) / np.var(
                    num_array[:, :, i]) ** .5
    num_array[np.isnan(num_array)] = 0.
    return (num_array)


def find_clin_folder(folder_i):
    # Find all files within folder.
    files_folder = os.listdir(folder_i)
    # Read mutation data in folder_i.
    clin_folder = None
    mtch = False
    ii = 0
    # see if we have the Raw Calls use them.
    while not mtch:
        mm = re.search('clin.', files_folder[ii])
        if mm != None:
            clin_folder = '%s' % files_folder[ii]
            mtch = True
        ii = ii + 1
        if ii == len(files_folder):
            mtch = True
    return (clin_folder)


def clin_prep(folder_i, pats):
    # find file.
    clin_file = find_clin_folder(folder_i)
    # read file.
    clinread = [line.rstrip('\n') for line in open(folder_i + '\\' + clin_file)]
    # now let the party begin!
    age_row = 'patient.age_at_initial_pathologic_diagnosis'
    age_row2 = 'patient.primary_pathology.age_at_initial_pathologic_diagnosis'
    barcode_row = 'patient.bcr_patient_barcode'
    dfu = 'days_to_last_followup'
    dtd = 'days_to_death'
    drugs = '.drug_name'
    # age and barcode are to be matched exactly.
    dfu_rows = []
    dtd_rows = []
    drugs_rows = []
    # go down the rows and find each row.
    for i in range(len(clinread)):
        div = re.split('\t', clinread[i])
        if div[0] == barcode_row:
            barcodes = div.copy()
            for j in range(len(barcodes)):
                barcodes[j] = barcodes[j].upper()
        if div[0] == age_row:
            ages = div.copy()
        if div[0] == age_row2:
            ages = div.copy()
        if re.search(drugs, div[0]) != None:
            drugs_rows.append(i)
        if re.search(dfu, div[0]) != None:
            dfu_rows.append(i)
        if re.search(dtd, div[0]) != None:
            dtd_rows.append(i)
    os_i = minimum_days(clinread, barcodes, pats, dfu_rows, dtd_rows)
    age_i = age_prep(ages, barcodes, pats)
    drugs_i = drugs_prep(clinread, barcodes, pats, drugs_rows)
    return (age_i, drugs_i, os_i)


def drugs_prep(clinread, barcodes, pats, drugs_rows):
    # start list with drugs.
    drugs_i = []
    for i in range(len(pats)):
        dj = []
        if barcodes.count(pats[i]) > 0:
            for j in range(len(drugs_rows)):
                div = re.split('\t', clinread[drugs_rows[j]])
                dj.append(div[barcodes.index(pats[i])])
        drugs_i.append(dj)
    return (drugs_i)


def age_prep(ages, barcodes, pats):
    # pre-allocate memory
    age_i = np.zeros((len(pats), 1))
    # do a for loop and match each patient to barcode.
    for i in range(len(pats)):
        if barcodes.count(pats[i]) > 0:
            if ages[barcodes.index(pats[i])] != 'NA':
                age_i[i, 0] = float(ages[barcodes.index(pats[i])])
    return (age_i)


def minimum_days(clinread, barcodes, pats, dfu_rows, dtd_rows):
    # match barcodes
    osurv = np.zeros((len(pats), 2))
    for i in range(len(pats)):
        if barcodes.count(pats[i]):
            idx = barcodes.index(pats[i])
            # go through days to death see if we find any number.
            go_death = True
            j = 0
            found = False
            while go_death:
                div = re.split('\t', clinread[dtd_rows[j]])
                if div[idx] != 'NA':
                    osurv[i, 0] = float(div[idx])
                    osurv[i, 1] = 1.
                    go_death = False
                    found = True
                j = j + 1
                if j == len(dtd_rows):
                    go_death = False
            if not found:
                dfus = []
                for j in range(len(dfu_rows)):
                    div = re.split('\t', clinread[dfu_rows[j]])
                    if div[idx] != 'NA':
                        if np.sign(float(div[idx])) != -1:
                            dfus.append(float(div[idx]))
                if len(dfus) > 0:
                    osurv[i, 0] = max(dfus)
    return (osurv)


def exp_prep(genes, pats, path):
    # pre-allocate memory
    exp_i = np.zeros((len(pats), len(genes)))
    # read expression folder.
    expread = [line.rstrip('\n') for line in open(path + 'GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt')]
    # get patient names
    pat_exp = re.split('\t', expread[0])
    # keep only the first 15 characteres
    pat_exp_match = []
    pat_input_match = []
    for i in range(len(pat_exp)):
        pat_exp[i] = pat_exp[i][0:12]
        if pats.count(pat_exp[i]) > 0:
            pat_exp_match.append(pats.index(pat_exp[i]))
            pat_input_match.append(i)
    # Boom Chacalaca!
    gene_names = []
    # Double for loop this bish!
    for i in range(len(expread)):
        div = re.split('\t', expread[i])
        # gene name is column 1!
        gene_names.append(div[0])
    # match the genes to the gene names
    gene_index = []
    for i in range(len(genes)):
        if gene_names.count(genes[i]) > 0:
            gene_index.append(gene_names.index(genes[i]))
        else:
            gene_index.append(None)  # in case it wasn't found.
    # now scan each of the matched genes and store in exp_i.
    for j in range(len(gene_index)):
        for i in range(len(pat_exp_match)):
            if gene_index[j] != None:
                div = re.split('\t', expread[gene_index[j]])
                exp_i[pat_exp_match[i], j] = float(div[pat_input_match[i]])
    # match the patients
    return (exp_i, pat_exp_match, pat_input_match)


def find_exp_folder(folder_i):
    # Find all files within folder.
    files_folder = os.listdir(folder_i)
    # Read mutation data in folder_i.
    exp_folder = None
    mtch = False
    ii = 0
    # see if we have the Raw Calls use them.
    while not mtch:
        mm = re.search('rnaseqv2', files_folder[ii])
        if mm != None:
            exp_folder = '%s' % files_folder[ii]
            mtch = True
        ii = ii + 1
        if ii == len(files_folder):
            mtch = True
    return (exp_folder)


def surv_discrete(osurv, time_2_units, max_units):
    # pre-allocate memory for sdis
    sdis = np.zeros((osurv.shape[0], max_units, 2))
    # for loop and allocate.
    for i in range(osurv.shape[0]):
        ti = osurv[i, 0]
        di = osurv[i, 1]
        # translate ti to units
        ti = int(np.round(ti / time_2_units))
        if ti > max_units - 1:
            ti = max_units - 1  # copy max_units
        sdis[i, 0:ti, 0] = 1.
        if di == 1.:
            sdis[i, :, 1] = 1.
        else:
            sdis[i, 0:ti, 1] = 1.
    return (sdis)


def zero_one_transform(x, axis=1):
    """ Function to transform input data so that is between 0 and 1 in a given axis.
        Args: x: a numpy array.
             axis: the axis along which we will set data between 0 and 1.
        Returns:
             num_array: a numpy array with values between 0 and 1.
    """
    num_array = np.zeros((x.shape))
    if axis == 0:
        for i in range(num_array.shape[0]):
            if (np.var(xx[i, :]) > 0):
                min_i = np.min(x[i, :])
                num_array[i, :] = x[i, :] - min_i
                num_array[i, :] = num_array[i, :] / np.max(num_array[i, :])
    elif axis == 1:
        for i in range(num_array.shape[1]):
            if (len(num_array.shape) == 2):
                if (np.var(x[:, i]) > 0):
                    min_i = np.min(x[:, i])
                    num_array[:, i] = x[:, i] - min_i
                    num_array[:, i] = num_array[:, i] / np.max(num_array[:, i])
            elif (len(num_array.shape) == 3):
                for j in range(num_array.shape[2]):
                    if (np.var(x[:, i, j]) > 0):
                        min_i = np.min(x[:, i, j])
                        num_array[:, i, j] = x[:, i, j] - min_i
                        num_array[:, i, j] = num_array[:, i, j] / np.max(num_array[:, i, j])
    elif axis == 2:
        for i in range(num_array.shape[2]):
            if (np.var(x[:, :, i]) > 0):
                min_i = np.min(x[:, i, j])
                num_array[:, :, i] = x[:, :, i] - min_i
                num_array[:, :, i] = num_array[:, :, i] / np.max(num_array[:, :, i])
    num_array[np.isnan(num_array)] = 0.
    return (num_array)

