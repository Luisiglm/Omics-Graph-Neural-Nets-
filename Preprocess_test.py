"""Tests for analysis.py."""

import unittest
import os
import numpy as np

import Preprocess

class PreprocessTest(unittest.TestCase):

    def test_find_mut_folder(self):
        expected = Preprocess.find_mut_folder(os.getcwd()+'//'+'testdata')
        actual = 'gdac.broadinstitute.org_ACC.Mutation_Packager_Calls.Level_3.2016012800.0.0'
        self.assertEqual(expected, actual)

    def test_mut_prep(self):
        res = Preprocess.mut_prep(os.getcwd()+'//'+'testdata', ['LXN'])
        # check patient's data is correct.
        # find patient:
        indx = res[0].index('TCGA-OR-A5J1')
        expected = [int(res[1][indx,0]), 1]
        expected.append(int(res[3][indx,0,1]))
        expected.append(int(res[4][indx,0,0]))
        expected.append(int(res[5][indx,0,0]))
        expected.append(res[6])
        actual = [158388792, 1, 1, 1, 1, '37']
        self.assertEqual(expected,actual)

    def test_find_mut_folder(self):
        expected = Preprocess.find_exp_folder(os.getcwd()+'//'+'testdata')
        actual = 'ACC.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt'
        self.assertEqual(expected, actual)

    def test_exp_prep(self):
        pats = ['TCGA-OR-A5J2']
        res = Preprocess.exp_prep(['A1BG'], pats,os.getcwd()+'//'+'testdata//')
        actual = 0.685559330167748
        expected = res[0][0,0]
        self.assertEqual(expected,actual)

    def test_clin_prep(self):
        res = Preprocess.clin_prep(os.getcwd()+'//'+'testdata',['TCGA-OR-A5K0'])
        expected = int(res[0])
        actual = 69
        self.assertEqual(expected, actual)

    def test_normalize(self):
        rand_x = np.random.normal(0,1,5).reshape((5,1))
        rand_x_normal = Preprocess.normalize(rand_x,axis = 1)
        expected = int(np.round(np.var(rand_x_normal[:,0])))
        actual = 1
        self.assertEqual(expected,actual)


if __name__ == '__main__':
  unittest.main()
