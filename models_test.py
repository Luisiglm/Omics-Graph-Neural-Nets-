"""layers_test.py."""

import tensorflow as tf
import numpy as np

import models

class ModelsTest(tf.test.TestCase):

    def test_gat_2(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            # simulate some inputs
            x_1 = np.ones((2,3,1))
            x_2 = np.ones((2,3,1))
            # create a gat model with 2 inputs.
            gt2 = models.gat_2([3,1],[3,1],adj,1)
            # check output is of size [2,1]
            outp = gt2([x_1,x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gt2 = models.gat_2([3,1],[3,1],adj,10)
            # check output is of size [2,1]
            outp = gt2([x_1,x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gt2.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gt2.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gt2.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gt2.layers[7]))
            expected_lay = "<class 'layers.gat'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gt2.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            gt2 = models.gat_2([3, 1], [3, 1], adj,1, depth = 1)
            outp = gt2([x_1, x_2])
            f1 = gt2.layers[0](x_1) # input layer
            f2 = gt2.layers[1](f1) # fully_3d
            f3 = gt2.layers[2](f2)  # BN
            f4 = gt2.layers[3](f3)  # relu
            f5 = gt2.layers[4](f4)  # Dropout
            f6 = gt2.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6),axis = 2) # concatenate inputs.
            f8, att = gt2.layers[7](f7) # graph gating.
            f9 = gt2.layers[8](f8) # BN
            f10 = gt2.layers[9](f9) # relu
            f11 = gt2.layers[10](f10) # Dropout
            f12 = gt2.layers[11](f11,att) # graph pooling.
            f13 = gt2.layers[12](f12)  # BN
            f14 = gt2.layers[13](f13)  # relu
            f15 = gt2.layers[14](f14)  # Droput
            f16 = gt2.layers[15](f15)  # Flatten
            expected_outp = gt2.layers[16](f16)  # Dense
            self.assertAllClose(expected_outp, outp)


    def test_gate_2(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            # simulate some inputs
            x_1 = np.ones((2,3,1))
            x_2 = np.ones((2,3,1))
            # create a gat model with 2 inputs.
            gte2 = models.gate_2([3,1],[3,1],adj,1)
            # check output is of size [2,1]
            outp = gte2([x_1,x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gte2 = models.gate_2([3,1],[3,1],adj,10)
            # check output is of size [2,10]
            outp = gte2([x_1,x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gte2.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gte2.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gte2.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gte2.layers[7]))
            expected_lay = "<class 'layers.gate'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gte2.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            # check operations are in the right order
            gt2 = models.gate_2([3, 1], [3, 1], adj,1, depth = 1)
            outp = gt2([x_1, x_2])
            f1 = gt2.layers[0](x_1) # input layer
            f2 = gt2.layers[1](f1) # fully_3d
            f3 = gt2.layers[2](f2)  # BN
            f4 = gt2.layers[3](f3)  # relu
            f5 = gt2.layers[4](f4)  # Dropout
            f6 = gt2.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6),axis = 2) # concatenate inputs.
            f8, att = gt2.layers[7](f7) # graph gating.
            f9 = gt2.layers[8](f8) # BN
            f10 = gt2.layers[9](f9) # relu
            f11 = gt2.layers[10](f10) # Dropout
            f12 = gt2.layers[11](f11,att) # graph pooling.
            f13 = gt2.layers[12](f12)  # BN
            f14 = gt2.layers[13](f13)  # relu
            f15 = gt2.layers[14](f14)  # Droput
            f16 = gt2.layers[15](f15)  # Flatten
            expected_outp = gt2.layers[16](f16)  # Dense
            self.assertAllClose(expected_outp, outp)


    def test_gpool_2(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            # simulate some inputs
            x_1 = np.ones((2,3,1))
            x_2 = np.ones((2,3,1))
            # create a gat model with 2 inputs.
            gpl2 = models.gpool_2([3,1],[3,1],adj,1)
            # check output is of size [2,1]
            outp = gpl2([x_1,x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gpl2 = models.gpool_2([3, 1], [3, 1], adj, 10)
            # check output is of size [2,1]
            outp = gpl2([x_1, x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gpl2.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gpl2.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gpl2.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gpl2.layers[7]))
            expected_lay = "<class 'layers.gpool'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gpl2.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            # check output
            gt2 = models.gpool_2([3, 1], [3, 1], adj,1, depth = 1)
            outp = gt2([x_1, x_2])
            f1 = gt2.layers[0](x_1) # input layer
            f2 = gt2.layers[1](f1) # fully_3d
            f3 = gt2.layers[2](f2)  # BN
            f4 = gt2.layers[3](f3)  # relu
            f5 = gt2.layers[4](f4)  # Dropout
            f6 = gt2.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6),axis = 2) # concatenate inputs.
            f8 = gt2.layers[7](f7) # graph pooling.
            f9 = gt2.layers[8](f8) # BN
            f10 = gt2.layers[9](f9) # relu
            f11 = gt2.layers[10](f10) # Dropout
            f12 = gt2.layers[11](f11) # graph pooling.
            f13 = gt2.layers[12](f12)  # BN
            f14 = gt2.layers[13](f13)  # relu
            f15 = gt2.layers[14](f14)  # Droput
            f16 = gt2.layers[15](f15)  # Flatten
            expected_outp = gt2.layers[16](f16)  # Dense
            self.assertAllClose(expected_outp, outp)


    def test_gcn_2(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            adj[1, 0] = 1.
            adj[2, 1] = 1.
            adj[0, 2] = 1.
            # simulate some inputs
            x_1 = np.ones((2, 3, 1))
            x_2 = np.ones((2, 3, 1))
            # create a gat model with 2 inputs.
            gcn2 = models.gcn_2([3, 1], [3, 1], adj, 1)
            # check output is of size [2,1]
            outp = gcn2([x_1, x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gcn2 = models.gcn_2([3, 1], [3, 1], adj, 10)
            # check output is of size [2,1]
            outp = gcn2([x_1, x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gcn2.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gcn2.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gcn2.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gcn2.layers[7]))
            expected_lay = "<class 'layers.gcn'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gcn2.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            # check the order of operations
            gt2 = models.gcn_2([3, 1], [3, 1], adj, 1, depth=1)
            outp = gt2([x_1, x_2])
            f1 = gt2.layers[0](x_1)  # input layer
            f2 = gt2.layers[1](f1)  # fully_3d
            f3 = gt2.layers[2](f2)  # BN
            f4 = gt2.layers[3](f3)  # relu
            f5 = gt2.layers[4](f4)  # Dropout
            f6 = gt2.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6), axis=2)  # concatenate inputs.
            f8 = gt2.layers[7](f7)  # graph pooling.
            f9 = gt2.layers[8](f8)  # BN
            f10 = gt2.layers[9](f9)  # relu
            f11 = gt2.layers[10](f10)  # Dropout
            f12 = gt2.layers[11](f11)  # graph pooling.
            f13 = gt2.layers[12](f12)  # BN
            f14 = gt2.layers[13](f13)  # relu
            f15 = gt2.layers[14](f14)  # Droput
            f16 = gt2.layers[15](f15)  # Flatten
            expected_outp = gt2.layers[16](f16)  # Dense
            self.assertAllClose(expected_outp, outp)


    def test_gat_3(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            # simulate some inputs
            x_1 = np.ones((2,3,1))
            x_2 = np.ones((2,3,1))
            x_3 = np.ones((2,3))
            # create a gat model with 2 inputs.
            gt3 = models.gat_3([3,1],[3,1],3,adj,1)
            # check output is of size [2,1]
            outp = gt3([x_1,x_2,x_3])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gt3 = models.gat_3([3,1],[3,1],3,adj,10)
            # check output is of size [2,1]
            outp = gt3([x_1,x_2,x_3])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gt3.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gt3.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gt3.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gt3.layers[7]))
            expected_lay = "<class 'layers.gat'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gt3.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            # see that you add the last input.
            gt3 = models.gat_3([3, 1], [3, 1], 3, adj,1, depth = 0)
            outp = gt3([x_1, x_2, x_3])
            f1 = gt3.layers[0](x_1) # input layer
            f2 = gt3.layers[1](f1) # fully_3d
            f3 = gt3.layers[2](f2)  # BN
            f4 = gt3.layers[3](f3)  # relu
            f5 = gt3.layers[4](f4)  # Dropout
            f6 = gt3.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6), axis=2)  # concatenate inputs.
            f8, att = gt3.layers[7](f7) # graph attention.
            f9 = gt3.layers[8](f8) # BN
            f10 = gt3.layers[9](f9) # relu
            f11 = gt3.layers[10](f10) # Dropout
            f12 = gt3.layers[11](f11) # flatten
            f13 = gt3.layers[12](x_3) # input layer
            f14 = gt3.layers[13](f12) # dense.
            f15 = gt3.layers[14](f13) # dense 2.
            f16 = tf.concat((f14,f15),axis =1)
            expected_outp = gt3.layers[16](f16)
            self.assertAllClose(expected_outp,outp)


    def test_gate_3(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            # simulate some inputs
            x_1 = np.ones((2,3,1))
            x_2 = np.ones((2,3,1))
            x_3 = np.ones((2,3))
            # create a gat model with 2 inputs.
            gte2 = models.gate_3([3,1],[3,1],3,adj,1)
            # check output is of size [2,1]
            outp = gte2([x_1,x_2,x_3])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gte2 = models.gate_3([3,1],[3,1],3,adj,10)
            # check output is of size [2,10]
            outp = gte2([x_1,x_2,x_3])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gte2.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gte2.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gte2.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gte2.layers[7]))
            expected_lay = "<class 'layers.gate'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gte2.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            # check order of operations
            gt3 = models.gate_3([3, 1], [3, 1], 3, adj, 1, depth=0)
            outp = gt3([x_1, x_2, x_3])
            f1 = gt3.layers[0](x_1)  # input layer
            f2 = gt3.layers[1](f1)  # fully_3d
            f3 = gt3.layers[2](f2)  # BN
            f4 = gt3.layers[3](f3)  # relu
            f5 = gt3.layers[4](f4)  # Dropout
            f6 = gt3.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6), axis=2)  # concatenate inputs.
            f8, att = gt3.layers[7](f7)  # graph attention.
            f9 = gt3.layers[8](f8)  # BN
            f10 = gt3.layers[9](f9)  # relu
            f11 = gt3.layers[10](f10)  # Dropout
            f12 = gt3.layers[11](f11)  # flatten
            f13 = gt3.layers[12](x_3)  # input layer
            f14 = gt3.layers[13](f12)  # dense.
            f15 = gt3.layers[14](f13)  # dense 2.
            f16 = tf.concat((f14, f15), axis=1)
            expected_outp = gt3.layers[16](f16)
            self.assertAllClose(expected_outp, outp)

    def test_gpool_3(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            # simulate some inputs
            x_1 = np.ones((2,3,1))
            x_2 = np.ones((2,3,1))
            x_3 = np.ones((2,3))
            # create a gat model with 2 inputs.
            gpl2 = models.gpool_2([3,1],[3,1],adj,1)
            # check output is of size [2,1]
            outp = gpl2([x_1,x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gpl2 = models.gpool_2([3, 1], [3, 1], adj, 10)
            # check output is of size [2,1]
            outp = gpl2([x_1, x_2])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gpl2.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gpl2.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gpl2.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gpl2.layers[7]))
            expected_lay = "<class 'layers.gpool'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gpl2.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            # check order of operations
            gt3 = models.gpool_3([3, 1], [3, 1], 3, adj, 1, depth=0)
            outp = gt3([x_1, x_2, x_3])
            f1 = gt3.layers[0](x_1)  # input layer
            f2 = gt3.layers[1](f1)  # fully_3d
            f3 = gt3.layers[2](f2)  # BN
            f4 = gt3.layers[3](f3)  # relu
            f5 = gt3.layers[4](f4)  # Dropout
            f6 = gt3.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6), axis=2)  # concatenate inputs.
            f8 = gt3.layers[7](f7)  # graph poling.
            f9 = gt3.layers[8](f8)  # BN
            f10 = gt3.layers[9](f9)  # relu
            f11 = gt3.layers[10](f10)  # Dropout
            f12 = gt3.layers[11](f11)  # flatten
            f13 = gt3.layers[12](x_3)  # input layer
            f14 = gt3.layers[13](f12)  # dense.
            f15 = gt3.layers[14](f13)  # dense 2.
            f16 = tf.concat((f14, f15), axis=1)
            expected_outp = gt3.layers[16](f16)
            self.assertAllClose(expected_outp, outp)

    def test_gcn_3(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3, 3))
            adj[0, 1] = 1.
            adj[1, 2] = 1.
            adj[2, 0] = 1.
            adj[1, 0] = 1.
            adj[2, 1] = 1.
            adj[0, 2] = 1.
            # simulate some inputs
            x_1 = np.ones((2, 3, 1))
            x_2 = np.ones((2, 3, 1))
            x_3 = np.ones((2, 3))
            # create a gat model with 2 inputs.
            gcn2 = models.gcn_3([3, 1], [3, 1], 3, adj, 1)
            # check output is of size [2,1]
            outp = gcn2([x_1, x_2, x_3])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 1)
            # check new size
            gcn2 = models.gcn_3([3, 1], [3, 1], 3, adj, 10)
            # check output is of size [2,1]
            outp = gcn2([x_1, x_2,x_3])
            # expected shape is 1.
            self.assertEqual(outp.shape[1], 10)
            # check that the second layer is a fully_3d.
            actual_lay = str(type(gcn2.layers[1]))
            expected_lay = "<class 'layers.fully_3d'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the third layer is a Batch Normalization Layer.
            actual_lay = str(type(gcn2.layers[2]))
            expected_lay = "<class 'keras.layers.normalization.batch_normalization.BatchNormalization'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fourth layer is a Dropout Layer.
            actual_lay = str(type(gcn2.layers[4]))
            expected_lay = "<class 'keras.layers.regularization.dropout.Dropout'>"
            self.assertEqual(actual_lay, expected_lay)
            # check the fifth layer is a Graph Attention Layer.
            actual_lay = str(type(gcn2.layers[7]))
            expected_lay = "<class 'layers.gcn'>"
            self.assertEqual(actual_lay, expected_lay)
            # count the layers after this. (there should be 2x4).
            expected_length = len(gcn2.layers) - 11 - 2
            self.assertEqual(actual_lay, expected_lay)
            # check order of operations.
            gt3 = models.gcn_3([3, 1], [3, 1], 3, adj, 1, depth=0)
            outp = gt3([x_1, x_2, x_3])
            f1 = gt3.layers[0](x_1)  # input layer
            f2 = gt3.layers[1](f1)  # fully_3d
            f3 = gt3.layers[2](f2)  # BN
            f4 = gt3.layers[3](f3)  # relu
            f5 = gt3.layers[4](f4)  # Dropout
            f6 = gt3.layers[5](x_2)  # input layer
            f7 = tf.concat((f5, f6), axis=2)  # concatenate inputs.
            f8 = gt3.layers[7](f7)  # graph convolution.
            f9 = gt3.layers[8](f8)  # BN
            f10 = gt3.layers[9](f9)  # relu
            f11 = gt3.layers[10](f10)  # Dropout
            f12 = gt3.layers[11](f11)  # flatten
            f13 = gt3.layers[12](x_3)  # input layer
            f14 = gt3.layers[13](f12)  # dense.
            f15 = gt3.layers[14](f13)  # dense 2.
            f16 = tf.concat((f14, f15), axis=1)
            expected_outp = gt3.layers[16](f16)
            self.assertAllClose(expected_outp, outp)

if __name__ == '__main__':
    tf.test.main()
