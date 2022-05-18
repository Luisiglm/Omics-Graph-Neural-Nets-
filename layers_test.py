"""layers_test.py."""

import tensorflow as tf
import numpy as np

import layers as layers

class LayersTest(tf.test.TestCase):

    def test_gat(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3,3))
            adj[0,1] = 1.
            adj[1,2] = 1.
            adj[2,0] = 1.
            # simulate some simple data with all ones.
            sim_data = tf.cast(np.ones((1,3,1)), 'float32')
            # make a gat layer object.
            gat_lay = layers.gat(adj,1)
            actual_1, actual_2 = gat_lay(sim_data)
            # the first output is the message passed output.
            temp_1 = tf.matmul(sim_data,gat_lay.w) # first stage.
            temp_att_self = tf.matmul(temp_1,gat_lay.ai) # first stage.
            temp_att_other = tf.transpose(tf.matmul(temp_1,gat_lay.aj), perm = [0, 2, 1])
            temp_att = temp_att_self+temp_att_other
            temp_att = temp_att - 1e9*(1-(gat_lay.adj+gat_lay.id))
            temp_att = tf.nn.softmax(temp_att, axis = 1)
            # check that the resulting network is the same as the output.
            self.assertAllClose(temp_att,actual_2)
            # check that we didn't add any extra edges.
            self.assertAllClose(1*(temp_att.numpy()[0,:,:]>0.),adj+np.eye(3))
            # let's check the message passing.
            temp_h =  tf.matmul(sim_data,gat_lay.w)
            temp_h = tf.transpose(temp_h, [0,2,1])
            temp_h = tf.matmul(temp_h, temp_att)
            expected_1 = tf.transpose(temp_h, [0,2,1])
            self.assertAllClose(expected_1,actual_1)



    def test_gpool(self):
        with self.test_session():
        # make a graph model for 0->1->2->0.
            adj = np.zeros((3,3))
            adj[0,1] = 1.
            adj[1,2] = 1.
            adj[2,0] = 1.
            adj = tf.cast(adj, 'float32')
            # simulate some simple data with all ones.
            sim_data = tf.cast(np.ones((1,3,1)), 'float32')
            # make a gpool_k layer object.
            gpool_lay = layers.gpool(adj,1)
            actual = gpool_lay(sim_data)
            temp = tf.matmul(sim_data,gpool_lay.w2)
            temp = tf.transpose(temp, [0,2,1])
            temp = tf.matmul(temp,adj)
            expected = tf.transpose(temp, [0,2,1]) + tf.matmul(sim_data,gpool_lay.w1)
            self.assertAllClose(expected,actual)


    def test_gpool_ad(self):
        with self.test_session():
        # make a graph model for 0->1->2->0.
            adj = np.zeros((3,3))
            adj[0,1] = 1.
            adj[1,2] = 1.
            adj[2,0] = 1.
            adj = tf.cast(adj, 'float32')
            # simulate some simple data with all ones.
            sim_data = tf.cast(np.ones((1,3,1)), 'float32')
            # make a gat layer object.
            gpool_lay = layers.gpool_ad(1)
            actual = gpool_lay(sim_data,adj)
            temp = tf.matmul(sim_data,gpool_lay.w)
            temp = tf.transpose(temp, [0,2,1])
            temp = tf.matmul(temp,adj)
            expected = tf.transpose(temp, [0,2,1])
            self.assertAllClose(expected,actual)


    def test_gate(self):
        with self.test_session():
            # make a graph model for 0->1->2->0.
            adj = np.zeros((3,3))
            adj[0,1] = 1.
            adj[1,2] = 1.
            adj[2,0] = 1.
            # simulate some simple data with all ones.
            sim_data = tf.cast(np.ones((1,3,1)), 'float32')
            # make a gat layer object.
            gate_lay = layers.gate(adj,1)
            actual_1, actual_2 = gate_lay(sim_data)
            # the first output is the message passed output.
            temp_1 = tf.matmul(sim_data,gate_lay.w) # first stage.
            temp_att_self = tf.matmul(temp_1,gate_lay.ai) # first stage.
            temp_att_other = tf.transpose(tf.matmul(temp_1,gate_lay.aj), perm = [0, 2, 1])
            temp_att = temp_att_self+temp_att_other
            temp_att = temp_att - 1e9*(1-gate_lay.adj)
            temp_att = tf.nn.sigmoid(temp_att + gate_lay.b_a)
            # check that the resulting network is the same as the output.
            self.assertAllClose(temp_att,actual_2)
            # check that we didn't add any extra edges.
            self.assertAllClose(1*(temp_att.numpy()[0,:,:]>0.),adj)
            # let's check the message passing.
            temp_h =  tf.matmul(sim_data,gate_lay.w)
            temp_h = tf.transpose(temp_h, [0,2,1])
            temp_h = tf.matmul(temp_h, temp_att)
            temp_h = tf.transpose(temp_h, [0,2,1])
            expected_1 = temp_h + tf.matmul(sim_data,gate_lay.w_2)
            self.assertAllClose(expected_1,actual_1)




    def test_gcn(self):
        with self.test_session():
            # make a graph model for 0<->1<->2<->0.
            adj = np.zeros((3,3))
            adj[0,1] = 1.
            adj[1,2] = 1.
            adj[2,0] = 1.
            adj[1,0] = 1.
            adj[2,1] = 1.
            adj[0,2] = 1.
            # simulate some simple data with all ones.
            sim_data = tf.cast(np.ones((1,3,1)), 'float32')
            # make a gat layer object.
            gcn_lay = layers.gcn(adj,1)
            actual = gcn_lay(sim_data)
            # first we will check if our normalized Laplacian is correct.
            D = np.diag(np.sum(adj,axis = 1)**-.5)
            L = np.eye(3) - np.matmul(np.matmul(D,adj),D)
            # check that the resulting Laplacian is correct.
            self.assertAllClose(L,gcn_lay.L)
            # let's check the message passing.
            temp_h1 =  tf.matmul(sim_data,gcn_lay.w)
            temp_h1 = tf.transpose(temp_h1, [0,2,1])
            temp_h1 = tf.matmul(temp_h1, L)
            expected = tf.transpose(temp_h1, [0,2,1])
            self.assertAllClose(expected,actual)


    def test_fully_3d(self):
        with self.test_session():
            # make some simple dataset.
            sim_data = tf.cast(np.ones((1,3,1)), 'float32')
            # create the layer.
            f3d_lay = layers.fully_3d(1,None)
            actual = f3d_lay(sim_data)
            # we have two sets of parameters and an input embedding here.
            inpt_embed = tf.math.add(sim_data,f3d_lay.w_g)
            expected = tf.matmul(inpt_embed,f3d_lay.w)+f3d_lay.b
            # check the dimensions are ok.
            self.assertEqual(expected.shape,[1,3,1])
            self.assertAllClose(expected,actual)




if __name__ == '__main__':
    tf.test.main()

