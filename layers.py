import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class gat(Layer):
    """ Graph Attention Layer adapted to genomic data.
        Takes as input a tf float 32 object of shape batch x genes x features
        Initialization Args:
            adj: a gene x gene numpy array
            units: the number of features to obtain.
            activation: a keras activation function.
        Parameters:
            w: a features x gat.units  tf variable .
            ai: a gat.units x genes tf variable that corresponds to the self attention parameter.
            aj: a gat.units x genes tf variable that is the attention to adjacent nodes parameter.
    """
    def __init__(self,adj,units, activation = None):
        super(gat, self).__init__()
        self.id = tf.cast(np.identity(adj.shape[1]), dtype = "float32")
        self.adj = tf.cast(adj, dtype ='float32')
        self.activation = tf.keras.activations.get(activation)
        self.units = units
        self.nodes = adj.shape[1]
    def build(self, input_shape):
        w_init = tf.keras.initializers.GlorotNormal(seed=None)
        self.w = tf.Variable(name = "weight",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        self.ai = tf.Variable(name = "self_attn",
                             initial_value = w_init(shape = (self.units,self.nodes), dtype = "float32"),
                             trainable = True)
        self.aj = tf.Variable(name = "other_attn",
                             initial_value = w_init(shape = (self.units,self.nodes), dtype = "float32"),
                             trainable = True)
        super().build(input_shape)
    def call(self, inputs):
        # batch x nodes x features //inputs
        f = tf.matmul(inputs,self.w) # batch x nodes x features'
        self_attn = tf.matmul(f,self.ai) # batch x nodes x nodes
        other_attn = tf.transpose(tf.matmul(f,self.aj), perm = [0, 2, 1]) # batch x nodes x nodes
        attn = tf.math.add(self_attn,other_attn)
        attn = tf.math.add(attn, -1e09*(1-(self.adj+self.id)))# it should be batch x nodes x nodes
        attn = tf.nn.softmax(attn, axis = 1) 
        # transpose this bad boy.
        f = tf.transpose(f,perm=[0, 2, 1])# batch x features' x nodes // 
        f = tf.matmul(f,attn)# message passing.
        f = tf.transpose(f,perm=[0, 2, 1]) # transpose again!
        return(self.activation(f), attn)# activate and poom!



class gate(Layer):
    """ Gated Graph Layer adapted to genomic data.
        Takes as input a tf float 32 object of shape batch x genes x features
        Initialization Args:
            adj: a gene x gene numpy array
            units: the number of features to obtain.
            activation: a keras activation function.
        Parameters:
            w: a features x gat.units  tf variable.
            w_2: a features x gat.units  tf variable.
            ai: a gat.units x genes tf variable that corresponds to the self gate parameter.
            aj: a gat.units x genes tf variable that is the gate to adjacent nodes parameter.
    """
    def __init__(self,adj,units, activation = None):
        super(gate, self).__init__()
        self.id = tf.cast(np.identity(adj.shape[1]), dtype = "float32")
        self.adj = tf.cast(adj, dtype ='float32')
        self.activation = tf.keras.activations.get(activation)
        self.units = units
    def build(self, input_shape):
        w_init = tf.keras.initializers.GlorotNormal(seed=None)
        self.w = tf.Variable(name = "weight",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        self.ai = tf.Variable(name = "self_attn",
                             initial_value = w_init(shape = (self.units,self.adj.shape[1]), dtype = "float32"),
                             trainable = True)
        self.aj = tf.Variable(name = "other_attn",
                             initial_value = w_init(shape = (self.units,self.adj.shape[1]), dtype = "float32"),
                             trainable = True)
        self.w_2 = tf.Variable(name = "weight_2",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        self.b_a = tf.Variable(name = "other_gate",
                             initial_value = w_init(shape = (self.adj.shape[0],self.adj.shape[1]), dtype = "float32"),
                             trainable = True)
        super().build(input_shape)
    def call(self, inputs):
        # batch x nodes x features //inputs
        f = tf.matmul(inputs,self.w) # batch x nodes x features'
        self_attn = tf.matmul(f,self.ai) # batch x nodes x nodes
        other_attn = tf.transpose(tf.matmul(f,self.aj), perm = [0, 2, 1]) # batch x nodes x nodes
        attn = tf.math.add(self_attn,other_attn)
        attn = tf.math.add(attn, -1e09*(1-(self.adj)))# it should be batch x nodes x nodes
        gate_h = tf.keras.activations.sigmoid(attn+self.b_a) 
        # transpose this bad boy.
        f = tf.transpose(f,perm=[0, 2, 1])# batch x features' x nodes // 
        f = tf.matmul(f,gate_h)# message passing.
        f = tf.transpose(f,perm=[0, 2, 1]) # transpose again!
        f = tf.add(f, tf.matmul(inputs, self.w_2))
        return(self.activation(f), gate_h)# activate and poom!




class gcn(Layer):
    """ Graph Convolution Layer adapted to genomic data.
        Takes as input a tf float 32 object of shape batch x genes x features
        Initialization Args:
            adj: a gene x gene numpy array
            units: the number of features to obtain.
            activation: a keras activation function.
        Parameters:
            w: a features x gat.units  tf variable.
    """
    def __init__(self,adj,units, activation = None):
        super(gcn, self).__init__()
        self.id = tf.cast(np.identity(adj.shape[1]), dtype = "float32")
        self.adj = tf.cast(adj, dtype ='float32')
        self.d = tf.math.multiply(self.id,tf.math.reduce_sum(self.adj, axis = 0)**-.5)
        self.L = self.id-tf.matmul(self.d,tf.matmul(self.adj,self.d))
        self.activation = tf.keras.activations.get(activation)
        self.units = units
        self.nodes = adj.shape[1]
    def build(self, input_shape):
        w_init = tf.keras.initializers.RandomNormal(  mean=0.0, stddev=1/np.sum(input_shape[-1]), seed=None)
        self.w = tf.Variable(name = "weight",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        super().build(input_shape)
    def call(self, inputs):
        # batch x nodes x features //inputs
        f = tf.matmul(inputs,self.w) # batch x nodes x features'
        # transpose this bad boy.
        f = tf.transpose(f,perm=[0, 2, 1])# batch x features' x nodes // 
        f = tf.matmul(f,self.L)# message passing.
        f = tf.transpose(f,perm=[0, 2, 1]) # transpose again!
        return(self.activation(f))# activate and poom!


class gpool(Layer):
    """ Graph Pooling Layer adapted to genomic data.
        Takes as input a tf float 32 object of shape batch x genes x features
        Args:
            adj: a batch x genes x genes tensor object.
            units: the number of features to obtain.
            activation: a keras activation function.
            w: a features x gat.units  tf variable.
    """
    def __init__(self,units, adj,activation = None):
        super(gpool, self).__init__()
        self.activation = tf.keras.activations.get(activation)
        self.units = units
        self.id = tf.cast(np.identity(adj.shape[1]), dtype = "float32")
        self.adj = tf.cast(adj, dtype = 'float32')
    def build(self, input_shape):
        w_init = tf.keras.initializers.GlorotNormal(seed=None)
        self.w1 = tf.Variable(name = "weight",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        self.w2 = tf.Variable(name = "weight",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        super().build(input_shape)
    def call(self, inputs):
        # batch x nodes x features //inputs  
        f = tf.matmul(inputs,self.w1) # batch x nodes x features'
        h = tf.transpose(inputs,perm=[0, 2, 1])# batch x features' x nodes // 
        h = tf.matmul(h,self.adj)# message passing.
        h = tf.transpose(h,perm=[0, 2, 1]) # transpose again!
        f = tf.add(tf.matmul(h, self.w2),f)
        # transpose this bad boy.
        return(self.activation(f))# activate and poom!



class gpool_ad(Layer):
    """ Graph Pooling Layer adapted to genomic data.
        Takes as input a tf float 32 object of shape batch x genes x features
        Args:
            adj: a batch x genes x genes tensor object.
            units: the number of features to obtain.
            activation: a keras activation function.
            w: a features x gat.units  tf variable.
    """
    def __init__(self,units, activation = None):
        super(gpool_ad, self).__init__()
        self.activation = tf.keras.activations.get(activation)
        self.units = units
    def build(self, input_shape):
        w_init = tf.keras.initializers.RandomNormal(  mean=0.0, stddev=1/np.sum(input_shape[-1]), seed=None)
        self.w = tf.Variable(name = "weight",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        super().build(input_shape)
    def call(self, inputs, adj):
        # batch x nodes x features //inputs
        f = tf.matmul(inputs,self.w) # batch x nodes x features'
        # transpose this bad boy.
        f = tf.transpose(f,perm=[0, 2, 1])# batch x features' x nodes // 
        f = tf.matmul(f,adj)# message passing.
        f = tf.transpose(f,perm=[0, 2, 1]) # transpose again!
        return(self.activation(f))# activate and poom!



class gated_pool_ad(Layer):
    """ Gated Pooling Layer adapted to genomic data.
        Takes as input a tf float 32 object of shape batch x genes x features
        Args:
            adj: a batch x genes x genes tensor object.
            units: the number of features to obtain.
            activation: a keras activation function.
            w: a features x gat.units  tf variable.
            w_2: a features x gat.units tf Variable.
    """
    def __init__(self,units, activation = None):
        super(gated_pool_ad, self).__init__()
        self.activation = tf.keras.activations.get(activation)
        self.units = units
    def build(self, input_shape):
        w_init = tf.keras.initializers.RandomNormal(  mean=0.0, stddev=1/np.sum(input_shape[-1]), seed=None)
        self.w = tf.Variable(name = "weight",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        self.w_2 = tf.Variable(name = "weight_2",
                             initial_value = w_init(shape = (input_shape[2],self.units), dtype = "float32"),
                             trainable = True)
        super().build(input_shape)
    def call(self, inputs, adj):
        # batch x nodes x features //inputs
        f = tf.matmul(inputs,self.w) # batch x nodes x features'
        # transpose this bad boy.
        f = tf.transpose(f,perm=[0, 2, 1])# batch x features' x nodes // 
        f = tf.matmul(f,adj)# message passing.
        f = tf.transpose(f,perm=[0, 2, 1]) # transpose again!
        f = tf.add(f, tf.matmul(inputs, self.w_2))
        return(self.activation(f))# activate and poom!
    
    

class fully_3d(Layer):
    def __init__(self, units, activation = None):
        super(fully_3d,self).__init__()
        self.activation = tf.keras.activations.get(activation)
        self.units = units
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w_g = tf.Variable(name = "kernel",
                             initial_value = w_init(shape = (input_shape[1],input_shape[2]),
                             dtype = "float32"),
                             trainable = True)
        self.w = tf.Variable(name = "kernel",
                             initial_value = w_init(shape = (input_shape[2],self.units),
                             dtype = "float32"),
                             trainable = True)
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(name = "bias",
                             initial_value = b_init(shape = (self.units,), dtype = "float32"),
                             trainable = True)
        super().build(input_shape)
    def call(self,inputs):
        inpts = tf.math.add(inputs,self.w_g)#+self.b_g# zero out the variables we aren't using. 
        return(self.activation(tf.matmul(inpts, self.w)+self.b))

    
