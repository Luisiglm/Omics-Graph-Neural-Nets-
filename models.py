import numpy as np
import tensorflow as tf
import layers
from tensorflow.keras.layers import Layer

def gat_k_model(input_shape,  adj,activation, out_shape, units = 10,hops = 5,depth = 2, act_out = 'sigmoid', units_last = 50):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    x, attn = layers.gat_k(adj,units,hops,activation)(inputs)
    #x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool_k(units,hops,activation)(x, attn)
        #x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units_last,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=activation)(x)
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model(inputs,outputs))



def gat_k_model_msk(input_shape,  adj,activation, out_shape,mask, units = 10,hops = 5,depth = 2, act_out = 'sigmoid', units_last = 50):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    x, attn = layers.gat_k(adj,units,hops,activation)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool_k(units,hops,activation)(x, attn)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = layers.masked_fc(units,mask)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units_last,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=activation)(x)
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model(inputs,outputs))


def gat_k_model_msk_2inputs(input_shape, input_shape_2, adj,activation, out_shape,mask, units = 10,hops = 5,depth = 2, depth2 = 2, units_2 = 10,act_out = 'sigmoid'):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    x, attn = layers.gat_k(adj,units,hops,activation)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool_k(units,hops,activation)(x, attn)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = layers.masked_fc(units,mask)(x)
    x = tf.keras.layers.Flatten()(x)
    inputs_2 = tf.keras.Input(shape = input_shape_2, dtype ="float32" )
    x2 = tf.keras.layers.Dense(units, activation,kernel_initializer = tf.keras.initializers.RandomNormal(0,1e-6))(inputs_2)
    for i in range(depth2):
        x2 = tf.keras.layers.Dense(units_2,activation,kernel_initializer = tf.keras.initializers.RandomNormal(0,1e-6))(x2)
    x = tf.concat((x,x2),axis = 1)    
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model([inputs, inputs_2],outputs))
