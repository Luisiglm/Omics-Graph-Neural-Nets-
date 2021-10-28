import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

def gat_model(input_shape,  adj,activation, out_shape, units = 10,depth = 2, act_out = 'sigmoid'):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = scalar_layer_3d()(inputs)
    x , attn= gat(adj,units,activation)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x  =  gpool_ad(units,activation)(x, attn)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model(inputs,outputs))


def gate_model(input_shape,  adj,activation, out_shape, units = 10,depth = 2, act_out = 'sigmoid'):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    #x = tf.keras.layers.BatchNormalization()(inputs)
    x = scalar_layer_3d()(inputs)
    x , attn= gate(adj,units,activation)(x)
    for i in range(depth):
        x  =  gated_pool_ad(units,activation)(x, attn)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model(inputs,outputs))

def sign_gate_model(input_shape,  adj,activation, out_shape, units = 10,depth = 2, act_out = 'sigmoid'):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    #x = tf.keras.layers.BatchNormalization()(inputs)
    x = scalar_layer_3d()(inputs)
    x , attn= sign_gate(adj,units,activation)(x)
    for i in range(depth):
        x  =  gated_pool_ad(units,activation)(x, attn)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model(inputs,outputs))
