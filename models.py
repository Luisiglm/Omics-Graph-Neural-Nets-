import numpy as np
import tensorflow as tf
import layers
from tensorflow.keras.layers import Layer

def gcn_modelfc_2inputs(input_shape, input_shape_2,  adj,activation, out_shape, units = 10,depth = 2, act_out = 'sigmoid', units_last = 50):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    x = layers.fully_3d( units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape = input_shape_2, dtype ="float32" )
    x = tf.concat((x,x2),axis = 2)   
    x  = layers.gpool(units,adj,None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool (units,adj,None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model([inputs,x2],outputs))

def gat_modelfc_3inputs(input_shape, input_shape_2, input_shape_3,  adj,out_shape,units = 10,depth = 2, act_out = 'sigmoid'):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    x = fully_3d( units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape = input_shape_2, dtype ="float32" )
    x = tf.concat((x,x2),axis = 2)   
    x, attn = layers.gat(adj,units,None)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool_ad(units,None)(x, attn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x3 = tf.keras.Input(shape = input_shape_3, dtype ="float32" )
    outputs_1 = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    outputs_2 = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x3)
    outputs = outputs_1 + outputs_2
    return(tf.keras.Model([inputs,x2,x3],outputs))

def mlp_local_modelfc(input_shape,  paths,activation, out_shape, units = 10,depth = 2, act_out = 'sigmoid'):
    inputs = tf.keras.Input(shape = input_shape, dtype ="float32" )
    x = layers.fully_3d( units, activation)(inputs)
    x = path_fc(pahts,units,activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for i in range(depth):
        x = tf.keras.layers.Dense(units_last,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=activation)(x)
    outputs = tf.keras.layers.Dense(out_shape,kernel_initializer = tf.keras.initializers.GlorotNormal(), activation=act_out)(x)
    return(tf.keras.Model(inputs,outputs))



