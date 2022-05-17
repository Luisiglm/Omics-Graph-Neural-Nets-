import tensorflow as tf
import layers



# Models for Cases with Graph Located Features

####################
# 2 INPUTS
####################

# e.g. Patients with gene-wise data, and gene's connected through a graph.

# Layers with two separated sets of inputs.

def gate_2(input_shape, input_shape_2, adj, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Graph Neural Network Model adapted to genomic data.
            It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
            Args:
                input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                input_shape_2: a tuple with the shape of the input 2 [genes x features]
                adj:  a gene x gene numpy array with the graph's adjacency matrix.
                units: the number of features to obtain.
                out_shape: the number of columns for the output.
                depth: the number of layers of graph pooling.
                act_out: activation function for the last layer.
            Returns:
                A Keras model object.
        """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x, attn = layers.gate(adj, units, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool_ad(units, None)(x, attn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model([inputs, x2], outputs))


def gat_2(input_shape, input_shape_2, adj, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Gated Graph Neural Network Model adapted to genomic data.
            It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
            Args:
                input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                input_shape_2: a tuple with the shape of the input 2 [genes x features]
                adj:  a gene x gene numpy array with the graph's adjacency matrix.
                units: the number of features to obtain.
                out_shape: the number of columns for the output.
                depth: the number of layers of graph pooling.
                act_out: activation function for the last layer.
            Returns:
                A Keras model object.
        """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x, attn = layers.gat(adj, units, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gated_pool_ad(units, None)(x, attn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model([inputs, x2], outputs))


def gpool_2(input_shape, input_shape_2, adj, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Neural Network Model adapted to genomic data.
            It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
            Args:
                input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                input_shape_2: a tuple with the shape of the input 2 [genes x features]
                adj: a gene x gene numpy array with the graph's adjacency matrix.
                units: the number of features to obtain.
                out_shape: the number of columns for the output.
                depth: the number of layers of graph pooling.
                act_out: activation function for the last layer.
            Returns:
                A Keras model object.
        """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x = layers.gpool(units, adj, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool(units, adj, None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model([inputs, x2], outputs))


def gcn_2(input_shape, input_shape_2, adj,  out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Convolutional Neural Network Model adapted to genomic data.
            It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
            Args:
                input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                input_shape_2: a tuple with the shape of the input 2 [genes x features]
                adj: a gene x gene numpy array with the undirected graph's adjacency matrix.
                units: the number of features to obtain.
                out_shape: the number of columns for the output.
                depth: the number of layers of graph pooling.
                act_out: activation function for the last layer.
            Returns:
                A Keras model object.
        """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x = layers.gcn(units, adj, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gcn(units, adj, None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model([inputs, x2], outputs))


####################
# 3 INPUTS
####################
# Models with three separated sets of inputs. (i.e. Mutations, Gene Expression and Clinical Features).


def gat_3(input_shape, input_shape_2, input_shape_3, adj, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Attention Neural Network Model adapted to genomic data.
              It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
              The last input will be combined at the end with the graph message passing. (e.g. clinical data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  input_shape_3: a tuple with the shape of the input 3 [features]
                  adj: a gene x gene numpy array with the graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
          """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x, attn = layers.gat(adj, units, None)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool_ad(units, None)(x, attn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x3 = tf.keras.Input(shape=input_shape_3, dtype="float32")
    outputs_1 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x)
    outputs_2 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x3)
    outputs = outputs_1 + outputs_2
    return (tf.keras.Model([inputs, x2, x3], outputs))


def gpool_3(input_shape, input_shape_2,input_shape_3, adj, activation, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Neural Network Model adapted to genomic data.
              It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
              The last input will be combined at the end with the graph message passing. (e.g. clinical data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  input_shape_3: a tuple with the shape of the input 3 [features]
                  adj: a gene x gene numpy array with the graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
          """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x = layers.gpool(units, adj, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool(units, adj, None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
    x3 = tf.keras.Input(shape=input_shape_3, dtype="float32")
    outputs_1 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x)
    outputs_2 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x3)
    outputs = outputs_1 + outputs_2
    return (tf.keras.Model([inputs, x2, x3], outputs))


def gate_3(input_shape, input_shape_2,input_shape_3, adj, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Gated Neural Network Model adapted to genomic data.
              It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
              The last input will be combined at the end with the graph message passing. (e.g. clinical data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  input_shape_3: a tuple with the shape of the input 3 [features]
                  adj: a gene x gene numpy array with the graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
          """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x, attn = layers.gate(adj, units, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gated_pool_ad(units, None)(x, attn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x3 = tf.keras.Input(shape=input_shape_3, dtype="float32")
    outputs_1 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x)
    outputs_2 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x3)
    outputs = outputs_1 + outputs_2
    return (tf.keras.Model([inputs, x2, x3], outputs))


def gcn_3(input_shape, input_shape_2, input_shape_3, adj, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Convolution Neural Network Model adapted to genomic data.
              It takes three inputs, the first will be processed with a linear layer. (e.g. mutation data)
              The last input will be combined at the end with the graph message passing. (e.g. clinical data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  input_shape_3: a tuple with the shape of the input 3 [features]
                  adj: a gene x gene numpy array with the undirected graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
          """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x = layers.gcn(units, adj, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gcn(units, adj, None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
    x3 = tf.keras.Input(shape=input_shape_3, dtype="float32")
    outputs_1 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x)
    outputs_2 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x3)
    outputs = outputs_1 + outputs_2
    return (tf.keras.Model([inputs, x2, x3], outputs))


# Models for Node Regressions for Depmap Dataset.
####################
# 2 INPUTS
####################

def gate_2_node(input_shape, input_shape_2, adj, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Gated Graph Neural Network Model adapted to genomic data for node models.
              It takes two inputs, the first will be processed with a linear layer. (e.g. mutation data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  adj: a gene x gene numpy array with the graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
          """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x, attn = layers.gate(adj, units, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gated_pool_ad(units, None)(x, attn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = layers.gated_pool_ad(out_shape, act_out)(x, attn)
    outputs = tf.keras.layers.Flatten()(x)
    return (tf.keras.Model([inputs, x2], outputs))


def gat_2_node(input_shape, input_shape_2, adj,  out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Attention Neural Network Model adapted to genomic data for node models.
              It takes two inputs, the first will be processed with a linear layer. (e.g. mutation data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  adj: a gene x gene numpy array with the graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
          """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x, attn = layers.gat(adj, units, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool_ad(units, None)(x, attn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = layers.gpool_ad(out_shape, act_out)(x, attn)
    outputs = tf.keras.layers.Flatten()(x)
    return (tf.keras.Model([inputs, x2], outputs))


def gpool_2_node(input_shape, input_shape_2, adj,  out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Neural Network Model adapted to genomic data for node models.
              It takes two inputs, the first will be processed with a linear layer. (e.g. mutation data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  adj: a gene x gene numpy array with the graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
          """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x = layers.gpool(units, adj, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gpool(units, adj, None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model([inputs, x2], outputs))


def gcn_2_node(input_shape, input_shape_2, adj,  out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Graph Convolutional Neural Network Model adapted to genomic data for node models.
              It takes two inputs, the first will be processed with a linear layer. (e.g. mutation data)
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features] (linear layer preprocessing).
                  input_shape_2: a tuple with the shape of the input 2 [genes x features]
                  adj: a gene x gene numpy array with the undirected graph's adjacency matrix.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
    """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x2 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    x = tf.concat((x, x2), axis=2)
    x = layers.gcn(units, adj, None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(depth):
        x = layers.gcn(units, adj, None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model([inputs, x2], outputs))


####################
# Locally-Connected Multi-Layer Perceptron.
####################

def lmlp(input_shape, paths, activation, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Local Multi-Layer Perceptron Model adapted to genomic data for node models.
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features].
                  paths: a gene x pathways numpy array with what gene's belong to what pathways.
                  activation: a keras activation function.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
    """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, activation)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = layers.path_fc(paths, units, activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for i in range(depth):
        x = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                  activation=activation)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model(inputs, outputs))


def lmlp_2(input_shape, input_shape_2, paths, activation, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Local Multi-Layer Perceptron Model adapted to genomic data for node models.
              It takes two inputs, the first is associated with the genetic information, the second is for
              clinical data.
              Args:
                  input_shape: a tuple with the shape of the input 1 [genes x features].
                  input_shape_2: a tuple with the shape of the input 2 [features].
                  paths: a gene x pathways numpy array with what gene's belong to what pathways.
                  activation: a keras activation function.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
    """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = layers.fully_3d(units, activation)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = layers.path_fc(paths, units, activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for i in range(depth):
        x = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                  activation=activation)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x3 = tf.keras.Input(shape=input_shape_2, dtype="float32")
    outputs_1 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x)
    outputs_2 = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      activation=act_out)(x3)
    outputs = outputs_1 + outputs_2
    return (tf.keras.Model(inputs, outputs))


def mlp(input_shape, activation, out_shape, units=10, depth=2, act_out='sigmoid'):
    """ Multi-Layer Perceptron Model adapted to genomic data for node models.
              Args:
                  input_shape: a tuple with the shape of the input [features].
                  activation: a keras activation function.
                  units: the number of features to obtain.
                  out_shape: the number of columns for the output.
                  depth: the number of layers of graph pooling.
                  act_out: activation function for the last layer.
              Returns:
                  A Keras model object.
    """
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")
    x = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                              activation=activation)(inputs)
    for i in range(depth):
        x = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                  activation=activation)(x)
    outputs = tf.keras.layers.Dense(out_shape, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    activation=act_out)(x)
    return (tf.keras.Model(inputs, outputs))




