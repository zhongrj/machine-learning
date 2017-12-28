import numpy as np
import tensorflow as tf

"""
    神经网络工具
"""


def init_weights(name, shape, trans=False, collections=None):
    """默认初始化 W"""
    if trans:
        init = tf.zeros_initializer
    else:
        init = tf.contrib.layers.variance_scaling_initializer()
    W = tf.get_variable(name, shape, tf.float32, init, collections=collections)
    return W


def init_bias(name, shape, trans=False, collections=None, init_value=np.array([[1., 0, 0],
                                                                               [0, 1., 0]])):
    """默认初始化 b"""
    init = tf.zeros_initializer
    b = tf.get_variable(name, shape, tf.float32, init)

    if trans:
        x = init_value
        x = x.astype('float32').flatten()
        b = tf.Variable(initial_value=x, collections=collections)

    return b


def BatchNormalization(input_tensor, phase=True, use_relu=False, name=None):
    """归一化"""
    normed = tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=phase, scope=name)

    if use_relu:
        normed = tf.nn.relu(normed)

    return normed


def Conv2D(input_tensor, input_shape, filter_size, num_filters, strides=1, name=None, collections=None):
    """卷积层"""
    shape = [filter_size, filter_size, input_shape, num_filters]

    W = init_weights(name=name + '_W', shape=shape, collections=collections)
    b = init_bias(name=name + '_b', shape=shape[-1], collections=collections)

    conv = tf.nn.conv2d(input_tensor, W, strides=[1, strides, strides, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, b, name=name)
    return conv


def MaxPooling2D(input_tensor, k=2, use_relu=False, name=None):
    """池化层"""
    pool = tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    if use_relu:
        pool = tf.nn.relu(pool)

    return pool


def Flatten(layer):
    """平铺"""
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def Dense(input_tensor, num_inputs, num_outputs, use_relu=True, trans=False, name=None, collections=None,
          init_value=np.array([[1., 0, 0],
                               [0, 1., 0]])):
    """全连接层"""
    shape = [num_inputs, num_outputs]

    W = init_weights(name=name + '_W', shape=shape, trans=trans, collections=collections)
    b = init_bias(name=name + '_b', shape=shape[-1], trans=trans, init_value=init_value, collections=collections)

    fc = tf.matmul(input_tensor, W, name=name) + b

    if use_relu:
        fc = tf.nn.relu(fc)

    return fc
