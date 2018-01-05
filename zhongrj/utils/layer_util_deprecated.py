import tensorflow as tf
from zhongrj.utils.math_util import *

"""
    神经网络层工具
    (废弃, 使用tf.layers自带)
"""


def init_weights(name, shape, trans=False, collections=None):
    """默认初始化 W"""
    if trans:
        init = tf.zeros_initializer
    else:
        init = tf.contrib.layers.variance_scaling_initializer()
        # init = tf.truncated_normal_initializer(stddev=0.1)
    W = tf.get_variable(name, shape, tf.float32, init, collections=collections)
    return W


def init_bias(name, shape, trans=False, collections=None, init_value=np.array([[1., 0, 0],
                                                                               [0, 1., 0]])):
    """默认初始化 b"""
    init = tf.zeros_initializer
    b = tf.get_variable(name, shape, tf.float32, init, collections=collections)

    if trans:
        x = init_value
        x = x.astype('float32').flatten()
        b = tf.Variable(initial_value=x, collections=collections)

    return b


def BatchNormalization(input_tensor, phase, act=tf.identity, name=None):
    """归一化"""
    return act(tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=phase, scope=name))

    # return act(tf.layers.batch_normalization(input_tensor, center=True, scale=True, ))
    # return act(tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, epsilon))


def Conv2D(input_tensor, input_shape, filter_size, num_filters, name='Conv', strides=1, collections=None):
    """卷积层"""
    shape = [filter_size, filter_size, input_shape, num_filters]

    W = init_weights(name=name + '_W', shape=shape, collections=collections)
    b = init_bias(name=name + '_b', shape=shape[-1], collections=collections)

    conv = tf.nn.conv2d(input_tensor, W, strides=[1, strides, strides, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, b, name=name)
    return conv


def MaxPooling2D(input_tensor, k=2, act=tf.identity, name=None):
    """池化层"""
    pool = act(tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name))

    return pool


def DeConv2D(input_tensor, output_shape, filter_size, name='DeConv', strides=2, collections=None):
    """反卷积层"""
    output_shape = [tf.shape(input_tensor)[0]] + output_shape
    shape = [filter_size, filter_size, output_shape[3], input_tensor.get_shape()[3]]

    W = init_weights(name=name + '_W', shape=shape, collections=collections)
    b = init_bias(name=name + '_b', shape=shape[-2], collections=collections)

    deconv = tf.nn.conv2d_transpose(input_tensor, W, output_shape=output_shape, strides=[1, strides, strides, 1],
                                    padding='SAME')
    deconv = tf.nn.bias_add(deconv, b, name=name)
    return deconv


def Flatten(layer):
    """平铺"""
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def Dense(input_tensor, num_inputs, num_outputs, name='fc', act=tf.identity, trans=False, collections=None,
          init_value=np.array([[1., 0, 0],
                               [0, 1., 0]])):
    """全连接层"""
    shape = [num_inputs, num_outputs]

    W = init_weights(name=name + '_W', shape=shape, trans=trans, collections=collections)
    b = init_bias(name=name + '_b', shape=shape[-1], trans=trans, init_value=init_value, collections=collections)

    return act(tf.nn.bias_add(tf.matmul(input_tensor, W), b, name=name))


def __CNN_old(cnn_input, is_train, classes,
              name='CNN', cnn_layers=2, dnn_layers=2,
              act=tf.nn.relu, reuse=None, collections=None):
    """(废弃)CNN识别层"""
    with tf.variable_scope(name, reuse=reuse):
        input_tensor, input_channel, output_channel = cnn_input, cnn_input.get_shape()[3], 16
        for i in range(1, cnn_layers + 1):
            filter_size = 5 if i <= 2 else 3
            conv = Conv2D(input_tensor, input_channel, filter_size, output_channel,
                          name='conv%s' % i, collections=collections)
            bn = BatchNormalization(conv, is_train)
            input_tensor = MaxPooling2D(bn, act=act, name='pool%s' % i)
            input_channel = output_channel
            output_channel = output_channel * 2

        cnn_flat, cnn_size = Flatten(input_tensor)

        input_tensor, input_size, output_size = cnn_flat, cnn_size, 512 * 2 ** (dnn_layers - 2)

        for i in range(1, dnn_layers + 1):
            if i == dnn_layers:
                input_tensor = Dense(input_tensor, input_size, classes, name='fc%s' % i, collections=collections)
            else:
                input_tensor = Dense(input_tensor, input_size, output_size, name='fc%s' % i, collections=collections)
                input_tensor = BatchNormalization(input_tensor, is_train, act=act)
                input_size = output_size
                output_size = output_size / 2

        return input_tensor


def __DeCNN_old(decnn_input, is_train, output_dims,
                name='DeCNN', cnn_layers=2, dnn_layers=2,
                act=tf.nn.relu, reuse=False, collections=None):
    """(废弃)反CNN生成层"""
    with tf.variable_scope(name, reuse=reuse):
        if cnn_layers == 0:
            fc_output_image_dims = output_dims
        else:
            fc_output_image_dims = [output_dims[0] // 2 ** cnn_layers, output_dims[1] // 2 ** cnn_layers,
                                    16 * 2 ** (cnn_layers - 1)]

        input_tensor, input_size, output_size = decnn_input, decnn_input.get_shape()[1], 512

        for i in reversed(range(1, dnn_layers + 1)):
            if i == 1:
                input_tensor = Dense(input_tensor, input_size, reduce_product(fc_output_image_dims), name='fc%s' % i,
                                     collections=collections)
            else:
                input_tensor = Dense(input_tensor, input_size, output_size, name='fc%s' % i, collections=collections)
                input_tensor = BatchNormalization(input_tensor, is_train, act=act)
                input_size = output_size
                output_size = output_size * 2

        input_tensor = tf.reshape(input_tensor, [-1] + fc_output_image_dims)

        for i in reversed(range(1, cnn_layers + 1)):
            filter_size = 5 if i <= 2 else 3

            if i == 1:
                output_shape = output_dims
            else:
                output_shape = [output_dims[0] // 2 ** (i - 1), output_dims[1] // 2 ** (i - 1),
                                16 * 2 ** (i - 2)]

            input_tensor = DeConv2D(input_tensor, output_shape, filter_size, strides=2, name='deconv%s' % i,
                                    collections=collections)
            if i != 1:
                input_tensor = BatchNormalization(input_tensor, is_train)

        return tf.reshape(input_tensor, [-1] + output_dims)


