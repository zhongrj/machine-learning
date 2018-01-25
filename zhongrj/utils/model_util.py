from zhongrj.utils.math_util import *
from zhongrj.utils.tf_util import *

"""
    模型Utils
"""

W_initializer = tf.truncated_normal_initializer(stddev=0.2)


# W_initializer = None


def Flatten(layer):
    return tf.reshape(layer, [-1, layer.get_shape()[1:].num_elements()])


def Dense(input_tensor,
          units,
          name=None,
          kernel_initializer=W_initializer):
    return tf.layers.dense(input_tensor, units, name=name, kernel_initializer=kernel_initializer)


def MaxPooling2D(input_tensor,
                 pool_size=2,
                 strides=2,
                 padding='valid'):
    return tf.layers.max_pooling2d(input_tensor,
                                   pool_size=pool_size,
                                   strides=strides,
                                   padding=padding)


def Conv2d(input_tensor,
           filters,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='SAME',
           name=None,
           kernel_initializer=W_initializer):
    return tf.layers.conv2d(input_tensor,
                            filters,
                            kernel_size,
                            strides,
                            padding,
                            name=name,
                            kernel_initializer=kernel_initializer)


def DeConv2d(input_tensor,
             filters,
             kernel_size=(2, 2),
             strides=(2, 2),
             padding='SAME',
             name=None,
             kernel_initializer=W_initializer):
    return tf.layers.conv2d_transpose(input_tensor,
                                      filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      name=name,
                                      kernel_initializer=kernel_initializer)


def CNN(cnn_input,
        classes,
        name='CNN',
        cnn_units=list(),
        dnn_units=list(),
        act=tf.nn.relu,
        batch_noraml=True,
        is_train=True,
        reuse=None):
    batch_noraml = tf.layers.batch_normalization if batch_noraml else lambda o, **kwargs: o

    with tf.variable_scope(name, reuse=reuse):
        output = cnn_input
        for i, units in enumerate(cnn_units):
            output = Conv2d(output, units, name='conv{}'.format(i))
            output = MaxPooling2D(batch_noraml(output, training=is_train, name='conv{}_bn'.format(i)))
            output = act(output)

        output = Flatten(output)

        for i, units in enumerate(dnn_units):
            output = Dense(output, units, name='fc{}'.format(i))
            output = act(batch_noraml(output, training=is_train, name='fc{}_bn'.format(i)))

        output = Dense(output, classes, name='fc{}'.format(len(dnn_units)))
    return output


def DeCNN(decnn_input,
          output_dims,
          name='DeCNN',
          cnn_units=list(),
          dnn_units=list(),
          act=tf.nn.relu,
          batch_noraml=True,
          is_train=True,
          reuse=None):
    batch_noraml = tf.layers.batch_normalization if batch_noraml else lambda o, **kwargs: o

    w, h, c = output_dims
    for i in range(len(cnn_units)):
        w, h, c = (w + 1) // 2, (h + 1) // 2, cnn_units[i]

    with tf.variable_scope(name, reuse=reuse):
        output = decnn_input
        for i, units in enumerate(reversed([w * h * c] + dnn_units)):
            output = Dense(output, units, name='fc{}'.format(i))
            if len(cnn_units) != 0 or i != len(dnn_units):  # 有cnn层 或 非最后一层dnn
                output = act(batch_noraml(output, training=is_train, name='fc{}_bn'.format(i)))

        output = tf.reshape(output, [-1, w, h, c])

        for i, units in enumerate(reversed(cnn_units[:-1])):
            output = DeConv2d(output, units, name='deconv{}'.format(i))
            output = act(batch_noraml(output, training=is_train, name='deconv{}_bn'.format(i)))

        output = DeConv2d(output, output_dims[2], name='deconv{}'.format(len(cnn_units) - 1))
    return tf.slice(output, [0, 0, 0, 0], [-1] + output_dims)


def IMG_g_IMG(image,
              name='img_g_img',
              cnn_units=list(),
              act=tf.nn.relu,
              batch_noraml=True,
              is_train=True,
              reuse=None):
    batch_noraml = tf.layers.batch_normalization if batch_noraml else lambda o, **kwargs: o

    # with tf.variable_scope(name, reuse=reuse):
    #     output = image
    #     for i, units in enumerate(cnn_units):
    #         output = Conv2d(output, units, name='conv{}'.format(i))
    #         output = act(batch_noraml(output, training=is_train, name='conv{}_bn'.format(i)))
    #     # output_shape, output = tf.shape(output), Flatten(output)
    #     # output = Dense(output, output.shape[1], name='fc')
    #     # output = batch_noraml(output, training=is_train, name='fc_bn')
    #     # output = act(output)
    #     # output = tf.reshape(output, output_shape)
    #     for i, units in enumerate(reversed(cnn_units[:-1])):
    #         output = DeConv2d(output, units, name='deconv{}'.format(i))
    #         output = act(batch_noraml(output, training=is_train, name='deconv{}_bn'.format(i)))
    #     output = DeConv2d(output, image.shape[3], name='deconv{}'.format(len(cnn_units) - 1))

    with tf.variable_scope(name, reuse=reuse):
        output = image
        for i, units in enumerate(cnn_units):
            output = Conv2d(output, units, strides=(1, 1), name='conv{}'.format(i))
            output = act(batch_noraml(output, training=is_train, name='conv{}_bn'.format(i)))
        output = Conv2d(output, image.shape[3], strides=(1, 1), name='conv{}'.format(len(cnn_units)))

    output = tf.nn.tanh(output)
    return output


def STN(cnn_input,
        init_value,
        name='STN',
        cnn_units=list(),
        dnn_units=list(),
        limit_rotate=False,
        act=tf.nn.relu,
        batch_noraml=True,
        is_train=True,
        reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        output = CNN(cnn_input,
                     6,
                     name='CNN',
                     cnn_units=cnn_units,
                     dnn_units=dnn_units,
                     act=act,
                     batch_noraml=batch_noraml,
                     is_train=is_train)

        # 转成默认值
        output = output * tf.Variable(tf.zeros([6])) + tf.reshape(tf.Variable(init_value), [-1])
        if limit_rotate:  # 限制旋转 todo 有bug
            output = output * tf.reshape(tf.constant([[1., 0, 1.],
                                                      [0, 1., 1.]]), [-1])
        return output


"""
    ########################################## 以下废弃 ##########################################
"""


def CNN_deprecated(cnn_input, is_train, classes,
                   name='CNN', cnn_layers=2, dnn_layers=2,
                   batch_noraml=True, act=tf.nn.relu, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        input_tensor, filter_size = cnn_input, 5
        for i in range(1, cnn_layers + 1):
            output_channel = int(16 * 2 ** (i - 1))
            input_tensor = Conv2d(input_tensor, output_channel, (filter_size, filter_size), name='conv%s' % i)
            if batch_noraml:
                input_tensor = tf.layers.batch_normalization(input_tensor, training=is_train, name='conv%s_bn' % i)
            input_tensor = act(input_tensor)

        input_tensor = Flatten(input_tensor)

        for i in range(1, dnn_layers + 1):
            output_size = classes if i == dnn_layers else int(256 * 2 ** (dnn_layers - i - 1))

            input_tensor = Dense(input_tensor, output_size, name='dense%s' % i)
            if i != dnn_layers:
                if batch_noraml:
                    input_tensor = tf.layers.batch_normalization(input_tensor, training=is_train,
                                                                 name='dense%s_bn' % i)
                input_tensor = act(input_tensor)

        return input_tensor


def DeCNN_deprecated(decnn_input, is_train, output_dims,
                     name='DeCNN', cnn_layers=2, dnn_layers=2,
                     batch_noraml=True, act=tf.nn.relu, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        if cnn_layers == 0:
            fc_output_image_dims = output_dims
        else:
            times = 2 ** cnn_layers
            fc_output_image_dims = [(output_dims[0] + times - 1) // times,
                                    (output_dims[1] + times - 1) // times,
                                    16 * 2 ** (cnn_layers - 1)]

        input_tensor = decnn_input

        for i in reversed(range(1, dnn_layers + 1)):
            output_size = reduce_product(fc_output_image_dims) if i == 1 else int(256 * 2 ** (dnn_layers - i))

            input_tensor = Dense(input_tensor, output_size, name='dense%s' % i)
            if cnn_layers != 0 or i != 1:
                if batch_noraml:
                    input_tensor = tf.layers.batch_normalization(input_tensor, training=is_train,
                                                                 name='dense%s_bn' % i)
                input_tensor = act(input_tensor)

        input_tensor, filter_size = tf.reshape(input_tensor, [-1] + fc_output_image_dims), 5

        for i in reversed(range(1, cnn_layers + 1)):
            output_channel = output_dims[2] if i == 1 else int(16 * 2 ** (i - 2))
            input_tensor = tf.layers.conv2d_transpose(input_tensor, output_channel, [filter_size, filter_size],
                                                      strides=(2, 2), padding='SAME', name='deconv%s' % i)
            if i != 1:
                if batch_noraml:
                    input_tensor = tf.layers.batch_normalization(input_tensor, training=is_train,
                                                                 name='deconv%s_bn' % i)
                input_tensor = act(input_tensor)

        return tf.slice(input_tensor, [0, 0, 0, 0], [-1] + output_dims)
