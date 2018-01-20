import tensorflow as tf


def init_sess(saver, checkpoint_dir):
    sess = tf.InteractiveSession()
    try:
        saver.restore(sess, checkpoint_dir)
        print('Loading session from ', checkpoint_dir)
    except:
        print('Loading session Exception, initializing all parameters...')
        sess.run(tf.global_variables_initializer())
    return sess


def softmax_cross_entropy_mean(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def sigmoid_cross_entropy_mean(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))


# 增强平方差loss的鲁棒性
# 不需要控制gradient和cross_entropy一致?
def huber_loss(logits, labels, max_gradient=1.0):
    err = tf.abs(labels - logits)
    mg = tf.constant(max_gradient)
    lin = mg * (err - 0.5 * mg)
    quad = 0.5 * err * err
    return tf.reduce_mean(tf.where(err < mg, quad, lin))


def get_trainable_collection(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def image_preprocess(image):
    with tf.name_scope('image_preprocess'):
        if image.shape[3] == 1:
            image = (image / 1 * 2) - 1
        else:
            image = (image / 255 * 2) - 1
    return image


def image_deprocess(image):
    with tf.name_scope('image_deprocess'):
        if image.shape[3] == 1:
            image = (image + 1) / 2 * 1
        else:
            image = (image + 1) / 2 * 255
    return image
