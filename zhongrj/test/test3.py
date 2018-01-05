import tensorflow as tf

a = tf.constant([0.1, 0.5, 0.9, 0.34])
b = tf.cast(tf.less_equal(a, 0.5), tf.float32)
w = tf.Variable(tf.zeros([4]), collections=['test', tf.GraphKeys.GLOBAL_VARIABLES])
c = tf.add(a, w)
# with tf.Session() as sess:
#     print(sess.run(b))
#     print(sess.run(tf.reduce_mean(b)))


f = tf.contrib.layers.batch_norm(c, is_training=False, scope='test')
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

# from zhongrj.data.mnist import load_data
# from zhongrj.utils.view_util import *
#
# data = load_data()
# data = data['train_x'].reshape([-1, 28, 28])
# for i in range(10):
#     show_image(data[np.random.choice(len(data), 48)], n_each_row=8)
