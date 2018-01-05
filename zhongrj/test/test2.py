from zhongrj.utils.model_util import *

a = tf.Variable([1, 2, 3])
b = tf.constant([[1, 2, 3], [2, 3, 4]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b * a))
