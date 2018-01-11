from zhongrj.utils.model_util import *

a = tf.Variable([1, 2, 3])
b = tf.constant([[1, 2, 3], [2, 3, 4]])

images = tf.constant([[[1, 2, 3],
                       [2, 3, 4]],
                      [[3, 4, 5],
                       [4, 5, 6]]])
images_pad = tf.pad(images, [[0, 0], [2, 2], [2, 2]], 'SYMMETRIC')

c = tf.constant([[[1, 2, 3],
                  [4, 5, 6]],
                 [[7, 8, 9],
                  [10, 11, 12]]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b * a))
    print(sess.run(images_pad))
    print(sess.run(tf.transpose(c, [1, 2, 0])))
