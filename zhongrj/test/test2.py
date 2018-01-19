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

d3x3 = np.arange(9).reshape((3, 3))
print(d3x3 == 0)
print(d3x3[d3x3 == 0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b * a))
    print(sess.run(images_pad))
    print(sess.run(tf.transpose(c, [1, 2, 0])))
    print(sess.run(tf.reduce_max(b, 1)))
    # print(sess.run(b[sess.run(tf.one_hot([1, 2, 2], 3, True, False))]))
    print(sess.run(tf.range(2)[:, tf.newaxis]))
    indics = tf.concat(
        [tf.range(2)[:, tf.newaxis],
         tf.constant([0, 0])[:, tf.newaxis]],
        1)
    print(sess.run(indics))
    print(sess.run(tf.gather_nd(b, indics)))
    # print(sess.run(tf.assign(a, tf.range(3))))
    # print(sess.run(a))
    print(sess.run(tf.cast(tf.constant([True, False]), tf.float32)))

    print(sess.run(tf.nn.softmax([2., 3., 5.])))
