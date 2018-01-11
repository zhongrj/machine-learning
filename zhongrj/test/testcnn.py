from zhongrj.data.mnist import load_data
from zhongrj.utils.model_util import *

x = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
y_actual = tf.placeholder(tf.float32, [None, 10])
is_train = tf.placeholder(tf.bool, name='is_train')

x_image = tf.reshape(x, [-1, 28, 28, 1])
y_predict = CNN(x_image,
                10,
                'CNN',
                [16, 32],
                [1024, 256],
                batch_noraml=True,
                is_train=is_train)

# y_predict = CNN_deprecated(x_image, is_train, 10, 'CNN', 3, 2, tf.nn.relu)
[print(param) for param in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'CNN')]

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_actual)
loss = tf.reduce_mean(cross_entropy)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

data = load_data()
train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        mask = np.random.choice(len(train_x), 100)
        feed_dict = {
            x: train_x[mask],
            y_actual: train_y[mask],
            is_train: True
        }
        _ = sess.run(optimizer, feed_dict)
        if i % 100 == 0:
            accuracy_ = sess.run(accuracy, feed_dict)
            print(accuracy_)

    print('\n\n\nFinal Test: ', sess.run(accuracy, feed_dict={
        x: test_x,
        y_actual: test_y,
        is_train: True
    }))
