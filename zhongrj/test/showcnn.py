from zhongrj.model.BaseModel import *


class showcnn(BaseModel):
    def __init__(self,
                 name,
                 x_dims,
                 y_classes,
                 cnn_units,
                 dnn_units,
                 learning_rate,
                 batch):
        BaseModel.__init__(self, name, batch)
        self.x_width, self.x_height, self.x_channel = x_dims
        self.y_classes = y_classes
        self.cnn_units = cnn_units
        self.dnn_units = dnn_units
        self.learning_rate = learning_rate

        self.__build()
        self._init_sess()

    def __build(self):
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, None, None, self.x_channel])
            self.y_actual = tf.placeholder(tf.float32, [None, self.y_classes])
            self.is_train = tf.placeholder(tf.bool, name='is_train')

        leakyReLU = lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU')
        with tf.variable_scope('CNN'):
            output = self.x
            for i, units in enumerate(self.cnn_units):
                output = Conv2d(output, units, (5, 5), name='conv{}'.format(i))
                output = MaxPooling2D(
                    tf.layers.batch_normalization(output, training=self.is_train, name='conv{}_bn'.format(i)))
                output = leakyReLU(output)

            output = tf.reshape(output, [-1, 3 * 3 * self.cnn_units[-1]])

            for i, units in enumerate(self.dnn_units):
                output = Dense(output, units, name='fc{}'.format(i))
                output = leakyReLU(
                    tf.layers.batch_normalization(output, training=self.is_train, name='fc{}_bn'.format(i)))

            self.y_predict = Dense(output, self.y_classes, name='fc{}'.format(len(self.dnn_units)))

        with tf.name_scope('loss'):
            self.loss = softmax_cross_entropy_mean(logits=self.y_predict, labels=self.y_actual)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.y_predict, 1), tf.argmax(self.y_actual, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        [print(param) for param in get_trainable_collection('CNN')]

    def train(self, images, labels):
        images = images.reshape([-1, self.x_height, self.x_width, self.x_channel])
        labels = labels.reshape([-1, self.y_classes])
        while True:
            batch_mask = np.random.choice(len(images), self.batch)
            batch = images[batch_mask], labels[batch_mask]
            feed_dict = {
                self.x: batch[0],
                self.y_actual: batch[1],
                self.is_train: True
            }
            _, i_global, accuracy_, loss_ = self.sess.run(
                [self.optimizer, self.global_step, self.accuracy, self.loss], feed_dict)

            if i_global % 10 == 0:
                print('step ', i_global)
                print('accuracy ', accuracy_)
                print('loss ', loss_)

            if i_global % 100 == 0:
                self.save_sess()

    def test(self, images, labels):
        images = images.reshape([-1, self.x_height, self.x_width, self.x_channel])
        print('accuracy: {}'.format(self.sess.run(self.accuracy, {
            self.x: images,
            self.y_actual: labels,
            self.is_train: False
        })))

    def back(self, images):
        # CNN/leakyReLU:0
        # CNN/leakyReLU_1:0
        # CNN/leakyReLU_2:0
        # CNN/leakyReLU_3:0
        # CNN/fc2/BiasAdd:0
        learning_rate, n_step = 10, 1000
        tensor = self.sess.graph.get_tensor_by_name('CNN/leakyReLU:0')
        feed_dict = {self.x: images, self.is_train: False}

        indics = tf.placeholder(tf.int32, [None, 4])
        active_tensor = tf.gather_nd(tensor, indics)
        # strict = tf.maximum(tf.abs(self.x - 0), tf.abs(self.x - 255))
        # strict = tf.reduce_mean(tf.pow(strict, 2))
        # gradient = tf.gradients(active_tensor - strict, self.x)
        gradient = tf.gradients(active_tensor, self.x)
        feed_dict[indics] = np.argwhere(np.less(0, self.sess.run(tensor, feed_dict)))
        act0 = self.sess.run(active_tensor, feed_dict)

        for i in range(n_step):
            gradient_, act1 = self.sess.run([gradient, active_tensor], feed_dict)
            feed_dict[self.x] = feed_dict[self.x] + gradient_[0] * learning_rate
            if i % 100 == 0:
                print('step{}: '.format(i))
                changes = np.sum(act1 - act0)
                print(changes)
                if changes > 10000:
                    break
        show_image(np.concatenate([images, feed_dict[self.x]]), 4)

    def show_layer(self):
        # initial
        max_value = 1 if self.x_channel == 1 else 255

        'CNN/conv0/kernel:0'
        'CNN/conv0/bias:0'
        'CNN/conv1/kernel:0'
        'CNN/conv1/bias:0'
        'CNN/conv2/kernel:0'
        'CNN/conv2/bias:0'
        'CNN/fc0/kernel:0'
        'CNN/fc0/bias:0'
        'CNN/fc1/kernel:0'
        'CNN/fc1/bias:0'
        'CNN/fc2/kernel:0'
        'CNN/fc2/bias:0'
        # CNN/leakyReLU:0
        # CNN/leakyReLU_1:0
        # CNN/Reshape:0
        # CNN/leakyReLU_3:0
        # CNN/leakyReLU_4:0
        # CNN/fc2/BiasAdd:0

        learning_rate, n_step = 10, 200
        restrict_times = 100

        # conv1
        # filters, bias = get_trainable_collection('CNN/conv0')[:2]
        # image_shape = (filters.shape[3].value, 5, 5, self.x_channel)
        # tensor = self.x
        # learning_rate = 1
        # conv2
        # filters, bias = get_trainable_collection('CNN/conv1')[:2]
        # image_shape = (filters.shape[3].value, 10, 10, self.x_channel)
        # tensor = self.sess.graph.get_tensor_by_name('CNN/leakyReLU:0')
        # learning_rate = 10
        # conv3
        # filters, bias = get_trainable_collection('CNN/conv2')[:2]
        # image_shape = (filters.shape[3].value, 20, 20, self.x_channel)
        # tensor = self.sess.graph.get_tensor_by_name('CNN/leakyReLU_1:0')
        # learning_rate = 10

        # conv-common
        # filters_shape = [i.value for i in filters.shape]
        # filters = tf.reshape(tf.transpose(tf.reshape(filters, (-1, filters_shape[3]))),
        #                      (filters_shape[3], filters_shape[0], filters_shape[1], filters_shape[2]))

        # fc1
        # filters, bias = get_trainable_collection('CNN/fc0')[:2]
        # image_shape = (filters.shape[1].value, 28, 28, self.x_channel)
        # tensor = self.sess.graph.get_tensor_by_name('CNN/Reshape:0')
        # restrict_times = 1000
        # fc2
        # filters, bias = get_trainable_collection('CNN/fc1')[:2]
        # image_shape = (filters.shape[1].value, 28, 28, self.x_channel)
        # tensor = self.sess.graph.get_tensor_by_name('CNN/leakyReLU_3:0')
        # classify
        filters, bias = get_trainable_collection('CNN/fc2')[:2]
        image_shape = (filters.shape[1].value, 28, 28, self.x_channel)
        tensor = self.sess.graph.get_tensor_by_name('CNN/leakyReLU_4:0')

        # fc-common
        filters = tf.transpose(filters)

        images = np.tile([0], image_shape)
        feed_dict = {self.x: images, self.is_train: False}

        leakyReLU = lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU')
        # sigmoid = tf.nn.sigmoid
        sigmoid = tf.identity
        act = tf.reduce_mean(sigmoid(leakyReLU(tf.reduce_sum(
            tf.reshape(tensor * filters, [image_shape[0], -1]), 1
        ) + bias)))
        restrict = tf.reduce_mean(tf.pow(tf.minimum(tf.minimum(self.x, max_value - self.x), 0), 2))
        gradient = tf.gradients(act - restrict_times * restrict, self.x)
        for i in range(n_step):
            gradient_, act_, restrict_ = self.sess.run([gradient, act, restrict], feed_dict)
            if i % 10 == 0:
                print('step {}:'.format(i))
                print('act:\t\t{}'.format(act_))
                print('restrict:\t{}'.format(restrict_))
                print('total:\t\t{}'.format(act_ - restrict_))
                save_image(feed_dict[self.x][:30], self.output_dir + str(i), 5)
            feed_dict[self.x] = feed_dict[self.x] + gradient_[0] * learning_rate

        i, n = 1, 30
        save_image(feed_dict[self.x][:n], self.output_dir + 'final_{}'.format(i), 1)
        while True:
            i += 1
            if len(feed_dict[self.x]) > n:
                save_image(feed_dict[self.x][n:n + 30], self.output_dir + 'final_{}'.format(i), 5)
                n += 30
            else:
                break


def classifier():
    model = showcnn(
        name='showcnn_classifier',
        x_dims=[28, 28, 1],
        y_classes=10,
        cnn_units=[10, 15, 20],
        dnn_units=[100, 50],
        learning_rate=1e-2,
        batch=100
    )

    from zhongrj.data.mnist import load_data

    if MODE == 'train':
        data = load_data()
        model.train(data['train_x'], data['train_y'])
    elif MODE == 'test':
        data = load_data()
        model.test(data['test_x'][:100], data['test_y'][:100])
    elif MODE == 'back':
        # mask = np.array([1, 2, 5001, 5002])
        # model.back(train_x[mask])
        pass
    elif MODE == 'show_layer':
        model.show_layer()


MODE = 'show_layer'

if __name__ == '__main__':
    classifier()
