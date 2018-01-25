from zhongrj.model.BaseModel import *

"""
    Encoder可用于Semi-Supervise Learning
"""


class AutoEncoder(BaseModel):
    def __init__(self,
                 name,
                 x_dims,
                 y_dims,
                 cnn_units,
                 dnn_units,
                 learning_rate,
                 batch):
        BaseModel.__init__(self, name, batch)
        self.x_width, self.x_height, self.x_channel = x_dims
        self.y_dims = y_dims
        self.cnn_units = cnn_units
        self.dnn_units = dnn_units
        self.learning_rate = learning_rate

        self.__build()
        self._init_sess()

    def __build(self):
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.x_width, self.x_height, self.x_channel])
            self.is_train = tf.placeholder(tf.bool)
        self.x_img = image_preprocess(self.x)

        with tf.variable_scope('encoder'):
            self.code = tf.nn.tanh(CNN(self.x_img,
                                       self.y_dims,
                                       cnn_units=self.cnn_units,
                                       dnn_units=self.dnn_units,
                                       batch_noraml=True,
                                       is_train=self.is_train))

        with tf.variable_scope('decoder'):
            self.decode = tf.nn.tanh(DeCNN(self.code,
                                           [self.x_width, self.x_height, self.x_channel],
                                           cnn_units=self.cnn_units,
                                           dnn_units=self.dnn_units,
                                           batch_noraml=True,
                                           is_train=self.is_train))

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.x_img, self.decode))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)

        [print(param) for param in get_trainable_collection()]

    def __generate_image(self, name, feed_dict):
        decode_ = self.sess.run(image_deprocess(self.decode), feed_dict)
        save_image(
            np.concatenate(
                [feed_dict[self.x][:24],
                 decode_[:24]]
            ), self.output_dir + name, n_each_row=8
        )

    def train(self, x):
        x = x.reshape([-1, self.x_width, self.x_height, self.x_channel])
        sample_feed_dict = {
            self.x: x[self.sess.run(self.sample)],
            self.is_train: False
        }
        while True:
            feed_dict = {
                self.x: x[np.random.choice(len(x), self.batch)],
                self.is_train: True
            }

            _, i_global, loss = self.sess.run([self.optimizer, self.global_step, self.loss], feed_dict)
            if i_global % 10 == 0:
                print(loss)
            sample_interval = 100
            if i_global % sample_interval == 0:
                self.__generate_image('sample_{}'.format(i_global // sample_interval), sample_feed_dict)
            if i_global % 100 == 0:
                self.save_sess()

    def test(self):
        n = 20
        basic = np.linspace(-1, 1, n)
        x = np.tile(basic[:, np.newaxis], (1, len(basic))).reshape([-1])
        y = np.tile(basic, (len(basic),))
        save_image_join(self.sess.run(self.decode, {
            self.x: np.array([]).reshape([-1, self.x_width, self.x_height, self.x_channel]),
            self.code: np.dstack((x, y)).squeeze(),
            self.is_train: False
        }), name=self.output_dir + 'test', n_each_row=len(basic))


MODE = 'train'


def mnist_autoencoder_model():
    return AutoEncoder(
        name='AutoEncoder_mnist',
        x_dims=[28, 28, 1],
        y_dims=2,
        cnn_units=[8, 8],
        dnn_units=[400, 100],
        learning_rate=1e-3,
        batch=100
    )


def encode_decode_mnist():
    from zhongrj.data.mnist import load_data

    model = mnist_autoencoder_model()

    if MODE == 'train':
        model.train(load_data()['train_x'])
    elif MODE == 'test':
        model.test()


def semi_supervised_mnist():
    from zhongrj.data.mnist import load_data

    data = load_data()
    n = 1000
    train_x, train_y, test_x, test_y = data['train_x'][:n], data['train_y'][:n], data['test_x'], data['test_y']
    train_x, test_x = train_x.reshape([-1, 28, 28, 1]), test_x.reshape([-1, 28, 28, 1])

    encoder_model = mnist_autoencoder_model()
    code = encoder_model.sess.run(encoder_model.code, {
        encoder_model.x: np.vstack((train_x, test_x)),
        encoder_model.is_train: False
    })
    train_code, test_code = code[:n], code[n:]

    # ==================================== simple classifier ====================================
    x = tf.placeholder(tf.float32, [None, encoder_model.y_dims])
    y_actual = tf.placeholder(tf.float32, [None, 10])
    is_train = tf.placeholder(tf.bool)
    y_predict = CNN(x, 10, dnn_units=[20, 20], batch_noraml=True, is_train=is_train)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_actual))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1)), tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            mask = np.random.choice(n, 100)
            _, accuracy_, loss_ = sess.run([optimizer, accuracy, loss], {
                x: train_code[mask],
                y_actual: train_y[mask],
                is_train: True
            })
            print(loss_)
            # print(accuracy_)
        print('Total Accuracy: ', sess.run(accuracy, {x: test_code, y_actual: test_y, is_train: True}))


def anime():
    from zhongrj.data.anime_face import load_data

    model = AutoEncoder(
        name='AutoEncoder_anime',
        x_dims=[48, 48, 3],
        y_dims=10,
        cnn_units=[40, 40, 40],
        dnn_units=[1024, 256],
        learning_rate=4e-4,
        batch=50
    )

    if MODE == 'train':
        model.train(load_data()['train_x'])
    elif MODE == 'test':
        model.test()


if __name__ == '__main__':
    anime()
