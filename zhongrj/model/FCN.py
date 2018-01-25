from zhongrj.model.BaseModel import *


class FCN(BaseModel):
    def __init__(self,
                 name,
                 x_dims,
                 y_classes,
                 cnn_units,
                 learning_rate,
                 batch):
        BaseModel.__init__(self, name, batch)

        self.x_width, self.x_height, self.x_channel = x_dims
        self.y_classes = y_classes
        self.cnn_units = cnn_units
        self.learning_rate = learning_rate

        self.__build()
        self._init_sess()

    def __build(self):
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.x_height, self.x_width, self.x_channel])
            self.y_actual = tf.placeholder(tf.int32, [None, self.x_height, self.x_width])
            self.is_train = tf.placeholder(tf.bool)

        output = self.x
        with tf.variable_scope('fcn'):
            for i, units in enumerate(self.cnn_units):
                output = Conv2d(output, units, name='conv{}'.format(i))
                output = tf.layers.batch_normalization(output, training=self.is_train, name='conv{}_bn'.format(i))
                output = tf.nn.relu(output)
            self.y_predict = Conv2d(output, self.y_classes, name='conv{}'.format(len(self.cnn_units)))

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_predict, labels=self.y_actual))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)

    def __generate_image(self, name, feed_dict):
        y_predict_ = self.sess.run(self.y_predict, feed_dict)
        y_predict_ = y_predict_.argmax(axis=3)
        save_image(
            np.concatenate(
                [feed_dict[self.x][:18].squeeze(),
                 y_predict_[:18]]
            ), self.output_dir + name, 6)

    def train(self, images, labels):
        images = images.reshape([-1, self.x_width, self.x_height, self.x_channel])
        labels = labels.reshape([-1, self.x_width, self.x_height])
        sample_feed_dict = {
            self.x: images[self.sess.run(self.sample)],
            self.is_train: False
        }
        while True:
            mask = np.random.choice(len(images), self.batch)
            feed_dict = {
                self.x: images[mask],
                self.y_actual: labels[mask],
                self.is_train: True
            }
            _, i_global, loss_ = self.sess.run([self.optimizer, self.global_step, self.loss], feed_dict)
            if i_global % 5 == 0:
                print('step {}:'.format(i_global))
                print(loss_)
            if i_global % 10 == 0:
                self.__generate_image('sample_{}'.format(i_global), sample_feed_dict)

    def test(self):
        pass


MODE = 'train'


def mnist_segmentation():
    from zhongrj.data.mnist import load_data

    model = FCN(
        name='FCN_mnist',
        x_dims=[28, 28, 1],
        y_classes=2,
        cnn_units=[20, 20],
        learning_rate=1e-3,
        batch=100
    )

    print('Loading Data ...')
    data = load_data()

    train_y = data['train_x'].copy()
    train_y[train_y > 0] = 1

    if MODE == 'train':
        model.train(data['train_x'], train_y)
    elif MODE == 'test':
        pass


if __name__ == '__main__':
    mnist_segmentation()
