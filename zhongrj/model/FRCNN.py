from zhongrj.model.BaseModel import *


class FRCNN(BaseModel):
    def __init__(self,
                 name,
                 x_dims,
                 y_classes,
                 learning_rate,
                 batch):
        BaseModel.__init__(self, name, batch)
        self.x_width, self.x_height, self.x_channel = x_dims
        self.y_classes = y_classes

        self.learning_rate = learning_rate

        self.__build()
        self._init_sess()

    def __build(self):
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.x_height, self.x_width, self.x_channel])
            self.x_location = tf.placeholder(tf.float32, [None, 4])
            self.y_actual = tf.placeholder(tf.float32, [None, ])
            self.anchors = tf.constant(np)
            self.is_train = tf.placeholder(tf.bool)

        with tf.variable_scope('feature_map'):
            output = Conv2d(self.x, 16)
            output = tf.layers.batch_normalization(output, training=self.is_train)
            output = tf.nn.relu(MaxPooling2D(output))

            output = Conv2d(output, 32)
            output = tf.layers.batch_normalization(output, training=self.is_train)
            output = tf.nn.relu(MaxPooling2D(output))

            # batch * 60 * 60 * 32
            self.features = output

        with tf.variable_scope('rpn'):
            # batch * 60 * 60 * 16
            # anchor 5 * 5
            output = Conv2d(self.features, 64, (5, 5), (5, 5))
            output = tf.layers.batch_normalization(output, training=self.is_train)

            output = Conv2d(output, 16, (1, 1))
            output = tf.layers.batch_normalization(output, training=self.is_train)
            output = tf.nn.relu(output)

            # batch * 60 * 60 * (1+4)
            self.predict1 = Conv2d(output, 1, (1, 1))
            self.bounding_box = Conv2d(output, 4, (1, 1))

        with tf.name_scope('roi'):
            pass

        with tf.variable_scope('frcnn'):
            # bounding_predict = tf.round(self.bounding_predict)
            pass

        with tf.name_scope('loss'):
            self.predict1_loss = sigmoid_cross_entropy_mean(
                tf.equal(self.y_actual, self.y_classes),
                self.predict1
            )
            self.box_loss = {}
            self.total_loss = {}
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, self.global_step)

    def __generate_image(self):
        pass

    def train(self, images, labels):
        pass

    def test(self, images, labels):
        pass


MODE = 'train'


def detect_multi_mnist():
    from zhongrj.data.mnist_multi import load_data

    model = FRCNN(
        name='SSD_detect_multi_mnist',
        x_dims=[],
        y_classes=10,
        learning_rate=1e-2,
        batch=50
    )

    print('Loading data ...')
    data = load_data()

    if MODE == 'train':
        model.train(data['train_x'], data['train_y'])
    elif MODE == 'test':
        model.test(data['train_x'], data['train_y'])


if __name__ == '__main__':
    detect_multi_mnist()
