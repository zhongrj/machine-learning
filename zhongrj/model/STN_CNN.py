import tensorflow as tf
from zhongrj.reference.spatial_transformer_network import spatial_transformer_network as stn
from zhongrj.utils.layer_util import *
from zhongrj.utils.tf_util import *
from zhongrj.utils.view_util import *

MODE = 'train'

FILE_DIR = get_file_dir(__file__)
CHECKPOINT_DIR = FILE_DIR + 'stn_cnn_%s_checkpoint/sess'


class STN_CNN:
    def __init__(
            self,
            name,
            x_dims,
            y_classes,
            learning_rate,
            trans_dims=None,
            batch=50,
            limit_rotate=False,
            cnn_layer=3):
        self.name = name
        self.output_dir = self.name + '/'
        self.checkpoint_dir = CHECKPOINT_DIR % self.name

        self.x_width, self.x_height, self.x_channel = x_dims
        self.y_classes = y_classes
        self.learning_rate = learning_rate
        self.batch = batch
        if trans_dims is None:
            trans_dims = x_dims
        self.trans_width, self.trans_height, self.trans_channel = trans_dims
        self.limit_rotate = limit_rotate
        self.cnn_layers = cnn_layer

        self.__build()
        self.__init_sess()

    def __build(self):
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.x_width * self.x_height * self.x_channel])
            self.y_actual = tf.placeholder(tf.float32, [None, self.y_classes])
            self.is_trans = tf.placeholder(tf.bool, name='is_trans')
        self.x_image = tf.reshape(self.x, [-1, self.x_height, self.x_width, self.x_channel])
        self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        self.__build_stn()
        self.__is_trans()
        self.__build_cnn()
        self.__def_optimizer()

    def __build_stn(self):
        print('Building STN ...')
        with tf.variable_scope('stn'):
            stn_collections = ['stn_collections_params', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = Conv2D(self.x_image, self.x_channel, 5, 32, name='conv1', collections=stn_collections)
            bn1 = BatchNormalization(conv1, name='bn1')
            pool1 = MaxPooling2D(bn1, use_relu=True, name='pool1')
            conv2 = Conv2D(pool1, 32, 5, 64, name='conv2', collections=stn_collections)
            bn2 = BatchNormalization(conv2, name='bn2')
            pool2 = MaxPooling2D(bn2, use_relu=True, name='pool2')

            pool2_flat, pool2_size = Flatten(pool2)

            fc1 = Dense(pool2_flat, pool2_size, 2048, use_relu=False, name='fc1',
                        collections=stn_collections)
            bn3 = BatchNormalization(fc1, use_relu=True, name='bn3')
            fc2 = Dense(bn3, 2048, 512, use_relu=True, name='fc2', collections=stn_collections)
            bn4 = BatchNormalization(fc2, use_relu=True, name='bn4')
            self.theta = Dense(bn4, 512, 6, use_relu=False, trans=True, name='fc3', collections=stn_collections,
                               init_value=np.array([[self.trans_width / self.x_width, 0, 0],
                                                    [0, self.trans_height / self.x_height, 0]]))
            if self.limit_rotate:  # 限制旋转
                self.theta = self.theta * tf.reshape(tf.constant(np.array([[1., 0, 1.], [0, 1., 1.]]), tf.float32),
                                                     [-1])

            self.x_image_trans = tf.reshape(stn(self.x_image, self.theta, [self.trans_height, self.trans_width]),
                                            [-1, self.trans_height, self.trans_width, self.trans_channel])

    def __is_trans(self):
        self.cnn_inputs = self.x_image_trans
        # self.cnn_inputs = tf.cond(self.is_trans, lambda: self.x_image_trans, lambda: self.x_image)

    def __build_cnn(self):
        print('Building CNN ...')
        with tf.variable_scope('cnn'):
            cnn_collections = ['cnn_collections_params', tf.GraphKeys.GLOBAL_VARIABLES]

            input_tensor, input_channel, output_channel = self.cnn_inputs, self.x_channel, 32
            for i in range(1, self.cnn_layers + 1):
                filter_size = 5 if i <= 2 else 3
                conv = Conv2D(input_tensor, input_channel, filter_size, output_channel,
                              name='conv%s' % i, collections=cnn_collections)
                bn = BatchNormalization(conv, name='bn%s' % i)
                input_tensor = MaxPooling2D(bn, use_relu=True, name='pool%s' % i)
                input_channel = output_channel
                output_channel = output_channel * 2

            cnn_flat, cnn_size = Flatten(input_tensor)

            fc1 = Dense(cnn_flat, cnn_size, 2048, use_relu=False, name='fc1', collections=cnn_collections)
            bn4 = BatchNormalization(fc1, use_relu=True, name='bn4')
            fc2 = Dense(bn4, 2048, 512, use_relu=False, name='fc2', collections=cnn_collections)
            bn5 = BatchNormalization(fc2, use_relu=True, name='bn5')
            self.y_predict = Dense(bn5, 512, self.y_classes, name='fc3', use_relu=False, collections=cnn_collections)

    def __def_optimizer(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predict, labels=self.y_actual)
        self.loss = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, self.global_step)
        # self.optimizer = tf.cond(
        #     self.is_trans,
        #     lambda: tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, self.global_step,
        #                                                                               var_list=tf.get_collection(
        #                                                                                   'stn_collections_params')),
        #     lambda: tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, self.global_step,
        #                                                                               var_list=tf.get_collection(
        #                                                                                   'cnn_collections_params')),
        # )

        correct_pred = tf.equal(tf.argmax(self.y_predict, 1), tf.argmax(self.y_actual, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def __init_sess(self):
        self.saver = tf.train.Saver()
        self.sess = init_sess(self.saver, self.checkpoint_dir)

    def __save_sess(self):
        self.saver.save(self.sess, self.checkpoint_dir)
        print('Saved Success .')

    # 这个方法搞得我差点死掉...
    def __draw_detected(self, images, thetas):
        color = 1 if self.x_channel == 1 else (255, 0, 0)

        def center_affine(theta, x, y):
            x_ = x - self.x_width / 2
            y_ = y - self.x_height / 2
            return (np.sum(theta.reshape([2, 3]) * np.array([[x_, y_, self.x_width / 2],
                                                             [x_, y_, self.x_height / 2]]),
                           axis=1).squeeze() + np.array([self.x_width / 2, self.x_height / 2])).astype(np.int)

        return [draw_rectangle(
            images[i],
            center_affine(thetas[i], 0, 0) + np.array([-1, -1]),
            center_affine(thetas[i], 0, self.x_height) + np.array([-1, 1]),
            center_affine(thetas[i], self.x_width, self.x_height) + np.array([1, 1]),
            center_affine(thetas[i], self.x_width, 0) + np.array([1, -1]),
            color=color
        ).reshape([self.x_height, self.x_width, self.x_channel]) for i in range(len(images))]

    def train(self, images, labels):
        print('Training ...')
        images = images.reshape([-1, self.x_width * self.x_height * self.x_channel])
        labels = labels.reshape([-1, self.y_classes])
        accumulated_accuracy = 1 / self.y_classes
        while True:
            batch_mask = np.random.choice(len(images), self.batch)
            batch = images[batch_mask], labels[batch_mask]
            feed_dict = {
                self.x: batch[0],
                self.y_actual: batch[1],
                self.is_trans: True
            }
            _, i_global = self.sess.run([self.optimizer, self.global_step], feed_dict)
            # i_global = self.sess.run(self.global_step, feed_dict)

            if i_global % 10 == 0:
                accuracy_, loss_, predict_, image_, theta_, image_trans_ = self.sess.run(
                    [self.accuracy, self.loss, self.y_predict, self.x_image, self.theta, self.x_image_trans],
                    feed_dict=feed_dict)
                print('step ', i_global)
                print('accuracy ', accuracy_)
                print('loss ', loss_)
                print('predict ', predict_[:10].argmax(axis=1))
                accumulated_accuracy = accumulated_accuracy * 0.8 + accuracy_ * 0.2
                print('total_accuracy ', accumulated_accuracy)
                save_image(
                    [im for im in image_[:10]] +
                    self.__draw_detected(image_[:10], theta_[:10]) +
                    [im for im in image_trans_[:10]],
                    self.output_dir + 'step_%s' % i_global, n_each_row=10, text=predict_[:10].argmax(axis=1))
            if i_global % 50 == 0:
                self.__save_sess()

    def test(self, images, labels):
        print('Test ...')
        test_mask = np.random.choice(len(images), 100)
        accuracy, image_, theta_, image_trans_, predict_ = self.sess.run(
            [self.accuracy, self.x_image, self.theta, self.x_image_trans, self.y_predict],
            feed_dict={
                self.x: images[test_mask],
                self.y_actual: labels[test_mask],
                self.is_trans: True
            })
        print('accuracy ', accuracy)
        save_image(
            [im for im in image_[:10]] +
            self.__draw_detected(image_[:10], theta_[:10]) +
            [im for im in image_trans_[:10]],
            self.output_dir + 'test', n_each_row=10, text=predict_[:10].argmax(axis=1))


def mnist_distortions():
    """扭曲数字识别"""
    from zhongrj.data.mnist_distortions import load_data

    model = STN_CNN(
        name='mnist_distortions',
        x_dims=[40, 40, 1],
        trans_dims=[25, 30, 1],
        y_classes=10,
        learning_rate=1e-4
    )

    print('Loading Data ...')
    data = load_data()

    if MODE == 'train':
        model.train(data['train_x'].reshape([-1, 40 * 40]), data['train_y'])
    elif MODE == 'test':
        model.test(data['test_x'].reshape([-1, 40 * 40]), data['test_y'])


def catvsdog():
    """猫狗大战"""
    from zhongrj.data.catvsdog import load_data

    model = STN_CNN(
        name='catvsdog',
        x_dims=[150, 150, 3],
        trans_dims=[60, 60, 3],
        y_classes=2,
        learning_rate=1e-4,
        batch=40,
        limit_rotate=True,
        cnn_layer=3
    )

    print('Loading Data ...')
    data = load_data()

    if MODE == 'train':
        model.train(data['train_x'].reshape([-1, 150 * 150]), data['train_y'])
    elif MODE == 'test':
        model.test(data['test_x'].reshape([-1, 150 * 150]), data['test_y'])


if __name__ == '__main__':
    catvsdog()
