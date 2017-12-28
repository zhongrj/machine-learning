import tensorflow as tf
from zhongrj.reference.spatial_transformer_network import spatial_transformer_network as stn
from zhongrj.utils.layer_util import *
from zhongrj.utils.tf_util import *
from zhongrj.utils.view_util import *

MODE = 'train'

FILE_DIR = get_file_dir(__file__)
CHECKPOINT_DIR = FILE_DIR + 'stn_cnn_checkpoint/sess'


class STN_CNN:
    def __init__(self, x_dims, y_classes, learning_rate, trans_dims=None):
        self.x_width, self.x_height, self.x_channel = x_dims
        self.y_classes = y_classes
        self.learning_rate = learning_rate
        if trans_dims is None:
            trans_dims = x_dims
        self.trans_width, self.trans_height, self.trans_channel = trans_dims

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
            conv1 = Conv2D(self.x_image, 1, 5, 32, name='conv1', collections=stn_collections)
            pool1 = MaxPooling2D(conv1, use_relu=True, name='pool1')
            conv2 = Conv2D(pool1, 32, 5, 64, name='conv2', collections=stn_collections)
            pool2 = MaxPooling2D(conv2, use_relu=True, name='pool2')

            pool2_flat, pool2_size = Flatten(pool2)

            fc1 = Dense(pool2_flat, pool2_size, 2048, use_relu=False, name='fc1',
                        collections=stn_collections)
            fc2 = Dense(fc1, 2048, 512, use_relu=True, name='fc2', collections=stn_collections)
            self.theta = Dense(fc2, 512, 6, use_relu=False, trans=True, name='fc3', collections=stn_collections,
                               init_value=np.array([[.7, 0, 0],
                                                    [0, .7, 0]]))

            self.x_image_trans = tf.reshape(stn(self.x_image, self.theta, [self.trans_height, self.trans_width]),
                                            [-1, self.trans_height, self.trans_width, self.trans_channel])

    def __is_trans(self):
        self.cnn_inputs = self.x_image_trans
        # self.cnn_inputs = tf.cond(self.is_trans, lambda: self.x_image_trans, lambda: self.x_image)

    def __build_cnn(self):
        print('Building CNN ...')
        with tf.variable_scope('cnn'):
            cnn_collections = ['cnn_collections_params', tf.GraphKeys.GLOBAL_VARIABLES]

            conv1 = Conv2D(self.cnn_inputs, 1, 5, 32, name='conv1', collections=cnn_collections)
            bn1 = BatchNormalization(conv1, name='bn1')
            pool1 = MaxPooling2D(bn1, use_relu=True, name='pool1')

            conv2 = Conv2D(pool1, 32, 5, 64, name='conv2', collections=cnn_collections)
            bn2 = BatchNormalization(conv2, name='bn2')
            pool2 = MaxPooling2D(bn2, use_relu=True, name='pool2')

            # conv3 = Conv2D(pool2, 64, 3, 128, name='conv3', collections=cnn_collections)
            # bn3 = BatchNormalization(conv3, name='bn3')
            # pool3 = MaxPooling2D(bn3, use_relu=True, name='pool3')

            pool3_flat, pool3_size = Flatten(pool2)

            fc1 = Dense(pool3_flat, pool3_size, 2048, use_relu=False, name='fc1', collections=cnn_collections)
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
        self.sess = init_sess(self.saver, CHECKPOINT_DIR)

    def __save_sess(self):
        self.saver.save(sess=self.sess, save_path=CHECKPOINT_DIR)
        print('Saved Success ...')

    # 这个方法搞得我差点死掉...
    def __draw_detected(self, images, thetas):
        def center_affine(theta, x, y):
            x_ = x - self.trans_width / 2
            y_ = y - self.trans_height / 2
            return np.sum(theta.reshape([2, 3]) * np.array([[x_, y_, self.trans_width / 2],
                                                            [x_, y_, self.trans_height / 2]]),
                          axis=1).squeeze() + np.array([self.x_width / 2, self.x_height / 2])

        return [draw_rectangle(
            images[i],
            center_affine(thetas[i], 0, 0) + np.array([-1, -1]),
            center_affine(thetas[i], 0, self.trans_height) + np.array([-1, 1]),
            center_affine(thetas[i], self.trans_width, self.trans_height) + np.array([1, 1]),
            center_affine(thetas[i], self.trans_width, 0) + np.array([1, -1]),
            color=1
        ).reshape([-1, self.x_height, self.x_width, 1]) for i in range(len(images))]

    def train(self, images, labels):
        print('Training ...')
        while True:
            batch_mask = np.random.choice(len(images), 50)
            batch = images[batch_mask], labels[batch_mask]
            feed_dict = {
                self.x: batch[0],
                self.y_actual: batch[1],
                self.is_trans: True
            }
            _, i_global = self.sess.run([self.optimizer, self.global_step], feed_dict)
            # i_global = self.sess.run(self.global_step, feed_dict)

            if i_global % 10 == 0:
                # feed_dict = {
                #     self.x: images[1000:1050],
                #     self.y_actual: labels[1000:1050],
                #     self.is_trans: True
                # }
                accuracy_, loss_, predict_, image_, theta_, image_trans_ = self.sess.run(
                    [self.accuracy, self.loss, self.y_predict, self.x_image, self.theta, self.x_image_trans],
                    feed_dict=feed_dict)
                print('step ', i_global)
                print('accuracy ', accuracy_)
                print('loss ', loss_)
                print('predict ', predict_[:10].argmax(axis=1))

                save_image(
                    [im for im in image_[:10]] +
                    self.__draw_detected(image_[:10], theta_[:10]) +
                    [im for im in image_trans_[:10]],
                    'step_%s' % i_global, n_each_row=10)
                if i_global % 50 == 0:
                    self.__save_sess()

    def test(self, images, labels):
        print('Test ...')
        test_mask = np.random.choice(len(images), 100)
        accuracy, image_, theta_, image_trans_ = self.sess.run(
            [self.accuracy, self.x_image, self.theta, self.x_image_trans],
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
            'test', n_each_row=10)


def sample():
    import zhongrj.mnist.mnist_distortions as mnist_distortions

    model = STN_CNN(
        x_dims=[40, 40, 1],
        trans_dims=[30, 40, 1],
        y_classes=10,
        learning_rate=1e-4
    )

    print('Loading Data ...')
    mnist = mnist_distortions.load_distortions_data()

    if MODE == 'train':
        model.train(mnist['train_x'].reshape([-1, 1600]), mnist['train_y'])
    elif MODE == 'test':
        model.test(mnist['test_x'].reshape([-1, 1600]), mnist['test_y'])


if __name__ == '__main__':
    sample()
