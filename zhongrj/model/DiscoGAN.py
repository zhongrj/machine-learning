from zhongrj.model.BaseModel import *

"""
    总结和问题：
        CycleGAN的做法：
            1. 使用了resnet, 大概是防止gradient vanishing
"""


class DiscoGAN(BaseModel):
    def __init__(self,
                 name,
                 A_dims,
                 B_dims,
                 g_cnn_units,
                 d_cnn_units,
                 d_dnn_units,
                 g_learning_rate,
                 d_learning_rate,
                 batch):
        BaseModel.__init__(self, name, batch, sample_num=20)

        self.A_width, self.A_height, self.A_channel = A_dims
        self.B_width, self.B_height, self.B_channel = B_dims
        self.g_cnn_units = g_cnn_units
        self.d_cnn_units = d_cnn_units
        self.d_dnn_units = d_dnn_units
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate

        self.__build()
        self._init_sess(graph=False)

    def __build(self):
        with tf.variable_scope('inputs'):
            self.A = tf.placeholder(tf.float32, [None, self.A_width, self.A_height, self.A_channel], name='A')
            self.B = tf.placeholder(tf.float32, [None, self.B_width, self.B_height, self.B_channel], name='B')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
        # self.A_img = self.A
        # self.B_img = self.B
        self.A_img = image_preprocess(self.A)
        self.B_img = image_preprocess(self.B)

        def A_g_B(A, reuse=None):
            return IMG_g_IMG(A,
                             name='A_g_B',
                             cnn_units=self.g_cnn_units,
                             batch_noraml=True,
                             is_train=self.is_train,
                             act=lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU'),
                             reuse=reuse)

        def B_g_A(B, reuse=None):
            return IMG_g_IMG(B,
                             name='B_g_A',
                             cnn_units=self.g_cnn_units,
                             batch_noraml=True,
                             is_train=self.is_train,
                             act=lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU'),
                             reuse=reuse)

        def d_A(A, A_fake, reuse=None):
            output = CNN(tf.concat([A, A_fake], 0),
                         1,
                         name='d_A',
                         cnn_units=self.d_cnn_units,
                         dnn_units=self.d_dnn_units,
                         batch_noraml=True,
                         is_train=self.is_train,
                         act=lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU'),
                         reuse=reuse)
            return tf.slice(output, [0, 0], [tf.shape(A)[0], -1], name='d_A_real'), \
                   tf.slice(output, [tf.shape(A)[0], 0], [-1, -1], name='d_A_fake')

        def d_B(B, B_fake, reuse=None):
            output = CNN(tf.concat([B, B_fake], 0),
                         1,
                         name='d_B',
                         cnn_units=self.d_cnn_units,
                         dnn_units=self.d_dnn_units,
                         batch_noraml=True,
                         is_train=self.is_train,
                         act=lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU'),
                         reuse=reuse)
            return tf.slice(output, [0, 0], [tf.shape(B)[0], -1], name='d_B_real'), \
                   tf.slice(output, [tf.shape(B)[0], 0], [-1, -1], name='d_B_fake')

        self.A_g_B = A_g_B(self.A_img)
        self.d_B_real, self.A_g_B_fake = d_B(self.B_img, self.A_g_B)
        self.A_g_B_g_A = B_g_A(self.A_g_B)
        _, self.A_g_B_g_A_fake = d_A(self.A_img, self.A_g_B_g_A)

        with tf.variable_scope('A_g_B_loss'):
            self.A_g_B_loss = sigmoid_cross_entropy_mean(tf.ones_like(self.A_g_B_fake), self.A_g_B_fake) + \
                              huber_loss(self.A_g_B_g_A, self.A_img)
        with tf.variable_scope('d_B_loss'):
            self.d_B_loss = sigmoid_cross_entropy_mean(tf.ones_like(self.d_B_real), self.d_B_real) + \
                            sigmoid_cross_entropy_mean(tf.zeros_like(self.A_g_B_fake), self.A_g_B_fake)

        self.B_g_A = B_g_A(self.B_img, True)
        self.d_A_real, self.B_g_A_fake = d_A(self.A_img, self.B_g_A, True)
        self.B_g_A_g_B = A_g_B(self.B_g_A, True)
        _, self.B_g_A_g_B_fake = d_B(self.B_img, self.B_g_A_g_B, True)

        with tf.name_scope('B_g_A_loss'):
            self.B_g_A_loss = sigmoid_cross_entropy_mean(tf.ones_like(self.B_g_A_fake), self.B_g_A_fake) + \
                              huber_loss(self.B_g_A_g_B, self.B_img)
        with tf.name_scope('d_A_loss'):
            self.d_A_loss = sigmoid_cross_entropy_mean(tf.ones_like(self.d_A_real), self.d_A_real) + \
                            sigmoid_cross_entropy_mean(tf.zeros_like(self.B_g_A_fake), self.B_g_A_fake)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.g_optimizer = tf.train.AdamOptimizer(self.g_learning_rate) \
                .minimize(self.A_g_B_loss + self.B_g_A_loss, self.global_step,
                          get_trainable_collection('A_g_B') + get_trainable_collection('B_g_A'))

            self.d_optimizer = tf.train.AdamOptimizer(self.d_learning_rate) \
                .minimize(self.d_A_loss + self.d_B_loss, self.global_step,
                          get_trainable_collection('d_A') + get_trainable_collection('d_B'))

        with tf.name_scope('accuracy'):
            self.A_accuracy = tf.reduce_mean(
                tf.cast(tf.concat([tf.less(self.B_g_A_fake, 0.), tf.less(0., self.d_A_real)], 0), tf.float32))
            self.B_accuracy = tf.reduce_mean(
                tf.cast(tf.concat([tf.less(self.A_g_B_fake, 0.), tf.less(0., self.d_B_real)], 0), tf.float32))

        [print(param) for param in get_trainable_collection()]

    def __generate_image(self, name, feed_dict):
        print('Painting ...')
        A_g_B_, A_g_B_g_A_, B_g_A_, B_g_A_g_B_ = self.sess.run(
            [image_deprocess(self.A_g_B), image_deprocess(self.A_g_B_g_A),
             image_deprocess(self.B_g_A), image_deprocess(self.B_g_A_g_B)], feed_dict)
        # A_g_B_, B_g_A_ = self.sess.run([image_deprocess(self.A_g_B), image_deprocess(self.B_g_A)], feed_dict)
        # A_g_B_, B_g_A_ = self.sess.run([self.A_g_B, self.B_g_A], feed_dict)
        save_image(
            np.concatenate(
                [feed_dict[self.A][:16],
                 A_g_B_[:16],
                 A_g_B_g_A_[:16],
                 feed_dict[self.B][:16],
                 B_g_A_[:16],
                 B_g_A_g_B_[:16]]
            ),
            name=self.output_dir + name,
            n_each_row=8,
        )

    def train(self, image_A, image_B):
        print('Train ...')
        one_epoch_step = max(len(image_A), len(image_B)) // self.batch
        print('one_epoch_step ', one_epoch_step)
        image_A = image_A.reshape([-1, self.A_width, self.A_height, self.A_channel])
        image_B = image_B.reshape([-1, self.B_width, self.B_height, self.B_channel])
        sample_mask = self.sess.run(self.sample)
        sample_feed_dict = {
            self.A: image_A[sample_mask],
            self.B: image_B[sample_mask],
            self.is_train: False
        }
        valid_tensor = [self.A_g_B_loss, self.B_g_A_loss, self.d_A_loss, self.d_B_loss,
                        self.A_accuracy, self.B_accuracy]
        valid_dict, temp = {self.A: image_A[np.random.choice(min(len(image_A), len(image_B)), 500)],
                            self.B: image_B[np.random.choice(min(len(image_A), len(image_B)), 500)],
                            self.is_train: False}, {'result': [0] * 10}

        def training_valid(optimizer, feed_dict):
            i_globle_, n = self.sess.run(self.global_step), 3
            i_step_ = i_globle_ // n
            if i_globle_ % n == 0:
                print('\nstep {}'.format(i_step_))
            if False:  # 以下用于展示descent
                if i_step_ % 5 == 0:
                    print('{}: '.format(i_globle_ % n))
                    result = self.sess.run(valid_tensor, valid_dict)
                    print('A_g_B_loss: ', result[0])
                    print('B_g_A_loss: ', result[1])
                    print('d_A_loss: ', result[2])
                    print('d_B_loss: ', result[3])
                    print('A_accuracy: ', result[4])
                    print('B_accuracy: ', result[5])
                    if i_globle_ % n == 1:
                        self._write_history(tf.Summary(value=[
                            tf.Summary.Value(tag='d_A_loss_descent', simple_value=result[2] - temp['result'][2]),
                            tf.Summary.Value(tag='d_B_loss_descent', simple_value=result[3] - temp['result'][3]),
                        ]), i_step_)
                    elif i_globle_ % n == 2:
                        self._write_history(tf.Summary(value=[
                            tf.Summary.Value(tag='A_g_B_loss_descent', simple_value=result[0] - temp['result'][0]),
                            tf.Summary.Value(tag='B_g_A_loss_descent', simple_value=result[1] - temp['result'][1]),
                            tf.Summary.Value(tag='A_g_B_loss', simple_value=result[0]),
                            tf.Summary.Value(tag='B_g_A_loss', simple_value=result[1]),
                            tf.Summary.Value(tag='d_A_loss', simple_value=result[2]),
                            tf.Summary.Value(tag='d_B_loss', simple_value=result[3]),
                        ]), i_step_)
                    temp['result'] = result
            self.sess.run(optimizer, feed_dict)
            return i_step_

        while True:
            feed_dict = {
                self.A: image_A[np.random.choice(len(image_A), self.batch)],
                self.B: image_B[np.random.choice(len(image_B), self.batch)],
                self.is_train: True
            }

            training_valid(self.d_optimizer, feed_dict)
            for i in range(2):
                i_step = training_valid(self.g_optimizer, feed_dict)

            # sample_interval = one_epoch_step // 10
            sample_interval = 25
            if i_step % sample_interval == 0:
                self.__generate_image('sample_{}'.format(i_step // sample_interval), sample_feed_dict)
                # self.__generate_image('random_{}'.format(i_step // sample_interval), feed_dict)
            if i_step % 50 == 0:
                self.save_sess()

    def test(self, A_img, B_img):
        print('Test ...')
        self.__generate_image('test', feed_dict={
            self.A: A_img,
            self.B: B_img,
            self.is_train: False
        })


MODE = 'train'


def transform_mnist():
    from zhongrj.data.mnist import load_data as load_mnist
    from zhongrj.data.mnist_transform import load_data as load_mnist_transform

    model = DiscoGAN(
        name='DiscoGAN_transform_mnist',
        A_dims=[28, 28, 1],
        B_dims=[28, 28, 1],
        g_cnn_units=[16, 32, 16],
        d_cnn_units=[8, 16],
        d_dnn_units=[512, 128],
        g_learning_rate=1e-3,
        d_learning_rate=1e-3,
        batch=50
    )

    print('Loading data ...')
    mnist = load_mnist()['train_x']
    mnist_transform = load_mnist_transform()['train_x']

    if MODE == 'train':
        model.train(mnist, mnist_transform)
    elif MODE == 'test':
        pass


def transform_face():
    from zhongrj.data.lovely_girl import load_data as load_lovely_gril
    from zhongrj.data.anime_face import load_data as load_anime_face

    model = DiscoGAN(
        name='DiscoGAN_transform_face',
        A_dims=[48, 48, 3],
        B_dims=[48, 48, 3],
        # g_cnn_units=[50, 80, 50],
        # d_cnn_units=[20, 40],
        # d_dnn_units=[2000, 500],
        # g_learning_rate=1e-3,
        # d_learning_rate=1e-3,
        # g_cnn_units=[20, 20, 20],
        # d_cnn_units=[20, 20, 20],
        # d_dnn_units=[600, 200],
        # g_learning_rate=1e-3,
        # d_learning_rate=1e-3,
        g_cnn_units=[10, 10, 10],
        d_cnn_units=[10, 10],
        d_dnn_units=[600, 200],
        g_learning_rate=2e-4,
        d_learning_rate=2e-4,
        batch=100
    )

    print('Loading data ...')
    lovely_gril = load_lovely_gril()['train_x']
    anime_face = load_anime_face()['train_x']

    if MODE == 'train':
        model.train(lovely_gril, anime_face)
    elif MODE == 'test':
        A_imgs = lovely_gril[np.random.choice(len(lovely_gril), 16)]
        B_imgs = anime_face[np.random.choice(len(anime_face), 16)]
        for i in range(8):
            A_imgs[i] = resize(load_img('C:/Users/lenovo/Desktop/faces/test/{}.jpg'.format(i)),
                               (model.A_width, model.A_height))
        model.test(A_img=A_imgs, B_img=B_imgs)


if __name__ == '__main__':
    transform_face()
