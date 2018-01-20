from zhongrj.model.BaseModel import *

"""
    总结和问题：
        1.  batch_normalization是分training和testing的, 因为mean和variance需要train
        2.  记得不要漏了激活函数
        3.  先normalization后activation可以避免gradient vanishing?
        4.  tf.layers.conv2d不需要maxpooling? 根据strides来pooling?
        5.  params_initialize也是一件很重要的事情, 选择tf.truncated_normal_initializer()更好?
        6.  LeakyReLU很重要, 原因不详
        7.  大概明白为什么图像要经过tanh的预处理了，为了使input有正有负（如果input全正或全负，所有的w会朝着同一个方向走，
            但如果w实际上是有正有负的，那么会绕很多弯路）至于scale到[-1, 1]，1有没有其他讲究就不清楚了。
        9.  关于GAN调参：
                （试验中, 经常看到学来学去都是很模糊的图, 开始一直以为是g_learning_rate太小, 学得太慢，
                后来把g_learning_rate设得更小, 却意外的发现学得更快, g_loss下降得更快,
                仔细想想, 这件事也很合理, 事实上不会调得很小）
                我觉得应该不会有需要调参调到比1e-5小的时候, 需要的话可能是初始参数设置有问题
                后来成功并不是因为上面的这些总结, 调着调着参数就成功了, 上面只是为以后提个醒
                但是学到了一件事, 根据训练结果发现问题出在哪里...
            大概学会了一点tensorboard调参了...
            如果1e-4都valid_loss都无法下降，证明模型有问题，或者说不适合gradient descent
"""


class DCGAN(BaseModel):
    def __init__(self,
                 name,
                 z_dims,
                 image_dims,
                 cnn_units,
                 dnn_units,
                 g_learning_rate,
                 d_learning_rate,
                 batch):
        BaseModel.__init__(self, name, batch,
                           sample_init=np.random.normal(0, 1, size=(64, z_dims)).astype(np.float32))
        self.z_dims = z_dims
        self.x_width, self.x_height, self.x_channel = image_dims
        self.cnn_units = cnn_units
        self.dnn_units = dnn_units
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate

        self.__build()
        self._init_sess(graph=False)

    def __build(self):
        with tf.name_scope('inputs'):
            self.z = tf.placeholder(tf.float32, [None, self.z_dims], name='z')
            self.x = tf.placeholder(tf.float32, [None, self.x_width * self.x_height * self.x_channel], name='x')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.act_image = tf.reshape(self.x, [-1, self.x_width, self.x_height, self.x_channel])
        self.act_image = image_preprocess(self.act_image)
        self.__build_generator()
        self.__build_discriminator()
        self.__def_optimizer()

    def __build_generator(self):
        print('Building DeCNN ...')
        self.gen_images = DeCNN(self.z,
                                [self.x_width, self.x_height, self.x_channel],
                                name='DeCNN',
                                cnn_units=self.cnn_units,
                                dnn_units=self.dnn_units,
                                batch_noraml=True,
                                act=lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU'),
                                is_train=self.is_train)
        self.gen_images = tf.nn.tanh(self.gen_images)
        self.g_params = get_trainable_collection('DeCNN')
        [print(param) for param in self.g_params]

    def __build_discriminator(self):
        print('Building CNN ...')

        def get_cnn_output(input_tensor, reuse):
            return CNN(input_tensor,
                       1,
                       name='CNN',
                       cnn_units=self.cnn_units,
                       dnn_units=self.dnn_units,
                       batch_noraml=True,
                       is_train=self.is_train,
                       act=lambda x: tf.maximum(x, 0.2 * x, name='leakyReLU'),
                       reuse=reuse)

        self.y_generate = get_cnn_output(self.gen_images, False)
        self.y_actual = get_cnn_output(self.act_image, True)
        self.d_params = get_trainable_collection('CNN')
        [print(param) for param in self.d_params]

    def __def_optimizer(self):

        with tf.name_scope('g_loss'):
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.y_generate),
                                                                                 logits=self.y_generate))
        with tf.name_scope('d_loss'):
            self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.y_generate),
                                                                                 logits=self.y_generate)) + tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.y_actual), logits=self.y_actual))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.g_optimizer = tf.train.AdamOptimizer(self.g_learning_rate) \
                .minimize(self.g_loss, self.global_step, var_list=self.g_params)
            self.d_optimizer = tf.train.AdamOptimizer(self.d_learning_rate) \
                .minimize(self.d_loss, self.global_step, var_list=self.d_params)

        with tf.name_scope('accuracy'):
            self.g_accuracy = tf.reduce_mean(tf.cast(tf.less(0., self.y_generate), tf.float32))
            self.d_accuracy = tf.reduce_mean(tf.cast(tf.concat([tf.less(self.y_generate, 0.),
                                                                tf.less(0., self.y_actual)], 0), tf.float32))

        tf.summary.histogram('g_loss', self.g_loss)
        tf.summary.histogram('d_loss', self.d_loss)

    def __generate_image(self, z, name):
        images = self.sess.run(image_deprocess(self.gen_images), {
            self.z: z,
            self.is_train: False
        })[:64]
        save_image_join(images, name, n_each_row=8)

    def train(self, images):
        print('Train ...')
        images = images.reshape(-1, self.x_width * self.x_height * self.x_channel)
        one_epoch_step = len(images) // self.batch
        print('one_epoch_step ', one_epoch_step)
        z_sample = self.sess.run(self.sample)

        valid_tensor, valid_dict, temp = [self.g_loss, self.d_loss, self.g_accuracy, self.d_accuracy], {
            self.z: np.random.normal(0, 1, size=(1000, self.z_dims)).astype(np.float32),
            self.x: images[np.random.choice(len(images), self.batch)],
            self.is_train: False
        }, {'result': [0] * 4}

        def training_valid(optimizer, feed_dict):
            i_globle_, n = self.sess.run(self.global_step), 3
            i_step_ = i_globle_ // n
            if i_globle_ % n == 0:
                print('\nstep {}'.format(i_step_))
            if False:  # 以下用于展示descent
                if i_step_ % 5 == 0:
                    print('{}: '.format(i_globle_ % n))
                    result = self.sess.run(valid_tensor, valid_dict)
                    print('g_loss: ', result[0])
                    print('d_loss: ', result[1])
                    print('g_accuracy: ', result[2])
                    print('d_accuracy: ', result[3])
                    if i_globle_ % n == 1:
                        self._write_history(tf.Summary(value=[
                            tf.Summary.Value(tag='d_loss_descent', simple_value=result[1] - temp['result'][1]),
                        ]), i_step_)
                    elif i_globle_ % n == 2:
                        self._write_history(tf.Summary(value=[
                            tf.Summary.Value(tag='g_loss_descent', simple_value=result[0] - temp['result'][0]),
                            tf.Summary.Value(tag='g_loss', simple_value=result[0]),
                            tf.Summary.Value(tag='d_loss', simple_value=result[1])
                        ]), i_step_)
                    temp['result'] = result
            self.sess.run(optimizer, feed_dict)
            return i_step_

        while True:
            feed_dict = {
                self.z: np.random.normal(0, 1, size=(self.batch, self.z_dims)).astype(np.float32),
                self.x: images[np.random.choice(len(images), self.batch)],
                self.is_train: True
            }

            training_valid(self.d_optimizer, feed_dict)
            for i in range(2):
                i_step = training_valid(self.g_optimizer, feed_dict)

            sample_interval = 50
            if i_step % sample_interval == 0:
                self.__generate_image(z_sample, self.output_dir + 'sample_%s' % (i_step // sample_interval))
                # self.__generate_image(feed_dict[self.z], self.output_dir + 'random_%s' % (i_step // sample_interval))
            if i_step % 100 == 0:
                self.save_sess()

    def test(self):
        print('Test ...')
        for i in range(10):
            self.__generate_image(
                np.random.normal(0, 1, size=(self.batch, self.z_dims)).astype(np.float32),
                self.output_dir + 'test_%s' % i
            )


MODE = 'train'


def generate_mnist():
    """手写数字生成"""
    from zhongrj.data.mnist import load_data

    model = DCGAN(
        name='DCGAN_generate_mnist',
        z_dims=100,
        image_dims=[28, 28, 1],
        cnn_units=[8, 16],
        dnn_units=[512, 256],
        g_learning_rate=1e-3,
        d_learning_rate=1e-3,
        batch=100
    )

    print('Loading Data ...')
    data = load_data()

    if MODE == 'train':
        model.train(data['train_x'])
    elif MODE == 'test':
        model.test()


def generate_anime_face():
    """生成动漫头像"""
    from zhongrj.data.anime_face import load_data

    model = DCGAN(
        name='DCGAN_generate_anime_face',
        z_dims=100,
        # image_dims=[24, 24, 1],
        # cnn_units=[16, 32],
        # dnn_units=[800, 300],
        # g_learning_rate=1e-3,
        # d_learning_rate=1e-3,
        # image_dims=[48, 48, 3],
        # cnn_units=[40, 80, 120],
        # dnn_units=[2000, 500],
        # g_learning_rate=2e-4,
        # d_learning_rate=2e-4,
        image_dims=[48, 48, 3],
        cnn_units=[20, 30, 40],
        dnn_units=[1000, 400],
        g_learning_rate=2e-4,
        d_learning_rate=2e-4,
        batch=64
    )

    print('Loading Data ...')
    data = load_data()

    if MODE == 'train':
        model.train(data['train_x'])
    elif MODE == 'test':
        model.test()


if __name__ == '__main__':
    generate_anime_face()
