from zhongrj.model.BaseModel import BaseModel
from zhongrj.utils.model_util import *
from zhongrj.utils.tf_util import *
from zhongrj.utils.view_util import *

"""
    总结和问题：
        1.  batch_normalization时而好train时而难train, 这个问题有待思考
        2.  batch_normalization是分training和testing的, 因为mean和variance需要train
        3.  有无激活函数差别是很大的 (只有隐藏层应该放activation?)
        4.  先normalization后activation可以避免gradient vanishing?
        5.  tf.layers.conv2d不需要maxpooling? 根据strides来pooling?
        6.  params_initialize也是一件很重要的事情, 选择tf.truncated_normal_initializer()更好?
        9.  一个一直不懂的问题: 怎么调参...
            关于GAN调参：
                g和d的learning_rate设为不一样的值更好，分别观察其对loss的变化
                （试验中, 经常看到学来学去都是很模糊的图, 开始一直以为是g_learning_rate太小, 学得太慢，
                后来把g_learning_rate设得更小, 却意外的发现学得更快, g_loss下降得更快,
                仔细想想, 这件事也很合理, 事实上不会调得很小）
                我觉得应该不会有需要调参调到比1e-5小的时候, 需要的话可能是初始参数设置有问题
                后来成功并不是因为上面的这些总结, 调着调着参数就成功了, 上面只是为以后提个醒
                但是学到了一件事, 根据训练结果发现问题出在哪里...
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
        BaseModel.__init__(self, name)
        self.z_dims = z_dims
        self.x_width, self.x_height, self.x_channel = image_dims
        self.cnn_units = cnn_units
        self.dnn_units = dnn_units
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate
        self.batch = batch

        self.__build()
        self._init_sess(graph=True)

    def __build(self):
        with tf.name_scope('inputs'):
            self.z = tf.placeholder(tf.float32, [None, self.z_dims], name='z')
            self.x = tf.placeholder(tf.float32, [None, self.x_width * self.x_height * self.x_channel], name='x')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.act_image = tf.reshape(self.x, [-1, self.x_width, self.x_height, self.x_channel])
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
                                is_train=self.is_train)
        # self.gen_images = tf.nn.tanh(self.gen_images)
        self.g_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DeCNN')
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
                       reuse=reuse)

        self.y_generate = get_cnn_output(self.gen_images, False)
        self.y_actual = get_cnn_output(self.act_image, True)
        self.d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')
        [print(param) for param in self.d_params]

    def __def_optimizer(self):

        with tf.name_scope('g_loss'):
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.y_generate),
                                                                                 logits=self.y_generate))
        with tf.name_scope('d_loss'):
            self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.y_generate),
                                                                                 logits=self.y_generate)) + tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.y_actual), logits=self.y_actual))

        with tf.name_scope('optimizer'):
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
        save_image_join(self.sess.run(self.gen_images, {
            self.z: z,
            self.is_train: True  # False
        })[:64], name, n_each_row=8)

    def train(self, images):
        print('Train ...')
        one_epoch_step = len(images) // self.batch
        z_sample = np.random.normal(0, 1, size=(self.batch, self.z_dims)).astype(np.float32)

        valid_tensor = [self.global_step, self.g_loss, self.d_loss,
                        self.g_accuracy, self.d_accuracy, tf.summary.merge_all()]

        def show_training(result):
            i_global = result[1] - 1
            i_step = i_global // 3
            if i_step % 20 == 0:
                print('\nstep %s: ' % i_step)
                print('g_loss: ', result[2])
                print('d_loss: ', result[3])
                print('g_accuracy: ', result[4])
                print('d_accuracy: ', result[5])
            # self._write_history(result[6], i_global)
            return i_step

        while True:
            feed_dict = {
                self.z: np.random.normal(0, 1, size=(self.batch, self.z_dims)).astype(np.float32),
                self.x: images[np.random.choice(len(images), self.batch)],
                self.is_train: True
            }

            i_step = show_training(self.sess.run([self.d_optimizer] + valid_tensor, feed_dict))
            for i in range(2):
                i_step = show_training(self.sess.run([self.g_optimizer] + valid_tensor, feed_dict))

            save_interval = one_epoch_step // 20
            if i_step % save_interval == 0:
                self._save_sess()
                self.__generate_image(z_sample, self.output_dir + 'sample_%s' % (i_step // save_interval))
                self.__generate_image(feed_dict[self.z], self.output_dir + 'random_%s' % (i_step // save_interval))

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

    """
        可行的参数
            (1). 
                z_dims=100, image_dims=[28, 28, 1], cnn_units=[16, 32], dnn_units=[1024, 256],
                g_learning_rate=2e-3, d_learning_rate=1e-3, batch=200
    """

    model = DCGAN(
        name='generate_mnist',
        z_dims=100,
        image_dims=[28, 28, 1],
        cnn_units=[16, 32],
        dnn_units=[1024, 256],
        g_learning_rate=2e-3,
        d_learning_rate=1e-3,
        batch=200
    )

    print('Loading Data ...')
    data = load_data()

    if MODE == 'train':
        model.train(data['train_x'].reshape([-1, 28 * 28]))
    elif MODE == 'test':
        model.test()


if __name__ == '__main__':
    generate_mnist()
