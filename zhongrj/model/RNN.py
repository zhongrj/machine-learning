from zhongrj.model.BaseModel import *


class RNN(BaseModel):
    def __init__(self,
                 name,
                 x_dims,
                 y_classes,
                 input_units,
                 cell_units,
                 learning_rate,
                 batch):
        BaseModel.__init__(self, name, batch)
        self.x_step, self.input_dims = x_dims
        self.y_classes = y_classes
        self.input_units = input_units
        self.cell_units = cell_units
        self.learning_rate = learning_rate

        self.__build()
        self._init_sess(graph=True)

    def __build(self):
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.x_step, self.input_dims])
            self.y_actual = tf.placeholder(tf.float32, [None, self.y_classes])

        with tf.variable_scope('input_layer'):
            inputs = tf.reshape(self.x, [-1, self.input_dims])
            inputs = Dense(inputs, self.input_units)
            inputs = tf.reshape(inputs, [-1, self.x_step, self.input_units])

        with tf.variable_scope('rnn_cell'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.cell_units, forget_bias=1., state_is_tuple=True)
            mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
            init_state = mlstm_cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(mlstm_cell, inputs, initial_state=init_state, time_major=False)

        with tf.variable_scope('output_layer'):
            outputs = tf.transpose(outputs, [1, 0, 2])
            self.y_predict = Dense(outputs[-1], self.y_classes)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predict, labels=self.y_actual))

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y_predict, 1), tf.argmax(self.y_actual, 1)), tf.float32))

        [print(param) for param in get_trainable_collection()]

    def train(self, x, y):
        print('Train ...')

        x = x.reshape([-1, self.x_step, self.input_dims])

        for i in range(100):
            mask = np.random.choice(len(x), self.batch)
            feed_dict = {
                self.x: x[mask],
                self.y_actual: y[mask],
            }

            _, accuracy_ = self.sess.run([self.optimizer, self.accuracy], feed_dict)

            print(accuracy_)

        print('Total Accuracy: ', self.sess.run(self.accuracy, feed_dict={
            self.x: x[:2000],
            self.y_actual: y[:2000]
        }))


MODE = 'train'


def mnist_classify():
    from zhongrj.data.mnist import load_data

    model = RNN(
        name='RNN_mnist_classify',
        x_dims=[28, 28],
        y_classes=10,
        input_units=128,
        cell_units=128,
        learning_rate=1e-2,
        batch=100
    )

    print('Loading data ...')
    data = load_data()

    if MODE == 'train':
        model.train(data['train_x'], data['train_y'])
    elif MODE == 'test':
        pass


if __name__ == '__main__':
    mnist_classify()
