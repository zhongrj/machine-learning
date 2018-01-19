from zhongrj.model.BaseModel import *

"""
    总结和问题：
"""


class DQN(BaseModel):
    def __init__(self,
                 name,
                 n_features,
                 n_actions,
                 dnn_units,
                 learning_rate,
                 memory_size,
                 batch,
                 assign_step=50,
                 reward_decay=0.9,
                 random_p=0.1):
        BaseModel.__init__(self, name, batch)
        self.n_features, self.n_actions = n_features, n_actions
        self.dnn_units = dnn_units
        self.learning_rate = learning_rate
        self.memory = np.zeros((memory_size, n_features + 3 + n_features))
        self.memory_size, self.memory_i = memory_size, 0

        self.assign_step, self.reward_decay, self.random_p = assign_step, reward_decay, random_p

        self.__build()
        self._init_sess()

    def __build(self):
        with tf.name_scope('inputs'):
            self.o = tf.placeholder(tf.float32, [None, self.n_features])
            self.a = tf.placeholder(tf.int32, [None, ])
            self.r = tf.placeholder(tf.float32, [None, ])
            self.d = tf.placeholder(tf.bool, [None, ])
            self.o_ = tf.placeholder(tf.float32, [None, self.n_features])
            self.is_train = tf.placeholder(tf.bool)

        self.eval = self.__build_dnn(self.o, 'NEW')
        self.eval_ = self.__build_dnn(self.o_, 'OLD')

        self.assign_op = [tf.assign(old, new) for old, new in
                          zip(get_trainable_collection('OLD'), get_trainable_collection('NEW'))]

        with tf.name_scope('loss'):
            # loss只取action(这里应该可以用稀疏矩阵)
            indices = tf.concat([tf.range(tf.shape(self.a)[0])[:, tf.newaxis],
                                 self.a[:, tf.newaxis]], 1)
            # 结束时o_的reward为0
            done = tf.cast(~self.d, tf.float32)
            self.loss = tf.reduce_mean(tf.squared_difference(
                tf.gather_nd(self.eval, indices),
                self.reward_decay * tf.reduce_max(self.eval_, 1) * done + self.r
            ))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.loss, self.global_step, var_list=get_trainable_collection('NEW'))

        [print(param) for param in get_trainable_collection()]

    def __build_dnn(self, observation, scope):
        with tf.variable_scope(scope):
            return CNN(observation,
                       self.n_actions,
                       dnn_units=self.dnn_units,
                       batch_noraml=True,
                       is_train=self.is_train)

    def store_transition(self, o, a, r, d, o_):
        self.memory[self.memory_i % self.memory_size] = np.hstack((o, a, r, d, o_))
        self.memory_i += 1

    def train(self):
        mask = np.random.choice(min(self.memory_i, self.memory_size), self.batch)
        batch = self.memory[mask]
        feed_dict = {
            self.o: batch[:, :self.n_features],
            self.a: batch[:, self.n_features],
            self.r: batch[:, self.n_features + 1],
            self.d: batch[:, self.n_features + 2],
            self.o_: batch[:, -self.n_features:],
            self.is_train: True
        }
        _, i_global, loss = self.sess.run([self.optimizer, self.global_step, self.loss], feed_dict)
        if i_global % self.assign_step == 0:
            print('Assign ...')
            self.sess.run(self.assign_op)
        if i_global % 1000 == 0:
            self.save_sess()

    def choose_step(self, observation, random, tips=None):
        probability = np.random.uniform()
        if random and probability < self.random_p:
            return tips(observation) if tips else np.random.randint(0, self.n_actions)
        actions_value = self.sess.run(self.eval, feed_dict={
            self.o: observation[np.newaxis, :],
            self.is_train: False
        })
        return np.argmax(actions_value)


MODE = 'train'


def maze():
    from zhongrj.reference.maze_env import Maze
    env = Maze()

    model = DQN(
        name='DQN_MAZE',
        n_features=env.n_features,
        n_actions=env.n_actions,
        dnn_units=[10],
        learning_rate=1e-2,
        memory_size=2000,
        batch=100
    )

    if MODE == 'train':
        Critic.train(env, model, render=False)
    elif MODE == 'test':
        Critic.play(env, model, random=False)


def mountain_car():
    import gym
    env = gym.make('MountainCar-v0').unwrapped

    model = DQN(
        name='DQN_MountainCar',
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        dnn_units=[20, 40, 20],
        learning_rate=4e-4,
        memory_size=2000,
        batch=100,
        random_p=0.05
    )

    if MODE == 'train':
        Critic.train(env, model, render=False,
                     # render_threshold=-100,
                     # tips=lambda o: 0 if o[1] < 0 else 2
                     )
    elif MODE == 'test':
        Critic.play(env, model, random=False)


def cart_pole():
    import gym
    env = gym.make('CartPole-v0').unwrapped

    model = DQN(
        name='DQN_CartPole',
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        dnn_units=[20, 40, 20],
        learning_rate=4e-4,
        memory_size=2000,
        batch=100,
        random_p=0.05
    )

    if MODE == 'train':
        Critic.train(env, model, render=False, render_threshold=4000)
    elif MODE == 'test':
        Critic.play(env, model, random=False)


if __name__ == '__main__':
    mountain_car()
