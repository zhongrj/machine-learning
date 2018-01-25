from zhongrj.model.BaseModel import *


class DDPG(BaseModel):
    def __init__(self,
                 name,
                 n_features,
                 n_actions,
                 dnn_units,
                 learning_rate,
                 reward_decay=0.9):
        BaseModel.__init__(self, name)
        self.n_features, self.n_actions = n_features, n_actions
        self.dnn_units = dnn_units
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.memory = []

        self.__build()
        self._init_sess()

    def __build(self):
        with tf.name_scope('inputs'):
            self.o = tf.placeholder(tf.float32, [None, self.n_features])
            self.a = tf.placeholder(tf.int32, [None, ])
            self.r = tf.placeholder(tf.float32, [None, ])
            self.is_train = tf.placeholder(tf.bool)

        logits = CNN(self.o,
                     self.n_actions,
                     dnn_units=self.dnn_units,
                     act=tf.nn.tanh,
                     batch_noraml=True,
                     is_train=self.is_train)
        self.predict = tf.nn.softmax(logits)

        with tf.name_scope('loss'):
            # a_prob = tf.reduce_sum(-tf.log(self.predict) * tf.one_hot(self.a, self.n_actions), axis=1)
            a_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.a)
            self.loss = tf.reduce_mean(a_prob * self.r)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)), tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)

        [print(param) for param in get_trainable_collection()]

    def store_transition(self, o, a, r):
        self.memory.insert(0, np.hstack((o, a, r)))

    def train(self):
        batch, next_reward = np.array(self.memory), 0
        for row in batch:
            row[self.n_features + 1] = next_reward = row[self.n_features + 1] + next_reward * self.reward_decay
        rewards = batch[:, self.n_features + 1]
        batch[:, self.n_features + 1] = (rewards - rewards.mean()) / rewards.std()

        _, i_global, loss = self.sess.run([self.optimizer, self.global_step, self.loss], {
            self.o: batch[:, :self.n_features],
            self.a: batch[:, self.n_features],
            self.r: batch[:, self.n_features + 1],
            self.is_train: True
        })

        self.memory = []
        if i_global % 100 == 0:
            self.save_sess()

    def choose_step(self, observation, tips=None):
        if tips and np.random.uniform() < 0.1:
            return tips(observation)
        predict = self.sess.run(self.predict, {
            self.o: observation[np.newaxis],
            self.is_train: False
        })
        return np.random.choice(self.n_actions, p=predict.ravel())


MODE = 'train'


def maze():
    from zhongrj.reference.maze_env import Maze
    env = Maze()

    model = DDPG(
        name='DDPG_MAZE',
        n_features=env.n_features,
        n_actions=env.n_actions,
        dnn_units=[10],
        learning_rate=1e-2
    )

    if MODE == 'train':
        Actor.train(env, model, render=True)
    elif MODE == 'test':
        Actor.play(env, model)


def mountain_car():
    import gym
    env = gym.make('MountainCar-v0').unwrapped

    model = DDPG(
        name='DDPG_MountainCar',
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        dnn_units=[20, 20],
        learning_rate=1e-2,
    )

    if MODE == 'train':
        Actor.train(env, model, render=False,
                    render_threshold=-100,
                    tips=lambda o: 0 if o[1] < 0 else 2
                    )
    elif MODE == 'test':
        Actor.play(env, model)


def cart_pole():
    import gym
    env = gym.make('CartPole-v0').unwrapped

    model = DDPG(
        name='DDPG_CartPole',
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        dnn_units=[20, 20],
        learning_rate=1e-2,
    )

    if MODE == 'train':
        Actor.train(env, model, render=False, render_threshold=1000)
    elif MODE == 'test':
        Actor.play(env, model)


if __name__ == '__main__':
    mountain_car()
