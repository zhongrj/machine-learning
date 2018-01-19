from zhongrj.utils.model_util import *
from zhongrj.utils.view_util import *

PROJECT_DIR = get_project_dir()
CHECKPOINT_DIR = PROJECT_DIR + 'checkpoint/%s/'
LOG_DIR = PROJECT_DIR + 'logs/%s/'


class BaseModel:
    def __init__(self, name, batch=None, sample_init=None, sample_num=64):
        self.name = name
        self.batch = batch
        self.output_dir = self.name + '/'
        self.checkpoint_dir = CHECKPOINT_DIR % self.name
        self.log_dir = LOG_DIR % self.name
        self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

        # 存储sample数据
        if sample_init is None:
            sample_init = np.random.choice(5000, sample_num)
        self.sample = tf.Variable(
            initial_value=sample_init,
            name='sample_mask',
            trainable=False
        )

    def _init_sess(self, graph=False):
        self.saver = tf.train.Saver()
        self.sess = init_sess(self.saver, self.checkpoint_dir)
        self.__save_graph(graph)

    def save_sess(self):
        make_dir(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir)
        print('Saved Success .\n')

    def __save_graph(self, graph):
        # tensorboard --logdir zhongrj/model/***_logs/
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph if graph else None)

    def _write_history(self, merge, step):
        self.writer.add_summary(merge, step)


class Critic:
    @staticmethod
    def train(env, model, render=False, render_threshold=np.inf, tips=None):
        i = 0
        while True:
            total_reward = 0
            observation = env.reset()
            while True:
                i += 1
                if render:
                    env.render()
                action = model.choose_step(observation, True, tips=tips)
                observation_, reward, done, info = env.step(action)
                model.store_transition(observation, action, reward, done, observation_)
                observation = observation_
                total_reward += reward
                if i > model.batch and i % 5 == 0:
                    model.train()
                if done:
                    print('done ----------------------------- total_reward: {}'.format(total_reward))
                    if total_reward > render_threshold:
                        model.save_sess()
                        Critic.play(env, model, False)
                    break

    @staticmethod
    def play(env, model, random=True):
        while True:
            total_reward = 0
            observation = env.reset()
            while True:
                env.render()
                action = model.choose_step(observation, random)
                observation_, reward, done, info = env.step(action)
                observation = observation_
                total_reward += reward
                if done:
                    print('done ----------------------------- total_reward: {}'.format(total_reward))
                    break


class Actor:
    @staticmethod
    def train(env, model, render=False, render_threshold=np.inf, tips=None):
        i = 0
        while True:
            total_reward = 0
            observation = env.reset()
            while True:
                i += 1
                if render:
                    env.render()
                action = model.choose_step(observation, tips=tips)
                observation_, reward, done, info = env.step(action)
                model.store_transition(observation, action, reward)
                observation = observation_
                total_reward += reward
                if done:
                    print('done ----------------------------- total_reward: {}'.format(total_reward))
                    model.train()
                    if total_reward > render_threshold:
                        model.save_sess()
                        Critic.play(env, model, False)
                    break

    @staticmethod
    def play(env, model):
        while True:
            total_reward = 0
            observation = env.reset()
            while True:
                env.render()
                action = model.choose_step(observation)
                observation_, reward, done, info = env.step(action)
                observation = observation_
                total_reward += reward
                if done:
                    print('done ----------------------------- total_reward: {}'.format(total_reward))
                    break
