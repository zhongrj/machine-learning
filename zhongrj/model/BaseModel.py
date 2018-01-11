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

    def _save_sess(self):
        make_dir(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir)
        print('Saved Success .\n')

    def __save_graph(self, graph):
        # tensorboard --logdir zhongrj/model/***_logs/
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph if graph else None)

    def _write_history(self, merge, step):
        self.writer.add_summary(merge, step)
