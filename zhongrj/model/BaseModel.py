from zhongrj.utils.tf_util import *
from zhongrj.utils.path_util import *

FILE_DIR = get_file_dir(__file__)
CHECKPOINT_DIR = FILE_DIR + '%s_checkpoint/sess'
LOG_DIR = FILE_DIR + '%s_logs/'


class BaseModel:
    def __init__(self, name):
        self.name = name
        self.output_dir = self.name + '/'
        self.checkpoint_dir = CHECKPOINT_DIR % self.name
        self.log_dir = LOG_DIR % self.name
        self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    def _init_sess(self, graph=False):
        self.saver = tf.train.Saver()
        self.sess = init_sess(self.saver, self.checkpoint_dir)
        if graph:
            self.__save_graph()

    def _save_sess(self):
        self.saver.save(self.sess, self.checkpoint_dir)
        print('Saved Success .\n')

    def __save_graph(self):
        # tensorboard --logdir zhongrj/model/***_logs/
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def _write_history(self, merge, step):
        self.writer.add_summary(merge, step)
