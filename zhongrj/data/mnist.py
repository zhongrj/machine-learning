import tensorflow.examples.tutorials.mnist.input_data as input_data
from zhongrj.utils.path_util import *

FILE_DIR = get_file_dir(__file__)
STORE_DIR = FILE_DIR + 'MNIST_data/'
make_dir(STORE_DIR)


def load_data():
    """
    若无法下载, 手动下载：https://github.com/zrj19931211/resource/tree/master/data/MNIST_data
    """
    mnist = input_data.read_data_sets(STORE_DIR, one_hot=True)  # 下载并加载mnist数据
    return {
        'train_x': mnist.train.images,
        'train_y': mnist.train.labels,
        'test_x': mnist.test.images,
        'test_y': mnist.test.labels
    }
