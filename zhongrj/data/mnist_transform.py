import numpy as np
from zhongrj.utils.path_util import *
import zhongrj.data.mnist as mnist_data

FILE_DIR = get_file_dir(__file__)
STORE_DIR = FILE_DIR + 'MNIST_data_transform/'
make_dir(STORE_DIR)
FILE_NAME = 'mnist_transform_28X28.npz'


def load_data():
    """读取数据"""
    try:
        return np.load(STORE_DIR + FILE_NAME)
    except:
        print('找不到', STORE_DIR + FILE_NAME, ' 生成数据中 慢慢等吧 ...')
        return __store_data()


def __store_data():
    mnist = mnist_data.load_data()
    data = mnist['train_x']
    data_transform = []
    for i, img in enumerate(data):
        data_transform.append(1 - img)
        if i % 100 == 0:
            print('\r{}% 完成~'.format(i * 100 / len(data)), end='')

    data_transform = {
        'train_x': np.array(data_transform)
    }
    np.savez(STORE_DIR + FILE_NAME, **data_transform)
    return data_transform


if __name__ == '__main__':
    data = load_data()['train_x']

    from zhongrj.utils.view_util import show_image

    show_image(data[np.random.choice(len(data), 18)].reshape([-1, 28, 28, 1]))
