import numpy as np
from zhongrj.utils.path_util import *

FILE_DIR = get_file_dir(__file__)
STORE_DIR = FILE_DIR + 'catvsdog_data/'
make_dir(STORE_DIR)
FILE_NAME = 'catvsdog_150x150.npz'


def load_data():
    """读取数据"""
    try:
        return np.load(STORE_DIR + FILE_NAME)
    except:
        raise FileNotFoundError('找不到', STORE_DIR + FILE_NAME, ' 自己想办法生成吧...')
