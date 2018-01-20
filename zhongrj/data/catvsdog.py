import matplotlib.image as mpimg
from zhongrj.utils.view_util import *

FILE_DIR = get_file_dir(__file__)
STORE_DIR = FILE_DIR + 'catvsdog_data/'
make_dir(STORE_DIR)
FILE_NAME = 'catvsdog_150x150x3.npz'


def load_data():
    """
    猫狗图片
    数据来源：kaggle 猫狗大战
    """
    try:
        return np.load(STORE_DIR + FILE_NAME)
    except:
        raise FileNotFoundError('找不到', STORE_DIR + FILE_NAME, ' 自己想办法生成吧...')


def __store_data():
    x, y = [], []

    for (path, dirs, files) in os.walk('C:/Users/lenovo/Desktop/训练数据集/猫狗大战/train'):
        for i in range(len(files)):
            image = mpimg.imread(path + "/" + files[i])
            image = resize(image, (150, 150))
            x.append(image)
            y.append([1, 0] if 'cat' in files[i] else [0, 1])
            if i % 100 == 0:
                print("\r{}% 完成~".format(i * 100 / len(files)), end='')

    x, y = np.array(x), np.array(y)
    train_x, train_y = np.vstack((x[:10000], x[-10000:])), np.vstack((y[:10000], y[-10000:]))
    test_x, test_y = x[10000:15000], y[10000:15000]
    np.savez(STORE_DIR + FILE_NAME, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


if __name__ == '__main__':
    # __store_data()

    data = load_data()['train_x']
    show_image(data[np.random.choice(len(data), 18)])
