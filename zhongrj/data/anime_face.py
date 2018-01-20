import matplotlib.image as mpimg
from zhongrj.utils.view_util import *

FILE_DIR = get_file_dir(__file__)
STORE_DIR = FILE_DIR + 'anime_face_data/'
make_dir(STORE_DIR)
FILE_NAME = 'anime_face_48x48x3.npz'


def load_data():
    """
    二次元妹子
    数据来源：https://zhuanlan.zhihu.com/p/24767059
    """
    try:
        return np.load(STORE_DIR + FILE_NAME)
    except:
        print('找不到', STORE_DIR + FILE_NAME)


def __store_data():
    train_x = []

    for (path, dirs, files) in os.walk('C:/Users/lenovo/Desktop/anime_faces'):
        for i in range(len(files)):
            image = mpimg.imread(path + "/" + files[i])
            image = resize(image, (48, 48))
            # image = bgr2gray(image)
            train_x.append(image)
            if i % 100 == 0:
                print("\r{}% 完成~".format(i * 100 / len(files)), end='')

    np.savez(STORE_DIR + FILE_NAME, train_x=np.array(train_x))


if __name__ == '__main__':
    # __store_data()

    data = load_data()['train_x']
    show_image(data[np.random.choice(len(data), 18)])
