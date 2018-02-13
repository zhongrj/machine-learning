from zhongrj.utils.view_util import *
import zhongrj.data.mnist as mnist_data

FILE_DIR = get_file_dir(__file__)
STORE_DIR = FILE_DIR + 'MNIST_data_multi/'
make_dir(STORE_DIR)
FILE_NAME = 'mnist_multi_160x160.npz'


def load_data():
    """读取数据"""
    try:
        return np.load(STORE_DIR + FILE_NAME)
    except:
        print('找不到', STORE_DIR + FILE_NAME, ' 生成数据中 慢慢等吧 ...')
        return __store_data()


def __store_data():
    mnist = mnist_data.load_data()
    mnist_train_x, mnist_train_y = mnist['train_x'].reshape([-1, 28, 28]), mnist['train_y']

    train_x, train_y = [], []
    for i in range(10):
        image, labels = __create_multi_mnist(mnist_train_x, mnist_train_y)
        for label in labels:
            image = draw_rectangle(image, label[1:3], label[3:5], 1, text=str(label[0]))
        train_x.append(image)
        train_y.append(labels)

    show_image(train_x[:1], 1)


def __create_multi_mnist(mnist_x, mnist_y):
    n = np.random.randint(2, 10)
    w, h = 240, 240
    sample, labels = np.zeros((w, h)), []
    mask = np.random.choice(len(mnist_x), n)
    for image, label in zip(mnist_x[mask], mnist_y[mask]):
        scale = np.random.uniform(0.5, 2)
        w_, h_ = int(np.random.randint(15, 25) * scale), int(np.random.randint(15, 25) * scale)
        image = resize(image, (w_, h_))
        x, y = np.random.randint(0, w - w_), np.random.randint(0, h - h_)
        sample[y:y + h_, x:x + w_] = np.max(np.dstack((image, sample[y:y + h_, x:x + w_])), axis=2)
        labels.append([label.argmax(), x, y, x + w_, y + h_])
    return sample, labels


if __name__ == '__main__':
    __store_data()

    # import zhongrj.utils.view_util as view
    #
    # mnist_multi = load_data()
    # view.show_image(mnist_multi['train_x'][:20], 10)
