from zhongrj.utils.view_util import *

FILE_DIR = get_file_dir(__file__)
STORE_DIR = FILE_DIR + 'mouse_track_data/'
make_dir(STORE_DIR)
FILE_NAME = 'mouse_track.npz'


def __show_data():
    """
    展示鼠标轨迹数据
    数据来源：黄国彬
    """
    data = read("C:/Users/lenovo/Desktop/data.txt")
    i = 0
    plt.figure(figsize=(16, 8))
    for line in data.split('\n'):
        spots = line.split(',')
        if len(spots) > 80:
            xs, ys, ts = [], [], []
            for spot in spots:
                try:
                    x, y, t = spot.split('|')
                    x, y, t = int(x), int(y), int(t)
                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                except:
                    continue
            plt.subplot(2, 4, i * 2+1)
            plt.title('X, Y随T散点图')
            plt.xlabel('Time')
            plt.ylabel('Position')
            # plt.scatter(ts, xs, marker='x', color='r', label='x')
            # plt.scatter(ts, ys, marker='o', color='b', label='y')
            # plt.legend(loc='upper right')
            plt.subplot(2, 4, i * 2 + 2)
            plt.title('轨迹图')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(0, 1500)
            plt.ylim(0, 1500)
            plt.scatter(xs, ys, marker='o', c=range(len(xs)), cmap=plt.cm.get_cmap('Reds'))
            i += 1
            if i == 4:
                break
    plt.show()


if __name__ == '__main__':
    __show_data()
