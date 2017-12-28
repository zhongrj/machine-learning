import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw
import cv2
import numpy as np
from zhongrj.utils.path_util import *

PROJECT_DIR = get_project_dir()
output_dir = PROJECT_DIR + 'output/'
make_dir(output_dir)


def show_image(image_list, n_each_row=6, cmap='gray'):
    """
    展示图片
    :param image_list: 图像列表
    :param n_each_row: 每行数量
    :param cmap: 
    :return: 
    """
    __draw_image(image_list, n_each_row, cmap)
    plt.show()


def save_image(image_list, name='temp', n_each_row=6, cmap='gray'):
    """
    生成图片
    :param image_list: 图像列表
    :param name: 文件名
    :param n_each_row: 每行数量
    :param cmap: 色调?
    :return: 
    """
    __draw_image(image_list, n_each_row, cmap)
    plt.savefig(output_dir + '%s.png' % name, bbox_inches='tight')


def show_dynamic_image(data_gen, n_each_row=6, cmap='gray', interval=1000):
    """
    动态画图
    :param data_gen: 数据生成器
    :param n_each_row: 每行数量
    :param cmap: 
    :param interval: 间隔
    :return: 
    """
    fig = plt.figure()

    def update_(data):
        __draw_image(data, n_each_row, cmap)

    ani = animation.FuncAnimation(fig, update_, data_gen, interval=interval)
    plt.show()


def draw_rectangle(image, first, twice, third, fourth, color):
    first = tuple([first[i] for i in range(2)])
    twice = tuple([twice[i] for i in range(2)])
    third = tuple([third[i] for i in range(2)])
    fourth = tuple([fourth[i] for i in range(2)])

    im = Image.fromarray(image.squeeze())
    draw = ImageDraw.Draw(im)
    draw.line((first, twice), color)
    draw.line((twice, third), color)
    draw.line((third, fourth), color)
    draw.line((fourth, first), color)
    return np.array(im)


def __draw_image(image_list, n_each_row, cmap):
    """
    画图
    :param image_list: 
    :param n_each_row: 
    :param cmap: 
    :return: 
    """
    n = len(image_list)
    col = n_each_row
    row = int(n / col) + 1
    for i in range(n):
        plt.subplot(row, col, i + 1)
        image = image_list[i]
        image = image.squeeze()
        plt.imshow(image, cmap=cmap)
        plt.axis('off')


if __name__ == '__main__':
    image = np.zeros([20, 40])
    image = draw_rectangle(image, np.array([1, 1]), (1, 15), [10, 20], [12, 3], 1)
    print(image[1][15])
    show_image([image])
