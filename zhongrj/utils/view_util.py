import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw
import cv2
import numpy as np
from zhongrj.utils.path_util import *

PROJECT_DIR = get_project_dir()
output_dir = PROJECT_DIR + 'output/'
make_dir(output_dir)


def show_image(image_list, n_each_row=6, cmap='gray', text=None):
    """
    展示图片
    :param image_list: 图像列表
    :param n_each_row: 每行数量
    :param cmap: 
    :return: 
    """
    __draw_image(image_list, n_each_row, cmap, text)
    plt.show()


def join_images(image_list, n_each_row=6):
    """
    合成图片
    :param image_list: 图像列表
    :param n_each_row: 每行数量
    :return: 合成图
    """
    n = len(image_list)
    col = n_each_row
    row = (n - 1) // col + 1

    shape = list(image_list[0].squeeze().shape)
    new_shape = [row * shape[0], col * shape[1]] + shape[2:]

    result = np.zeros(new_shape)
    for i in range(n):
        r, c = i // col, i % col
        result[r * shape[0]: (r + 1) * shape[0], c * shape[1]:(c + 1) * shape[1]] = image_list[i].squeeze()

    return result


def save_image_join(image_list, name='temp', n_each_row=6, cmap='gray', text=None):
    save_image([join_images(image_list, n_each_row=n_each_row)], name, 1, cmap=cmap, text=text)


def save_image(image_list, name='temp', n_each_row=6, cmap='gray', text=None):
    """
    生成图片
    :param image_list: 图像列表
    :param name: 文件名
    :param n_each_row: 每行数量
    :param cmap: 色调?
    :param text: 文本标注
    :return: 
    """
    image_list = [im if len(im.shape) == 2 or im.shape[2] == 1 else im.astype(np.uint8) for im in image_list]
    __draw_image(image_list, n_each_row, cmap, text)
    file = output_dir + '%s.jpg' % name
    make_dir(get_elder(file))
    plt.savefig(file, bbox_inches='tight')
    plt.clf()


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


def draw_rectangle(image, first, second, third, fourth, color):
    first = tuple([first[i] for i in range(2)])
    second = tuple([second[i] for i in range(2)])
    third = tuple([third[i] for i in range(2)])
    fourth = tuple([fourth[i] for i in range(2)])

    image = image.squeeze().copy()
    thickness = int(image.shape[0] / 50) + 1
    cv2.line(image, first, second, color, thickness)
    cv2.line(image, second, third, color, thickness)
    cv2.line(image, third, fourth, color, thickness)
    cv2.line(image, fourth, first, color, thickness)
    return image

    # im = Image.fromarray(image.squeeze())
    # draw = ImageDraw.Draw(im)
    # draw.line((first, second), color)
    # draw.line((second, third), color)
    # draw.line((third, fourth), color)
    # draw.line((fourth, first), color)
    # return np.array(im)


def __draw_image(image_list, n_each_row, cmap, text=None):
    """
    画图
    :param image_list: 
    :param n_each_row: 
    :param cmap: 
    :param text: 
    :return: 
    """
    n = len(image_list)
    col = n_each_row
    row = (n - 1) // col + 1
    if text is not None:
        row = row + (len(text) - 1) // col + 1
    if row > 1:
        plt.figure(figsize=(col, row))
    for i in range(n):
        plt.subplot(row, col, i + 1)
        plt.axis('off')
        image = image_list[i]
        image = image.squeeze()
        plt.imshow(image, cmap=cmap)
    if text is not None:
        for i in range(len(text)):
            plt.subplot(row, col, n + i + 1)
            plt.axis('off')
            plt.text(0, 0, text[i], fontdict={'size': 12})


if __name__ == '__main__':
    image = np.zeros([20, 40])
    image = draw_rectangle(image, np.array([1, 1]), (1, 15), [10, 20], [12, 3], 1)

    image2 = np.zeros([20, 40])
    image2 = draw_rectangle(image2, np.array([1, 1]), (1, 15), [10, 20], [12, 3], 1)
    show_image([join_images([image, image2], 3)], 1)
